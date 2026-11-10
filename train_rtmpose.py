"""Download and fine-tune RTMPose-S on the converted UFC/MMA dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rtmpose_s_combat.py"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "generated" / "rtmpose_coco" / "annotations"
DEFAULT_WORK_DIR = PROJECT_ROOT / "work_dirs" / "rtmpose_s_combat"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "rtmpose-s-coco.pth"
DEFAULT_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root prepended to image paths stored in the COCO JSON files.",
    )
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable AMP. Default: enabled on CUDA and disabled on CPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit training to the first N samples for a fast compatibility test.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=None,
        help="Limit validation to the first N samples for a fast metric check.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Disable validation during a short smoke test.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than zero")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    for option in ("train_samples", "validation_samples"):
        if getattr(args, option) is not None and getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be greater than zero")
    return args


def download_checkpoint(url: str, destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    if destination.is_file() and destination.stat().st_size > 1_000_000:
        print(f"Using existing checkpoint: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading RTMPose-S checkpoint to: {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded / total:6.1%}", end="\r", flush=True)
    if temporary.stat().st_size <= 1_000_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded checkpoint is unexpectedly small")
    temporary.replace(destination)
    print(f"Downloaded {destination.stat().st_size / 1024 / 1024:.1f} MiB")
    return destination


def validate_data(annotation_dir: Path, data_root: Path) -> None:
    expected = {
        "train": (4074, 8094),
        "val": (519, 1035),
        "test": (513, 1026),
    }
    for split, counts in expected.items():
        path = annotation_dir / f"{split}.json"
        if not path.is_file():
            raise RuntimeError(
                f"Missing {path}. Run 'python convert_to_coco.py' before training."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = (len(payload.get("images", [])), len(payload.get("annotations", [])))
        if found != counts:
            raise RuntimeError(f"{split}: expected image/pose counts {counts}, found {found}")
        missing = [
            image["file_name"]
            for image in payload["images"]
            if not (data_root / image["file_name"]).is_file()
        ]
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} COCO image paths do not exist")


def configure_device(requested: str) -> tuple[str, bool]:
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a platform-appropriate PyTorch build first."
        ) from exc
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
    return device, cuda_available


def build_config(args: argparse.Namespace, checkpoint: Path, device: str):
    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "OpenMMLab dependencies are missing. Install requirements-mmpose.txt."
        ) from exc

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Config not found: {config_path}")
    cfg = Config.fromfile(config_path)
    data_root = args.data_root.expanduser().resolve()
    annotation_dir = args.annotation_dir.expanduser().resolve()
    metadata_path = (PROJECT_ROOT / "custom_combat_skeleton.py").resolve()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg.work_dir = str(work_dir)
    cfg.launcher = "none"
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = max(1, min(5, args.epochs))
    cfg.default_hooks.checkpoint.interval = max(1, min(5, args.epochs))
    if os.name == "nt":
        cfg.env_cfg.mp_cfg.mp_start_method = "spawn"

    for loader_name, split in (
        ("train_dataloader", "train"),
        ("val_dataloader", "val"),
        ("test_dataloader", "test"),
    ):
        loader = cfg[loader_name]
        loader.batch_size = args.batch_size
        loader.num_workers = args.workers
        loader.persistent_workers = args.workers > 0
        loader.dataset.data_root = str(data_root)
        loader.dataset.ann_file = str(annotation_dir / f"{split}.json")
        loader.dataset.data_prefix = dict(img="")
        loader.dataset.metainfo = dict(from_file=str(metadata_path))

    if args.train_samples is not None:
        cfg.train_dataloader.dataset.indices = args.train_samples
    if args.validation_samples is not None:
        cfg.val_dataloader.dataset.indices = args.validation_samples

    if args.skip_validation:
        cfg.train_cfg.val_interval = args.epochs + 1
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.default_hooks.checkpoint.pop("save_best", None)
        cfg.default_hooks.checkpoint.pop("rule", None)

    if not args.skip_validation:
        cfg.val_evaluator.ann_file = str(annotation_dir / "val.json")
    cfg.test_evaluator.ann_file = str(annotation_dir / "test.json")

    scaled_lr = 5e-4 * args.batch_size / 64.0
    cfg.optim_wrapper.optimizer.lr = scaled_lr
    use_amp = args.amp if args.amp is not None else device == "cuda"
    if use_amp and device != "cuda":
        raise RuntimeError("AMP is only supported by this workflow on CUDA")
    if use_amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
    else:
        cfg.optim_wrapper.type = "OptimWrapper"
        cfg.optim_wrapper.pop("loss_scale", None)

    for scheduler in cfg.param_scheduler:
        if scheduler.type == "CosineAnnealingLR":
            scheduler.end = args.epochs
            scheduler.T_max = args.epochs
            scheduler.eta_min = scaled_lr * 0.05
    for hook in cfg.custom_hooks:
        if hook.type == "mmdet.PipelineSwitchHook":
            hook.switch_epoch = max(1, args.epochs - 10)

    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise RuntimeError(f"Resume checkpoint not found: {resume_path}")
        cfg.resume = True
        cfg.load_from = str(resume_path)
    else:
        cfg.resume = False
        cfg.load_from = str(checkpoint)

    print(f"Device: {device}; AMP: {use_amp}; batch: {args.batch_size}; lr: {scaled_lr:g}")
    print(f"Work directory: {work_dir}")
    return cfg


def main() -> int:
    args = parse_args()
    try:
        checkpoint = download_checkpoint(args.checkpoint_url, args.checkpoint)
        if args.download_only:
            return 0
        device, _ = configure_device(args.device)
        data_root = args.data_root.expanduser().resolve()
        annotation_dir = args.annotation_dir.expanduser().resolve()
        validate_data(annotation_dir, data_root)
        cfg = build_config(args, checkpoint, device)

        from mmengine.runner import Runner

        runner = Runner.from_cfg(cfg)
        runner.train()
        work_dir = Path(cfg.work_dir)
        best = sorted(work_dir.glob("best_coco_AP*.pth"), key=lambda path: path.stat().st_mtime)
        latest = work_dir / "last_checkpoint"
        if best:
            print(f"Best checkpoint: {best[-1].resolve()}")
        if latest.is_file():
            print(f"Latest checkpoint pointer: {latest.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
