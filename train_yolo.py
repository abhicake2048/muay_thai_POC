"""Prepare detector labels from UFC pose annotations and train YOLO11s.

The source dataset primarily contains YOLO-Pose labels, plus a small number of
box-only rows. This script creates a separate, detector-only dataset by
retaining the class and bounding-box fields from each row. Source images and
labels are never modified.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "mma-fighter-pose-estimation-dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "yolo_detector"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "yolo_detector"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}
EXPECTED_KEYPOINTS = 17
EXPECTED_KEYPOINT_DIMS = 3
EXPECTED_POSE_FIELDS = 1 + 4 + EXPECTED_KEYPOINTS * EXPECTED_KEYPOINT_DIMS


class DatasetError(RuntimeError):
    """Raised when the source dataset is incomplete or malformed."""


@dataclass
class SplitSummary:
    split: str
    images: int = 0
    labels: int = 0
    instances: int = 0
    pose_instances: int = 0
    box_only_instances: int = 0
    linked: int = 0
    copied: int = 0
    reused: int = 0
    pruned: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UFC YOLO-Pose labels to boxes and train YOLO11s."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Source UFC pose dataset containing data.yaml and split folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated detector dataset directory.",
    )
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training images to use (0 < fraction <= 1). Useful for smoke tests.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, for example 0, cpu, or mps. Default: Ultralytics auto-selection.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo11s_ufc_poc")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate detector data without importing Ultralytics or training.",
    )
    args = parser.parse_args()

    for name in ("epochs", "imgsz", "batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than zero")
    for name in ("workers", "patience"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be greater than zero and at most one")
    return args


def load_source_config(dataset_root: Path) -> tuple[dict[str, Any], dict[int, str]]:
    config_path = dataset_root / "data.yaml"
    if not config_path.is_file():
        raise DatasetError(f"Missing source configuration: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DatasetError(f"Expected a YAML mapping in {config_path}")

    kpt_shape = config.get("kpt_shape")
    if kpt_shape != [EXPECTED_KEYPOINTS, EXPECTED_KEYPOINT_DIMS]:
        raise DatasetError(
            f"Expected kpt_shape [{EXPECTED_KEYPOINTS}, {EXPECTED_KEYPOINT_DIMS}], "
            f"found {kpt_shape!r}"
        )

    raw_names = config.get("names")
    if isinstance(raw_names, list):
        class_names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            class_names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetError("Class-name keys in data.yaml must be integers") from exc
    else:
        raise DatasetError("data.yaml must define 'names' as a list or mapping")

    if config.get("nc") is not None and int(config["nc"]) != len(class_names):
        raise DatasetError("data.yaml 'nc' does not match the number of class names")
    if class_names != {0: "fighter"}:
        raise DatasetError(f"Expected one fighter class, found {class_names!r}")

    for config_key, source_dir in SPLITS.items():
        if config_key not in config:
            raise DatasetError(f"data.yaml is missing the '{config_key}' split")
        configured = Path(str(config[config_key]))
        configured = configured if configured.is_absolute() else dataset_root / configured
        expected = dataset_root / source_dir / "images"
        if not configured.resolve().is_dir() and expected.is_dir():
            print(
                f"Note: source data.yaml path for '{config_key}' does not exist; "
                f"using {expected}",
                file=sys.stderr,
            )

    return config, class_names


def indexed_files(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise DatasetError(f"Missing directory: {directory}")

    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in files:
            raise DatasetError(
                f"Duplicate filename stem '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    return files


def parse_pose_label(path: Path, class_names: dict[int, str]) -> tuple[str, int, int, int]:
    output_rows: list[str] = []
    pose_instances = 0
    box_only_instances = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) not in (5, EXPECTED_POSE_FIELDS):
                raise DatasetError(
                    f"{path}:{line_number}: expected either 5 box fields or "
                    f"{EXPECTED_POSE_FIELDS} pose fields, found {len(fields)}"
                )

            try:
                class_id = int(fields[0])
                box = [float(value) for value in fields[1:5]]
                keypoints = [float(value) for value in fields[5:]]
            except ValueError as exc:
                raise DatasetError(f"{path}:{line_number}: non-numeric label value") from exc

            if class_id not in class_names:
                raise DatasetError(f"{path}:{line_number}: unknown class ID {class_id}")
            if any(value < 0.0 or value > 1.0 for value in box):
                raise DatasetError(f"{path}:{line_number}: bounding box is not normalized")
            if box[2] <= 0.0 or box[3] <= 0.0:
                raise DatasetError(f"{path}:{line_number}: box width and height must be positive")

            if len(fields) == EXPECTED_POSE_FIELDS:
                for keypoint_index in range(EXPECTED_KEYPOINTS):
                    offset = keypoint_index * EXPECTED_KEYPOINT_DIMS
                    x, y, visibility = keypoints[offset : offset + EXPECTED_KEYPOINT_DIMS]
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        raise DatasetError(
                            f"{path}:{line_number}: keypoint {keypoint_index} is not normalized"
                        )
                    if visibility not in (0.0, 1.0, 2.0):
                        raise DatasetError(
                            f"{path}:{line_number}: keypoint {keypoint_index} has invalid "
                            f"visibility {visibility}"
                        )
                pose_instances += 1
            else:
                box_only_instances += 1

            output_rows.append(" ".join(fields[:5]))

    content = "\n".join(output_rows)
    if content:
        content += "\n"
    return content, len(output_rows), pose_instances, box_only_instances


def write_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        source_stat = source.stat()
        destination_stat = destination.stat()
        try:
            if os.path.samefile(source, destination):
                return "reused"
        except OSError:
            pass
        if (
            source_stat.st_size == destination_stat.st_size
            and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        ):
            return "reused"
        destination.unlink()

    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def prune_stale_files(directory: Path, expected_names: set[str], extensions: set[str]) -> int:
    if not directory.exists():
        return 0
    pruned = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions and path.name not in expected_names:
            path.unlink()
            pruned += 1
    return pruned


def prepare_split(
    dataset_root: Path,
    output_root: Path,
    config_split: str,
    source_split: str,
    class_names: dict[int, str],
) -> SplitSummary:
    source_images_dir = dataset_root / source_split / "images"
    source_labels_dir = dataset_root / source_split / "labels"
    output_images_dir = output_root / source_split / "images"
    output_labels_dir = output_root / source_split / "labels"

    images = indexed_files(source_images_dir, IMAGE_EXTENSIONS)
    labels = indexed_files(source_labels_dir, {".txt"})
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"{len(missing_labels)} images without labels")
        if missing_images:
            details.append(f"{len(missing_images)} labels without images")
        raise DatasetError(f"Split '{config_split}' has " + " and ".join(details))
    if not images:
        raise DatasetError(f"Split '{config_split}' contains no image-label pairs")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    summary = SplitSummary(split=config_split, images=len(images), labels=len(labels))

    for stem, source_image in images.items():
        link_status = link_or_copy(source_image, output_images_dir / source_image.name)
        setattr(summary, link_status, getattr(summary, link_status) + 1)

        detector_content, instances, pose_instances, box_only_instances = parse_pose_label(
            labels[stem], class_names
        )
        write_if_changed(output_labels_dir / f"{stem}.txt", detector_content)
        summary.instances += instances
        summary.pose_instances += pose_instances
        summary.box_only_instances += box_only_instances

    expected_images = {path.name for path in images.values()}
    expected_labels = {f"{stem}.txt" for stem in labels}
    summary.pruned += prune_stale_files(output_images_dir, expected_images, IMAGE_EXTENSIONS)
    summary.pruned += prune_stale_files(output_labels_dir, expected_labels, {".txt"})
    return summary


def write_detector_config(output_root: Path, class_names: dict[int, str]) -> Path:
    config = {
        "path": output_root.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_names,
    }
    config_path = output_root / "data.yaml"
    content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    write_if_changed(config_path, content)
    return config_path


def prepare_detector_dataset(dataset_root: Path, output_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    _, class_names = load_source_config(dataset_root)

    print(f"Preparing detector dataset from: {dataset_root}")
    print(f"Generated data directory:       {output_root}")
    summaries = [
        prepare_split(
            dataset_root,
            output_root,
            config_split,
            source_split,
            class_names,
        )
        for config_split, source_split in SPLITS.items()
    ]
    config_path = write_detector_config(output_root, class_names)

    for summary in summaries:
        print(
            f"{summary.split:>5}: {summary.images} images, {summary.labels} labels, "
            f"{summary.instances} fighter instances "
            f"({summary.pose_instances} pose, {summary.box_only_instances} box-only); "
            f"({summary.linked} linked, {summary.copied} copied, "
            f"{summary.reused} reused, {summary.pruned} stale files pruned)"
        )
    print(f"Detector data config: {config_path}")
    return config_path


def train(args: argparse.Namespace, data_config: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Create the Python 3.11 environment and run "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    print(f"Loading pretrained checkpoint: {args.model}")
    model = YOLO(args.model)
    train_options: dict[str, Any] = {
        "data": str(data_config),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "fraction": args.fraction,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "pretrained": True,
        "cache": False,
        "plots": True,
    }
    if args.device is not None:
        train_options["device"] = args.device

    print("Starting YOLO11 detector training.")
    results = model.train(**train_options)
    save_dir = Path(results.save_dir)
    best_checkpoint = save_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Training finished, but best.pt was not found in {save_dir}")
    print(f"Best checkpoint: {best_checkpoint.resolve()}")
    return best_checkpoint.resolve()


def main() -> int:
    args = parse_args()
    try:
        data_config = prepare_detector_dataset(args.dataset, args.output)
        if args.prepare_only:
            print("Preparation and validation completed; training was not requested.")
            return 0
        train(args, data_config)
        return 0
    except (DatasetError, OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
