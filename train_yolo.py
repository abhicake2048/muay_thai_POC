"""Prepare the MMA fighter detection dataset and fine-tune YOLO11-S.

The source dataset already contains five-field YOLO detection labels. Its
Roboflow train/validation/test split mixes frames from every source fight, so
this script combines the source folders and creates a leakage-free split by
source-video group. The pose-keypoint dataset is deliberately not used here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "mma-fighter-detection-dataset"
DEFAULT_SUPPLEMENTAL_DATASET = (
    PROJECT_ROOT / "punch-kick-player detection.v4i.yolov11"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "yolo_fighter_detector"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "yolo_fighter_detector"
DEFAULT_EXPORT = PROJECT_ROOT / "weights" / "yolo_fighter_detector" / "best.pt"
SOURCE_SPLITS = ("train", "valid", "test")
TARGET_SPLITS = ("train", "val", "test")
VAL_VIDEO_GROUPS = {"10_mp4", "12_mp4"}
TEST_VIDEO_GROUPS = {"8_mp4", "15_mp4"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_PRIMARY_COUNTS = {
    "train": {"images": 4074, "labels": 4074, "instances": 8124},
    "val": {"images": 519, "labels": 519, "instances": 1036},
    "test": {"images": 513, "labels": 513, "instances": 1026},
}
EXPECTED_COMBINED_COUNTS = {
    "train": {"images": 4590, "labels": 4590, "instances": 9119},
    "val": {"images": 519, "labels": 519, "instances": 1036},
    "test": {"images": 513, "labels": 513, "instances": 1026},
}


class DatasetError(RuntimeError):
    """Raised when the detection dataset is incomplete or malformed."""


@dataclass(frozen=True)
class SourcePair:
    image_path: Path
    label_path: Path
    video_group: str
    target_split: str
    dataset_key: str
    output_prefix: str
    class_map: dict[int, int]
    require_boxes: bool


@dataclass
class SplitSummary:
    split: str
    images: int = 0
    labels: int = 0
    instances: int = 0
    linked: int = 0
    copied: int = 0
    reused: int = 0
    pruned: int = 0
    ignored_instances: int = 0
    skipped_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="MMA fighter detection dataset containing five-field YOLO labels.",
    )
    parser.add_argument(
        "--supplemental-dataset",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_DATASET,
        help=(
            "Kick/punch/player dataset. Blue Player and Red Player are remapped "
            "to fighter; kick and punch boxes are ignored."
        ),
    )
    parser.add_argument(
        "--no-supplemental",
        action="store_true",
        help="Prepare and train from the primary MMA detection dataset only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size, or -1 for Ultralytics automatic GPU-memory sizing.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training images to use (0 < fraction <= 1).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device such as 0, cpu, or mps. Default: automatic.",
    )
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="Cache images. Use 'ram' on Colab after copying data to /content.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision during training.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolo11s_fighter_detection")
    parser.add_argument(
        "--export",
        type=Path,
        default=DEFAULT_EXPORT,
        help="Copy the best checkpoint here after training.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Allow a revised dataset whose counts differ from the audited copy.",
    )
    args = parser.parse_args()

    for name in ("epochs", "imgsz"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.batch != -1 and args.batch <= 0:
        parser.error("--batch must be -1 or greater than zero")
    for name in ("workers", "patience"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be greater than zero and at most one")
    return args


def load_source_config(dataset_root: Path) -> dict[int, str]:
    config_path = dataset_root / "data.yaml"
    if not config_path.is_file():
        raise DatasetError(f"Missing source configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DatasetError(f"Expected a YAML mapping in {config_path}")

    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetError("Class-name keys must be integers") from exc
    else:
        raise DatasetError("data.yaml must define 'names' as a list or mapping")
    if config.get("nc") is not None and int(config["nc"]) != len(names):
        raise DatasetError("data.yaml 'nc' does not match its class names")
    if names != {0: "fighter"}:
        raise DatasetError(f"Expected one fighter class, found {names!r}")

    for key in ("train", "val", "test"):
        configured = config.get(key)
        if configured is None:
            raise DatasetError(f"data.yaml is missing '{key}'")
        candidate = Path(str(configured))
        candidate = candidate if candidate.is_absolute() else dataset_root / candidate
        if not candidate.resolve().is_dir():
            print(
                f"Note: source data.yaml '{key}' path is invalid ({configured!r}); "
                "the generated YAML will correct it.",
                file=sys.stderr,
            )
    return names


def load_supplemental_config(dataset_root: Path) -> tuple[dict[int, str], dict[int, int]]:
    """Load a multi-class dataset and select only its player classes."""
    config_path = dataset_root / "data.yaml"
    if not config_path.is_file():
        raise DatasetError(f"Missing supplemental configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise DatasetError(f"Expected a YAML mapping in {config_path}")

    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(index): str(name) for index, name in raw_names.items()}
        except (TypeError, ValueError) as exc:
            raise DatasetError("Supplemental class-name keys must be integers") from exc
    else:
        raise DatasetError("Supplemental data.yaml must define 'names'")
    if config.get("nc") is not None and int(config["nc"]) != len(names):
        raise DatasetError("Supplemental data.yaml 'nc' does not match its names")

    player_ids = {
        class_id
        for class_id, name in names.items()
        if name.strip().casefold().endswith("player")
    }
    if len(player_ids) != 2:
        raise DatasetError(
            "Expected exactly two supplemental player classes, found "
            f"{[(class_id, names[class_id]) for class_id in sorted(player_ids)]}"
        )
    class_map = {class_id: 0 for class_id in player_ids}
    print(
        "Supplemental player mapping: "
        + ", ".join(f"{class_id}:{names[class_id]} -> 0:fighter" for class_id in sorted(player_ids))
    )
    return names, class_map


def indexed_files(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise DatasetError(f"Missing directory: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in files:
            raise DatasetError(f"Duplicate filename stem '{path.stem}' in {directory}")
        files[path.stem] = path
    return files


def video_group_from_stem(stem: str) -> str:
    base = stem.split("_jpg.rf.", maxsplit=1)[0]
    if "-" not in base:
        raise DatasetError(f"Cannot extract video group from filename: {stem}")
    group, frame = base.rsplit("-", maxsplit=1)
    if not frame.isdigit():
        raise DatasetError(f"Filename does not end in a numeric frame ID: {stem}")
    return group


def target_split(video_group: str) -> str:
    if video_group in VAL_VIDEO_GROUPS:
        return "val"
    if video_group in TEST_VIDEO_GROUPS:
        return "test"
    return "train"


def discover_pairs(dataset_root: Path) -> list[SourcePair]:
    pairs: list[SourcePair] = []
    seen_stems: set[str] = set()
    seen_frames: set[str] = set()
    for source_split in SOURCE_SPLITS:
        images = indexed_files(dataset_root / source_split / "images", IMAGE_EXTENSIONS)
        labels = indexed_files(dataset_root / source_split / "labels", {".txt"})
        missing_labels = images.keys() - labels.keys()
        missing_images = labels.keys() - images.keys()
        if missing_labels or missing_images:
            raise DatasetError(
                f"{source_split}: {len(missing_labels)} images lack labels and "
                f"{len(missing_images)} labels lack images"
            )
        for stem in sorted(images):
            if stem in seen_stems:
                raise DatasetError(f"Duplicate filename across source splits: {stem}")
            seen_stems.add(stem)
            frame_identity = stem.split("_jpg.rf.", maxsplit=1)[0]
            if frame_identity in seen_frames:
                raise DatasetError(f"Duplicate source frame: {frame_identity}")
            seen_frames.add(frame_identity)
            group = video_group_from_stem(stem)
            pairs.append(
                SourcePair(
                    image_path=images[stem],
                    label_path=labels[stem],
                    video_group=group,
                    target_split=target_split(group),
                    dataset_key="mma_detection",
                    output_prefix="",
                    class_map={0: 0},
                    require_boxes=True,
                )
            )
    return pairs


def discover_supplemental_pairs(
    dataset_root: Path, class_map: dict[int, int]
) -> list[SourcePair]:
    """Put all supplemental video groups in train to avoid cross-split leakage."""
    pairs: list[SourcePair] = []
    seen_stems: set[str] = set()
    seen_frames: set[str] = set()
    for source_split in SOURCE_SPLITS:
        images = indexed_files(dataset_root / source_split / "images", IMAGE_EXTENSIONS)
        labels = indexed_files(dataset_root / source_split / "labels", {".txt"})
        missing_labels = images.keys() - labels.keys()
        missing_images = labels.keys() - images.keys()
        if missing_labels or missing_images:
            raise DatasetError(
                f"supplemental/{source_split}: {len(missing_labels)} images lack "
                f"labels and {len(missing_images)} labels lack images"
            )
        for stem in sorted(images):
            if stem in seen_stems:
                raise DatasetError(f"Duplicate supplemental filename: {stem}")
            seen_stems.add(stem)
            frame_identity = stem.split("_jpg.rf.", maxsplit=1)[0]
            if frame_identity in seen_frames:
                raise DatasetError(f"Duplicate supplemental source frame: {frame_identity}")
            seen_frames.add(frame_identity)
            pairs.append(
                SourcePair(
                    image_path=images[stem],
                    label_path=labels[stem],
                    video_group=video_group_from_stem(stem),
                    target_split="train",
                    dataset_key="kick_punch_player",
                    output_prefix="kickpunch__",
                    class_map=dict(class_map),
                    require_boxes=False,
                )
            )
    return pairs


def validate_label(
    path: Path,
    source_class_names: dict[int, str],
    class_map: dict[int, int],
    require_boxes: bool,
) -> tuple[str, int, int]:
    rows: list[str] = []
    ignored = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 5:
                raise DatasetError(
                    f"{path}:{line_number}: expected 5 YOLO detection fields, "
                    f"found {len(fields)}. Pose labels are not accepted by this script."
                )
            try:
                class_id = int(fields[0])
                center_x, center_y, width, height = map(float, fields[1:])
            except ValueError as exc:
                raise DatasetError(f"{path}:{line_number}: non-numeric value") from exc
            if class_id not in source_class_names:
                raise DatasetError(f"{path}:{line_number}: unknown class ID {class_id}")
            values = (center_x, center_y, width, height)
            if any(value < 0.0 or value > 1.0 for value in values):
                raise DatasetError(f"{path}:{line_number}: box is not normalized")
            if width <= 0.0 or height <= 0.0:
                raise DatasetError(f"{path}:{line_number}: box has non-positive size")
            if class_id in class_map:
                rows.append(" ".join([str(class_map[class_id]), *fields[1:]]))
            else:
                ignored += 1
    if require_boxes and not rows:
        raise DatasetError(f"Detection image has no fighter boxes: {path}")
    content = "\n".join(rows)
    if content:
        content += "\n"
    return content, len(rows), ignored


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
        try:
            if os.path.samefile(source, destination):
                return "reused"
        except OSError:
            pass
        source_stat = source.stat()
        destination_stat = destination.stat()
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


def prune_stale(directory: Path, expected: set[str], extensions: set[str]) -> int:
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in extensions and path.name not in expected:
            path.unlink()
            removed += 1
    return removed


def prepare_detector_dataset(
    dataset_root: Path,
    supplemental_root: Path | None,
    output_root: Path,
    skip_count_check: bool = False,
) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    primary_names = load_source_config(dataset_root)
    pairs = discover_pairs(dataset_root)
    source_names = {"mma_detection": primary_names}
    resolved_supplemental: Path | None = None
    if supplemental_root is not None:
        resolved_supplemental = supplemental_root.expanduser().resolve()
        supplemental_names, supplemental_map = load_supplemental_config(
            resolved_supplemental
        )
        source_names["kick_punch_player"] = supplemental_names
        pairs.extend(
            discover_supplemental_pairs(resolved_supplemental, supplemental_map)
        )

    summaries = {split: SplitSummary(split) for split in TARGET_SPLITS}
    expected_images: dict[str, set[str]] = defaultdict(set)
    expected_labels: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)

    print(f"Detection source: {dataset_root}")
    if resolved_supplemental is not None:
        print(f"Supplemental source: {resolved_supplemental}")
    print(f"Grouped output:   {output_root}")
    for pair in pairs:
        split = pair.target_split
        image_dir = output_root / split / "images"
        label_dir = output_root / split / "labels"
        image_name = f"{pair.output_prefix}{pair.image_path.name}"
        label_name = f"{pair.output_prefix}{pair.image_path.stem}.txt"
        content, instances, ignored = validate_label(
            pair.label_path,
            source_names[pair.dataset_key],
            pair.class_map,
            pair.require_boxes,
        )
        summary = summaries[split]
        summary.ignored_instances += ignored
        if instances == 0:
            summary.skipped_images += 1
            continue

        expected_images[split].add(image_name)
        expected_labels[split].add(label_name)
        groups[split].add(f"{pair.dataset_key}:{pair.video_group}")
        status = link_or_copy(pair.image_path, image_dir / image_name)
        setattr(summary, status, getattr(summary, status) + 1)
        write_if_changed(label_dir / label_name, content)
        summary.images += 1
        summary.labels += 1
        summary.instances += instances

    for split, summary in summaries.items():
        summary.pruned += prune_stale(
            output_root / split / "images", expected_images[split], IMAGE_EXTENSIONS
        )
        summary.pruned += prune_stale(
            output_root / split / "labels", expected_labels[split], {".txt"}
        )

    overlaps = {
        "train_val": sorted(groups["train"] & groups["val"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "val_test": sorted(groups["val"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise DatasetError(f"Video-group leakage remains: {overlaps}")

    observed = {
        split: {
            "images": summaries[split].images,
            "labels": summaries[split].labels,
            "instances": summaries[split].instances,
        }
        for split in TARGET_SPLITS
    }
    expected_counts = (
        EXPECTED_COMBINED_COUNTS
        if resolved_supplemental is not None
        else EXPECTED_PRIMARY_COUNTS
    )
    if not skip_count_check and observed != expected_counts:
        raise DatasetError(f"Expected counts {expected_counts}, found {observed}")

    config = {
        "path": output_root.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "fighter"},
    }
    config_path = output_root / "data.yaml"
    write_if_changed(
        config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    report = {
        "source_datasets": {
            "mma_detection": dataset_root.as_posix(),
            "kick_punch_player": (
                resolved_supplemental.as_posix()
                if resolved_supplemental is not None
                else None
            ),
        },
        "split_strategy": (
            "fixed MMA video groups; supplemental videos restricted to train"
        ),
        "validation_video_groups": sorted(VAL_VIDEO_GROUPS),
        "test_video_groups": sorted(TEST_VIDEO_GROUPS),
        "video_groups": {split: sorted(groups[split]) for split in TARGET_SPLITS},
        "overlaps": overlaps,
        "splits": observed,
        "supplemental_filtering": {
            "retained_classes": ["Blue Player", "Red Player"],
            "remapped_class": "fighter",
            "ignored_classes": ["Blue Kick", "Blue punch", "Red Kick", "Red Punch"],
            "skipped_images_without_player_boxes": summaries["train"].skipped_images,
            "ignored_non_player_boxes": summaries["train"].ignored_instances,
        },
    }
    write_if_changed(
        output_root / "preparation_report.json",
        json.dumps(report, indent=2) + "\n",
    )

    for split in TARGET_SPLITS:
        summary = summaries[split]
        print(
            f"{split:>5}: {summary.images} images, {summary.instances} fighter boxes; "
            f"{summary.linked} linked, {summary.copied} copied, "
            f"{summary.reused} reused, {summary.pruned} stale files pruned; "
            f"{summary.skipped_images} images skipped, "
            f"{summary.ignored_instances} non-fighter boxes ignored"
        )
    print(f"Detector data config: {config_path}")
    return config_path


def train(args: argparse.Namespace, data_config: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run 'python -m pip install -r requirements.txt'."
        ) from exc

    cache: bool | str = False if args.cache == "none" else args.cache
    options: dict[str, Any] = {
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
        "cache": cache,
        "amp": args.amp,
        "plots": True,
    }
    if args.device is not None:
        options["device"] = args.device

    print(f"Loading pretrained detector: {args.model}")
    model = YOLO(args.model)
    results = model.train(**options)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished, but best.pt is missing from {results.save_dir}")

    export = args.export.expanduser().resolve()
    export.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() != export:
        shutil.copy2(best, export)
    print(f"Run checkpoint:      {best.resolve()}")
    print(f"Exported checkpoint: {export}")
    return export


def main() -> int:
    args = parse_args()
    try:
        data_config = prepare_detector_dataset(
            args.dataset,
            None if args.no_supplemental else args.supplemental_dataset,
            args.output,
            args.skip_count_check,
        )
        if args.prepare_only:
            print("Detection dataset validation and grouped preparation completed.")
            return 0
        train(args, data_config)
        return 0
    except (DatasetError, OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
