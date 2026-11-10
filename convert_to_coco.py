"""Convert UFC YOLO-Pose labels into leakage-free COCO Keypoint files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from custom_combat_skeleton import KEYPOINT_NAMES, SKELETON_LINKS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "mma-fighter-pose-estimation-dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "rtmpose_coco"
SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VAL_VIDEO_GROUPS = {"10_mp4", "12_mp4"}
TEST_VIDEO_GROUPS = {"8_mp4", "15_mp4"}
COCO_SKELETON = [
    [KEYPOINT_NAMES.index(start) + 1, KEYPOINT_NAMES.index(end) + 1]
    for start, end in SKELETON_LINKS
]
EXPECTED_COUNTS = {
    "train": {"images": 4074, "annotations": 8094, "dropped_box_rows": 30},
    "val": {"images": 519, "annotations": 1035, "dropped_box_rows": 1},
    "test": {"images": 513, "annotations": 1026, "dropped_box_rows": 0},
}


class ConversionError(RuntimeError):
    """Raised when source data is incomplete or malformed."""


@dataclass(frozen=True)
class SourceImage:
    old_split: str
    image_path: Path
    label_path: Path
    video_group: str
    new_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Allow conversion of a dataset revision with counts different from the audited copy.",
    )
    args = parser.parse_args()
    if args.render_samples < 0:
        parser.error("--render-samples cannot be negative")
    return args


def video_group_from_stem(stem: str) -> str:
    base = stem.split("_jpg.rf.", maxsplit=1)[0]
    if "-" not in base:
        raise ConversionError(f"Cannot extract source-video group from filename: {stem}")
    group, frame_number = base.rsplit("-", maxsplit=1)
    if not frame_number.isdigit():
        raise ConversionError(f"Filename does not end with a numeric frame ID: {stem}")
    return group


def assigned_split(video_group: str) -> str:
    if video_group in VAL_VIDEO_GROUPS:
        return "val"
    if video_group in TEST_VIDEO_GROUPS:
        return "test"
    return "train"


def discover_source_images(dataset_root: Path) -> list[SourceImage]:
    records: list[SourceImage] = []
    seen_relative_paths: set[str] = set()
    for old_split in SOURCE_SPLITS:
        image_dir = dataset_root / old_split / "images"
        label_dir = dataset_root / old_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ConversionError(f"Missing images or labels directory for split: {old_split}")

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if images.keys() != labels.keys():
            missing_labels = sorted(images.keys() - labels.keys())
            missing_images = sorted(labels.keys() - images.keys())
            raise ConversionError(
                f"{old_split}: {len(missing_labels)} missing labels and "
                f"{len(missing_images)} missing images"
            )

        for stem in sorted(images):
            relative = images[stem].relative_to(dataset_root.parent).as_posix()
            if relative in seen_relative_paths:
                raise ConversionError(f"Duplicate image path: {relative}")
            seen_relative_paths.add(relative)
            group = video_group_from_stem(stem)
            records.append(
                SourceImage(
                    old_split=old_split,
                    image_path=images[stem],
                    label_path=labels[stem],
                    video_group=group,
                    new_split=assigned_split(group),
                )
            )
    return sorted(records, key=lambda item: item.image_path.as_posix())


def normalized_box_to_coco(
    values: list[float], width: int, height: int, context: str
) -> list[float]:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ConversionError(f"{context}: bounding box is not normalized")
    center_x, center_y, box_width, box_height = values
    if box_width <= 0.0 or box_height <= 0.0:
        raise ConversionError(f"{context}: box width and height must be positive")
    x1 = max(0.0, (center_x - box_width / 2.0) * width)
    y1 = max(0.0, (center_y - box_height / 2.0) * height)
    x2 = min(float(width), (center_x + box_width / 2.0) * width)
    y2 = min(float(height), (center_y + box_height / 2.0) * height)
    if x2 <= x1 or y2 <= y1:
        raise ConversionError(f"{context}: clipped box has zero area")
    return [x1, y1, x2 - x1, y2 - y1]


def parse_pose_row(
    fields: list[str], width: int, height: int, context: str
) -> tuple[list[float], list[float], int]:
    if len(fields) != 56:
        raise ConversionError(f"{context}: expected 56 fields, found {len(fields)}")
    try:
        class_id = int(fields[0])
        box_values = [float(value) for value in fields[1:5]]
        pose_values = [float(value) for value in fields[5:]]
    except ValueError as exc:
        raise ConversionError(f"{context}: non-numeric annotation value") from exc
    if class_id != 0:
        raise ConversionError(f"{context}: expected fighter class 0, found {class_id}")

    bbox = normalized_box_to_coco(box_values, width, height, context)
    keypoints: list[float] = []
    num_keypoints = 0
    for index in range(17):
        x, y, visibility = pose_values[index * 3 : index * 3 + 3]
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConversionError(f"{context}: keypoint {index} is not normalized")
        if visibility not in (0.0, 1.0, 2.0):
            raise ConversionError(
                f"{context}: keypoint {index} has invalid visibility {visibility}"
            )
        if visibility == 0.0:
            keypoints.extend([0.0, 0.0, 0])
        else:
            keypoints.extend([x * width, y * height, int(visibility)])
            num_keypoints += 1
    return bbox, keypoints, num_keypoints


def category_record() -> dict[str, Any]:
    return {
        "id": 1,
        "name": "fighter",
        "supercategory": "person",
        "keypoints": KEYPOINT_NAMES,
        "skeleton": COCO_SKELETON,
    }


def convert_records(
    records: list[SourceImage], dataset_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    data = {
        split: {
            "info": {
                "description": "UFC/MMA fighter COCO-17 keypoints",
                "version": "1.0",
            },
            "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0"}],
            "images": [],
            "annotations": [],
            "categories": [category_record()],
        }
        for split in ("train", "val", "test")
    }
    next_image_id = Counter()
    next_annotation_id = Counter()
    dropped_box_rows = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    source_parent = dataset_root.parent

    for record in records:
        split = record.new_split
        next_image_id[split] += 1
        image_id = next_image_id[split]
        groups[split].add(record.video_group)

        with Image.open(record.image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ConversionError(f"Invalid image dimensions: {record.image_path}")
        file_name = record.image_path.relative_to(source_parent).as_posix()
        data[split]["images"].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )

        valid_poses = 0
        for line_number, raw_line in enumerate(
            record.label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) == 5:
                dropped_box_rows[split] += 1
                continue
            context = f"{record.label_path}:{line_number}"
            bbox, keypoints, num_keypoints = parse_pose_row(fields, width, height, context)
            next_annotation_id[split] += 1
            data[split]["annotations"].append(
                {
                    "id": next_annotation_id[split],
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "num_keypoints": num_keypoints,
                    "keypoints": keypoints,
                }
            )
            valid_poses += 1
        if valid_poses == 0:
            raise ConversionError(f"Image has no valid pose annotations: {record.image_path}")

    overlaps = {
        "train_val": sorted(groups["train"] & groups["val"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "val_test": sorted(groups["val"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise ConversionError(f"Source-video leakage detected: {overlaps}")

    report = {
        "source_dataset": dataset_root.as_posix(),
        "image_root": source_parent.as_posix(),
        "split_strategy": "fixed_source_video_groups",
        "validation_video_groups": sorted(VAL_VIDEO_GROUPS),
        "test_video_groups": sorted(TEST_VIDEO_GROUPS),
        "video_groups": {split: sorted(values) for split, values in groups.items()},
        "overlaps": overlaps,
        "splits": {
            split: {
                "images": len(data[split]["images"]),
                "annotations": len(data[split]["annotations"]),
                "dropped_box_rows": dropped_box_rows[split],
            }
            for split in ("train", "val", "test")
        },
    }
    return data, report


def validate_coco(data: dict[str, dict[str, Any]]) -> None:
    for split, payload in data.items():
        image_ids = {image["id"] for image in payload["images"]}
        annotation_ids = {annotation["id"] for annotation in payload["annotations"]}
        if len(image_ids) != len(payload["images"]):
            raise ConversionError(f"{split}: duplicate image IDs")
        if len(annotation_ids) != len(payload["annotations"]):
            raise ConversionError(f"{split}: duplicate annotation IDs")
        image_by_id = {image["id"]: image for image in payload["images"]}
        for annotation in payload["annotations"]:
            if annotation["image_id"] not in image_ids:
                raise ConversionError(f"{split}: annotation references a missing image")
            if len(annotation["keypoints"]) != 51:
                raise ConversionError(f"{split}: annotation does not contain 17 keypoints")
            image = image_by_id[annotation["image_id"]]
            x, y, width, height = annotation["bbox"]
            if x < 0 or y < 0 or x + width > image["width"] + 1e-6:
                raise ConversionError(f"{split}: bounding box exceeds image width")
            if y + height > image["height"] + 1e-6:
                raise ConversionError(f"{split}: bounding box exceeds image height")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def render_previews(
    data: dict[str, dict[str, Any]], image_root: Path, output_root: Path, count: int, seed: int
) -> None:
    if count == 0:
        return
    candidates: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for split, payload in data.items():
        annotations = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        for image in payload["images"]:
            candidates.append((split, image, annotations[image["id"]]))

    rng = random.Random(seed)
    selected = rng.sample(candidates, min(count, len(candidates)))
    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for index, (split, image_record, annotations) in enumerate(selected, start=1):
        source = image_root / image_record["file_name"]
        with Image.open(source).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for annotation in annotations:
                x, y, width, height = annotation["bbox"]
                draw.rectangle((x, y, x + width, y + height), outline=(255, 0, 0), width=2)
                points = [
                    tuple(annotation["keypoints"][offset : offset + 3])
                    for offset in range(0, 51, 3)
                ]
                for start, end in COCO_SKELETON:
                    p1, p2 = points[start - 1], points[end - 1]
                    if p1[2] > 0 and p2[2] > 0:
                        draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 255, 255), width=2)
                for point_x, point_y, visibility in points:
                    if visibility > 0:
                        radius = 3
                        draw.ellipse(
                            (
                                point_x - radius,
                                point_y - radius,
                                point_x + radius,
                                point_y + radius,
                            ),
                            fill=(0, 255, 0) if visibility == 2 else (255, 165, 0),
                        )
            output_name = f"{index:02d}_{split}_{Path(image_record['file_name']).name}"
            expected_names.add(output_name)
            image.save(preview_dir / output_name, quality=92)
    for stale in preview_dir.iterdir():
        if stale.is_file() and stale.name not in expected_names:
            stale.unlink()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    try:
        records = discover_source_images(dataset_root)
        data, report = convert_records(records, dataset_root)
        validate_coco(data)
        if not args.skip_count_check:
            for split, expected in EXPECTED_COUNTS.items():
                if report["splits"][split] != expected:
                    raise ConversionError(
                        f"{split}: expected {expected}, found {report['splits'][split]}"
                    )

        annotation_dir = output_root / "annotations"
        for split, payload in data.items():
            write_json(annotation_dir / f"{split}.json", payload)
        write_json(output_root / "conversion_report.json", report)
        render_previews(data, dataset_root.parent, output_root, args.render_samples, args.seed)

        print(f"COCO annotations written to: {annotation_dir}")
        for split in ("train", "val", "test"):
            values = report["splits"][split]
            print(
                f"{split:>5}: {values['images']} images, {values['annotations']} poses, "
                f"{values['dropped_box_rows']} box-only rows dropped"
            )
        print(f"Preview images written to: {output_root / 'previews'}")
        return 0
    except (ConversionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
