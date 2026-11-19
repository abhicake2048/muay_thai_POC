"""Build one clean COCO-17 pose dataset from every supported dataset in a folder.

Supported inputs:

* YOLO-Pose datasets with ``kpt_shape: [17, 3]``.
* COCO keypoint exports using COCO-17.
* The two 14-keypoint MMA schemas documented in ``SCHEMA_MAPPINGS`` below.
* ZIP archives containing any of the formats above.

The output contains one image store, one complete ``all.json`` annotation
database, and grouped train/validation/test views for RTMPose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

from custom_combat_skeleton import KEYPOINT_NAMES, SKELETON_LINKS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "combined_rtmpose_coco"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
COCO_SKELETON = [
    [KEYPOINT_NAMES.index(start) + 1, KEYPOINT_NAMES.index(end) + 1]
    for start, end in SKELETON_LINKS
]


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


CANONICAL_INDEX = {_normalized_name(name): index for index, name in enumerate(KEYPOINT_NAMES)}

# These are positional mappings into the canonical COCO-17 order. Values not
# listed here (chin, neck, and a single center-waist point) are intentionally
# ignored instead of fabricating left/right joints.
MMA_HEAD_CHIN_SCHEMA = (
    "head",
    "chin",
    "neck",
    "left_elbow",
    "left_shoulder",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
    "waist",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)
MMA_HEAD_CHIN_MAPPING = {
    0: 0,
    3: 7,
    4: 5,
    5: 9,
    6: 6,
    7: 8,
    8: 10,
    10: 13,
    11: 14,
    12: 15,
    13: 16,
}

MMA_DUPLICATE_LIMB_SCHEMA = (
    "head",
    "neck",
    "shoulder",
    "shoulder",
    "elbow",
    "hand",
    "elbow",
    "hand",
    "hip",
    "hip",
    "knee",
    "foot",
    "foot",
    "knee",
)
# This export lost the side names, but its skeleton links preserve two complete
# limbs. The first linked limb is treated as left and the second as right.
MMA_DUPLICATE_LIMB_MAPPING = {
    0: 0,
    3: 5,
    4: 7,
    5: 9,
    8: 11,
    10: 13,
    11: 15,
    2: 6,
    6: 8,
    7: 10,
    9: 12,
    13: 14,
    12: 16,
}

SCHEMA_MAPPINGS = {
    "mma_head_chin_14": MMA_HEAD_CHIN_MAPPING,
    "mma_duplicate_limb_14": MMA_DUPLICATE_LIMB_MAPPING,
}


class PreparationError(RuntimeError):
    """Raised when the requested combined dataset cannot be built safely."""


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    root: Path
    name: str


@dataclass
class PoseInstance:
    bbox: list[float]
    keypoints: list[float]
    num_keypoints: int


@dataclass
class ImageSample:
    source_name: str
    source_path: Path
    original_name: str
    source_split: str
    width: int
    height: int
    pixel_hash: str
    group: str
    instances: list[PoseInstance]
    split: str = ""
    output_name: str = ""

    @property
    def quality(self) -> tuple[int, int]:
        return (len(self.instances), sum(item.num_keypoints for item in self.instances))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-keypoints",
        type=int,
        default=4,
        help="Drop a pose instance with fewer than this many visible keypoints.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-samples", type=int, default=24)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset. Required for repeatable Colab reruns.",
    )
    args = parser.parse_args()
    if args.min_keypoints < 1 or args.min_keypoints > 17:
        parser.error("--min-keypoints must be between 1 and 17")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("--train-ratio + --val-ratio must be less than 1")
    if args.render_samples < 0:
        parser.error("--render-samples cannot be negative")
    return args


def category_record() -> dict[str, Any]:
    return {
        "id": 1,
        "name": "fighter",
        "supercategory": "person",
        "keypoints": KEYPOINT_NAMES,
        "skeleton": COCO_SKELETON,
    }


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PreparationError(f"Cannot read YAML file {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def is_yolo_pose_root(root: Path) -> bool:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        return False
    payload = read_yaml(yaml_path)
    shape = payload.get("kpt_shape")
    try:
        return int(shape[0]) == 17 and int(shape[1]) in (2, 3)
    except (IndexError, TypeError, ValueError):
        return False


def discover_directory_sources(root: Path, prefix: str = "") -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    coco_roots: set[Path] = set()
    for annotation in root.rglob("_annotations.coco.json"):
        if annotation.is_file():
            coco_roots.add(annotation.parent.parent.resolve())
    for coco_root in sorted(coco_roots, key=lambda item: item.as_posix()):
        relative = coco_root.name
        try:
            relative = coco_root.relative_to(root.resolve()).as_posix() or coco_root.name
        except ValueError:
            pass
        if relative == ".":
            relative = ""
        name = f"{prefix}{relative}".rstrip(":") if prefix else (relative or root.name)
        sources.append(SourceSpec("coco", coco_root, name))

    yolo_roots: set[Path] = set()
    for yaml_path in root.rglob("data.yaml"):
        if yaml_path.is_file() and is_yolo_pose_root(yaml_path.parent):
            yolo_roots.add(yaml_path.parent.resolve())
    for yolo_root in sorted(yolo_roots, key=lambda item: item.as_posix()):
        relative = yolo_root.name
        try:
            relative = yolo_root.relative_to(root.resolve()).as_posix() or yolo_root.name
        except ValueError:
            pass
        if relative == ".":
            relative = ""
        name = f"{prefix}{relative}".rstrip(":") if prefix else (relative or root.name)
        sources.append(SourceSpec("yolo_pose", yolo_root, name))
    return sources


def archive_has_pose_data(archive: Path) -> bool:
    try:
        with zipfile.ZipFile(archive) as handle:
            names = [item.filename.replace("\\", "/") for item in handle.infolist()]
            if any(name.endswith("_annotations.coco.json") for name in names):
                return True
            for info in handle.infolist():
                if info.filename.replace("\\", "/").endswith("data.yaml"):
                    with handle.open(info) as stream:
                        text = stream.read(128 * 1024).decode("utf-8", errors="replace")
                    if re.search(r"(?m)^\s*kpt_shape\s*:\s*\[?\s*17\s*[, ]", text):
                        return True
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f"Cannot inspect ZIP archive {archive}: {exc}") from exc
    return False


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            member = info.filename.replace("\\", "/")
            target = (destination / member).resolve()
            if target != destination and destination not in target.parents:
                raise PreparationError(f"Unsafe path in {archive.name}: {member}")
        handle.extractall(destination)


def discover_all_sources(
    input_root: Path, staging_root: Path, report: dict[str, Any]
) -> list[SourceSpec]:
    sources = discover_directory_sources(input_root)
    archives = sorted(input_root.rglob("*.zip"), key=lambda item: item.as_posix())
    for index, archive in enumerate(archives, start=1):
        extracted_peer = archive.parent / archive.stem
        if extracted_peer.is_dir():
            report["skipped_archives_existing_directory"].append(archive.as_posix())
            continue
        if not archive_has_pose_data(archive):
            report["skipped_non_pose_archives"].append(archive.as_posix())
            continue
        destination = staging_root / f"{index:03d}_{_normalized_name(archive.stem)}"
        destination.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(archive, destination)
        found = discover_directory_sources(destination, prefix=f"{archive.name}::")
        if not found:
            raise PreparationError(
                f"{archive} looked like a pose archive, but no supported dataset root was found"
            )
        sources.extend(found)
    unique: dict[tuple[str, Path], SourceSpec] = {}
    for source in sources:
        unique[(source.kind, source.root.resolve())] = source
    return sorted(unique.values(), key=lambda item: (item.name, item.kind))


def schema_mapping(names: list[str]) -> tuple[str, dict[int, int]] | None:
    normalized = tuple(_normalized_name(name) for name in names)
    if len(normalized) == 17 and set(normalized) == set(CANONICAL_INDEX):
        return "coco17", {
            source_index: CANONICAL_INDEX[name]
            for source_index, name in enumerate(normalized)
        }
    if normalized == MMA_HEAD_CHIN_SCHEMA:
        return "mma_head_chin_14", MMA_HEAD_CHIN_MAPPING
    if normalized == MMA_DUPLICATE_LIMB_SCHEMA:
        return "mma_duplicate_limb_14", MMA_DUPLICATE_LIMB_MAPPING
    return None


def visible_count(keypoints: list[float]) -> int:
    return sum(1 for offset in range(2, len(keypoints), 3) if keypoints[offset] > 0)


def map_and_clean_keypoints(
    values: list[Any], mapping: dict[int, int], width: int, height: int
) -> list[float]:
    output: list[float] = [0.0] * 51
    for source_index, canonical_index in mapping.items():
        offset = source_index * 3
        if offset + 2 >= len(values):
            continue
        try:
            x = float(values[offset])
            y = float(values[offset + 1])
            visibility = int(float(values[offset + 2]))
        except (TypeError, ValueError, OverflowError):
            continue
        if visibility <= 0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        if x < 0 or y < 0 or x > width or y > height:
            continue
        target = canonical_index * 3
        output[target] = min(max(x, 0.0), max(float(width) - 1e-3, 0.0))
        output[target + 1] = min(max(y, 0.0), max(float(height) - 1e-3, 0.0))
        output[target + 2] = 2 if visibility >= 2 else 1
    return output


def clean_bbox(
    raw_bbox: Any, keypoints: list[float], width: int, height: int
) -> list[float] | None:
    visible = [
        (float(keypoints[offset]), float(keypoints[offset + 1]))
        for offset in range(0, 51, 3)
        if keypoints[offset + 2] > 0
    ]
    corners: list[float] | None = None
    try:
        x, y, box_width, box_height = (float(value) for value in raw_bbox)
        if all(math.isfinite(value) for value in (x, y, box_width, box_height)):
            if box_width > 0 and box_height > 0:
                corners = [x, y, x + box_width, y + box_height]
    except (TypeError, ValueError, OverflowError):
        corners = None
    if visible:
        xs = [point[0] for point in visible]
        ys = [point[1] for point in visible]
        keypoint_box = [min(xs), min(ys), max(xs), max(ys)]
        pad_x = max(2.0, (keypoint_box[2] - keypoint_box[0]) * 0.05)
        pad_y = max(2.0, (keypoint_box[3] - keypoint_box[1]) * 0.05)
        keypoint_box = [
            keypoint_box[0] - pad_x,
            keypoint_box[1] - pad_y,
            keypoint_box[2] + pad_x,
            keypoint_box[3] + pad_y,
        ]
        if corners is None:
            corners = keypoint_box
        else:
            corners = [
                min(corners[0], keypoint_box[0]),
                min(corners[1], keypoint_box[1]),
                max(corners[2], keypoint_box[2]),
                max(corners[3], keypoint_box[3]),
            ]
    if corners is None:
        return None
    x1 = min(max(corners[0], 0.0), float(width))
    y1 = min(max(corners[1], 0.0), float(height))
    x2 = min(max(corners[2], 0.0), float(width))
    y2 = min(max(corners[3], 0.0), float(height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def inspect_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if width <= 1 or height <= 1:
            raise ValueError("image dimensions are too small")
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{width}x{height}:RGB:".encode("ascii"))
        digest.update(rgb.tobytes())
        return width, height, digest.hexdigest()


def infer_group(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.split(r"\.rf\.[0-9a-f]+", stem, maxsplit=1, flags=re.IGNORECASE)[0]
    changed = True
    while changed:
        changed = False
        for suffix in ("_jpg", "_jpeg", "_png", "_bmp", "_webp"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    video_match = re.match(r"^(.+?_mp4)[-_]\d+$", stem, flags=re.IGNORECASE)
    if video_match:
        stem = video_match.group(1)
    normalized = _normalized_name(stem)
    return normalized or hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]


def resolve_coco_image(root: Path, annotation_path: Path, file_name: str) -> Path | None:
    candidates = [annotation_path.parent / file_name, root / file_name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    basename = Path(file_name).name
    direct = annotation_path.parent / basename
    if direct.is_file():
        return direct
    matches = list(root.rglob(basename))
    return matches[0] if len(matches) == 1 else None


def process_coco_source(
    source: SourceSpec, min_keypoints: int
) -> tuple[list[ImageSample], Counter[str], set[str]]:
    samples: list[ImageSample] = []
    stats: Counter[str] = Counter()
    schemas: set[str] = set()
    annotation_paths = sorted(source.root.rglob("_annotations.coco.json"))
    for annotation_path in annotation_paths:
        source_split = annotation_path.parent.name.lower()
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreparationError(f"Cannot read COCO file {annotation_path}: {exc}") from exc
        categories = {
            category.get("id"): category for category in payload.get("categories", [])
        }
        mappings: dict[Any, tuple[str, dict[int, int]]] = {}
        for category_id, category in categories.items():
            names = category.get("keypoints")
            if isinstance(names, list):
                found = schema_mapping([str(name) for name in names])
                if found is not None:
                    mappings[category_id] = found
                    schemas.add(found[0])
                else:
                    stats["unsupported_keypoint_categories"] += 1

        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[annotation.get("image_id")].append(annotation)

        for image_record in payload.get("images", []):
            stats["images_seen"] += 1
            file_name = str(image_record.get("file_name", ""))
            image_path = resolve_coco_image(source.root, annotation_path, file_name)
            if image_path is None:
                stats["missing_images"] += 1
                continue
            try:
                width, height, pixel_hash = inspect_image(image_path)
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                stats["corrupt_or_invalid_images"] += 1
                continue

            instances: list[PoseInstance] = []
            for annotation in annotations_by_image.get(image_record.get("id"), []):
                mapping_record = mappings.get(annotation.get("category_id"))
                if mapping_record is None:
                    stats["unsupported_or_non_pose_annotations"] += 1
                    continue
                _, mapping = mapping_record
                values = annotation.get("keypoints")
                if not isinstance(values, list):
                    stats["malformed_pose_annotations"] += 1
                    continue
                keypoints = map_and_clean_keypoints(values, mapping, width, height)
                count = visible_count(keypoints)
                if count < min_keypoints:
                    stats["low_keypoint_pose_annotations"] += 1
                    continue
                bbox = clean_bbox(annotation.get("bbox"), keypoints, width, height)
                if bbox is None:
                    stats["invalid_bbox_annotations"] += 1
                    continue
                instances.append(PoseInstance(bbox, keypoints, count))
                stats["pose_annotations_kept"] += 1
            if not instances:
                stats["images_without_usable_pose"] += 1
                continue
            samples.append(
                ImageSample(
                    source_name=source.name,
                    source_path=image_path,
                    original_name=file_name,
                    source_split=source_split,
                    width=width,
                    height=height,
                    pixel_hash=pixel_hash,
                    group=infer_group(file_name),
                    instances=instances,
                )
            )
            stats["images_kept_before_deduplication"] += 1
    return samples, stats, schemas


def yolo_bbox(values: list[float], width: int, height: int) -> list[float] | None:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    center_x, center_y, box_width, box_height = values
    if box_width <= 0 or box_height <= 0:
        return None
    return [
        (center_x - box_width / 2.0) * width,
        (center_y - box_height / 2.0) * height,
        box_width * width,
        box_height * height,
    ]


def process_yolo_source(
    source: SourceSpec, min_keypoints: int
) -> tuple[list[ImageSample], Counter[str], set[str]]:
    samples: list[ImageSample] = []
    stats: Counter[str] = Counter()
    yaml_payload = read_yaml(source.root / "data.yaml")
    shape = yaml_payload.get("kpt_shape", [17, 3])
    dimensions = int(shape[1])
    if dimensions not in (2, 3):
        raise PreparationError(f"Unsupported YOLO keypoint shape in {source.root / 'data.yaml'}")

    for folder_name in ("train", "valid", "val", "test"):
        image_dir = source.root / folder_name / "images"
        label_dir = source.root / folder_name / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda item: item.name):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stats["images_seen"] += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                stats["missing_labels"] += 1
                continue
            try:
                width, height, pixel_hash = inspect_image(image_path)
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                stats["corrupt_or_invalid_images"] += 1
                continue
            instances: list[PoseInstance] = []
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                fields = raw_line.strip().split()
                if not fields:
                    continue
                if len(fields) == 5:
                    stats["box_only_annotations"] += 1
                    continue
                expected = 5 + 17 * dimensions
                if len(fields) != expected:
                    stats["malformed_pose_annotations"] += 1
                    continue
                try:
                    box_values = [float(value) for value in fields[1:5]]
                    raw_pose = [float(value) for value in fields[5:]]
                except ValueError:
                    stats["malformed_pose_annotations"] += 1
                    continue
                expanded: list[float] = []
                for index in range(17):
                    values = raw_pose[index * dimensions : index * dimensions + dimensions]
                    x, y = values[:2]
                    visibility = values[2] if dimensions == 3 else int(x != 0 or y != 0) * 2
                    expanded.extend([x * width, y * height, visibility])
                keypoints = map_and_clean_keypoints(
                    expanded, {index: index for index in range(17)}, width, height
                )
                count = visible_count(keypoints)
                if count < min_keypoints:
                    stats["low_keypoint_pose_annotations"] += 1
                    continue
                bbox = clean_bbox(yolo_bbox(box_values, width, height), keypoints, width, height)
                if bbox is None:
                    stats["invalid_bbox_annotations"] += 1
                    continue
                instances.append(PoseInstance(bbox, keypoints, count))
                stats["pose_annotations_kept"] += 1
            if not instances:
                stats["images_without_usable_pose"] += 1
                continue
            samples.append(
                ImageSample(
                    source_name=source.name,
                    source_path=image_path,
                    original_name=image_path.name,
                    source_split=folder_name,
                    width=width,
                    height=height,
                    pixel_hash=pixel_hash,
                    group=infer_group(image_path.name),
                    instances=instances,
                )
            )
            stats["images_kept_before_deduplication"] += 1
    return samples, stats, {"coco17_yolo_pose"}


def deduplicate_samples(
    samples: list[ImageSample], report: dict[str, Any]
) -> list[ImageSample]:
    selected: dict[str, ImageSample] = {}
    for sample in samples:
        current = selected.get(sample.pixel_hash)
        if current is None:
            selected[sample.pixel_hash] = sample
            continue
        report["cleaning"]["exact_duplicate_images"] += 1
        if sample.quality > current.quality:
            selected[sample.pixel_hash] = sample
            report["cleaning"]["duplicates_replaced_by_better_annotation"] += 1
    return sorted(selected.values(), key=lambda item: item.pixel_hash)


def assign_grouped_splits(
    samples: list[ImageSample], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[ImageSample]]:
    groups: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)
    if len(groups) < 3:
        raise PreparationError(
            "At least three independent filename/video groups are required for "
            "leakage-aware train/validation/test splits"
        )
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    targets = {split: len(samples) * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in SPLIT_NAMES}
    output: dict[str, list[ImageSample]] = {split: [] for split in SPLIT_NAMES}

    def group_order(item: tuple[str, list[ImageSample]]) -> tuple[int, str]:
        name, records = item
        tie = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
        return (-len(records), tie)

    ordered_groups = sorted(groups.items(), key=group_order)
    for group_index, (group_name, records) in enumerate(ordered_groups):
        remaining = {split: targets[split] - counts[split] for split in SPLIT_NAMES}
        groups_left = len(ordered_groups) - group_index
        empty_splits = [split for split in SPLIT_NAMES if not output[split]]
        eligible = empty_splits if len(empty_splits) == groups_left else list(SPLIT_NAMES)
        best_remaining = max(remaining[split] for split in eligible)
        candidates = [
            split
            for split in eligible
            if abs(remaining[split] - best_remaining) < 1e-9
        ]
        chosen = min(
            candidates,
            key=lambda split: hashlib.sha256(
                f"{seed}:{group_name}:{split}".encode("utf-8")
            ).hexdigest(),
        )
        for sample in records:
            sample.split = chosen
            output[chosen].append(sample)
        counts[chosen] += len(records)

    if any(not output[split] for split in SPLIT_NAMES):
        raise PreparationError(f"Grouped split produced an empty partition: {counts}")
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_coco_payload(
    samples: list[ImageSample], image_ids: dict[str, int], annotation_start: int = 0
) -> tuple[dict[str, Any], int]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = annotation_start
    for sample in sorted(samples, key=lambda item: item.pixel_hash):
        image_id = image_ids[sample.pixel_hash]
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{sample.output_name}",
                "width": sample.width,
                "height": sample.height,
                "split": sample.split,
                "source_dataset": sample.source_name,
                "original_file_name": sample.original_name,
                "group": sample.group,
            }
        )
        for instance in sample.instances:
            annotation_id += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": instance.bbox,
                    "area": instance.bbox[2] * instance.bbox[3],
                    "iscrowd": 0,
                    "num_keypoints": instance.num_keypoints,
                    "keypoints": instance.keypoints,
                }
            )
    return {
        "info": {
            "description": "Combined and cleaned combat-sports COCO-17 pose dataset",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [category_record()],
    }, annotation_id


def validate_payload(payload: dict[str, Any], output_root: Path) -> None:
    image_ids = {image["id"] for image in payload["images"]}
    if len(image_ids) != len(payload["images"]):
        raise PreparationError("Duplicate image IDs in generated COCO data")
    annotation_ids = {annotation["id"] for annotation in payload["annotations"]}
    if len(annotation_ids) != len(payload["annotations"]):
        raise PreparationError("Duplicate annotation IDs in generated COCO data")
    annotated_images = {annotation["image_id"] for annotation in payload["annotations"]}
    if annotated_images != image_ids:
        raise PreparationError("Generated COCO data contains an image without a pose")
    for image in payload["images"]:
        if not (output_root / image["file_name"]).is_file():
            raise PreparationError(f"Generated image is missing: {image['file_name']}")
    for annotation in payload["annotations"]:
        if annotation["image_id"] not in image_ids:
            raise PreparationError("Annotation references a missing image")
        if len(annotation["keypoints"]) != 51:
            raise PreparationError("Annotation does not contain COCO-17 keypoints")
        if annotation["num_keypoints"] != visible_count(annotation["keypoints"]):
            raise PreparationError("Incorrect num_keypoints in generated annotation")


def render_previews(
    samples: list[ImageSample], output_root: Path, count: int, seed: int
) -> None:
    if count == 0:
        return
    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected: list[ImageSample] = []
    by_source: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        by_source[sample.source_name].append(sample)
    if count >= len(by_source):
        for source_name in sorted(by_source):
            selected.append(rng.choice(by_source[source_name]))
    remaining = [sample for sample in samples if sample not in selected]
    selected.extend(rng.sample(remaining, min(count - len(selected), len(remaining))))
    for index, sample in enumerate(selected, start=1):
        with Image.open(output_root / "images" / sample.output_name).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for instance in sample.instances:
                x, y, width, height = instance.bbox
                draw.rectangle((x, y, x + width, y + height), outline=(255, 64, 64), width=2)
                points = [
                    tuple(instance.keypoints[offset : offset + 3])
                    for offset in range(0, 51, 3)
                ]
                for start, end in COCO_SKELETON:
                    first, second = points[start - 1], points[end - 1]
                    if first[2] > 0 and second[2] > 0:
                        draw.line(
                            (first[0], first[1], second[0], second[1]),
                            fill=(0, 255, 255),
                            width=2,
                        )
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
                            fill=(0, 255, 0) if visibility == 2 else (255, 180, 0),
                        )
            image.save(
                preview_dir
                / (
                    f"{index:02d}_{sample.split}_"
                    f"{_normalized_name(sample.source_name)[:32]}_{sample.output_name}"
                ),
                quality=92,
            )


def prepare_output(output_root: Path, input_root: Path, overwrite: bool) -> None:
    output_root = output_root.resolve()
    input_root = input_root.resolve()
    if output_root in (PROJECT_ROOT.resolve(), input_root, Path(output_root.anchor)):
        raise PreparationError(f"Refusing unsafe output path: {output_root}")
    if output_root.exists():
        if any(output_root.iterdir()):
            if not overwrite:
                raise PreparationError(
                    f"Output directory is not empty: {output_root}. Use --overwrite to replace it."
                )
            shutil.rmtree(output_root)
        else:
            output_root.rmdir()
    (output_root / "images").mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not input_root.is_dir():
        raise PreparationError(f"Input dataset directory does not exist: {input_root}")

    report: dict[str, Any] = {
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "settings": {
            "min_keypoints": args.min_keypoints,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "seed": args.seed,
        },
        "skipped_non_pose_archives": [],
        "skipped_archives_existing_directory": [],
        "sources": [],
        "cleaning": Counter(),
    }

    prepare_output(output_root, input_root, args.overwrite)
    # Keep staging beside the output. On Windows this avoids the very long
    # sandbox Temp prefix exceeding legacy path-length limits for Roboflow names.
    with tempfile.TemporaryDirectory(
        prefix=".pose_merge_", dir=output_root.parent
    ) as temporary:
        staging_root = Path(temporary)
        sources = discover_all_sources(input_root, staging_root, report)
        if not sources:
            raise PreparationError(
                f"No supported pose datasets were found under {input_root}"
            )

        all_samples: list[ImageSample] = []
        for source in sources:
            if source.kind == "coco":
                samples, stats, schemas = process_coco_source(source, args.min_keypoints)
            elif source.kind == "yolo_pose":
                samples, stats, schemas = process_yolo_source(source, args.min_keypoints)
            else:
                raise PreparationError(f"Unknown source kind: {source.kind}")
            all_samples.extend(samples)
            report["sources"].append(
                {
                    "name": source.name,
                    "format": source.kind,
                    "schemas": sorted(schemas),
                    "statistics": dict(sorted(stats.items())),
                }
            )
            report["cleaning"].update(stats)
            print(
                f"{source.name}: kept {len(samples)} images / "
                f"{stats['pose_annotations_kept']} poses",
                flush=True,
            )

        samples = deduplicate_samples(all_samples, report)
        if not samples:
            raise PreparationError("Every discovered image was removed by cleaning")
        split_samples = assign_grouped_splits(
            samples, args.train_ratio, args.val_ratio, args.seed
        )

        for sample in samples:
            extension = sample.source_path.suffix.lower()
            if extension == ".jpeg":
                extension = ".jpg"
            if extension not in IMAGE_EXTENSIONS:
                extension = ".jpg"
            sample.output_name = f"{sample.pixel_hash[:24]}{extension}"
            shutil.copy2(sample.source_path, output_root / "images" / sample.output_name)

        image_ids = {
            sample.pixel_hash: index
            for index, sample in enumerate(sorted(samples, key=lambda item: item.pixel_hash), start=1)
        }
        all_payload, _ = build_coco_payload(samples, image_ids)
        validate_payload(all_payload, output_root)
        write_json(output_root / "annotations" / "all.json", all_payload)

        split_report: dict[str, Any] = {}
        group_sets: dict[str, set[str]] = {}
        for split in SPLIT_NAMES:
            payload, _ = build_coco_payload(split_samples[split], image_ids)
            validate_payload(payload, output_root)
            write_json(output_root / "annotations" / f"{split}.json", payload)
            group_sets[split] = {sample.group for sample in split_samples[split]}
            split_report[split] = {
                "images": len(payload["images"]),
                "poses": len(payload["annotations"]),
                "groups": len(group_sets[split]),
            }

        overlaps = {
            "train_val": sorted(group_sets["train"] & group_sets["val"]),
            "train_test": sorted(group_sets["train"] & group_sets["test"]),
            "val_test": sorted(group_sets["val"] & group_sets["test"]),
        }
        if any(overlaps.values()):
            raise PreparationError(f"Group leakage detected after splitting: {overlaps}")
        report["splits"] = split_report
        report["group_overlaps"] = overlaps
        report["combined"] = {
            "images": len(all_payload["images"]),
            "poses": len(all_payload["annotations"]),
            "groups": len({sample.group for sample in samples}),
        }
        report["cleaning"] = dict(sorted(report["cleaning"].items()))
        write_json(output_root / "dataset_report.json", report)
        render_previews(samples, output_root, args.render_samples, args.seed)

    print(f"\nCombined dataset: {output_root}")
    print(
        f"Kept {report['combined']['images']} images and "
        f"{report['combined']['poses']} pose instances"
    )
    for split in SPLIT_NAMES:
        values = report["splits"][split]
        print(
            f"{split:>5}: {values['images']} images, {values['poses']} poses, "
            f"{values['groups']} groups"
        )
    return report


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except (PreparationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
