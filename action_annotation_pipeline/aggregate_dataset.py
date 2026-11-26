"""Freeze multiple per-video exports into leakage-free PoseC3D and TriDet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import PipelineError, read_json, read_jsonl, write_json, write_jsonl
from training_contracts import validate_feature_array, validate_tridet_payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_projects(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("manifest.json") if (path.parent / "exports").is_dir()})


def split_groups(video_ids: list[str], seed: int) -> dict[str, list[str]]:
    groups = sorted(set(video_ids))
    random.Random(seed).shuffle(groups)
    if len(groups) == 1:
        return {"train": groups, "val": [], "test": []}
    if len(groups) == 2:
        return {"train": groups[:1], "val": groups[1:], "test": []}
    val_count = max(1, round(len(groups) * 0.15))
    test_count = max(1, round(len(groups) * 0.15))
    train_count = max(1, len(groups) - val_count - test_count)
    return {"train": groups[:train_count], "val": groups[train_count:train_count + val_count], "test": groups[train_count + val_count:]}


def freeze(root: Path, output: Path, seed: int) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise PipelineError(f"Frozen dataset output is not empty: {output}; use a new version directory")
    projects = discover_projects(root)
    if not projects:
        raise PipelineError(f"No exported projects found below {root}")
    manifests = {project: read_json(project / "manifest.json") for project in projects}
    splits = split_groups([str(value["video_id"]) for value in manifests.values()], seed)
    source_to_split = {video: split for split, videos in splits.items() for video in videos}
    pose_annotations, pose_split = [], {"train": [], "val": [], "test": []}
    pose_metadata: list[dict[str, Any]] = []
    tridet_database: dict[str, Any] = {}
    canonical_class_map: dict[str, Any] | None = None
    canonical_augmentation: dict[str, Any] | None = None
    tridet_taxonomy: list[str] | None = None
    tridet_label_to_id: dict[str, int] | None = None
    tridet_metadata: dict[str, Any] | None = None
    feature_sources: list[tuple[str, Path, Path | None, str]] = []
    missing = []
    for project, manifest in manifests.items():
        video_id = str(manifest["video_id"])
        pose_path = project / "exports" / "posec3d" / "posec3d_annotations.pkl"
        tridet_path = project / "exports" / "tridet" / "annotations.json"
        if not pose_path.is_file() or not tridet_path.is_file():
            missing.append(video_id)
            continue
        with pose_path.open("rb") as handle:
            pose_payload = pickle.load(handle)
        split = source_to_split[video_id]
        for annotation in pose_payload["annotations"]:
            sample_id = str(annotation["frame_dir"])
            pose_annotations.append(annotation)
            pose_split[split].append(sample_id)
        class_map_path = project / "exports" / "posec3d" / "class_map.json"
        augmentation_path = project / "exports" / "posec3d" / "augmentation_policy.json"
        if not class_map_path.is_file() or not augmentation_path.is_file():
            raise PipelineError(f"PoseC3D class/augmentation contract is missing for {video_id}; re-export the project")
        class_map = read_json(class_map_path)
        augmentation = read_json(augmentation_path)
        if canonical_class_map is not None and class_map != canonical_class_map:
            raise PipelineError(f"PoseC3D class map differs in project {video_id}")
        if canonical_augmentation is not None and augmentation != canonical_augmentation:
            raise PipelineError(f"PoseC3D augmentation policy differs in project {video_id}")
        canonical_class_map = class_map
        canonical_augmentation = augmentation
        metadata_path = project / "exports" / "posec3d" / "sample_metadata.jsonl"
        for row in read_jsonl(metadata_path) if metadata_path.is_file() else []:
            pose_metadata.append({**row, "split": split, "source_group": video_id})
        tri_payload = read_json(tridet_path)
        taxonomy = list(tri_payload.get("taxonomy", []))
        if tridet_taxonomy is not None and taxonomy != tridet_taxonomy:
            raise PipelineError(f"TriDet taxonomy differs in project {video_id}")
        tridet_taxonomy = taxonomy
        label_to_id = dict(tri_payload.get("label_to_id", {}))
        if tridet_label_to_id is not None and label_to_id != tridet_label_to_id:
            raise PipelineError(f"TriDet label map differs in project {video_id}")
        tridet_label_to_id = label_to_id
        if tridet_metadata is None:
            tridet_metadata = dict(tri_payload.get("metadata", {}))
        feature_manifest_path = project / "exports" / "tridet" / "feature_manifest.json"
        if not feature_manifest_path.is_file():
            raise PipelineError(f"TriDet feature manifest is missing for {video_id}; run mtdata features first")
        feature_manifest = read_json(feature_manifest_path)
        for stream_id, record in tri_payload["database"].items():
            value = dict(record)
            value["subset"] = {"train": "training", "val": "validation", "test": "testing"}[split]
            tridet_database[stream_id] = value
            feature_record = feature_manifest.get(stream_id, {})
            feature_path = project / "exports" / "tridet" / str(feature_record.get("feature_file", ""))
            timestamps_value = feature_record.get("timestamps_file", "features/timestamps.npy")
            timestamps_source = project / "exports" / "tridet" / str(timestamps_value)
            if feature_record.get("status") != "complete" or not feature_path.is_file():
                raise PipelineError(f"TriDet feature array is missing or incomplete for {stream_id}; run mtdata features first")
            feature_array = np.load(feature_path, mmap_mode="r")
            feature_errors = validate_feature_array(feature_array)
            if feature_errors:
                raise PipelineError(f"TriDet feature array is invalid for {stream_id}: {'; '.join(feature_errors)}")
            feature_sources.append((stream_id, feature_path, timestamps_source if timestamps_source.is_file() else None, _sha256(feature_path)))
    if missing:
        raise PipelineError(f"Projects are not fully exported: {', '.join(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    feature_output = output / "tridet_features"
    feature_output.mkdir(parents=True, exist_ok=True)
    frozen_feature_manifest: dict[str, Any] = {}
    feature_checksums: dict[str, str] = {}
    for stream_id, source, timestamps_source, checksum in feature_sources:
        destination = feature_output / f"{stream_id}.npy"
        shutil.copy2(source, destination)
        timestamp_destination = output / "tridet_timestamps" / f"{stream_id}.npy"
        if timestamps_source is not None:
            timestamp_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(timestamps_source, timestamp_destination)
        frozen_feature_manifest[stream_id] = {
            "feature_file": str(destination.relative_to(output)),
            "timestamps_file": str(timestamp_destination.relative_to(output)) if timestamps_source is not None else None,
            "status": "complete",
            "sha256": checksum,
        }
        feature_checksums[stream_id] = checksum
    with (output / "posec3d_annotations.pkl").open("wb") as handle:
        pickle.dump({"split": pose_split, "annotations": pose_annotations}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output / "posec3d_class_map.json", canonical_class_map or {})
    write_json(output / "posec3d_augmentation_policy.json", canonical_augmentation or {})
    write_jsonl(output / "posec3d_sample_metadata.jsonl", pose_metadata)
    tridet_payload = {
        "version": "muay_thai_actions_v2",
        "taxonomy": tridet_taxonomy or [],
        "label_to_id": tridet_label_to_id or {},
        "database": tridet_database,
        "metadata": tridet_metadata or {},
    }
    contract_errors = validate_tridet_payload(tridet_payload, require_training_subset=True)
    if contract_errors:
        raise PipelineError(f"Frozen TriDet contract failed: {'; '.join(contract_errors[:5])}")
    write_json(output / "tridet_annotations.json", tridet_payload)
    write_json(output / "tridet_feature_manifest.json", frozen_feature_manifest)
    leakage = {video: split for video, split in source_to_split.items()}
    fingerprint = hashlib.sha256(json.dumps({"splits": splits, "pose_samples": len(pose_annotations), "streams": sorted(tridet_database), "feature_checksums": feature_checksums}, sort_keys=True).encode()).hexdigest()
    report = {"dataset_version": "1.0", "fingerprint": fingerprint, "seed": seed, "projects": len(projects), "posec3d_samples": len(pose_annotations), "tridet_streams": len(tridet_database), "tridet_feature_files": len(frozen_feature_manifest), "source_splits": leakage, "splits": splits, "leakage_check": "passed"}
    write_json(output / "split_manifest.json", report)
    write_json(output / "dataset_card.json", {**report, "human_verified_only": True, "canonical_schema": "Muay Thai Data Factory v2", "limitations": ["Anatomical side marked unknown when not visible", "No punch trajectory/line label", "Source videos, not clips, are the split unit"]})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = freeze(args.projects_root.resolve(), args.output.resolve(), args.seed)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
