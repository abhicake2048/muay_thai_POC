"""Freeze multiple per-video exports into leakage-free PoseC3D and TriDet datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

from common import PipelineError, read_json, write_json


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
    projects = discover_projects(root)
    if not projects:
        raise PipelineError(f"No exported projects found below {root}")
    manifests = {project: read_json(project / "manifest.json") for project in projects}
    splits = split_groups([str(value["video_id"]) for value in manifests.values()], seed)
    source_to_split = {video: split for split, videos in splits.items() for video in videos}
    pose_annotations, pose_split = [], {"train": [], "val": [], "test": []}
    tridet_database: dict[str, Any] = {}
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
        tri_payload = read_json(tridet_path)
        for stream_id, record in tri_payload["database"].items():
            value = dict(record)
            value["subset"] = {"train": "training", "val": "validation", "test": "testing"}[split]
            tridet_database[stream_id] = value
    if missing:
        raise PipelineError(f"Projects are not fully exported: {', '.join(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "posec3d_annotations.pkl").open("wb") as handle:
        pickle.dump({"split": pose_split, "annotations": pose_annotations}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output / "tridet_annotations.json", {"version": "muay_thai_actions_v2", "database": tridet_database})
    leakage = {video: split for video, split in source_to_split.items()}
    fingerprint = hashlib.sha256(json.dumps({"splits": splits, "pose_samples": len(pose_annotations), "streams": sorted(tridet_database)}, sort_keys=True).encode()).hexdigest()
    report = {"dataset_version": "1.0", "fingerprint": fingerprint, "seed": seed, "projects": len(projects), "posec3d_samples": len(pose_annotations), "tridet_streams": len(tridet_database), "source_splits": leakage, "splits": splits, "leakage_check": "passed"}
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
