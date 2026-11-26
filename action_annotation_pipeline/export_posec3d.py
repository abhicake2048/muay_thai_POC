"""Export verified action events and dense poses to MMAction2 PoseC3D format."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import PipelineError, load_fighter_map, load_manifest, raw_track_for_fighter, read_json, read_jsonl, write_json, write_jsonl
from validate_events import validate
from schema import canonicalize_event, flatten_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--context-before", type=float, default=0.40)
    parser.add_argument("--context-after", type=float, default=0.45)
    parser.add_argument("--background-ratio", type=float, default=0.35)
    parser.add_argument("--background-window", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=48, help="Uniform temporal samples per clip; 0 keeps every proxy frame")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def overlaps(start: float, end: float, intervals: list[tuple[float, float]], margin: float = 0.0) -> bool:
    return any(start < other_end + margin and end > other_start - margin for other_start, other_end in intervals)


def make_annotation(
    sample_id: str,
    start: float,
    end: float,
    actor: int,
    target: int,
    label_id: int,
    fps: float,
    width: int,
    height: int,
    pose_lookup: dict[tuple[int, int], int],
    pose_data: Any,
    fighter_map: dict[str, Any],
    sample_frames: int = 0,
    canonical_lookup: bool = False,
) -> dict[str, Any]:
    first_frame = max(0, int(np.floor(start * fps)))
    last_frame = max(first_frame, int(np.ceil(end * fps)))
    dense_indices = np.arange(first_frame, last_frame + 1, dtype=np.int32)
    if sample_frames > 0 and len(dense_indices) > 0:
        frame_indices = dense_indices[np.rint(np.linspace(0, len(dense_indices) - 1, sample_frames)).astype(np.int32)]
    else:
        frame_indices = dense_indices
    keypoint = np.zeros((2, len(frame_indices), 17, 2), dtype=np.float32)
    keypoint_score = np.zeros((2, len(frame_indices), 17), dtype=np.float32)
    for time_index, frame_index in enumerate(frame_indices):
        timestamp = float(frame_index / fps)
        for person_slot, fighter_id in enumerate((actor, target)):
            if fighter_id not in (0, 1):
                continue
            if canonical_lookup:
                pose_index = pose_lookup.get((int(frame_index), int(fighter_id)))
            else:
                raw_track = raw_track_for_fighter(fighter_map, fighter_id, timestamp)
                if raw_track is None:
                    continue
                pose_index = pose_lookup.get((int(frame_index), int(raw_track)))
            if pose_index is None:
                continue
            keypoint[person_slot, time_index] = pose_data["keypoints"][pose_index]
            keypoint_score[person_slot, time_index] = pose_data["keypoint_scores"][pose_index]
    return {
        "frame_dir": sample_id,
        "total_frames": int(len(frame_indices)),
        "img_shape": (height, width),
        "original_shape": (height, width),
        "label": int(label_id),
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
    }


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        output_dir = project / "exports" / "posec3d"
        output_pickle = output_dir / "posec3d_annotations.pkl"
        if output_pickle.exists() and not args.overwrite:
            raise PipelineError(f"Output exists: {output_pickle}; pass --overwrite")
        errors, _, report = validate(project, project / "annotations" / "events.jsonl", False, False)
        if errors:
            raise PipelineError(f"Event validation failed with {len(errors)} errors; inspect qa/event_validation_report.json")
        manifest = load_manifest(project)
        schema = read_json(project / "annotations" / "classes.json")
        fighter_map = load_fighter_map(project)
        events = read_jsonl(project / "annotations" / "events.jsonl")
        pose_path = project / "poses" / "poses.npz"
        if not pose_path.is_file():
            raise PipelineError(f"Pose file not found: {pose_path}")
        pose_data = np.load(pose_path)
        has_canonical_fighters = "fighter_ids" in pose_data.files
        lookup_id_values = pose_data["fighter_ids"] if has_canonical_fighters else pose_data["track_ids"]
        pose_lookup = {
            (int(frame), int(identity)): index
            for index, (frame, identity) in enumerate(zip(pose_data["frame_indices"], lookup_id_values))
        }
        fps = float(manifest["proxy_video"]["fps"])
        width = int(manifest["proxy_video"]["width"])
        height = int(manifest["proxy_video"]["height"])
        duration = float(manifest["proxy_video"]["duration_sec"])
        labels = list(schema["posec3d_classes"])
        label_map = {label: index for index, label in enumerate(labels)}
        verified_actions = [
            canonicalize_event(event, schema) for event in events
            if event.get("event_type") in {"action", "balance"} and event.get("review_status") == "verified"
        ]
        attribute_values = {
            "anatomical_side": list(schema["anatomical_sides"]),
            "stance_role": list(schema["stance_roles"]),
            "target_zone": list(schema["target_zones"]),
            "outcome": list(schema["outcomes"]),
        }
        attribute_maps = {name: {value: index for index, value in enumerate(values)} for name, values in attribute_values.items()}
        annotations: list[dict[str, Any]] = []
        metadata_rows: list[dict[str, Any]] = []
        for event in verified_actions:
            start = max(0.0, float(event["start_sec"]) - args.context_before)
            end = min(duration, float(event["end_sec"]) + args.context_after)
            label = str(event.get("technique") or event["label"])
            sample_id = f"{manifest['video_id']}__{event['event_id']}__actor{event['actor_id']}"
            annotation = make_annotation(
                sample_id, start, end, int(event["actor_id"]), int(event.get("target_id", -1)),
                label_map[label], fps, width, height, pose_lookup, pose_data, fighter_map, args.num_frames,
                canonical_lookup=has_canonical_fighters,
            )
            annotation["source_event_id"] = str(event["event_id"])
            annotation["attribute_labels"] = {}
            annotation["attribute_masks"] = {}
            for name, value_map in attribute_maps.items():
                value = str(event.get(name, "uncertain" if name == "outcome" else "unknown"))
                if value not in value_map:
                    value = "uncertain" if name == "outcome" else "unknown"
                annotation["attribute_labels"][name] = int(value_map[value])
                annotation["attribute_masks"][name] = value not in {"unknown", "uncertain", "not_applicable"}
            annotations.append(annotation)
            metadata_rows.append({
                "sample_id": sample_id, "source_event_id": event["event_id"], "label": label,
                "actor_id": int(event["actor_id"]), "target_id": int(event.get("target_id", -1)),
                "clip_start_sec": start, "clip_end_sec": end,
                "event_type": event.get("event_type"), "action_family": event.get("action_family", "balance"), "technique": label,
                "anatomical_side": event.get("anatomical_side"), "effector_limb": event.get("effector_limb"),
                "effector_id": event.get("effector_id"), "stance_at_start": event.get("stance_at_start"),
                "stance_role": event.get("stance_role"), "target_zone": event.get("target_zone"),
                "outcome": event.get("outcome"), "contact_surface": event.get("contact_surface"),
                "visibility": event.get("visibility"), "flat_label": flatten_label(event),
                "source_video": manifest["video_id"],
            })

        desired_background = int(round(len(annotations) * max(0.0, args.background_ratio)))
        review_ranges = [(float(e["start_sec"]), float(e["end_sec"])) for e in events if e.get("event_type") == "review_range" and e.get("review_status") == "verified"]
        excluded = [(float(e["start_sec"]), float(e["end_sec"])) for e in events if e.get("event_type") in ("action", "ignore") and e.get("review_status") in ("verified", "uncertain")]
        candidates: list[tuple[float, float]] = []
        step = max(0.25, args.background_window / 2)
        for range_start, range_end in review_ranges:
            start = range_start
            while start + args.background_window <= range_end + 1e-6:
                end = start + args.background_window
                if not overlaps(start, end, excluded, margin=0.15):
                    candidates.append((start, end))
                start += step
        random.Random(args.seed).shuffle(candidates)
        for background_index, (start, end) in enumerate(candidates[:desired_background], start=1):
            actor = (background_index - 1) % 2
            target = 1 - actor
            sample_id = f"{manifest['video_id']}__background_{background_index:05d}__actor{actor}"
            annotations.append(make_annotation(
                sample_id, start, end, actor, target, label_map["background"], fps, width, height,
                pose_lookup, pose_data, fighter_map, args.num_frames,
                canonical_lookup=has_canonical_fighters,
            ))
            annotations[-1]["source_event_id"] = None
            annotations[-1]["attribute_labels"] = {name: -1 for name in attribute_maps}
            annotations[-1]["attribute_masks"] = {name: False for name in attribute_maps}
            metadata_rows.append({
                "sample_id": sample_id, "source_event_id": None, "label": "background",
                "actor_id": actor, "target_id": target, "clip_start_sec": start,
                "clip_end_sec": end, "source_video": manifest["video_id"],
            })

        if not annotations:
            raise PipelineError("No verified PoseC3D samples were produced")
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = [item["frame_dir"] for item in annotations]
        payload = {"split": {"train": sample_ids, "val": [], "test": []}, "annotations": annotations}
        with output_pickle.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(output_dir / "class_map.json", {
            "schema_version": "2.0", "classes": labels, "label_to_id": label_map,
            "attribute_classes": attribute_values, "attribute_to_id": attribute_maps,
            "masked_multitask_contract": "Use label for technique CE. For each attribute, compute CE only where attribute_masks[head] is true."
        })
        write_json(output_dir / "augmentation_policy.json", {
            "apply_at_training_time_only": True,
            "apply_after_source_video_split": True,
            "policy": {
                "temporal_center_jitter_sec": [-0.12, 0.12],
                "temporal_speed_range": [0.90, 1.10],
                "spatial_scale_range": [0.90, 1.10],
                "rotation_degrees": [-8.0, 8.0],
                "translation_fraction": [-0.02, 0.02],
                "keypoint_noise_sigma_fraction": 0.004,
                "keypoint_dropout_probability": 0.03,
                "horizontal_flip": False,
            },
            "safety_notes": [
                "Do not augment validation or test samples.",
                "Do not create augmented copies before source-video grouping.",
                "Horizontal flip is disabled because side/stance attributes require synchronized semantic handling.",
                "Keep actor in slot 0 and opponent in slot 1 after every transform.",
            ],
        })
        write_jsonl(output_dir / "sample_metadata.jsonl", metadata_rows)
        summary = {
            "samples": len(annotations), "action_samples": len(verified_actions),
            "verified_event_samples": len(verified_actions),
            "background_samples": len(annotations) - len(verified_actions),
            "output": str(output_pickle), "source_group": manifest["video_id"],
            "split_warning": "All samples belong to one source video and must remain in one split.",
            "reviewed_duration_sec": report["reviewed_duration_sec"],
            "uniform_sample_frames": args.num_frames,
            "identity_lookup": "canonical_fighter_id" if has_canonical_fighters else "legacy_raw_track_segments",
            "augmentation_policy": str(output_dir / "augmentation_policy.json"),
            "tensor_contract": {"keypoint": [2, args.num_frames or "T", 17, 2], "keypoint_score": [2, args.num_frames or "T", 17]},
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, KeyError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
