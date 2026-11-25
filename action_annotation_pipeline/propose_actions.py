"""Generate high-motion action-window proposals from dense RTMPose sequences."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    PipelineError,
    canonical_fighter_for_track,
    load_fighter_map,
    load_manifest,
    write_jsonl,
)
from schema import effector_id


UPPER_JOINTS = np.asarray([7, 8, 9, 10], dtype=int)  # elbows and wrists
LOWER_JOINTS = np.asarray([13, 14, 15, 16], dtype=int)  # knees and ankles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=88.0)
    parser.add_argument("--minimum-gap", type=float, default=0.45)
    parser.add_argument("--pre-seconds", type=float, default=0.45)
    parser.add_argument("--post-seconds", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, help="Defaults to proposals/branches/pose_motion.jsonl")
    return parser.parse_args()


def normalized_speed(
    points_a: np.ndarray,
    points_b: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    dt: float,
    diagonal: float,
) -> np.ndarray:
    valid = (scores_a >= 0.15) & (scores_b >= 0.15)
    speed = np.zeros(points_a.shape[0], dtype=np.float32)
    if dt > 0 and diagonal > 1:
        speed[valid] = np.linalg.norm(points_b[valid] - points_a[valid], axis=1) / diagonal / dt
    return speed


def main() -> int:
    args = parse_args()
    try:
        if not 50 <= args.percentile <= 99.9:
            raise PipelineError("--percentile must be between 50 and 99.9")
        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        fighter_map = load_fighter_map(project)
        poses_path = project / "poses" / "poses.npz"
        output = args.output.expanduser().resolve() if args.output else project / "proposals" / "branches" / "pose_motion.jsonl"
        if not poses_path.is_file():
            raise PipelineError(f"Pose file not found: {poses_path}")
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Proposals already exist: {output}; pass --overwrite")
        data = np.load(poses_path)
        frames = data["frame_indices"]
        times = data["timestamps"]
        raw_tracks = data["track_ids"]
        boxes = data["bboxes_xyxy"]
        points = data["keypoints"]
        scores = data["keypoint_scores"]
        duration = float(manifest["proxy_video"]["duration_sec"])

        grouped: dict[int, list[int]] = defaultdict(list)
        for index, raw_track in enumerate(raw_tracks):
            grouped[int(raw_track)].append(index)

        proposals: list[dict[str, object]] = []
        serial = 1
        for raw_track, indices_list in grouped.items():
            indices = np.asarray(sorted(indices_list, key=lambda i: frames[i]), dtype=int)
            if len(indices) < 4:
                continue
            motion_rows: list[tuple[float, float, float]] = []
            for previous, current in zip(indices[:-1], indices[1:]):
                dt = float(times[current] - times[previous])
                if dt <= 0 or dt > 0.25:
                    continue
                x1, y1, x2, y2 = boxes[current]
                diagonal = float(np.hypot(x2 - x1, y2 - y1))
                velocity = normalized_speed(points[previous], points[current], scores[previous], scores[current], dt, diagonal)
                upper = float(np.max(velocity[UPPER_JOINTS]))
                lower = float(np.max(velocity[LOWER_JOINTS]))
                motion_rows.append((float(times[current]), upper, lower))
            if len(motion_rows) < 4:
                continue
            values = np.asarray([max(upper, lower) for _, upper, lower in motion_rows], dtype=np.float32)
            threshold = max(0.15, float(np.percentile(values, args.percentile)))
            candidate_indices = [i for i in range(1, len(motion_rows) - 1) if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]]
            selected: list[int] = []
            for candidate in sorted(candidate_indices, key=lambda i: values[i], reverse=True):
                timestamp = motion_rows[candidate][0]
                if all(abs(timestamp - motion_rows[other][0]) >= args.minimum_gap for other in selected):
                    selected.append(candidate)
            for candidate in sorted(selected, key=lambda i: motion_rows[i][0]):
                peak, upper, lower = motion_rows[candidate]
                fighter_id = canonical_fighter_for_track(fighter_map, raw_track, peak)
                if fighter_id is None:
                    continue
                family = "punch_or_elbow" if upper > lower * 1.12 else "kick_or_knee" if lower > upper * 1.12 else "unknown_action"
                limb = "hand" if family == "punch_or_elbow" else "leg" if family == "kick_or_knee" else "unknown"
                start = max(0.0, peak - args.pre_seconds)
                end = min(duration, peak + args.post_seconds)
                proposals.append({
                    "event_id": f"proposal_{serial:06d}",
                    "video_id": manifest["video_id"],
                    "event_type": "action",
                    "actor_id": fighter_id,
                    "target_id": 1 - fighter_id if fighter_id in (0, 1) else -1,
                    "start_sec": round(start, 6),
                    "peak_sec": round(peak, 6),
                    "end_sec": round(end, 6),
                    "label": "unknown_action",
                    "technique": "unknown_action",
                    "action_family": "unknown",
                    "suggested_family": family,
                    "anatomical_side": "unknown",
                    "side": "unknown",
                    "effector_limb": limb,
                    "effector_id": effector_id("unknown", limb),
                    "stance_at_start": "unknown",
                    "stance_role": "unknown",
                    "target_zone": "unknown",
                    "outcome": "uncertain",
                    "contact_surface": "unknown",
                    "visibility": "unknown",
                    "occluded": False,
                    "uncertainty_flags": ["technique_unclassified", "side_unclassified"],
                    "round": 0,
                    "combo_id": None,
                    "counter_to_event_id": None,
                    "possible_cause": None,
                    "review_status": "proposed",
                    "annotator": "automatic_pose_motion",
                    "notes": f"motion score={values[candidate]:.4f}; family={family}; raw_track={raw_track}",
                    "source": "proposal",
                    "proposal_branch": "pose_motion",
                    "proposal_confidence": round(float(min(1.0, values[candidate] / max(threshold, 1e-6))), 6),
                    "annotation_revision": 1,
                    "provenance": {"source": "pose_motion", "raw_track_id": raw_track},
                })
                serial += 1
        proposals.sort(key=lambda row: (float(row["start_sec"]), int(row["actor_id"])))
        write_jsonl(output, proposals)
        legacy_output = project / "proposals" / "action_proposals.jsonl"
        if output != legacy_output:
            write_jsonl(legacy_output, proposals)
        print(json.dumps({"proposals": len(proposals), "output": str(output)}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
