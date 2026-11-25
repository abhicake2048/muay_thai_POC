"""Infer anatomical side, moving limb and provisional stance from COCO17 motion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import PipelineError, load_fighter_map, raw_track_for_fighter, read_jsonl, write_jsonl
from schema import effector_id


SIDE_JOINTS = {
    "left": {"hand": (7, 9), "elbow": (5, 7), "leg": (13, 15), "knee": (11, 13)},
    "right": {"hand": (8, 10), "elbow": (6, 8), "leg": (14, 16), "knee": (12, 14)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _motion_score(points: np.ndarray, scores: np.ndarray, indices: tuple[int, ...]) -> float:
    if len(points) < 2:
        return 0.0
    selected = points[:, indices]
    selected_scores = scores[:, indices]
    delta = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
    valid = (selected_scores[:-1] >= 0.15) & (selected_scores[1:] >= 0.15)
    values = delta[valid]
    return float(np.percentile(values, 90)) if values.size else 0.0


def _stance(points: np.ndarray, scores: np.ndarray, opponent_x: float | None) -> tuple[str, str, float]:
    if not len(points) or opponent_x is None or scores[-1, 15] < 0.15 or scores[-1, 16] < 0.15:
        return "unknown", "unknown", 0.0
    left_x, right_x = float(points[-1, 15, 0]), float(points[-1, 16, 0])
    center = float(np.mean(points[-1, [11, 12], 0]))
    direction = 1.0 if opponent_x >= center else -1.0
    left_progress = direction * left_x
    right_progress = direction * right_x
    separation = abs(left_progress - right_progress)
    lead = "left" if left_progress >= right_progress else "right"
    confidence = min(1.0, separation / 40.0)
    return ("orthodox" if lead == "left" else "southpaw"), lead, confidence


def infer(proposals: list[dict[str, Any]], project: Path, window: float) -> list[dict[str, Any]]:
    pose_path = project / "poses" / "poses.npz"
    if not pose_path.is_file():
        raise PipelineError(f"Pose file not found: {pose_path}")
    data = np.load(pose_path)
    fighter_map = load_fighter_map(project)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, track_id in enumerate(data["track_ids"]):
        grouped[int(track_id)].append(index)
    results: list[dict[str, Any]] = []
    for proposal in proposals:
        value = dict(proposal)
        actor = int(value.get("actor_id", -1))
        peak = float(value.get("peak_sec", 0.0))
        raw_track = raw_track_for_fighter(fighter_map, actor, peak)
        if raw_track is None:
            results.append(value)
            continue
        candidates = [index for index in grouped.get(raw_track, []) if abs(float(data["timestamps"][index]) - peak) <= window]
        candidates.sort(key=lambda index: float(data["timestamps"][index]))
        if len(candidates) < 2:
            results.append(value)
            continue
        points = data["keypoints"][candidates]
        scores = data["keypoint_scores"][candidates]
        family_hint = str(value.get("suggested_family", "unknown_action"))
        limb_choices = ("hand", "elbow") if family_hint == "punch_or_elbow" else ("leg", "knee") if family_hint == "kick_or_knee" else ("hand", "leg")
        scores_by_effector = {
            (side, limb): _motion_score(points, scores, SIDE_JOINTS[side][limb])
            for side in ("left", "right") for limb in limb_choices
        }
        (estimated_side, estimated_limb), best = max(scores_by_effector.items(), key=lambda item: item[1])
        ordered = sorted(scores_by_effector.values(), reverse=True)
        margin = (best - ordered[1]) / max(best, 1e-6) if len(ordered) > 1 else 0.0
        if best <= 0 or margin < 0.08:
            estimated_side = "unknown"
        proposed_side = str(value.get("anatomical_side", value.get("side", "unknown")))
        proposed_limb = str(value.get("effector_limb", "unknown"))
        # The actionness branch knows which anatomical limb generated its
        # peak. Preserve that evidence; the window-wide estimator is a fallback
        # for legacy motion proposals.
        side = proposed_side if proposed_side in {"left", "right"} else estimated_side
        limb = proposed_limb if proposed_limb not in {"", "unknown", "not_applicable"} else estimated_limb
        target = int(value.get("target_id", -1))
        target_track = raw_track_for_fighter(fighter_map, target, peak) if target >= 0 else None
        opponent_x = None
        if target_track is not None:
            opponent_candidates = [index for index in grouped.get(target_track, []) if abs(float(data["timestamps"][index]) - peak) <= 0.05]
            if opponent_candidates:
                opponent_x = float(np.mean(data["keypoints"][opponent_candidates[0], [11, 12], 0]))
        stance, lead_side, stance_confidence = _stance(points, scores, opponent_x)
        role = "lead" if side == lead_side and side != "unknown" else "rear" if side != "unknown" and lead_side != "unknown" else "unknown"
        value.update({
            "anatomical_side": side,
            "side": side,
            "effector_limb": limb,
            "effector_id": effector_id(side, limb),
            "stance_at_start": stance,
            "stance_role": role,
            "attribute_confidence": round(float(max(0.0, min(1.0, margin))), 6),
            "stance_confidence": round(float(stance_confidence), 6),
            "proposal_branch": "pose_attributes",
        })
        flags = [flag for flag in value.get("uncertainty_flags", []) if not (flag == "side_unclassified" and side != "unknown")]
        if side == "unknown":
            flags.append("side_motion_ambiguous")
        if stance == "unknown" or stance_confidence < 0.25:
            flags.append("stance_uncertain")
        value["uncertainty_flags"] = sorted(set(flags))
        value.setdefault("provenance", {})["pose_attribute_estimator"] = {
            "raw_track_id": raw_track, "motion_scores": {f"{key[0]}_{key[1]}": round(score, 4) for key, score in scores_by_effector.items()}
        }
        results.append(value)
    return results


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        source = args.input.expanduser().resolve() if args.input else project / "proposals" / "branches" / "pose_actionness.jsonl"
        output = args.output.expanduser().resolve() if args.output else project / "proposals" / "branches" / "pose_attributes.jsonl"
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Output exists: {output}; use --overwrite")
        rows = infer(read_jsonl(source), project, args.window)
        write_jsonl(output, rows)
        print(json.dumps({"proposals": len(rows), "output": str(output)}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
