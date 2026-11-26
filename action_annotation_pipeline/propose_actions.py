"""Generate high-recall action proposals from canonical RTMPose timelines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from action_signals import (
    action_components,
    build_fighter_timelines,
    compute_action_signals,
    family_candidates,
    fighter_map_coverage,
    hysteresis_bounds,
    require_fighter_map_coverage,
)
from common import PipelineError, load_fighter_map, load_manifest, write_json, write_jsonl
from schema import effector_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--high-threshold", type=float, default=2.0)
    parser.add_argument("--low-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-physical-speed", type=float, default=0.40)
    parser.add_argument("--minimum-peak-gap", type=float, default=0.18)
    parser.add_argument("--pre-seconds", type=float, default=0.35)
    parser.add_argument("--post-seconds", type=float, default=0.45)
    parser.add_argument("--max-gap-frames", type=int, default=6)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, help="Defaults to proposals/branches/pose_actionness.jsonl")
    return parser.parse_args()


def _target_zone(signal: dict[str, Any], peak: int) -> tuple[str, dict[str, float]]:
    values = {zone: float(scores[peak]) for zone, scores in signal["zone_convergence"].items()}
    zone, score = max(values.items(), key=lambda item: item[1])
    return (zone if score > 0.05 else "unknown"), values


def _event(
    serial: int,
    manifest: dict[str, Any],
    actor: int,
    effector_name: str,
    signal: dict[str, Any],
    peak: int,
    onset: int,
    offset: int,
    fps: float,
    duration: float,
    pre_seconds: float,
    post_seconds: float,
    component_id: str,
    subaction_index: int,
) -> dict[str, Any]:
    peak_sec = peak / fps
    # Store physical action boundaries in the canonical interval. Context is
    # retained separately for VLM/PoseC3D clip construction and must not leak
    # into TriDet ground-truth segments.
    action_start = max(0.0, (onset - 0.5) / fps)
    action_end = min(duration, (offset + 0.5) / fps)
    families, limb, family_hint = family_candidates(signal, peak)
    target_zone, target_evidence = _target_zone(signal, peak)
    side = str(signal["side"])
    confidence = min(0.99, 0.45 + 0.12 * float(signal["actionness"][peak]))
    return {
        "event_id": f"pose_action_{serial:06d}",
        "video_id": manifest["video_id"],
        "event_type": "action",
        "actor_id": actor,
        "target_id": 1 - actor if actor in (0, 1) else -1,
        "start_sec": round(action_start, 6),
        "peak_sec": round(peak_sec, 6),
        "end_sec": round(action_end, 6),
        "action_start_sec": round(action_start, 6),
        "action_end_sec": round(action_end, 6),
        "context_start_sec": round(max(0.0, action_start - pre_seconds), 6),
        "context_end_sec": round(min(duration, action_end + post_seconds), 6),
        "proxy_fps": fps,
        "proxy_start_frame": int(round(action_start * fps)),
        "proxy_peak_frame": int(peak),
        "proxy_end_frame": int(round(action_end * fps)),
        "label": "unknown_action",
        "technique": "unknown_action",
        "technique_candidates": [],
        "action_family": "unknown",
        "family_candidates": families,
        "suggested_family": family_hint,
        "anatomical_side": side,
        "side": side,
        "effector_limb": limb,
        "effector_id": effector_id(side, limb),
        "pose_effector_id": effector_name,
        "stance_at_start": "unknown",
        "stance_role": "unknown",
        "target_zone": target_zone,
        "outcome": "uncertain",
        "contact_surface": "unknown",
        "visibility": "visible" if float(signal["confidence"][peak]) >= 0.35 else "partially_occluded",
        "occluded": bool(float(signal["confidence"][peak]) < 0.35),
        "uncertainty_flags": ["technique_unclassified"],
        "round": 0,
        "combo_id": None,
        "counter_to_event_id": None,
        "possible_cause": None,
        "review_status": "proposed",
        "annotator": "automatic_pose_actionness",
        "notes": f"pose actionness={float(signal['actionness'][peak]):.3f}; effector={effector_name}",
        "source": "proposal",
        "proposal_branch": "pose_actionness",
        "proposal_confidence": round(confidence, 6),
        "attribute_confidence": round(float(signal["confidence"][peak]), 6),
        "actionness_score": round(float(signal["actionness"][peak]), 6),
        "annotation_revision": 1,
        "provenance": {
            "source": "pose_actionness",
            "effector": effector_name,
            "motion_component_id": component_id,
            "subaction_index": subaction_index,
            "peak_frame": peak,
            "evidence": {
                "raw_score": round(float(signal["raw"][peak]), 6),
                "distal_speed": round(float(signal["distal_speed"][peak]), 6),
                "middle_speed": round(float(signal["middle_speed"][peak]), 6),
                "acceleration": round(float(signal["acceleration"][peak]), 6),
                "extension_speed": round(float(signal["extension_speed"][peak]), 6),
                "target_convergence": round(float(signal["convergence"][peak]), 6),
                "target_zone_convergence": {key: round(value, 6) for key, value in target_evidence.items()},
                "pose_confidence": round(float(signal["confidence"][peak]), 6),
                "robust_statistics": signal["statistics"],
            },
        },
    }


def generate_proposals(
    data: Any,
    manifest: dict[str, Any],
    fighter_map: dict[str, Any],
    *,
    high_threshold: float = 2.0,
    low_threshold: float = 0.75,
    minimum_physical_speed: float = 0.40,
    minimum_peak_gap: float = 0.18,
    pre_seconds: float = 0.35,
    post_seconds: float = 0.45,
    max_gap_frames: int = 6,
    smoothing_window: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fps = float(manifest["proxy_video"]["fps"])
    duration = float(manifest["proxy_video"]["duration_sec"])
    expected = int(manifest.get("expected_fighters", 2))
    coverage = fighter_map_coverage(
        fighter_map,
        np.asarray(data["timestamps"]),
        np.asarray(data["track_ids"]),
        expected,
        canonical_fighters=np.asarray(data["fighter_ids"])
        if "fighter_ids" in (set(data.files) if hasattr(data, "files") else set(data.keys())) else None,
    )
    require_fighter_map_coverage(coverage)
    timelines, timeline_report = build_fighter_timelines(
        data,
        fighter_map,
        expected,
        fps,
        max_gap_frames=max_gap_frames,
        smoothing_window=smoothing_window,
    )
    proposals: list[dict[str, Any]] = []
    signal_summary: dict[str, Any] = {}
    serial = 1
    separation_frames = max(1, int(math.ceil(minimum_peak_gap * fps)))
    for actor, timeline in timelines.items():
        opponent = timelines.get(1 - actor) if actor in (0, 1) else None
        signals = compute_action_signals(timeline, opponent, fps)
        signal_summary[str(actor)] = {}
        for effector_name, signal in signals.items():
            components = action_components(
                signal["actionness"], high_threshold, low_threshold, separation_frames
            )
            components = [
                component for component in components
                if float(signal["distal_speed"][component[0]]) >= minimum_physical_speed
            ]
            # A low-variance sequence can make a strict robust threshold too
            # conservative. Keep one physically meaningful fallback peak so
            # the reviewer sees evidence instead of a silent empty file.
            if not components and len(signal["raw"]) > 2:
                fallback = int(np.argmax(signal["raw"]))
                if 0 < fallback < len(signal["raw"]) - 1 and float(signal["distal_speed"][fallback]) >= minimum_physical_speed:
                    onset, offset = hysteresis_bounds(signal["actionness"], fallback, low_threshold)
                    components = [(fallback, onset, offset)]
            signal_summary[str(actor)][effector_name] = {
                "components": len(components),
                "maximum_actionness": round(float(np.max(signal["actionness"])), 6),
                "maximum_raw_score": round(float(np.max(signal["raw"])), 6),
                "statistics": signal["statistics"],
            }
            for component_index, (peak, onset, offset) in enumerate(components):
                component_id = f"fighter{actor}:{effector_name}:{onset}-{offset}"
                proposals.append(_event(
                    serial, manifest, actor, effector_name, signal, peak, onset, offset,
                    fps, duration, pre_seconds, post_seconds, component_id, component_index,
                ))
                serial += 1
    deduplicated: dict[tuple[int, str, float, float], dict[str, Any]] = {}
    for proposal in proposals:
        key = (
            int(proposal["actor_id"]),
            str(proposal["pose_effector_id"]),
            round(float(proposal["action_start_sec"]), 4),
            round(float(proposal["action_end_sec"]), 4),
        )
        current = deduplicated.get(key)
        if current is None or float(proposal["actionness_score"]) > float(current["actionness_score"]):
            deduplicated[key] = proposal
    proposals = list(deduplicated.values())
    proposals.sort(key=lambda row: (float(row["start_sec"]), int(row["actor_id"]), float(row["peak_sec"])))
    report = {"coverage": coverage, "timeline": timeline_report, "signals": signal_summary, "proposal_count": len(proposals)}
    return proposals, report


def main() -> int:
    args = parse_args()
    try:
        if args.high_threshold <= args.low_threshold:
            raise PipelineError("--high-threshold must be greater than --low-threshold")
        if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
            raise PipelineError("--smoothing-window must be a positive odd integer")
        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        fighter_map = load_fighter_map(project)
        poses_path = project / "poses" / "poses.npz"
        output = args.output.expanduser().resolve() if args.output else project / "proposals" / "branches" / "pose_actionness.jsonl"
        if not poses_path.is_file():
            raise PipelineError(f"Pose file not found: {poses_path}")
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Proposals already exist: {output}; pass --overwrite")
        data = np.load(poses_path)
        proposals, report = generate_proposals(
            data,
            manifest,
            fighter_map,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            minimum_physical_speed=args.minimum_physical_speed,
            minimum_peak_gap=args.minimum_peak_gap,
            pre_seconds=args.pre_seconds,
            post_seconds=args.post_seconds,
            max_gap_frames=args.max_gap_frames,
            smoothing_window=args.smoothing_window,
        )
        write_jsonl(output, proposals)
        write_jsonl(project / "proposals" / "action_proposals.jsonl", proposals)
        write_json(project / "qa" / "pose_actionness_report.json", report)
        print(json.dumps({"proposals": len(proposals), "output": str(output), "report": str(project / 'qa' / 'pose_actionness_report.json')}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
