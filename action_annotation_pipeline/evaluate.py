"""Evaluate proposal quality and dataset readiness against reviewed events."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from common import PipelineError, read_json, read_jsonl, write_json
from schema import canonicalize_event
from validate_events import validate


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersection = max(0.0, min(float(left["end_sec"]), float(right["end_sec"])) - max(float(left["start_sec"]), float(right["start_sec"])))
    union = max(float(left["end_sec"]), float(right["end_sec"])) - min(float(left["start_sec"]), float(right["start_sec"]))
    return intersection / union if union > 0 else 0.0


def evaluate(project: Path, threshold: float = 0.30) -> dict[str, Any]:
    schema = read_json(project / "annotations" / "classes.json")
    events = [canonicalize_event(row, schema) for row in read_jsonl(project / "annotations" / "events.jsonl")]
    truth = [row for row in events if row.get("event_type") == "action" and row.get("review_status") == "verified"]
    proposal_path = project / "proposals" / "fused_proposals.jsonl"
    if not proposal_path.is_file() or not proposal_path.stat().st_size:
        proposal_path = project / "proposals" / "action_proposals.jsonl"
    proposals = [canonicalize_event(row, schema) for row in read_jsonl(proposal_path)] if proposal_path.is_file() and proposal_path.stat().st_size else []
    matches = []
    for event in truth:
        candidates = [row for row in proposals if int(row.get("actor_id", -2)) == int(event.get("actor_id", -1))]
        best = max(candidates, key=lambda row: temporal_iou(event, row), default=None)
        iou = temporal_iou(event, best) if best else 0.0
        technique_candidates = []
        if best:
            technique_candidates.append(best.get("technique", best.get("label")))
            technique_candidates.extend(item.get("technique") for item in best.get("branch_evidence", []) if item.get("technique"))
        top3 = list(dict.fromkeys(technique_candidates))[:3]
        visible_side = event.get("anatomical_side") in {"left", "right", "bilateral"} and event.get("visibility") != "heavily_occluded"
        matches.append({
            "event_id": event["event_id"], "matched": iou >= threshold, "tiou": iou,
            "peak_error_frames": abs(float(event.get("peak_sec", 0)) - float(best.get("peak_sec", 0))) * 30.0 if best else None,
            "top3_technique_hit": event.get("technique") in top3,
            "side_evaluated": visible_side,
            "side_correct": bool(best and best.get("anatomical_side") == event.get("anatomical_side")) if visible_side else None,
        })
    recalled = [row for row in matches if row["matched"]]
    peak_errors = [row["peak_error_frames"] for row in recalled if row["peak_error_frames"] is not None]
    side_rows = [row for row in matches if row["side_evaluated"]]
    validation_errors, validation_warnings, validation_report = validate(project, project / "annotations" / "events.jsonl", False, False)
    metrics = {
        "verified_actions": len(truth),
        "proposals": len(proposals),
        "proposal_recall_tiou_0.3": len(recalled) / len(truth) if truth else None,
        "top3_technique_recall": sum(row["top3_technique_hit"] for row in matches) / len(truth) if truth else None,
        "anatomical_side_accuracy_visible": sum(bool(row["side_correct"]) for row in side_rows) / len(side_rows) if side_rows else None,
        "median_peak_error_frames_30fps": statistics.median(peak_errors) if peak_errors else None,
        "reviewed_duration_sec": validation_report["reviewed_duration_sec"],
        "video_duration_sec": validation_report["video_duration_sec"],
    }
    gates = {
        "valid_annotations": not validation_errors,
        "proposal_recall_at_least_0.95": metrics["proposal_recall_tiou_0.3"] is not None and metrics["proposal_recall_tiou_0.3"] >= 0.95,
        "top3_recall_at_least_0.90": metrics["top3_technique_recall"] is not None and metrics["top3_technique_recall"] >= 0.90,
        "side_accuracy_at_least_0.90": metrics["anatomical_side_accuracy_visible"] is not None and metrics["anatomical_side_accuracy_visible"] >= 0.90,
        "median_peak_error_at_most_4_frames": metrics["median_peak_error_frames_30fps"] is not None and metrics["median_peak_error_frames_30fps"] <= 4.0,
    }
    return {"metrics": metrics, "acceptance_gates": gates, "pilot_pass": all(gates.values()), "validation_errors": validation_errors, "validation_warnings": validation_warnings, "matches": matches}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        report = evaluate(project)
        write_json(project / "qa" / "pilot_evaluation.json", report)
        print(json.dumps(report, indent=2))
        return 0 if not report["validation_errors"] else 1
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
