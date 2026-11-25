"""Fuse independent proposal branches while preserving per-branch evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from common import PipelineError, read_jsonl, write_jsonl
from schema import canonicalize_event
from common import read_json


UNKNOWN = {"", "unknown", "unknown_action", "uncertain", "not_applicable", None}


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    start = max(float(left["start_sec"]), float(right["start_sec"]))
    end = min(float(left["end_sec"]), float(right["end_sec"]))
    intersection = max(0.0, end - start)
    union = max(float(left["end_sec"]), float(right["end_sec"])) - min(float(left["start_sec"]), float(right["start_sec"]))
    return intersection / union if union > 0 else 0.0


def _vote(rows: list[dict[str, Any]], key: str, confidence_key: str = "proposal_confidence") -> tuple[Any, float]:
    totals: Counter[Any] = Counter()
    for row in rows:
        value = row.get(key)
        if value not in UNKNOWN:
            totals[value] += max(0.01, float(row.get(confidence_key, row.get("attribute_confidence", 0.5))))
    if not totals:
        return "unknown", 0.0
    value, score = totals.most_common(1)[0]
    total = sum(totals.values())
    return value, float(score / total) if total else 0.0


def fuse(branch_rows: list[dict[str, Any]], schema: dict[str, Any], iou_threshold: float = 0.30) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    ordered = sorted(branch_rows, key=lambda row: (int(row.get("actor_id", -1)), float(row.get("start_sec", 0))))
    for row in ordered:
        source_id = row.get("source_event_id") or row.get("event_id")
        match = next((cluster for cluster in clusters if int(cluster[0].get("actor_id", -2)) == int(row.get("actor_id", -1)) and (
            any((member.get("source_event_id") or member.get("event_id")) == source_id for member in cluster)
            or temporal_iou(cluster[0], row) >= iou_threshold
            or abs(float(cluster[0].get("peak_sec", 0)) - float(row.get("peak_sec", 0))) <= 0.20
        )), None)
        if match is None:
            clusters.append([row])
        else:
            match.append(row)
    output: list[dict[str, Any]] = []
    for serial, cluster in enumerate(clusters, start=1):
        best = max(cluster, key=lambda row: float(row.get("proposal_confidence", 0.0)))
        fused = dict(best)
        fused["event_id"] = f"fused_{serial:06d}"
        fused["start_sec"] = round(min(float(row["start_sec"]) for row in cluster), 6)
        fused["end_sec"] = round(max(float(row["end_sec"]) for row in cluster), 6)
        peaks = sorted(float(row.get("peak_sec", 0.0)) for row in cluster)
        fused["peak_sec"] = round(peaks[len(peaks) // 2], 6)
        field_confidence: dict[str, float] = {}
        for key in ("technique", "action_family", "anatomical_side", "effector_limb", "stance_at_start", "stance_role", "target_zone", "outcome", "contact_surface", "visibility"):
            confidence_key = "attribute_confidence" if key in {"anatomical_side", "effector_limb", "stance_role"} else "stance_confidence" if key == "stance_at_start" else "proposal_confidence"
            value, confidence = _vote(cluster, key, confidence_key)
            if value not in UNKNOWN:
                fused[key] = value
            field_confidence[key] = round(confidence, 6)
        fused["proposal_confidence"] = round(max(field_confidence.values(), default=0.0), 6)
        fused["field_confidence"] = field_confidence
        fused["branch_evidence"] = [{
            "branch": row.get("proposal_branch", row.get("source", "unknown")),
            "event_id": row.get("event_id"),
            "confidence": row.get("proposal_confidence", row.get("attribute_confidence", 0.0)),
            "technique": row.get("technique", row.get("label")),
            "anatomical_side": row.get("anatomical_side", row.get("side")),
        } for row in cluster]
        fused["review_status"] = "proposed"
        fused["source"] = "fused_proposal"
        fused["proposal_branch"] = "fusion"
        flags = list(fused.get("uncertainty_flags", []))
        if fused.get("anatomical_side") not in UNKNOWN:
            flags = [flag for flag in flags if flag != "side_unclassified"]
        if fused.get("technique") not in UNKNOWN:
            flags = [flag for flag in flags if flag != "technique_unclassified"]
        fused["uncertainty_flags"] = sorted(set(flags))
        output.append(canonicalize_event(fused, schema))
    return sorted(output, key=lambda row: (float(row["start_sec"]), int(row.get("actor_id", -1))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--branches", type=Path, nargs="*")
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        output = project / "proposals" / "fused_proposals.jsonl"
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Output exists: {output}; use --overwrite")
        branch_paths = [path.resolve() for path in args.branches] if args.branches else sorted((project / "proposals" / "branches").glob("*.jsonl"))
        rows: list[dict[str, Any]] = []
        for path in branch_paths:
            if path.is_file():
                rows.extend(read_jsonl(path))
        if not rows:
            raise PipelineError("No proposal branch rows were found")
        schema = read_json(project / "annotations" / "classes.json")
        fused = fuse(rows, schema, args.iou)
        write_jsonl(output, fused)
        print(json.dumps({"input_rows": len(rows), "fused_proposals": len(fused), "output": str(output)}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
