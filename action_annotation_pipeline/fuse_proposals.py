"""Fuse independent proposal branches while preserving per-branch evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import PipelineError, read_jsonl, write_jsonl
from schema import canonicalize_event, technique_family
from common import read_json


UNKNOWN = {"", "unknown", "unknown_action", "uncertain", "not_applicable", None}
LEARNED_BRANCHES = {"qwen_actions", "qwen3_vl", "posec3d", "boxing6_posec3d", "tridet"}


def _limb_group(row: dict[str, Any]) -> str:
    limb = str(row.get("effector_limb", "unknown"))
    if limb in {"hand", "elbow", "both_arms"}:
        return "upper"
    if limb in {"leg", "foot", "knee"}:
        return "lower"
    return "unknown"


def _compatible(left: dict[str, Any], right: dict[str, Any], iou_threshold: float = 0.30) -> bool:
    left_group, right_group = _limb_group(left), _limb_group(right)
    if left_group != "unknown" and right_group != "unknown" and left_group != right_group:
        return False
    left_pose_effector, right_pose_effector = left.get("pose_effector_id"), right.get("pose_effector_id")
    if left_pose_effector and right_pose_effector and left_pose_effector != right_pose_effector:
        return False
    if left.get("proposal_branch") in LEARNED_BRANCHES and right.get("proposal_branch") in LEARNED_BRANCHES:
        left_effector, right_effector = str(left.get("effector_id", "unknown")), str(right.get("effector_id", "unknown"))
        if left_effector not in UNKNOWN and right_effector not in UNKNOWN and left_effector != right_effector:
            return False
    peak_difference = abs(float(left.get("peak_sec", 0)) - float(right.get("peak_sec", 0)))
    left_branch, right_branch = str(left.get("proposal_branch", "")), str(right.get("proposal_branch", ""))
    # Two independently localized visual events at distinct peaks are a
    # combination, even when their broad context windows overlap heavily.
    if left_branch == right_branch == "qwen_actions" and peak_difference >= 0.18:
        return False
    return temporal_iou(left, right) >= iou_threshold or peak_difference <= 0.15


def _cluster_compatible(cluster: list[dict[str, Any]], row: dict[str, Any], iou_threshold: float) -> bool:
    if str(row.get("proposal_branch", "")) in {"qwen_actions", "qwen3_vl"}:
        for member in cluster:
            if str(member.get("proposal_branch", "")) in {"qwen_actions", "qwen3_vl"} and abs(float(member.get("peak_sec", 0)) - float(row.get("peak_sec", 0))) >= 0.18:
                return False
    return any(_compatible(member, row, iou_threshold) for member in cluster)


def _ranked_techniques(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    sources: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        branch = str(row.get("proposal_branch", row.get("source", "unknown")))
        confidence = max(0.01, float(row.get("proposal_confidence", 0.0)))
        technique = str(row.get("technique", row.get("label", "unknown_action")))
        if technique not in UNKNOWN and technique_family(schema, technique):
            scores[technique] += confidence
            sources[technique].add(branch)
        for candidate in row.get("technique_candidates", []):
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("technique", candidate.get("label", "")))
            if technique_family(schema, label):
                scores[label] += max(0.0, float(candidate.get("confidence", 0.0)))
                sources[label].add(str(candidate.get("source", branch)))
    total = max(sum(scores.values()), 1e-6)
    return [
        {"technique": label, "confidence": round(score / total, 6), "sources": sorted(sources[label])}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    ]


def _pose_families(rows: list[dict[str, Any]]) -> set[str]:
    families: set[str] = set()
    for row in rows:
        if str(row.get("proposal_branch", "")).startswith("pose"):
            for candidate in row.get("family_candidates", []):
                if isinstance(candidate, dict) and candidate.get("label"):
                    families.add(str(candidate["label"]))
            hint = str(row.get("suggested_family", ""))
            if hint == "punch_or_elbow":
                families.update(("punch", "elbow"))
            elif hint == "kick_or_knee":
                families.update(("kick", "knee"))
    return families


def _select_technique(rows: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[str, float, list[dict[str, Any]]]:
    candidates = _ranked_techniques(rows, schema)
    pose_families = _pose_families(rows)
    visual = [
        row for row in rows
        if row.get("proposal_branch") in LEARNED_BRANCHES
        and row.get("technique") not in UNKNOWN
        and float(row.get("proposal_confidence", 0.0)) >= 0.55
    ]
    visual.sort(key=lambda row: float(row.get("proposal_confidence", 0.0)), reverse=True)
    for row in visual:
        technique = str(row["technique"])
        family = technique_family(schema, technique)
        if family and (not pose_families or family in pose_families):
            return technique, float(row.get("proposal_confidence", 0.0)), candidates
    return "unknown_action", 0.0, candidates


def _select_side(rows: list[dict[str, Any]]) -> tuple[str, float]:
    pose = [(str(row.get("anatomical_side", "unknown")), float(row.get("attribute_confidence", 0.0))) for row in rows if str(row.get("proposal_branch", "")).startswith("pose")]
    visual = [(str(row.get("anatomical_side", "unknown")), float(row.get("attribute_confidence", row.get("proposal_confidence", 0.0)))) for row in rows if row.get("proposal_branch") in LEARNED_BRANCHES]
    pose = [item for item in pose if item[0] in {"left", "right", "bilateral"}]
    visual = [item for item in visual if item[0] in {"left", "right", "bilateral"}]
    if pose and visual and pose[0][0] == visual[0][0]:
        return pose[0][0], min(1.0, (pose[0][1] + visual[0][1]) / 2 + 0.15)
    combined = sorted([*pose, *visual], key=lambda item: item[1], reverse=True)
    if combined and combined[0][1] >= 0.80:
        return combined[0]
    return "unknown", 0.0


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
        match = next((cluster for cluster in clusters if int(cluster[0].get("actor_id", -2)) == int(row.get("actor_id", -1)) and _cluster_compatible(cluster, row, iou_threshold)), None)
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
        technique, technique_confidence, technique_candidates = _select_technique(cluster, schema)
        fused["technique"] = technique
        fused["label"] = technique
        fused["action_family"] = technique_family(schema, technique) or "unknown"
        fused["technique_candidates"] = technique_candidates
        field_confidence["technique"] = round(technique_confidence, 6)
        field_confidence["action_family"] = round(technique_confidence, 6)
        side, side_confidence = _select_side(cluster)
        fused["anatomical_side"] = side
        fused["side"] = side
        field_confidence["anatomical_side"] = round(side_confidence, 6)
        fused["proposal_confidence"] = round(max(field_confidence.values(), default=0.0), 6)
        fused["field_confidence"] = field_confidence
        fused["branch_evidence"] = [{
            "branch": row.get("proposal_branch", row.get("source", "unknown")),
            "event_id": row.get("event_id"),
            "confidence": row.get("proposal_confidence", row.get("attribute_confidence", 0.0)),
            "technique": row.get("technique", row.get("label")),
            "anatomical_side": row.get("anatomical_side", row.get("side")),
            "technique_candidates": row.get("technique_candidates", []),
            "effector_limb": row.get("effector_limb"),
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
