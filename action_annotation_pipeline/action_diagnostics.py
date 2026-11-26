"""Diagnose temporal coverage and proposal-branch health for one project."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from action_signals import fighter_map_coverage
from common import PipelineError, load_fighter_map, load_manifest, read_json, read_jsonl, write_json


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() and path.stat().st_size else []


def _duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        groups[(
            int(row.get("actor_id", -1)),
            str(row.get("pose_effector_id", row.get("effector_id", "unknown"))),
            round(float(row.get("action_start_sec", row.get("start_sec", 0.0))), 3),
            round(float(row.get("action_end_sec", row.get("end_sec", 0.0))), 3),
        )] += 1
    duplicate_groups = [count for count in groups.values() if count > 1]
    duplicate_rows = sum(count - 1 for count in duplicate_groups)
    return {
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_rows": duplicate_rows,
        "maximum_copies": max(duplicate_groups, default=1),
        "duplicate_rate": round(duplicate_rows / max(len(rows), 1), 6),
    }


def _branch_summary(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    peaks = [float(row.get("peak_sec", 0.0)) for row in rows]
    return {
        "path": str(path),
        "rows": len(rows),
        "actors": dict(Counter(str(row.get("actor_id", -1)) for row in rows)),
        "techniques": dict(Counter(str(row.get("technique", row.get("label", "unknown_action"))) for row in rows)),
        "first_peak_sec": min(peaks) if peaks else None,
        "last_peak_sec": max(peaks) if peaks else None,
        "duplicates": _duplicate_summary(rows),
    }


def build_report(project: Path, expected_classifier: str = "auto") -> dict[str, Any]:
    manifest = load_manifest(project)
    fighter_map = load_fighter_map(project)
    poses_path = project / "poses" / "poses.npz"
    if not poses_path.is_file():
        raise PipelineError(f"Pose file not found: {poses_path}")
    data = np.load(poses_path)
    expected = int(manifest.get("expected_fighters", 2))
    coverage = fighter_map_coverage(
        fighter_map,
        np.asarray(data["timestamps"]),
        np.asarray(data["track_ids"]),
        expected,
        canonical_fighters=np.asarray(data["fighter_ids"]) if "fighter_ids" in data.files else None,
    )
    tracks = _rows(project / "tracks" / "tracks.jsonl")
    detection_counts = [len(row.get("detections", [])) for row in tracks]
    branch_paths = sorted((project / "proposals" / "branches").glob("*.jsonl"))
    branches = {path.stem: _branch_summary(path) for path in branch_paths}
    fused = _rows(project / "proposals" / "fused_proposals.jsonl")
    classified = [row for row in fused if str(row.get("technique", "unknown_action")) != "unknown_action"]
    unresolved = [row for row in fused if str(row.get("technique", "unknown_action")) == "unknown_action"]
    fused_peaks = [float(row.get("peak_sec", 0.0)) for row in fused]
    learned_names = {"qwen_actions", "qwen3_vl", "posec3d", "boxing6_posec3d", "tridet"}
    classifier_branches = {
        str(evidence.get("branch", ""))
        for row in fused
        for evidence in row.get("branch_evidence", [])
        if str(evidence.get("branch", "")) in learned_names
    }
    warnings: list[str] = []
    critical: list[str] = []
    recommendations: list[str] = []
    if not coverage["complete"]:
        message = "fighter_map.json does not cover the pose timeline"
        warnings.append(message)
        critical.append(message)
        recommendations.append("Re-run tracks and poses with the canonical fighter-identity stitcher before action discovery.")
    if coverage["unmapped_pose_instances"]:
        warnings.append(f"{coverage['unmapped_pose_instances']} pose instances are not mapped to a canonical fighter")
    if not fused:
        message = "no fused action proposals were produced"
        warnings.append(message)
        critical.append(message)
    elif fused_peaks and max(fused_peaks) < coverage["pose_end_sec"] - 1.0:
        message = "proposals stop before the pose timeline ends"
        warnings.append(message)
        critical.append(message)
    classifier_expected = expected_classifier in {"hybrid", "required"}
    if classifier_expected and fused and not classified:
        warnings.append("all proposals are unknown_action; no successful technique classifier output was fused")
        critical.append("hybrid mode produced no classified actions")
    if classifier_expected and not classifier_branches:
        warnings.append("no technique-classifier branch ran; pose actionness can only suggest family/limb")
    qwen_summary_path = project / "proposals" / "raw" / "qwen" / "summary.json"
    qwen_summary = read_json(qwen_summary_path) if qwen_summary_path.is_file() else None
    qwen_relevant = expected_classifier in {"hybrid", "required"} or (
        expected_classifier == "auto" and bool(classifier_branches)
    )
    if qwen_relevant and qwen_summary and int(qwen_summary.get("tasks", 0)) > 0:
        if int(qwen_summary.get("events", 0)) == 0:
            message = "Qwen processed windows but produced zero normalized action events"
            warnings.append(message)
            critical.append(message)
            recommendations.append("Inspect proposals/raw/qwen records and previews; check crop coverage, parse failures, and valid zero-event counts.")
        if int(qwen_summary.get("failures", 0)):
            warnings.append(
                f"Qwen failed on {int(qwen_summary['failures'])}/{int(qwen_summary.get('tasks', 0))} tasks"
            )
    if classifier_expected and unresolved:
        warnings.append(
            f"{len(unresolved)} motion candidates remain unclassified and are isolated from classified actions"
        )
    fused_duplicates = _duplicate_summary(fused)
    if fused_duplicates["duplicate_rate"] > 0.15:
        message = f"fused proposals have a high exact-duplicate rate ({fused_duplicates['duplicate_rate']:.1%})"
        warnings.append(message)
        critical.append(message)
        recommendations.append("Re-run pose actionness and fusion with component-first segmentation; do not fuse pose_actionness and pose_attributes together.")
    pose_report_path = project / "qa" / "pose_actionness_report.json"
    pose_report = read_json(pose_report_path) if pose_report_path.is_file() else None
    identity_report_path = project / "tracks" / "fighter_identity_report.json"
    identity_report = read_json(identity_report_path) if identity_report_path.is_file() else None
    return {
        "project": str(project),
        "video_id": manifest["video_id"],
        "duration_sec": float(manifest["proxy_video"]["duration_sec"]),
        "tracks": {
            "frames": len(tracks),
            "first_timestamp": float(tracks[0]["timestamp"]) if tracks else None,
            "last_timestamp": float(tracks[-1]["timestamp"]) if tracks else None,
            "mean_detections_per_frame": float(np.mean(detection_counts)) if detection_counts else 0.0,
            "frames_without_detections": int(sum(count == 0 for count in detection_counts)),
        },
        "poses": {
            "instances": int(len(data["timestamps"])),
            "first_timestamp": coverage["pose_start_sec"],
            "last_timestamp": coverage["pose_end_sec"],
            "raw_tracks": sorted({int(value) for value in data["track_ids"]}),
            "canonical_fighters": sorted({int(value) for value in data["fighter_ids"]}) if "fighter_ids" in data.files else [],
        },
        "fighter_map": {**coverage, "path": str(project / "tracks" / "fighter_map.json")},
        "fighter_identity_stitching": identity_report,
        "branches": branches,
        "fused": {
            "rows": len(fused),
            "first_peak_sec": min(fused_peaks) if fused_peaks else None,
            "last_peak_sec": max(fused_peaks) if fused_peaks else None,
            "actors": dict(Counter(str(row.get("actor_id", -1)) for row in fused)),
            "techniques": dict(Counter(str(row.get("technique", "unknown_action")) for row in fused)),
            "duplicates": fused_duplicates,
        },
        "technique_classifier": {"ran": bool(classifier_branches), "branches": sorted(classifier_branches)},
        "classification": {
            "expected_mode": expected_classifier,
            "classified_actions": len(classified),
            "unresolved_motion_candidates": len(unresolved),
            "classification_coverage": round(len(classified) / max(len(fused), 1), 6),
            "classified_output": str(project / "proposals" / "classified_actions.jsonl"),
            "unresolved_output": str(project / "proposals" / "unresolved_motion.jsonl"),
        },
        "qwen": qwen_summary if qwen_relevant else {"ignored_stale_summary_in_pose_mode": bool(qwen_summary)},
        "pose_actionness": pose_report,
        "warnings": warnings,
        "critical_failures": critical,
        "recommendations": list(dict.fromkeys(recommendations)),
        "ok": not critical,
    }


def render_timeline(report: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    duration = max(0.001, float(report["duration_sec"]))
    bars = []
    rows = []
    for event in proposals:
        start = max(0.0, float(event.get("start_sec", 0.0)))
        end = min(duration, float(event.get("end_sec", start)))
        left, width = 100.0 * start / duration, max(0.15, 100.0 * (end - start) / duration)
        actor = int(event.get("actor_id", -1))
        label = str(event.get("technique", event.get("label", "unknown_action")))
        candidates = event.get("technique_candidates", [])
        candidate_text = ", ".join(
            f"{item.get('technique', item.get('label', '?'))} {float(item.get('confidence', 0)):.2f}"
            for item in candidates[:3] if isinstance(item, dict)
        )
        bars.append(
            f'<div class="event actor{actor}" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="{html.escape(label)} @ {float(event.get("peak_sec", 0)):.3f}s"></div>'
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('event_id', '')))}</td>"
            f"<td>{float(event.get('peak_sec', 0)):.3f}</td><td>{actor}</td>"
            f"<td>{html.escape(label)}</td><td>{html.escape(candidate_text)}</td>"
            f"<td>{float(event.get('proposal_confidence', 0)):.3f}</td>"
            "</tr>"
        )
    warnings = "".join(f"<li>{html.escape(value)}</li>" for value in report.get("warnings", [])) or "<li>None</li>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Action diagnostics</title>
<style>body{{font:14px Segoe UI,sans-serif;background:#101722;color:#e7edf5;padding:20px}}.timeline{{height:74px;background:#243247;position:relative;border-radius:6px}}.event{{position:absolute;height:28px;top:8px;min-width:2px;opacity:.8}}.actor0{{background:#34d399}}.actor1{{background:#fb923c;top:40px}}table{{border-collapse:collapse;width:100%;margin-top:20px}}td,th{{padding:6px;border-bottom:1px solid #33445c;text-align:left}}code{{color:#93c5fd}}</style>
</head><body><h1>{html.escape(str(report['video_id']))} action diagnostics</h1>
<p>Duration: {duration:.3f}s · proposals: {len(proposals)} · classifier: {html.escape(str(report['technique_classifier']))}</p>
<h2>Warnings</h2><ul>{warnings}</ul><div class="timeline">{''.join(bars)}</div>
<table><thead><tr><th>Event</th><th>Peak</th><th>Actor</th><th>Technique</th><th>Top candidates</th><th>Confidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-classifier", choices=("auto", "pose", "hybrid", "required"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for resumable stage compatibility")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        report = build_report(project, args.expected_classifier)
        output = project / "qa" / "action_diagnostics.json"
        timeline = project / "qa" / "action_timeline.html"
        write_json(output, report)
        timeline.write_text(
            render_timeline(report, _rows(project / "proposals" / "fused_proposals.jsonl")),
            encoding="utf-8",
        )
        print(json.dumps({"report": str(output), "timeline": str(timeline), "ok": report["ok"], "warnings": report["warnings"]}, indent=2))
        return 0 if report["ok"] else 2
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
