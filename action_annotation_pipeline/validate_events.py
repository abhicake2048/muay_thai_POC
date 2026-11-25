"""Validate reviewed temporal events before PoseC3D or TriDet export."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import PipelineError, interval_covered, load_manifest, merge_intervals, read_json, read_jsonl, write_json
    from .schema import canonicalize_event, technique_family
except ImportError:
    from common import PipelineError, interval_covered, load_manifest, merge_intervals, read_json, read_jsonl, write_json
    from schema import canonicalize_event, technique_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--require-full-review", action="store_true")
    parser.add_argument("--allow-proposals", action="store_true")
    return parser.parse_args()


def allowed_labels(schema: dict[str, Any], event_type: str) -> set[str]:
    key = {
        "action": "primary_action_classes",
        "balance": "balance_classes",
        "position": "position_classes",
        "movement": "movement_classes",
        "stance": "stance_classes",
        "sequence": "sequence_classes",
    }.get(event_type)
    if key:
        return set(schema.get(key, []))
    if event_type == "review_range":
        return {"reviewed"}
    if event_type == "ignore":
        return {"unreviewed", "unusable", "camera_cut", "replay"}
    return set()


def validate(project: Path, events_path: Path, require_full_review: bool, allow_proposals: bool) -> tuple[list[str], list[str], dict[str, Any]]:
    manifest = load_manifest(project)
    schema = read_json(project / "annotations" / "classes.json")
    events = read_jsonl(events_path)
    duration = float(manifest["proxy_video"]["duration_sec"])
    errors: list[str] = []
    warnings: list[str] = []
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    review_ranges: list[tuple[float, float]] = []
    verified_actions: list[dict[str, Any]] = []
    is_v2 = str(schema.get("schema_version", "1.0")).startswith("2")

    for row_number, raw_event in enumerate(events, start=1):
        event = canonicalize_event(raw_event, schema) if is_v2 else raw_event
        prefix = f"row {row_number} ({event.get('event_id', 'missing id')})"
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            errors.append(f"{prefix}: missing event_id")
        elif event_id in ids:
            errors.append(f"{prefix}: duplicate event_id")
        ids.add(event_id)
        event_type = str(event.get("event_type", ""))
        if event_type not in schema.get("event_types", []):
            errors.append(f"{prefix}: invalid event_type {event_type!r}")
            continue
        try:
            start, end = float(event["start_sec"]), float(event["end_sec"])
            peak = float(event.get("peak_sec", (start + end) / 2))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: start_sec/peak_sec/end_sec must be numeric")
            continue
        if not 0 <= start < end <= duration + 0.05:
            errors.append(f"{prefix}: invalid interval {start:.3f}-{end:.3f} for duration {duration:.3f}")
        if not start <= peak <= end:
            errors.append(f"{prefix}: peak_sec must be inside start/end")
        label = str(event.get("technique") or event.get("label", ""))
        if label not in allowed_labels(schema, event_type):
            errors.append(f"{prefix}: label {label!r} is invalid for {event_type}")
        status = str(event.get("review_status", ""))
        if status not in schema.get("review_statuses", []):
            errors.append(f"{prefix}: invalid review_status {status!r}")
        if status == "proposed" and not allow_proposals:
            errors.append(f"{prefix}: automatic proposal has not been reviewed")
        if event_type == "action":
            try:
                actor, target = int(event["actor_id"]), int(event["target_id"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{prefix}: action actor_id/target_id must be integers")
                actor, target = -99, -99
            if actor not in (0, 1):
                errors.append(f"{prefix}: actor_id must be canonical fighter 0 or 1")
            if target not in (-1, 0, 1) or target == actor:
                errors.append(f"{prefix}: target_id must be the opponent or -1")
            side = event.get("anatomical_side", event.get("side"))
            if side not in schema.get("anatomical_sides", schema.get("sides", [])):
                errors.append(f"{prefix}: invalid anatomical_side")
            if event.get("target_zone") not in schema.get("target_zones", []):
                errors.append(f"{prefix}: invalid target_zone")
            if event.get("outcome") not in schema.get("outcomes", []):
                errors.append(f"{prefix}: invalid outcome")
            if is_v2:
                family = str(event.get("action_family", ""))
                expected_family = technique_family(schema, label)
                if not expected_family:
                    errors.append(f"{prefix}: technique {label!r} has no action family")
                elif family != expected_family:
                    errors.append(f"{prefix}: action_family {family!r} does not match {expected_family!r}")
                if event.get("effector_limb") not in schema.get("effector_limbs", []):
                    errors.append(f"{prefix}: invalid effector_limb")
                if event.get("stance_at_start") not in schema.get("stance_classes", []):
                    errors.append(f"{prefix}: invalid stance_at_start")
                if event.get("stance_role") not in schema.get("stance_roles", []):
                    errors.append(f"{prefix}: invalid stance_role")
                if event.get("contact_surface") not in schema.get("contact_surfaces", []):
                    errors.append(f"{prefix}: invalid contact_surface")
                if event.get("visibility") not in schema.get("visibility_levels", []):
                    errors.append(f"{prefix}: invalid visibility")
                if event.get("effector_id") != canonicalize_event(event, schema).get("effector_id"):
                    errors.append(f"{prefix}: effector_id is inconsistent with anatomical_side/effector_limb")
            if status == "verified":
                verified_actions.append(event)
        elif event_type == "balance":
            try:
                actor = int(event["actor_id"])
            except (KeyError, TypeError, ValueError):
                actor = -99
                errors.append(f"{prefix}: balance actor_id must be a canonical fighter")
            if actor not in (0, 1):
                errors.append(f"{prefix}: balance actor_id must be 0 or 1")
            if status == "verified":
                verified_actions.append(event)
        if event_type == "review_range" and status == "verified":
            review_ranges.append((start, end))
        counts[f"{event_type}:{label}:{status}"] += 1

    merged_review = merge_intervals(review_ranges)
    if verified_actions and not merged_review:
        errors.append("Verified actions exist but no verified review_range was annotated")
    for event in verified_actions:
        if not interval_covered(float(event["start_sec"]), float(event["end_sec"]), merged_review):
            errors.append(f"{event['event_id']}: verified action lies outside verified review_range")
    covered_duration = sum(end - start for start, end in merged_review)
    if require_full_review and (not merged_review or covered_duration < duration - 0.10):
        errors.append(f"Full review required, but review ranges cover {covered_duration:.3f}/{duration:.3f} seconds")
    elif covered_duration < duration - 0.10:
        warnings.append(f"Only {covered_duration:.3f}/{duration:.3f} seconds are marked reviewed; partial data is not ready for full-video TriDet training")
    report = {
        "events": len(events),
        "verified_actions": len(verified_actions),
        "schema_version": schema.get("schema_version", "1.0"),
        "video_duration_sec": duration,
        "reviewed_duration_sec": covered_duration,
        "review_ranges": merged_review,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
    }
    return errors, warnings, report


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        events = args.events.expanduser().resolve() if args.events else project / "annotations" / "events.jsonl"
        errors, warnings, report = validate(project, events, args.require_full_review, args.allow_proposals)
        write_json(project / "qa" / "event_validation_report.json", report)
        print(json.dumps(report, indent=2))
        return 1 if errors else 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
