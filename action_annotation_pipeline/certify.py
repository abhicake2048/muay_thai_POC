"""Fail-closed certification of action, review, export, and feature artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

from action_diagnostics import build_report
from common import PipelineError, read_json, read_jsonl, write_json
from training_contracts import validate_feature_array, validate_posec3d_payload, validate_tridet_payload
from validate_events import validate
from vlm_qwen import validate_vlm_payload


STAGES = ("actions", "reviewed", "exported", "features")


def _safe_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() and path.stat().st_size else []


def _record(gates: dict[str, dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    gates[name] = {"passed": bool(passed), "evidence": evidence}


def _structured_qwen_audit(project: Path, assessments: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    allowed = set(schema.get("trainable_event_classes", schema.get("primary_action_classes", [])))
    valid = 0
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for row in assessments:
        task_id = str(row.get("task_id", ""))
        raw_value = row.get("raw_record") or f"proposals/raw/qwen/{task_id}.json"
        raw_path = project / str(raw_value)
        if not raw_path.is_file():
            missing.append(task_id)
            continue
        if row.get("status") == "failed":
            invalid.append({"task_id": task_id, "error": str(row.get("error", "failed window"))})
            continue
        try:
            record = read_json(raw_path)
            payload = record.get("parsed")
            if not isinstance(payload, dict):
                raise ValueError("parsed payload is missing")
            validate_vlm_payload(
                payload,
                float(row["end_sec"]) - float(row["start_sec"]),
                allowed,
                schema,
            )
            expected_action = row.get("status") == "action"
            if bool(payload.get("events")) != expected_action:
                raise ValueError("ledger status disagrees with parsed events")
            valid += 1
        except (KeyError, OSError, TypeError, ValueError, PipelineError) as exc:
            invalid.append({"task_id": task_id, "error": str(exc)})
    return {
        "assessment_rows": len(assessments),
        "valid_structured_rows": valid,
        "missing_raw_records": missing,
        "invalid_records": invalid,
    }


def _certify_actions(project: Path, gates: dict[str, dict[str, Any]], minimum_classified: int) -> dict[str, Any]:
    schema = read_json(project / "annotations" / "classes.json")
    diagnostics = build_report(project, "hybrid")
    summary_path = project / "proposals" / "raw" / "qwen" / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    assessment_path = project / "proposals" / "branches" / "qwen_window_assessments.jsonl"
    assessments = _safe_rows(assessment_path)
    classified = _safe_rows(project / "proposals" / "classified_actions.jsonl")
    structured = _structured_qwen_audit(project, assessments, schema)
    tasks = int(summary.get("tasks", 0))
    failed = int(summary.get("failed_windows", summary.get("failures", 0)))
    coverage = diagnostics.get("qwen", {}).get("successful_temporal_coverage", {})
    expected = int(read_json(project / "manifest.json").get("expected_fighters", 2))
    coverage_ok = all(float(coverage.get(str(fighter), {}).get("coverage_ratio", 0.0)) >= 0.98 for fighter in range(expected))
    statuses_total = sum(int(summary.get(name, 0)) for name in ("action_windows", "no_action_windows", "failed_windows"))
    _record(gates, "action_diagnostics_ok", bool(diagnostics.get("ok")), diagnostics.get("critical_failures", []))
    _record(gates, "qwen_summary_present", bool(summary), str(summary_path))
    _record(gates, "qwen_tasks_processed", tasks > 0, tasks)
    _record(gates, "qwen_ledger_complete", tasks > 0 and len(assessments) == tasks and statuses_total == tasks, {
        "tasks": tasks, "assessment_rows": len(assessments), "status_total": statuses_total,
    })
    _record(gates, "qwen_no_failed_windows", tasks > 0 and failed == 0, failed)
    _record(gates, "qwen_structured_outputs_valid", tasks > 0 and structured["valid_structured_rows"] == tasks, structured)
    _record(gates, "qwen_temporal_coverage", coverage_ok, coverage)
    _record(gates, "technique_classifier_ran", bool(diagnostics.get("technique_classifier", {}).get("ran")), diagnostics.get("technique_classifier"))
    _record(gates, "minimum_classified_actions", len(classified) >= minimum_classified, {
        "required": minimum_classified, "actual": len(classified),
    })
    return {"diagnostics": diagnostics, "qwen_summary": summary, "structured_audit": structured}


def _certify_review(project: Path, gates: dict[str, dict[str, Any]], minimum_verified: int) -> dict[str, Any]:
    events_path = project / "annotations" / "events.jsonl"
    if not events_path.is_file():
        _record(gates, "reviewed_events_present", False, str(events_path))
        _record(gates, "review_contract_valid", False, "events.jsonl is missing")
        _record(gates, "minimum_verified_events", False, {"required": minimum_verified, "actual": 0})
        return {}
    errors, warnings, report = validate(project, events_path, True, False)
    _record(gates, "reviewed_events_present", True, str(events_path))
    _record(gates, "review_contract_valid", not errors, {"errors": errors, "warnings": warnings})
    _record(gates, "minimum_verified_events", int(report.get("verified_actions", 0)) >= minimum_verified, {
        "required": minimum_verified, "actual": int(report.get("verified_actions", 0)),
    })
    return report


def _certify_exports(project: Path, gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pose_path = project / "exports" / "posec3d" / "posec3d_annotations.pkl"
    class_path = project / "exports" / "posec3d" / "class_map.json"
    pose_errors: list[str] = []
    pose_samples = 0
    if pose_path.is_file() and class_path.is_file():
        try:
            with pose_path.open("rb") as handle:
                pose_payload = pickle.load(handle)
            class_count = len(read_json(class_path).get("classes", []))
            pose_errors = validate_posec3d_payload(pose_payload, class_count)
            pose_samples = len(pose_payload.get("annotations", []))
        except (OSError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            pose_errors = [str(exc)]
    else:
        pose_errors = ["PoseC3D pickle or class map is missing"]
    _record(gates, "posec3d_export_valid", not pose_errors and pose_samples > 0, {
        "samples": pose_samples, "errors": pose_errors, "path": str(pose_path),
    })

    tridet_path = project / "exports" / "tridet" / "annotations.json"
    tri_errors: list[str] = []
    tri_streams = tri_actions = 0
    if tridet_path.is_file():
        try:
            tri_payload = read_json(tridet_path)
            tri_errors = validate_tridet_payload(tri_payload)
            database = tri_payload.get("database", {})
            tri_streams = len(database)
            tri_actions = sum(len(record.get("annotations", [])) for record in database.values())
        except (OSError, TypeError, ValueError, PipelineError) as exc:
            tri_errors = [str(exc)]
    else:
        tri_errors = ["TriDet annotations.json is missing"]
    _record(gates, "tridet_export_valid", not tri_errors and tri_streams > 0 and tri_actions > 0, {
        "streams": tri_streams, "actions": tri_actions, "errors": tri_errors, "path": str(tridet_path),
    })
    return {"posec3d_samples": pose_samples, "tridet_streams": tri_streams, "tridet_actions": tri_actions}


def _certify_features(project: Path, gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest_path = project / "exports" / "tridet" / "feature_manifest.json"
    annotations_path = project / "exports" / "tridet" / "annotations.json"
    errors: list[str] = []
    checked = 0
    expected_streams: set[str] = set()
    if annotations_path.is_file():
        expected_streams = set(read_json(annotations_path).get("database", {}))
    feature_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    for stream_id in sorted(expected_streams):
        record = feature_manifest.get(stream_id, {})
        feature_path = project / "exports" / "tridet" / str(record.get("feature_file", ""))
        timestamp_path = project / "exports" / "tridet" / str(record.get("timestamps_file", ""))
        if record.get("status") != "complete" or not feature_path.is_file() or not timestamp_path.is_file():
            errors.append(f"{stream_id}: feature/timestamp output is incomplete")
            continue
        try:
            timestamps = np.load(timestamp_path, mmap_mode="r")
            array = np.load(feature_path, mmap_mode="r")
            item_errors = validate_feature_array(array, len(timestamps))
            if timestamps.ndim != 1 or not np.isfinite(np.asarray(timestamps)).all():
                item_errors.append("timestamps must be a finite 1D array")
            errors.extend(f"{stream_id}: {value}" for value in item_errors)
            checked += not item_errors
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{stream_id}: {exc}")
    _record(gates, "tridet_features_valid", bool(expected_streams) and checked == len(expected_streams) and not errors, {
        "expected_streams": len(expected_streams), "valid_streams": checked, "errors": errors,
    })
    return {"expected_streams": len(expected_streams), "valid_streams": checked, "errors": errors}


def certify(project: Path, stage: str, minimum_classified: int = 1, minimum_verified: int = 1) -> dict[str, Any]:
    if stage not in STAGES:
        raise PipelineError(f"Unknown certification stage: {stage}")
    gates: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {
        "actions": _certify_actions(project, gates, minimum_classified),
    }
    stage_index = STAGES.index(stage)
    if stage_index >= STAGES.index("reviewed"):
        evidence["review"] = _certify_review(project, gates, minimum_verified)
    if stage_index >= STAGES.index("exported"):
        evidence["exports"] = _certify_exports(project, gates)
    if stage_index >= STAGES.index("features"):
        evidence["features"] = _certify_features(project, gates)
    failed = [name for name, value in gates.items() if not value["passed"]]
    return {
        "project": str(project),
        "requested_stage": stage,
        "passed": not failed,
        "failed_gates": failed,
        "gates": gates,
        "evidence": evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="actions")
    parser.add_argument("--minimum-classified", type=int, default=1)
    parser.add_argument("--minimum-verified", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.minimum_classified < 0 or args.minimum_verified < 0:
            raise PipelineError("minimum counts must be non-negative")
        project = args.project.expanduser().resolve()
        report = certify(project, args.stage, args.minimum_classified, args.minimum_verified)
        write_json(project / "qa" / "pipeline_certification.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    except (OSError, TypeError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
