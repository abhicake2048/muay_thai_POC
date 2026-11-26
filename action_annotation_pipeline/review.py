"""Prepare or launch FiftyOne temporal review and emit a CVAT label contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from common import PipelineError, load_manifest, read_json, read_jsonl, write_json, write_jsonl
from schema import canonicalize_event


def _field(label: object, key: str, default=None):
    try:
        value = label.get_field(key)  # type: ignore[attr-defined]
        return default if value is None else value
    except (AttributeError, KeyError, ValueError):
        try:
            value = label[key]  # type: ignore[index]
            return default if value is None else value
        except (KeyError, TypeError):
            return default


def fiftyone_detections_to_events(detections: list[object], fps: float, schema: dict) -> list[dict]:
    """Convert saved FiftyOne temporal labels into sync-review JSONL rows."""
    rows: list[dict] = []
    review_labels = {"reviewed": "review_range"}
    ignore_labels = {"unreviewed", "unusable", "camera_cut", "replay"}
    for index, detection in enumerate(detections, start=1):
        label = str(getattr(detection, "label", _field(detection, "label", "")))
        support = list(getattr(detection, "support", _field(detection, "support", [])) or [])
        if len(support) != 2:
            raise PipelineError(f"FiftyOne temporal label {index} has invalid support")
        start_frame, end_frame = int(support[0]) - 1, int(support[1]) - 1
        start_sec, end_sec = start_frame / fps, end_frame / fps
        event_type = str(_field(detection, "event_type", ""))
        if label in review_labels:
            event_type = review_labels[label]
        elif label in ignore_labels:
            event_type = "ignore"
        elif not event_type:
            event_type = "action"
        peak_frame = int(_field(detection, "proxy_peak_frame", round((start_frame + end_frame) / 2)))
        row = {
            "event_id": str(_field(detection, "event_id", f"fiftyone_{index:06d}")),
            "event_type": event_type,
            "actor_id": int(_field(detection, "actor_id", -1)),
            "target_id": int(_field(detection, "target_id", -1)),
            "start_sec": round(start_sec, 6), "peak_sec": round(peak_frame / fps, 6), "end_sec": round(end_sec, 6),
            "proxy_fps": fps, "proxy_start_frame": start_frame, "proxy_peak_frame": peak_frame, "proxy_end_frame": end_frame,
            "label": label, "technique": label,
            "anatomical_side": _field(detection, "anatomical_side", "unknown"),
            "effector_limb": _field(detection, "effector_limb", "unknown"),
            "stance_at_start": _field(detection, "stance_at_start", "unknown"),
            "stance_role": _field(detection, "stance_role", "unknown"),
            "target_zone": _field(detection, "target_zone", "unknown"),
            "outcome": _field(detection, "outcome", "uncertain"),
            "contact_surface": _field(detection, "contact_surface", "unknown"),
            "visibility": _field(detection, "visibility", "unknown"),
            "review_status": _field(detection, "review_status", "verified" if event_type in {"review_range", "ignore"} else "proposed"),
            "annotator": _field(detection, "annotator", "fiftyone_reviewer"),
            "classification_reason": _field(detection, "classification_reason", ""),
            "boundary_reason": _field(detection, "boundary_reason", ""),
            "window_summary": _field(detection, "window_summary", ""),
            "combo_id": _field(detection, "combo_id"),
            "counter_to_event_id": _field(detection, "counter_to_event_id"),
            "possible_cause": _field(detection, "possible_cause"),
            "round": int(_field(detection, "round", 0)),
            "notes": _field(detection, "notes", ""),
            "source": "fiftyone_review",
            "annotation_revision": int(_field(detection, "annotation_revision", 0)) + 1,
            "provenance": {"source": "fiftyone_review", "imported_event_id": _field(detection, "event_id")},
        }
        for field in ("technique_candidates", "branch_evidence", "visual_evidence", "uncertainty_flags"):
            value = _field(detection, field, [])
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = []
            row[field] = value if isinstance(value, list) else []
        if event_type in {"review_range", "ignore"}:
            row["actor_id"] = -1
            row["target_id"] = -1
        rows.append(canonicalize_event(row, schema))
    return sorted(rows, key=lambda item: (float(item["start_sec"]), int(item.get("actor_id", -1))))


def cvat_contract(schema: dict) -> dict:
    attributes = [
        {"name": "anatomical_side", "input_type": "select", "values": schema["anatomical_sides"], "mutable": True},
        {"name": "effector_limb", "input_type": "select", "values": schema["effector_limbs"], "mutable": True},
        {"name": "stance_at_start", "input_type": "select", "values": schema["stance_classes"], "mutable": True},
        {"name": "stance_role", "input_type": "select", "values": schema["stance_roles"], "mutable": True},
        {"name": "target_zone", "input_type": "select", "values": schema["target_zones"], "mutable": True},
        {"name": "outcome", "input_type": "select", "values": schema["outcomes"], "mutable": True},
        {"name": "contact_surface", "input_type": "select", "values": schema["contact_surfaces"], "mutable": True},
        {"name": "visibility", "input_type": "select", "values": schema["visibility_levels"], "mutable": True},
    ]
    return {"version": "2.0", "labels": [{"name": technique, "type": "tag", "attributes": attributes} for technique in schema["primary_action_classes"]], "note": "Use CVAT for boxes/skeleton corrections. FiftyOne remains the canonical temporal-event review surface."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--launch", action="store_true", help="Launch the FiftyOne app and block until closed")
    parser.add_argument("--port", type=int, default=5151)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        schema = read_json(project / "annotations" / "classes.json")
        proposal_path = project / "proposals" / "fused_proposals.jsonl"
        if not proposal_path.is_file() or not proposal_path.stat().st_size:
            proposal_path = project / "proposals" / "action_proposals.jsonl"
        proposals = read_jsonl(proposal_path) if proposal_path.is_file() and proposal_path.stat().st_size else []
        assessments_path = project / "proposals" / "branches" / "qwen_window_assessments.jsonl"
        assessments = read_jsonl(assessments_path) if assessments_path.is_file() and assessments_path.stat().st_size else []
        # Refresh the dependency-free reviewer for existing projects as the
        # proposal evidence UI evolves; reviewed event files are untouched.
        shutil.copy2(Path(__file__).parent / "annotator" / "index.html", project / "review" / "annotate.html")
        write_json(project / "review" / "cvat_label_contract.json", cvat_contract(schema))
        review_manifest = {
            "schema_version": "2.0", "project": manifest["project_id"],
            "video": manifest["paths"]["proxy_video"], "proposals": str(proposal_path),
            "proposal_count": len(proposals), "fiftyone_dataset": f"mtdata_{manifest['project_id']}",
            "qwen_window_assessments": str(assessments_path) if assessments_path.is_file() else None,
            "qwen_window_assessment_count": len(assessments),
            "cvat_label_contract": str(project / "review" / "cvat_label_contract.json"),
            "human_verification_required": True,
        }
        write_json(project / "review" / "review_manifest.json", review_manifest)
        if args.launch:
            try:
                import fiftyone as fo
            except ImportError as exc:
                raise PipelineError(f"FiftyOne is not installed: {exc}") from exc
            name = review_manifest["fiftyone_dataset"]
            if fo.dataset_exists(name):
                dataset = fo.load_dataset(name)
                dataset.delete()
            dataset = fo.Dataset(name, persistent=True)
            sample = fo.Sample(filepath=str(manifest["paths"]["proxy_video"]))
            detections = []
            fps = float(manifest["proxy_video"]["fps"])
            for raw in proposals:
                event = canonicalize_event(raw, schema)
                detection = fo.TemporalDetection(
                    label=str(event.get("technique", "unknown_action")),
                    support=[int(round(float(event["start_sec"]) * fps)) + 1, int(round(float(event["end_sec"]) * fps)) + 1],
                    confidence=float(event.get("proposal_confidence", 0.0)),
                )
                for key in ("event_id", "event_type", "actor_id", "target_id", "anatomical_side", "effector_limb", "effector_id", "stance_at_start", "stance_role", "target_zone", "outcome", "contact_surface", "visibility", "review_status", "annotator", "annotation_revision", "proxy_start_frame", "proxy_peak_frame", "proxy_end_frame", "classification_reason", "boundary_reason", "window_summary", "combo_id", "counter_to_event_id", "possible_cause", "round", "notes"):
                    detection[key] = event.get(key)
                detection["technique_candidates"] = json.dumps(event.get("technique_candidates", []), ensure_ascii=False)
                detection["branch_evidence"] = json.dumps(event.get("branch_evidence", []), ensure_ascii=False)
                detection["visual_evidence"] = json.dumps(event.get("visual_evidence", []), ensure_ascii=False)
                detection["uncertainty_flags"] = json.dumps(event.get("uncertainty_flags", []), ensure_ascii=False)
                detections.append(detection)
            sample["action_proposals"] = fo.TemporalDetections(detections=detections)
            assessment_detections = []
            for assessment in assessments:
                status = str(assessment.get("status", "unknown"))
                value = fo.TemporalDetection(
                    label=f"qwen_window__{status}",
                    support=[int(round(float(assessment["start_sec"]) * fps)) + 1, int(round(float(assessment["end_sec"]) * fps)) + 1],
                    confidence=float(assessment.get("no_action_confidence", 0.0)) if status == "no_action" else None,
                )
                for key in ("task_id", "actor_id", "scan_kind", "window_summary", "no_action_reason", "error"):
                    value[key] = assessment.get(key)
                value["normalized_event_ids"] = json.dumps(assessment.get("normalized_event_ids", []), ensure_ascii=False)
                assessment_detections.append(value)
            sample["qwen_window_assessments"] = fo.TemporalDetections(detections=assessment_detections)
            dataset.add_sample(sample)
            session = fo.launch_app(dataset, port=args.port, address=os.environ.get("FIFTYONE_ADDRESS", "localhost"))
            session.wait()
            dataset.save()
            refreshed = dataset.first()
            reviewed_field = refreshed.get_field("action_proposals")
            reviewed_detections = list(reviewed_field.detections or []) if reviewed_field is not None else []
            reviewed_rows = fiftyone_detections_to_events(reviewed_detections, fps, schema)
            reviewed_output = project / "review" / "fiftyone_reviewed_events.jsonl"
            write_jsonl(reviewed_output, reviewed_rows)
            review_manifest["fiftyone_reviewed_events"] = str(reviewed_output)
            review_manifest["fiftyone_reviewed_event_count"] = len(reviewed_rows)
            write_json(project / "review" / "review_manifest.json", review_manifest)
        print(json.dumps(review_manifest, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
