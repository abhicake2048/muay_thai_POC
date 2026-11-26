"""Export verified, exhaustive action intervals to TriDet/OpenTAD-style JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import PipelineError, load_manifest, read_json, read_jsonl, write_json
from validate_events import validate
from schema import canonicalize_event, flatten_label
from training_contracts import validate_tridet_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic export only; not train-ready")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        output_dir = project / "exports" / "tridet"
        output_path = output_dir / "annotations.json"
        if output_path.exists() and not args.overwrite:
            raise PipelineError(f"Output exists: {output_path}; pass --overwrite")
        errors, _, report = validate(
            project,
            project / "annotations" / "events.jsonl",
            require_full_review=not args.allow_partial,
            allow_proposals=False,
        )
        if errors:
            raise PipelineError(f"Event validation failed with {len(errors)} errors; inspect qa/event_validation_report.json")
        manifest = load_manifest(project)
        schema = read_json(project / "annotations" / "classes.json")
        events = read_jsonl(project / "annotations" / "events.jsonl")
        duration = float(manifest["proxy_video"]["duration_sec"])
        fps = float(manifest["proxy_video"]["fps"])
        taxonomy = list(schema.get("trainable_event_classes", schema["primary_action_classes"]))
        label_to_id = {label: index for index, label in enumerate(taxonomy)}
        database: dict[str, object] = {}
        for fighter_id in range(int(manifest.get("expected_fighters", 2))):
            annotations = []
            flat_annotations = []
            for raw_event in events:
                event = canonicalize_event(raw_event, schema)
                if event.get("event_type") not in {"action", "balance"} or event.get("review_status") != "verified":
                    continue
                if int(event.get("actor_id", -1)) != fighter_id:
                    continue
                label = str(event.get("technique") or event["label"])
                annotations.append({
                    "segment": [float(event["start_sec"]), float(event["end_sec"])],
                    "segment_frames": [
                        int(event.get("proxy_start_frame", round(float(event["start_sec"]) * fps))),
                        int(event.get("proxy_end_frame", round(float(event["end_sec"]) * fps))),
                    ],
                    "source_segment_frames": [
                        int(event.get("source_start_frame", event.get("proxy_start_frame", round(float(event["start_sec"]) * fps)))),
                        int(event.get("source_end_frame", event.get("proxy_end_frame", round(float(event["end_sec"]) * fps)))),
                    ],
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "event_type": event.get("event_type"),
                    "flat_label": flatten_label(event),
                    "event_id": str(event["event_id"]),
                    "peak_sec": float(event.get("peak_sec", (float(event["start_sec"]) + float(event["end_sec"])) / 2)),
                    "peak_frame": int(event.get("proxy_peak_frame", round(float(event.get("peak_sec", 0.0)) * fps))),
                    "source_peak_frame": int(event.get("source_peak_frame", event.get("proxy_peak_frame", round(float(event.get("peak_sec", 0.0)) * fps)))),
                    "target_id": int(event.get("target_id", -1)),
                    "action_family": event.get("action_family"),
                    "anatomical_side": event.get("anatomical_side", "unknown"),
                    "effector_limb": event.get("effector_limb", "unknown"),
                    "effector_id": event.get("effector_id", "unknown"),
                    "stance_at_start": event.get("stance_at_start", "unknown"),
                    "stance_role": event.get("stance_role", "unknown"),
                    "target_zone": event.get("target_zone", "unknown"),
                    "outcome": event.get("outcome", "uncertain"),
                    "contact_surface": event.get("contact_surface", "unknown"),
                    "visibility": event.get("visibility", "unknown"),
                })
                flat_annotations.append({
                    "segment": [float(event["start_sec"]), float(event["end_sec"])],
                    "label": flatten_label(event), "event_id": str(event["event_id"]),
                })
            annotations.sort(key=lambda item: item["segment"][0])
            flat_annotations.sort(key=lambda item: item["segment"][0])
            stream_id = f"{manifest['video_id']}__fighter{fighter_id}"
            database[stream_id] = {
                "duration": duration,
                "fps": fps,
                "subset": manifest.get("subset", "training"),
                "source_video": manifest["video_id"],
                "actor_id": fighter_id,
                "annotations": annotations,
                "flattened_annotations": flat_annotations,
            }
        payload = {
            "version": "muay_thai_actions_v2",
            "taxonomy": taxonomy,
            "label_to_id": label_to_id,
            "database": database,
            "metadata": {
                "reviewed_duration_sec": report["reviewed_duration_sec"],
                "full_video_duration_sec": duration,
                "partial_export": bool(args.allow_partial),
                "feature_contract": "One actor-conditioned FP16 VideoMAEv2 .npy array per database key; initial feat_stride=4 proxy frames at 30 FPS.",
                "annotation_contract": "TriDet/ActivityNet JSON: database -> stream -> duration/fps/subset/annotations; every annotation has segment seconds, label and label_id.",
                "canonical_label_contract": "TriDet localizes technique. PoseC3D attribute heads classify anatomical side, stance role, target and outcome for each detected interval.",
                "flattened_baseline": "Each stream also contains flattened_annotations for an optional technique__effector baseline.",
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_errors = validate_tridet_payload(payload)
        write_json(project / "qa" / "tridet_contract_report.json", {
            "ok": not contract_errors,
            "errors": contract_errors,
            "streams": len(database),
            "contract": "TriDet/ActivityNet JSON with database, FPS, seconds segments, label and label_id",
        })
        if contract_errors:
            raise PipelineError(f"TriDet export contract failed with {len(contract_errors)} errors")
        write_json(output_path, payload)
        feature_manifest_path = output_dir / "feature_manifest.json"
        existing_features = read_json(feature_manifest_path) if feature_manifest_path.is_file() else {}
        feature_manifest = {}
        for stream_id, record in database.items():
            existing = existing_features.get(stream_id, {})
            feature_file = output_dir / str(existing.get("feature_file", f"features/{stream_id}.npy"))
            if existing.get("status") == "complete" and feature_file.is_file():
                feature_manifest[stream_id] = existing
            else:
                feature_manifest[stream_id] = {
                    "feature_file": f"features/{stream_id}.npy",
                    "status": "not_extracted",
                    "actor_id": record["actor_id"],
                    "duration": record["duration"],
                }
        write_json(feature_manifest_path, feature_manifest)
        summary = {
            "output": str(output_path),
            "streams": len(database),
            "actions": sum(len(record["annotations"]) for record in database.values()),
            "partial": bool(args.allow_partial),
        }
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, KeyError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
