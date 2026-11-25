"""Prepare or launch FiftyOne temporal review and emit a CVAT label contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from common import PipelineError, load_manifest, read_json, read_jsonl, write_json
from schema import canonicalize_event


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
        # Refresh the dependency-free reviewer for existing projects as the
        # proposal evidence UI evolves; reviewed event files are untouched.
        shutil.copy2(Path(__file__).parent / "annotator" / "index.html", project / "review" / "annotate.html")
        write_json(project / "review" / "cvat_label_contract.json", cvat_contract(schema))
        review_manifest = {
            "schema_version": "2.0", "project": manifest["project_id"],
            "video": manifest["paths"]["proxy_video"], "proposals": str(proposal_path),
            "proposal_count": len(proposals), "fiftyone_dataset": f"mtdata_{manifest['project_id']}",
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
                for key in ("event_id", "actor_id", "target_id", "anatomical_side", "effector_limb", "effector_id", "stance_at_start", "stance_role", "target_zone", "outcome", "contact_surface", "visibility", "review_status"):
                    detection[key] = event.get(key)
                detection["technique_candidates"] = json.dumps(event.get("technique_candidates", []), ensure_ascii=False)
                detection["branch_evidence"] = json.dumps(event.get("branch_evidence", []), ensure_ascii=False)
                detections.append(detection)
            sample["action_proposals"] = fo.TemporalDetections(detections=detections)
            dataset.add_sample(sample)
            session = fo.launch_app(dataset, port=args.port, address=os.environ.get("FIFTYONE_ADDRESS", "localhost"))
            session.wait()
        print(json.dumps(review_manifest, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
