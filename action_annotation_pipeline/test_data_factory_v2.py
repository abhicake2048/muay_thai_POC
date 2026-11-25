from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from aggregate_dataset import freeze  # noqa: E402
from common import read_json, write_json, write_jsonl  # noqa: E402
from fuse_proposals import fuse  # noqa: E402
from schema import canonicalize_event, display_label  # noqa: E402
from state import StageState, stage_signature  # noqa: E402
from validate_events import validate  # noqa: E402
from benchmark_trackers import score_tracker  # noqa: E402


class DataFactoryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = read_json(PIPELINE_ROOT / "classes.json")

    def test_southpaw_jab_keeps_anatomical_right_hand(self) -> None:
        event = canonicalize_event({
            "event_type": "action", "label": "jab", "side": "right",
            "stance_at_start": "southpaw", "stance_role": "lead",
            "target_zone": "head", "outcome": "missed", "visibility": "visible",
        }, self.schema)
        self.assertEqual(event["technique"], "jab")
        self.assertEqual(event["anatomical_side"], "right")
        self.assertEqual(event["effector_id"], "right_hand")
        self.assertEqual(display_label(event, include_target=False), "jab__right_hand")

    def test_boxing6_and_v1_migrations_are_factorized(self) -> None:
        hook = canonicalize_event({"event_type": "action", "label": "lead_hook", "side": "left"}, self.schema)
        kick = canonicalize_event({"event_type": "action", "label": "head_kick", "side": "right"}, self.schema)
        self.assertEqual((hook["technique"], hook["stance_role"], hook["effector_id"]), ("hook", "lead", "left_hand"))
        self.assertEqual((kick["technique"], kick["target_zone"], kick["effector_id"]), ("roundhouse_kick", "head", "right_leg"))

    def test_fusion_retains_branch_evidence_and_side(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.7, "event_type": "action", "review_status": "proposed"}
        rows = [
            {**base, "event_id": "motion", "technique": "unknown_action", "anatomical_side": "left", "effector_limb": "hand", "proposal_branch": "pose_attributes", "proposal_confidence": 0.8},
            {**base, "event_id": "vlm", "technique": "hook", "action_family": "punch", "anatomical_side": "left", "effector_limb": "hand", "proposal_branch": "qwen3_vl", "proposal_confidence": 0.9},
        ]
        output = fuse(rows, self.schema)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["technique"], "hook")
        self.assertEqual(output[0]["effector_id"], "left_hand")
        self.assertEqual(len(output[0]["branch_evidence"]), 2)

    def test_stage_state_only_accepts_current_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.txt", root / "output.txt"
            source.write_text("a", encoding="utf-8")
            output.write_text("b", encoding="utf-8")
            signature = stage_signature([source], {"mode": "test"})
            state = StageState(root)
            state.record("test", signature, "complete", outputs=[output])
            self.assertTrue(state.is_current("test", signature, [output]))
            source.write_text("changed", encoding="utf-8")
            self.assertNotEqual(signature, stage_signature([source], {"mode": "test"}))

    def test_v2_validator_accepts_explicit_factorized_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "annotations").mkdir()
            (project / "qa").mkdir()
            write_json(project / "manifest.json", {"proxy_video": {"duration_sec": 3.0}, "video_id": "test"})
            write_json(project / "annotations" / "classes.json", self.schema)
            write_jsonl(project / "annotations" / "events.jsonl", [
                {"event_id": "review", "event_type": "review_range", "start_sec": 0, "peak_sec": 1.5, "end_sec": 3, "label": "reviewed", "review_status": "verified"},
                {"event_id": "jab", "event_type": "action", "actor_id": 0, "target_id": 1, "start_sec": 1, "peak_sec": 1.2, "end_sec": 1.5,
                 "action_family": "punch", "technique": "jab", "label": "jab", "anatomical_side": "right", "side": "right", "effector_limb": "hand", "effector_id": "right_hand",
                 "stance_at_start": "southpaw", "stance_role": "lead", "target_zone": "head", "outcome": "blocked", "contact_surface": "fist", "visibility": "visible", "review_status": "verified"},
            ])
            errors, _, report = validate(project, project / "annotations" / "events.jsonl", True, False)
            self.assertEqual(errors, [])
            self.assertEqual(report["verified_actions"], 1)

    def test_freeze_splits_by_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "projects"
            for index in range(3):
                project = root / f"video{index}"
                (project / "exports" / "posec3d").mkdir(parents=True)
                (project / "exports" / "tridet").mkdir(parents=True)
                write_json(project / "manifest.json", {"video_id": f"video{index}"})
                sample = {"frame_dir": f"video{index}__sample", "label": 0}
                with (project / "exports" / "posec3d" / "posec3d_annotations.pkl").open("wb") as handle:
                    pickle.dump({"split": {"train": [sample["frame_dir"]]}, "annotations": [sample]}, handle)
                write_json(project / "exports" / "tridet" / "annotations.json", {"database": {f"video{index}__fighter0": {"source_video": f"video{index}", "annotations": []}}})
            report = freeze(root, Path(temporary) / "frozen", 42)
            self.assertEqual(report["leakage_check"], "passed")
            assigned = [video for split in report["splits"].values() for video in split]
            self.assertEqual(len(assigned), len(set(assigned)))

    def test_tracker_benchmark_counts_identity_switches(self) -> None:
        gold = [
            {"frame_index": 0, "fighters": [{"fighter_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_index": 1, "fighters": [{"fighter_id": 0, "bbox_xyxy": [1, 0, 11, 10]}]},
        ]
        stable = [
            {"frame_index": 0, "detections": [{"track_id": 4, "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_index": 1, "detections": [{"track_id": 4, "bbox_xyxy": [1, 0, 11, 10]}]},
        ]
        switched = [
            {"frame_index": 0, "detections": [{"track_id": 4, "bbox_xyxy": [0, 0, 10, 10]}]},
            {"frame_index": 1, "detections": [{"track_id": 8, "bbox_xyxy": [1, 0, 11, 10]}]},
        ]
        self.assertEqual(score_tracker(gold, stable, 0.3)["identity_switches"], 0)
        self.assertEqual(score_tracker(gold, switched, 0.3)["identity_switches"], 1)


if __name__ == "__main__":
    unittest.main()
