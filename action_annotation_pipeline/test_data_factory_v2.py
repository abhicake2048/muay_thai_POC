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
from action_signals import action_components, fighter_map_coverage  # noqa: E402
from extract_tracks import _was_automatic_map  # noqa: E402
from fighter_identity import stitch_fighter_tracks  # noqa: E402
from fuse_proposals import fuse  # noqa: E402
from schema import canonicalize_event, display_label  # noqa: E402
from state import StageState, stage_signature  # noqa: E402
from validate_events import validate  # noqa: E402
from benchmark_trackers import score_tracker  # noqa: E402
from vlm_qwen import _json_object, build_scan_tasks, normalize_events  # noqa: E402

import numpy as np


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

    def test_stale_smoke_map_does_not_cover_full_pose_timeline(self) -> None:
        mapping = {"fighters": [
            {"fighter_id": 0, "track_segments": [{"track_id": 10, "start_sec": 0, "end_sec": 4}]},
            {"fighter_id": 1, "track_segments": [{"track_id": 11, "start_sec": 0, "end_sec": 4}]},
        ]}
        report = fighter_map_coverage(
            mapping,
            np.asarray([0.0, 0.0, 3.9, 3.9, 10.0, 10.0]),
            np.asarray([10, 11, 10, 11, 10, 11]),
            2,
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["unmapped_pose_instances"], 2)

    def test_only_automatic_fighter_maps_are_replaced(self) -> None:
        previous = {"fighters": [{"fighter_id": 0, "track_segments": [{"track_id": 1, "start_sec": 0, "end_sec": 4}]}]}
        self.assertTrue(_was_automatic_map(dict(previous), previous))
        self.assertTrue(_was_automatic_map({**previous, "review_status": "suggested"}, None))
        self.assertFalse(_was_automatic_map({**previous, "review_status": "verified"}, previous))
        modified = {"fighters": [{"fighter_id": 0, "track_segments": [{"track_id": 9, "start_sec": 0, "end_sec": 4}]}]}
        self.assertFalse(_was_automatic_map(modified, previous))

    def test_dense_vlm_scan_exists_without_pose_candidates(self) -> None:
        tasks = build_scan_tasks([], 6.0, 2, 3.0, 1.5, "hybrid")
        self.assertGreaterEqual(len(tasks), 4)
        self.assertEqual({task["actor_id"] for task in tasks}, {0, 1})
        self.assertTrue(all(task["kind"] == "dense" for task in tasks))

    def test_candidate_vlm_windows_are_grouped_before_model_inference(self) -> None:
        rows = [
            {"event_id": f"p{index}", "actor_id": 0, "peak_sec": peak, "proposal_confidence": 0.8}
            for index, peak in enumerate((1.0, 1.1, 1.35))
        ]
        tasks = build_scan_tasks(rows, 5.0, 2, 3.0, 1.5, "candidates")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["source_event_ids"]), 3)

    def test_component_first_segmentation_removes_duplicate_peak_intervals(self) -> None:
        noisy_single = np.asarray([0, 0, 0.9, 2.5, 1.4, 2.2, 0.9, 0.9, 0, 0], dtype=np.float32)
        self.assertEqual(len(action_components(noisy_single, 2.0, 0.75, 2)), 1)
        true_pair = np.asarray([0, 0, 0.9, 2.5, 0.8, 0.8, 2.4, 0.9, 0, 0], dtype=np.float32)
        self.assertEqual(len(action_components(true_pair, 2.0, 0.75, 2)), 2)

    def test_fighter_identity_stitches_raw_id_fragments_and_ignores_bystander(self) -> None:
        records = []
        for frame in range(6):
            left_id = 10 if frame < 3 else 50
            records.append({
                "frame_index": frame,
                "timestamp": frame / 30.0,
                "detections": [
                    {"track_id": left_id, "bbox_xyxy": [100 + frame, 80, 300 + frame, 650], "confidence": 0.95},
                    {"track_id": 20, "bbox_xyxy": [800 - frame, 80, 1000 - frame, 650], "confidence": 0.94},
                    {"track_id": 99, "bbox_xyxy": [10, 10, 40, 80], "confidence": 0.70},
                ],
            })
        fighters, report = stitch_fighter_tracks(records, 1280, 720, 30.0)
        self.assertEqual(report["raw_track_ids_by_fighter"]["0"], [10, 50])
        self.assertEqual(report["raw_track_ids_by_fighter"]["1"], [20])
        self.assertEqual(report["unassigned_detections"], 6)
        self.assertGreaterEqual(len(fighters[0]["track_segments"]), 2)

    def test_vlm_window_can_return_multiple_actions(self) -> None:
        payload = {"events": [
            {"start_offset_sec": 0.2, "peak_offset_sec": 0.4, "end_offset_sec": 0.7, "technique": "jab", "confidence": 0.8, "anatomical_side": "left", "effector_limb": "hand"},
            {"start_offset_sec": 0.8, "peak_offset_sec": 1.0, "end_offset_sec": 1.3, "technique": "cross", "confidence": 0.75, "anatomical_side": "right", "effector_limb": "hand"},
        ]}
        task = {"task_id": "dense_1", "kind": "dense", "actor_id": 0, "start_sec": 10.0, "end_sec": 13.0, "source": None}
        events = normalize_events(payload, task, {"video_id": "test"}, self.schema)
        self.assertEqual(len(events), 2)
        self.assertEqual([event["technique"] for event in events], ["jab", "cross"])
        self.assertEqual(events[1]["peak_sec"], 11.0)

    def test_vlm_zero_event_and_malformed_json_handling(self) -> None:
        task = {"task_id": "dense_1", "kind": "dense", "actor_id": 0, "start_sec": 0.0, "end_sec": 3.0, "source": None}
        self.assertEqual(normalize_events({"events": []}, task, {"video_id": "test"}, self.schema), [])
        with self.assertRaises(ValueError):
            _json_object("not JSON")
        with self.assertRaises(ValueError):
            _json_object('{"events":"jab"}')

    def test_conflicting_low_confidence_side_stays_unknown(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.6, "event_type": "action", "review_status": "proposed", "effector_limb": "hand"}
        rows = [
            {**base, "event_id": "pose", "technique": "unknown_action", "anatomical_side": "left", "proposal_branch": "pose_attributes", "proposal_confidence": 0.7, "attribute_confidence": 0.6},
            {**base, "event_id": "qwen", "technique": "jab", "anatomical_side": "right", "proposal_branch": "qwen_actions", "proposal_confidence": 0.7, "attribute_confidence": 0.7},
        ]
        output = fuse(rows, self.schema)
        self.assertEqual(output[0]["anatomical_side"], "unknown")

    def test_vlm_technique_must_match_pose_family(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.6, "event_type": "action", "review_status": "proposed", "anatomical_side": "left"}
        rows = [
            {**base, "event_id": "pose", "technique": "unknown_action", "effector_limb": "hand", "family_candidates": [{"label": "punch", "confidence": 0.8}], "proposal_branch": "pose_actionness", "proposal_confidence": 0.8},
            {**base, "event_id": "qwen", "technique": "roundhouse_kick", "effector_limb": "hand", "proposal_branch": "qwen_actions", "proposal_confidence": 0.95},
        ]
        output = fuse(rows, self.schema)
        self.assertEqual(output[0]["technique"], "unknown_action")
        self.assertEqual(output[0]["technique_candidates"][0]["technique"], "roundhouse_kick")

    def test_fusion_preserves_nearby_visual_combination(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "end_sec": 2.0, "event_type": "action", "review_status": "proposed", "anatomical_side": "left", "effector_limb": "hand", "proposal_branch": "qwen_actions", "proposal_confidence": 0.9}
        rows = [
            {**base, "event_id": "jab1", "peak_sec": 1.20, "technique": "jab"},
            {**base, "event_id": "jab2", "peak_sec": 1.45, "technique": "jab"},
        ]
        output = fuse(rows, self.schema)
        self.assertEqual(len(output), 2)

    def test_fusion_merges_same_qwen_event_from_overlapping_scan_windows(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "end_sec": 1.5, "event_type": "action", "review_status": "proposed", "anatomical_side": "left", "effector_limb": "hand", "proposal_branch": "qwen_actions", "proposal_confidence": 0.9, "technique": "jab"}
        rows = [
            {**base, "event_id": "window1", "peak_sec": 1.22, "provenance": {"task_id": "dense_1"}},
            {**base, "event_id": "window2", "peak_sec": 1.25, "provenance": {"task_id": "dense_2"}},
        ]
        self.assertEqual(len(fuse(rows, self.schema)), 1)

    def test_pose_attributes_do_not_double_vote_with_source_actionness(self) -> None:
        base = {"event_id": "same", "actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.2, "end_sec": 1.4, "event_type": "action", "review_status": "proposed", "technique": "unknown_action", "effector_limb": "hand", "anatomical_side": "left", "proposal_confidence": 0.8}
        output = fuse([
            {**base, "proposal_branch": "pose_actionness"},
            {**base, "proposal_branch": "pose_attributes", "stance_at_start": "orthodox"},
        ], self.schema)
        self.assertEqual(len(output), 1)
        self.assertEqual(len(output[0]["branch_evidence"]), 1)
        self.assertEqual(output[0]["branch_evidence"][0]["branch"], "pose_attributes")

    def test_fusion_preserves_different_pose_effectors_at_same_time(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.6, "event_type": "action", "review_status": "proposed", "technique": "unknown_action", "proposal_branch": "pose_actionness", "proposal_confidence": 0.8}
        rows = [
            {**base, "event_id": "left", "pose_effector_id": "left_hand", "anatomical_side": "left", "effector_limb": "hand"},
            {**base, "event_id": "right", "pose_effector_id": "right_hand", "anatomical_side": "right", "effector_limb": "hand"},
        ]
        self.assertEqual(len(fuse(rows, self.schema)), 2)

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
                write_json(project / "exports" / "posec3d" / "class_map.json", {"classes": ["jab"]})
                write_json(project / "exports" / "posec3d" / "augmentation_policy.json", {"apply_at_training_time_only": True})
                write_json(project / "exports" / "tridet" / "annotations.json", {"taxonomy": ["jab"], "database": {f"video{index}__fighter0": {"source_video": f"video{index}", "annotations": []}}})
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
