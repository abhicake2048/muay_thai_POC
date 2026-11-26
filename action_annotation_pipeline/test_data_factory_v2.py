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
from action_diagnostics import _assessment_coverage  # noqa: E402
from vlm_qwen import _composite_frames, _content_fingerprint, _json_object, _task_fingerprint, build_scan_tasks, load_resumable_task, normalize_events, validate_vlm_payload  # noqa: E402
from extract_videomae_features import pool_video_tokens, slice_stream_features  # noqa: E402
from training_contracts import validate_tridet_payload  # noqa: E402
from review import fiftyone_detections_to_events  # noqa: E402
from evaluate import ranked_techniques  # noqa: E402

import numpy as np
import cv2


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

    def test_fighter_may_end_before_other_when_its_own_poses_are_fully_covered(self) -> None:
        mapping = {
            "processed_start_sec": 0.0,
            "processed_end_sec": 190.533,
            "fighters": [
                {"fighter_id": 0, "track_segments": [{"track_id": 10, "start_sec": 0, "end_sec": 190.55}]},
                {"fighter_id": 1, "track_segments": [{"track_id": 11, "start_sec": 0, "end_sec": 189.683}]},
            ],
        }
        report = fighter_map_coverage(
            mapping,
            np.asarray([0.0, 0.0, 189.667, 190.533]),
            np.asarray([10, 11, 11, 10]),
            2,
            canonical_fighters=np.asarray([0, 1, 1, 0]),
        )
        self.assertTrue(report["complete"])
        self.assertTrue(report["fighters"][1]["covers_observed_pose_span"])

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

    def test_qwen_resume_fingerprint_tracks_inputs_and_task_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl"
            source.write_text('{"event_id":"one"}\n', encoding="utf-8")
            first = _content_fingerprint([source], {"prompt": "v2", "fps": 8.0})
            self.assertEqual(first, _content_fingerprint([source], {"fps": 8.0, "prompt": "v2"}))
            task = {"task_id": "candidate_000001", "actor_id": 0, "start_sec": 1.0, "end_sec": 4.0, "source": {"volatile": True}}
            signature = _task_fingerprint(first, task)
            self.assertEqual(signature, _task_fingerprint(first, {**task, "source": {"different": True}}))
            self.assertNotEqual(signature, _task_fingerprint(first, {**task, "end_sec": 4.5}))
            source.write_text('{"event_id":"two"}\n', encoding="utf-8")
            self.assertNotEqual(first, _content_fingerprint([source], {"prompt": "v2", "fps": 8.0}))

    def test_qwen_resume_accepts_only_valid_action_assessments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "window.json"
            task = {"task_id": "dense_1", "kind": "dense", "actor_id": 1, "start_sec": 0.0, "end_sec": 3.0, "source": None}
            manifest = {"video_id": "test", "proxy_video": {"fps": 30.0}}
            no_action = {
                "contains_action": False,
                "window_summary": "Both fighters circle in guard.",
                "events": [],
                "no_action_confidence": 0.88,
                "no_action_reason": "No committed attack or defense is visible.",
            }
            write_json(raw, {"task_fingerprint": "match", "parsed": no_action, "crop_stats": {"actor_crop_coverage": 1.0}})
            result = load_resumable_task(raw, "match", task, manifest, self.schema, set(self.schema["trainable_event_classes"]))
            self.assertIsNotNone(result)
            self.assertEqual(result[2], [])
            self.assertIsNone(load_resumable_task(raw, "changed", task, manifest, self.schema, set(self.schema["trainable_event_classes"])))
            write_json(raw, {"task_fingerprint": "match", "parsed": {"contains_action": False, "events": []}})
            self.assertIsNone(load_resumable_task(raw, "match", task, manifest, self.schema, set(self.schema["trainable_event_classes"])))

    def test_qwen_assessment_coverage_counts_explicit_no_action_windows(self) -> None:
        rows = [
            {"actor_id": fighter, "start_sec": start, "end_sec": end, "status": "no_action"}
            for fighter in (0, 1) for start, end in ((0.0, 3.0), (3.0, 6.0))
        ]
        coverage = _assessment_coverage(rows, 2, 6.0)
        self.assertEqual(coverage["0"]["coverage_ratio"], 1.0)
        self.assertEqual(coverage["1"]["coverage_ratio"], 1.0)

    def test_candidate_vlm_windows_are_grouped_before_model_inference(self) -> None:
        rows = [
            {"event_id": f"p{index}", "actor_id": 0, "peak_sec": peak, "proposal_confidence": 0.8}
            for index, peak in enumerate((1.0, 1.1, 1.35))
        ]
        tasks = build_scan_tasks(rows, 5.0, 2, 3.0, 1.5, "candidates")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["source_event_ids"]), 3)

    def test_vlm_composite_preserves_timestamps_when_actor_box_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
            self.assertTrue(writer.isOpened())
            for frame in range(10):
                writer.write(np.full((64, 64, 3), frame * 10, dtype=np.uint8))
            writer.release()
            tracks = {0: [{"fighter_id": 0, "track_id": 1, "bbox_xyxy": [8, 4, 40, 60]}]}
            frames, stats = _composite_frames(video, 0.0, 1.0, 10.0, 0, 10.0, tracks, {"fighters": []})
            self.assertEqual(len(frames), 10)
            self.assertEqual(stats["decoded_frames"], 10)
            self.assertEqual(stats["actor_crop_coverage"], 0.7)

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

    def test_vlm_contract_requires_explicit_explained_action_or_no_action(self) -> None:
        action = {
            "contains_action": True,
            "window_summary": "Actor extends the lead hand and retracts.",
            "events": [{
                "start_offset_sec": 0.2, "peak_offset_sec": 0.4, "end_offset_sec": 0.7,
                "technique": "jab", "confidence": 0.8,
                "technique_candidates": [{"technique": "jab", "confidence": 0.8}],
                "anatomical_side": "left", "effector_limb": "hand",
                "stance_at_start": "orthodox", "stance_role": "lead",
                "target_zone": "head", "outcome": "attempted", "contact_surface": "fist",
                "visibility": "visible", "uncertainty_flags": [],
                "classification_reason": "Lead hand travels straight toward the head.",
                "boundary_reason": "Extension onset through completed retraction.",
                "visual_evidence": ["wrist advances", "elbow extends"],
            }],
            "no_action_confidence": 0.0,
            "no_action_reason": "",
        }
        self.assertIs(validate_vlm_payload(action, 3.0, set(self.schema["primary_action_classes"]), self.schema), action)
        no_action = {
            "contains_action": False, "window_summary": "Both fighters circle without a strike.",
            "events": [], "no_action_confidence": 0.91,
            "no_action_reason": "Only footwork and guard adjustment are visible.",
        }
        self.assertIs(validate_vlm_payload(no_action, 3.0, set(self.schema["primary_action_classes"]), self.schema), no_action)
        invalid = dict(no_action, contains_action=True)
        with self.assertRaises(ValueError):
            validate_vlm_payload(invalid, 3.0, set(self.schema["primary_action_classes"]), self.schema)

    def test_vlm_can_emit_balance_event_without_merging_into_strike_motion(self) -> None:
        prediction = {
            "start_offset_sec": 0.5, "peak_offset_sec": 0.8, "end_offset_sec": 1.2,
            "technique": "knockdown", "technique_candidates": [{"technique": "knockdown", "confidence": 0.9}],
            "anatomical_side": "not_applicable", "effector_limb": "whole_body",
            "stance_at_start": "unknown", "stance_role": "not_applicable", "target_zone": "not_applicable",
            "outcome": "uncertain", "contact_surface": "not_applicable", "visibility": "visible",
            "confidence": 0.9, "uncertainty_flags": [], "classification_reason": "Actor falls to the canvas.",
            "boundary_reason": "Loss of balance through ground contact.", "visual_evidence": ["torso descends", "body reaches canvas"],
        }
        payload = {"contains_action": True, "window_summary": "Actor is knocked down.", "events": [prediction], "no_action_confidence": 0.0, "no_action_reason": ""}
        allowed = set(self.schema["trainable_event_classes"])
        validate_vlm_payload(payload, 3.0, allowed, self.schema)
        task = {"task_id": "balance", "kind": "dense", "actor_id": 0, "start_sec": 5.0, "end_sec": 8.0, "source": None}
        event = normalize_events(payload, task, {"video_id": "test", "proxy_video": {"fps": 30.0}}, self.schema)[0]
        self.assertEqual(event["event_type"], "balance")
        self.assertEqual(event["target_id"], -1)
        strike = {**event, "event_id": "pose", "event_type": "action", "technique": "unknown_action", "proposal_branch": "pose_actionness"}
        self.assertEqual(len(fuse([event, strike], self.schema)), 2)

    def test_videomae_pooling_preserves_channel_dimension(self) -> None:
        class TensorStub:
            def __init__(self, array):
                self.array = np.asarray(array)
                self.ndim = self.array.ndim
                self.shape = self.array.shape
            def float(self):
                return self
            def mean(self, dim):
                return TensorStub(self.array.mean(axis=dim))
            def cpu(self):
                return self
            def numpy(self):
                return self.array
        vector = pool_video_tokens(TensorStub(np.arange(24).reshape(1, 3, 8)))
        self.assertEqual(vector.shape, (8,))
        self.assertTrue(np.allclose(vector, np.arange(8) + 8))

    def test_fiftyone_review_round_trip_emits_syncable_events(self) -> None:
        class Detection(dict):
            def __init__(self, label, support, **fields):
                super().__init__(fields)
                self.label = label
                self.support = support
            def get_field(self, key):
                if key == "label":
                    return self.label
                if key == "support":
                    return self.support
                return self.get(key)
        rows = fiftyone_detections_to_events([
            Detection("reviewed", [1, 61]),
            Detection("jab", [16, 31], event_id="jab", event_type="action", actor_id=0, target_id=1,
                      anatomical_side="left", effector_limb="hand", stance_at_start="orthodox", stance_role="lead",
                      target_zone="head", outcome="attempted", contact_surface="fist", visibility="visible",
                      review_status="verified", proxy_peak_frame=22),
            Detection("cross", [25, 31], event_id="cross", event_type="action", actor_id=0, target_id=1,
                      anatomical_side="right", effector_limb="hand", stance_at_start="orthodox", stance_role="rear",
                      target_zone="head", outcome="missed", contact_surface="fist", visibility="visible",
                      review_status="verified", proxy_peak_frame=10),
        ], 30.0, self.schema)
        self.assertEqual(rows[0]["event_type"], "review_range")
        self.assertEqual(rows[0]["review_status"], "verified")
        action = next(row for row in rows if row["event_type"] == "action")
        self.assertEqual((action["start_sec"], action["peak_sec"], action["end_sec"]), (0.5, 0.733333, 1.0))
        self.assertEqual(action["effector_id"], "left_hand")
        trimmed = next(row for row in rows if row["event_id"] == "cross")
        self.assertEqual(trimmed["peak_sec"], 0.9)

    def test_feature_slicing_excludes_clips_crossing_ignored_boundaries(self) -> None:
        features = np.arange(24, dtype=np.float32).reshape(6, 4)
        timestamps = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
        starts = timestamps - 0.25
        ends = timestamps + 0.25
        sliced, local_times = slice_stream_features(features, timestamps, starts, ends, 1.0, 2.5)
        self.assertEqual(sliced.shape, (3, 4))
        self.assertTrue(np.array_equal(sliced, features[2:5]))
        self.assertTrue(np.allclose(local_times, [0.25, 0.75, 1.25]))

    def test_conflicting_low_confidence_side_stays_unknown(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.6, "event_type": "action", "review_status": "proposed", "effector_limb": "hand"}
        rows = [
            {**base, "event_id": "pose", "technique": "unknown_action", "anatomical_side": "left", "proposal_branch": "pose_attributes", "proposal_confidence": 0.7, "attribute_confidence": 0.6},
            {**base, "event_id": "qwen", "technique": "jab", "anatomical_side": "right", "proposal_branch": "qwen_actions", "proposal_confidence": 0.7, "attribute_confidence": 0.7},
        ]
        output = fuse(rows, self.schema)
        self.assertEqual(output[0]["anatomical_side"], "unknown")

    def test_evaluation_uses_ranked_fused_and_branch_candidates(self) -> None:
        proposal = {
            "technique": "unknown_action",
            "technique_candidates": [
                {"technique": "hook", "confidence": 0.6},
                {"technique": "jab", "confidence": 0.9},
            ],
            "branch_evidence": [{
                "technique": "cross",
                "technique_candidates": [{"technique": "uppercut", "confidence": 0.8}],
            }],
        }
        self.assertEqual(ranked_techniques(proposal), ["jab", "hook", "cross"])

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

    def test_fusion_preserves_simultaneous_different_actions_from_one_vlm_window(self) -> None:
        base = {"actor_id": 0, "target_id": 1, "start_sec": 1.0, "peak_sec": 1.3, "end_sec": 1.6, "event_type": "action", "review_status": "proposed", "anatomical_side": "left", "effector_limb": "hand", "proposal_branch": "qwen_actions", "proposal_confidence": 0.9, "provenance": {"task_id": "same_window"}}
        rows = [
            {**base, "event_id": "jab", "technique": "jab"},
            {**base, "event_id": "parry", "technique": "parry"},
        ]
        self.assertEqual(len(fuse(rows, self.schema)), 2)

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
                stream_id = f"video{index}__fighter0"
                write_json(project / "exports" / "tridet" / "annotations.json", {
                    "taxonomy": ["jab"], "label_to_id": {"jab": 0},
                    "database": {stream_id: {"source_video": f"video{index}", "duration": 2.0, "fps": 30.0, "subset": "training", "annotations": []}},
                })
                feature_dir = project / "exports" / "tridet" / "features"
                feature_dir.mkdir()
                np.save(feature_dir / f"{stream_id}.npy", np.ones((4, 8), dtype=np.float16))
                np.save(feature_dir / "timestamps.npy", np.arange(4, dtype=np.float32))
                write_json(project / "exports" / "tridet" / "feature_manifest.json", {
                    stream_id: {"feature_file": f"features/{stream_id}.npy", "status": "complete"}
                })
            report = freeze(root, Path(temporary) / "frozen", 42)
            self.assertEqual(report["leakage_check"], "passed")
            self.assertEqual(report["tridet_feature_files"], 3)
            assigned = [video for split in report["splits"].values() for video in split]
            self.assertEqual(len(assigned), len(set(assigned)))
            frozen = read_json(Path(temporary) / "frozen" / "tridet_annotations.json")
            self.assertEqual(validate_tridet_payload(frozen, require_training_subset=True), [])

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
