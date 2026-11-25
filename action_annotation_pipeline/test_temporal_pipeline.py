from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from common import raw_track_for_fighter, write_json, write_jsonl  # noqa: E402
from export_posec3d import make_annotation  # noqa: E402
from extract_tracks import SimpleFighterTracker  # noqa: E402
from export_posec3d import main as export_posec3d_main  # noqa: E402
from export_tridet import main as export_tridet_main  # noqa: E402
from propose_actions import main as propose_actions_main  # noqa: E402
from validate_events import validate  # noqa: E402


class TemporalPipelineTests(unittest.TestCase):
    def make_synthetic_project(self, project: Path) -> None:
        for relative in ("annotations", "tracks", "poses", "proposals", "exports/posec3d", "exports/tridet", "qa"):
            (project / relative).mkdir(parents=True, exist_ok=True)
        write_json(project / "manifest.json", {
            "video_id": "test", "subset": "training", "expected_fighters": 2,
            "proxy_video": {"duration_sec": 2.0, "fps": 10.0, "width": 640, "height": 480},
            "paths": {},
        })
        write_json(project / "annotations" / "classes.json", json.loads((PIPELINE_ROOT / "classes.json").read_text(encoding="utf-8")))
        write_json(project / "tracks" / "fighter_map.json", {
            "fighters": [
                {"fighter_id": 0, "track_segments": [{"track_id": 10, "start_sec": 0, "end_sec": 2}]},
                {"fighter_id": 1, "track_segments": [{"track_id": 11, "start_sec": 0, "end_sec": 2}]},
            ]
        })
        rows = []
        for frame in range(20):
            for track_id in (10, 11):
                points = np.zeros((17, 2), dtype=np.float32)
                points[:, 0] = 100 + (track_id - 10) * 300
                points[:, 1] = np.arange(17) * 5 + 100
                if track_id == 10:
                    points[9, 0] += frame * 8
                rows.append((frame, frame / 10, track_id, points))
        np.savez_compressed(
            project / "poses" / "poses.npz",
            frame_indices=np.asarray([row[0] for row in rows], dtype=np.int32),
            timestamps=np.asarray([row[1] for row in rows], dtype=np.float64),
            track_ids=np.asarray([row[2] for row in rows], dtype=np.int32),
            bboxes_xyxy=np.asarray([[50, 50, 250, 450] if row[2] == 10 else [350, 50, 550, 450] for row in rows], dtype=np.float32),
            bbox_scores=np.ones(len(rows), dtype=np.float32),
            keypoints=np.asarray([row[3] for row in rows], dtype=np.float32),
            keypoint_scores=np.ones((len(rows), 17), dtype=np.float32),
        )
    def test_simple_tracker_keeps_two_ids(self) -> None:
        tracker = SimpleFighterTracker(1280, 720)
        first = tracker.update(
            np.asarray([[100, 100, 300, 600], [800, 100, 1000, 600]], dtype=np.float32),
            np.asarray([0.9, 0.8], dtype=np.float32),
        )
        second = tracker.update(
            np.asarray([[110, 100, 310, 600], [790, 100, 990, 600]], dtype=np.float32),
            np.asarray([0.9, 0.8], dtype=np.float32),
        )
        self.assertEqual({item[0] for item in first}, {0, 1})
        self.assertEqual({item[0] for item in second}, {0, 1})

    def test_fighter_map_segments(self) -> None:
        mapping = {
            "fighters": [
                {"fighter_id": 0, "track_segments": [
                    {"track_id": 2, "start_sec": 0, "end_sec": 4.9},
                    {"track_id": 7, "start_sec": 5.0, "end_sec": 10.0},
                ]}
            ]
        }
        self.assertEqual(raw_track_for_fighter(mapping, 0, 1.0), 2)
        self.assertEqual(raw_track_for_fighter(mapping, 0, 8.0), 7)

    def test_validator_accepts_reviewed_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "annotations").mkdir()
            write_json(project / "manifest.json", {
                "proxy_video": {"duration_sec": 10.0}, "paths": {}, "video_id": "test"
            })
            write_json(project / "annotations" / "classes.json", json.loads((PIPELINE_ROOT / "classes.json").read_text(encoding="utf-8")))
            write_jsonl(project / "annotations" / "events.jsonl", [
                {
                    "event_id": "review", "event_type": "review_range", "actor_id": -1,
                    "target_id": -1, "start_sec": 0.0, "peak_sec": 5.0, "end_sec": 10.0,
                    "label": "reviewed", "review_status": "verified",
                },
                {
                    "event_id": "action", "event_type": "action", "actor_id": 0,
                    "target_id": 1, "start_sec": 2.0, "peak_sec": 2.3, "end_sec": 2.7,
                    "label": "jab", "side": "left", "target_zone": "head",
                    "outcome": "blocked", "review_status": "verified",
                },
            ])
            errors, _, report = validate(project, project / "annotations" / "events.jsonl", True, False)
            self.assertEqual(errors, [])
            self.assertEqual(report["verified_actions"], 1)

    def test_posec3d_actor_is_person_zero(self) -> None:
        pose_data = {
            "keypoints": np.stack([
                np.full((17, 2), 1.0, dtype=np.float32),
                np.full((17, 2), 2.0, dtype=np.float32),
            ]),
            "keypoint_scores": np.ones((2, 17), dtype=np.float32),
        }
        mapping = {"fighters": [
            {"fighter_id": 0, "track_segments": [{"track_id": 10, "start_sec": 0, "end_sec": 1}]},
            {"fighter_id": 1, "track_segments": [{"track_id": 11, "start_sec": 0, "end_sec": 1}]},
        ]}
        annotation = make_annotation(
            "sample", 0.0, 0.0, 1, 0, 4, 30.0, 1280, 720,
            {(0, 10): 0, (0, 11): 1}, pose_data, mapping,
        )
        self.assertTrue(np.all(annotation["keypoint"][0, 0] == 2.0))
        self.assertTrue(np.all(annotation["keypoint"][1, 0] == 1.0))

    def test_proposal_and_export_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            self.make_synthetic_project(project)
            with patch.object(sys, "argv", ["propose_actions.py", "--project", str(project)]):
                self.assertEqual(propose_actions_main(), 0)
            self.assertTrue((project / "proposals" / "action_proposals.jsonl").is_file())
            write_jsonl(project / "annotations" / "events.jsonl", [
                {
                    "event_id": "review", "event_type": "review_range", "actor_id": -1,
                    "target_id": -1, "start_sec": 0.0, "peak_sec": 1.0, "end_sec": 2.0,
                    "label": "reviewed", "review_status": "verified",
                },
                {
                    "event_id": "jab", "event_type": "action", "actor_id": 0,
                    "target_id": 1, "start_sec": 0.5, "peak_sec": 0.8, "end_sec": 1.1,
                    "label": "jab", "side": "left", "target_zone": "head",
                    "outcome": "blocked", "review_status": "verified",
                },
            ])
            with patch.object(sys, "argv", ["export_posec3d.py", "--project", str(project)]):
                self.assertEqual(export_posec3d_main(), 0)
            with patch.object(sys, "argv", ["export_tridet.py", "--project", str(project)]):
                self.assertEqual(export_tridet_main(), 0)
            self.assertTrue((project / "exports" / "posec3d" / "posec3d_annotations.pkl").is_file())
            self.assertTrue((project / "exports" / "tridet" / "annotations.json").is_file())


if __name__ == "__main__":
    unittest.main()
