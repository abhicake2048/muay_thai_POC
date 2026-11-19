from __future__ import annotations

import unittest
from pathlib import Path

from prepare_combined_pose_dataset import (
    CANONICAL_INDEX,
    MMA_DUPLICATE_LIMB_MAPPING,
    MMA_DUPLICATE_LIMB_SCHEMA,
    MMA_HEAD_CHIN_MAPPING,
    MMA_HEAD_CHIN_SCHEMA,
    ImageSample,
    PoseInstance,
    assign_grouped_splits,
    infer_group,
    map_and_clean_keypoints,
    schema_mapping,
)


class CombinedDatasetTests(unittest.TestCase):
    def test_recognizes_all_supported_schemas(self) -> None:
        coco_names = list(CANONICAL_INDEX)
        self.assertEqual(schema_mapping(coco_names)[0], "coco17")
        self.assertEqual(
            schema_mapping(list(MMA_HEAD_CHIN_SCHEMA)),
            ("mma_head_chin_14", MMA_HEAD_CHIN_MAPPING),
        )
        self.assertEqual(
            schema_mapping(list(MMA_DUPLICATE_LIMB_SCHEMA)),
            ("mma_duplicate_limb_14", MMA_DUPLICATE_LIMB_MAPPING),
        )
        self.assertIsNone(schema_mapping(["head", "unknown_joint"]))

    def test_custom_mapping_outputs_coco17_and_ignores_out_of_frame_points(self) -> None:
        values = []
        for index in range(14):
            values.extend([10.0 + index, 20.0 + index, 2])
        values[0] = 999.0  # mapped head is outside the 100 px image
        output = map_and_clean_keypoints(values, MMA_HEAD_CHIN_MAPPING, 100, 100)
        self.assertEqual(len(output), 51)
        self.assertEqual(output[2], 0)
        self.assertEqual(output[5 * 3 + 2], 2)  # left shoulder
        self.assertEqual(output[11 * 3 + 2], 0)  # center waist is not duplicated to hips

    def test_video_frames_and_augmentations_share_a_group(self) -> None:
        first = "fight_mp4-0001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"
        second = "fight_mp4-0999_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
        self.assertEqual(infer_group(first), "fight_mp4")
        self.assertEqual(infer_group(second), "fight_mp4")
        self.assertEqual(
            infer_group("kick-15_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg"),
            "kick_15",
        )

    def test_grouped_split_is_complete_deterministic_and_leakage_free(self) -> None:
        pose = PoseInstance([0.0, 0.0, 20.0, 30.0], [0.0] * 51, 4)
        samples = []
        for group_index in range(9):
            for image_index in range(3):
                digest = f"{group_index:02d}{image_index:02d}".ljust(64, "0")
                samples.append(
                    ImageSample(
                        source_name="test",
                        source_path=Path(f"{digest}.jpg"),
                        original_name=f"group{group_index}-{image_index}.jpg",
                        source_split="train",
                        width=100,
                        height=100,
                        pixel_hash=digest,
                        group=f"group_{group_index}",
                        instances=[pose],
                    )
                )
        result = assign_grouped_splits(samples, 0.8, 0.1, 42)
        self.assertEqual(sum(len(records) for records in result.values()), len(samples))
        self.assertTrue(all(result[split] for split in ("train", "val", "test")))
        groups = {
            split: {sample.group for sample in records} for split, records in result.items()
        }
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])


if __name__ == "__main__":
    unittest.main()
