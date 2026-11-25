"""Small OpenCV-only pose augmentations used by the RTMPose config.

Albumentations always depends on ``opencv-python-headless`` while MMPose and
MMEngine depend on ``opencv-python``.  Installing both distributions gives two
packages ownership of the same ``cv2`` files.  This transform keeps the useful
blur/occlusion augmentation without creating that binary-package collision.
"""

from __future__ import annotations

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CombatBlurDropout(BaseTransform):
    """Apply light blur and one rectangular occlusion to a pose crop."""

    def __init__(
        self,
        blur_probability: float = 0.1,
        median_blur_probability: float = 0.1,
        dropout_probability: float = 0.5,
        dropout_height: tuple[float, float] = (0.2, 0.4),
        dropout_width: tuple[float, float] = (0.2, 0.4),
        fill_value: int = 0,
    ) -> None:
        for name, value in (
            ("blur_probability", blur_probability),
            ("median_blur_probability", median_blur_probability),
            ("dropout_probability", dropout_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, bounds in (
            ("dropout_height", dropout_height),
            ("dropout_width", dropout_width),
        ):
            if not 0.0 < bounds[0] <= bounds[1] <= 1.0:
                raise ValueError(f"{name} must be an increasing fraction pair")
        self.blur_probability = blur_probability
        self.median_blur_probability = median_blur_probability
        self.dropout_probability = dropout_probability
        self.dropout_height = dropout_height
        self.dropout_width = dropout_width
        self.fill_value = fill_value

    def transform(self, results: dict) -> dict:
        image = results["img"]
        if np.random.random() < self.blur_probability:
            image = cv2.blur(image, (3, 3))
        if np.random.random() < self.median_blur_probability:
            image = cv2.medianBlur(image, 3)
        if np.random.random() < self.dropout_probability:
            height, width = image.shape[:2]
            hole_height = max(
                1,
                int(round(height * np.random.uniform(*self.dropout_height))),
            )
            hole_width = max(
                1,
                int(round(width * np.random.uniform(*self.dropout_width))),
            )
            top = int(np.random.randint(0, max(1, height - hole_height + 1)))
            left = int(np.random.randint(0, max(1, width - hole_width + 1)))
            image = image.copy()
            image[top : top + hole_height, left : left + hole_width] = self.fill_value
        results["img"] = image
        return results
