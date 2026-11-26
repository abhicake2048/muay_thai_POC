"""Strict format checks for MMAction2 PoseC3D and TriDet exports."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_posec3d_payload(payload: dict[str, Any], class_count: int) -> list[str]:
    errors: list[str] = []
    splits = payload.get("split")
    annotations = payload.get("annotations")
    if not isinstance(splits, dict):
        return ["PoseC3D payload must contain a split object"]
    if not isinstance(annotations, list):
        return ["PoseC3D payload must contain an annotations array"]
    sample_ids: set[str] = set()
    for index, annotation in enumerate(annotations):
        prefix = f"annotations[{index}]"
        if not isinstance(annotation, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = str(annotation.get("frame_dir", ""))
        if not sample_id:
            errors.append(f"{prefix}.frame_dir is required")
        elif sample_id in sample_ids:
            errors.append(f"duplicate PoseC3D frame_dir: {sample_id}")
        sample_ids.add(sample_id)
        keypoint = np.asarray(annotation.get("keypoint"))
        score = np.asarray(annotation.get("keypoint_score"))
        if keypoint.ndim != 4 or keypoint.shape[0] != 2 or keypoint.shape[2:] != (17, 2):
            errors.append(f"{prefix}.keypoint must have shape [2,T,17,2], got {keypoint.shape}")
            continue
        expected_score = (2, keypoint.shape[1], 17)
        if score.shape != expected_score:
            errors.append(f"{prefix}.keypoint_score must have shape {expected_score}, got {score.shape}")
        if int(annotation.get("total_frames", -1)) != keypoint.shape[1]:
            errors.append(f"{prefix}.total_frames does not match keypoint T")
        label = annotation.get("label")
        if not isinstance(label, (int, np.integer)) or not 0 <= int(label) < class_count:
            errors.append(f"{prefix}.label is outside the class map")
        for shape_key in ("img_shape", "original_shape"):
            shape = annotation.get(shape_key)
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                errors.append(f"{prefix}.{shape_key} must be [height,width]")
    for split_name, values in splits.items():
        if not isinstance(values, list):
            errors.append(f"split.{split_name} must be an array")
            continue
        missing = set(map(str, values)) - sample_ids
        if missing:
            errors.append(f"split.{split_name} references unknown samples: {sorted(missing)[:3]}")
    return errors


def validate_tridet_payload(payload: dict[str, Any], require_training_subset: bool = False) -> list[str]:
    errors: list[str] = []
    database = payload.get("database")
    taxonomy = payload.get("taxonomy")
    label_to_id = payload.get("label_to_id")
    if not isinstance(database, dict):
        return ["TriDet payload must contain a database object"]
    if not isinstance(taxonomy, list) or not taxonomy:
        errors.append("TriDet taxonomy must be a non-empty array")
        taxonomy = []
    expected_map = {str(label): index for index, label in enumerate(taxonomy)}
    if label_to_id != expected_map:
        errors.append("TriDet label_to_id must exactly match taxonomy order")
    valid_subsets = {"training", "validation", "testing", "unassigned"}
    for stream_id, record in database.items():
        prefix = f"database.{stream_id}"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be an object")
            continue
        try:
            duration = float(record.get("duration", 0.0))
            fps = float(record.get("fps", 0.0))
        except (TypeError, ValueError):
            duration = fps = 0.0
        if duration <= 0:
            errors.append(f"{prefix}.duration must be positive")
        if fps <= 0:
            errors.append(f"{prefix}.fps must be positive")
        subset = str(record.get("subset", ""))
        if subset not in valid_subsets:
            errors.append(f"{prefix}.subset is invalid: {subset}")
        if require_training_subset and subset == "unassigned":
            errors.append(f"{prefix}.subset is unassigned; freeze the dataset before training")
        annotations = record.get("annotations")
        if not isinstance(annotations, list):
            errors.append(f"{prefix}.annotations must be an array")
            continue
        for index, annotation in enumerate(annotations):
            item = f"{prefix}.annotations[{index}]"
            if not isinstance(annotation, dict):
                errors.append(f"{item} must be an object")
                continue
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                errors.append(f"{item}.segment must be [start_sec,end_sec]")
            else:
                try:
                    start, end = map(float, segment)
                    if not 0.0 <= start < end <= duration + 1e-3:
                        errors.append(f"{item}.segment is outside stream duration")
                except (TypeError, ValueError):
                    errors.append(f"{item}.segment values must be numeric")
            label = str(annotation.get("label", ""))
            if label not in expected_map:
                errors.append(f"{item}.label is not in taxonomy: {label}")
            if annotation.get("label_id") != expected_map.get(label):
                errors.append(f"{item}.label_id does not match taxonomy")
            frame_segment = annotation.get("segment_frames")
            if not isinstance(frame_segment, list) or len(frame_segment) != 2:
                errors.append(f"{item}.segment_frames must be [start_frame,end_frame]")
    return errors


def validate_feature_array(array: np.ndarray, expected_steps: int | None = None) -> list[str]:
    errors: list[str] = []
    if array.ndim != 2:
        return [f"TriDet feature array must have shape [T,C], got {array.shape}"]
    if array.shape[0] < 1 or array.shape[1] < 2:
        errors.append(f"TriDet feature array must have positive T and C>=2, got {array.shape}")
    if expected_steps is not None and array.shape[0] != expected_steps:
        errors.append(f"TriDet feature T={array.shape[0]} does not match timestamps={expected_steps}")
    if not np.isfinite(array.astype(np.float32)).all():
        errors.append("TriDet feature array contains non-finite values")
    return errors
