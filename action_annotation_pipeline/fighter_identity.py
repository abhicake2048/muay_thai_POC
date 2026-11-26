"""Stitch fragmented detector tracks into two canonical fighter identities."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import numpy as np


def _center(box: list[float]) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def _area(box: list[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _quality(detection: dict[str, Any], width: int, height: int) -> float:
    box = list(detection["bbox_xyxy"])
    center = _center(box)
    area_ratio = _area(box) / max(1.0, float(width * height))
    centrality = 1.0 - min(
        1.0,
        math.hypot(center[0] / max(width, 1) - 0.5, center[1] / max(height, 1) - 0.52) / 0.72,
    )
    return (
        0.50 * float(detection.get("confidence", 0.0))
        + 0.35 * min(1.0, area_ratio / 0.12)
        + 0.15 * centrality
    )


def _match_score(
    fighter: int,
    detection: dict[str, Any],
    state: dict[str, Any] | None,
    raw_owner: dict[int, tuple[int, int]],
    frame_index: int,
    width: int,
    height: int,
    owner_memory_frames: int,
) -> float:
    score = 0.75 * _quality(detection, width, height)
    track_id = int(detection["track_id"])
    previous_owner = raw_owner.get(track_id)
    if previous_owner and frame_index - previous_owner[1] <= owner_memory_frames:
        score += 3.5 if previous_owner[0] == fighter else -4.0
    if state is None:
        return score

    gap = max(1, frame_index - int(state["last_frame"]))
    box = list(detection["bbox_xyxy"])
    center = _center(box)
    predicted = state["center"] + state["velocity"] * min(gap, 8)
    diagonal = max(1.0, float(np.hypot(width, height)))
    normalized_distance = float(np.linalg.norm(center - predicted)) / diagonal
    geometry_weight = max(0.35, 1.0 - gap / 90.0)
    score += geometry_weight * 3.0 * max(0.0, 1.0 - normalized_distance / 0.30)
    score += geometry_weight * 1.8 * _iou(list(state["box"]), box)
    previous_area, current_area = max(_area(list(state["box"])), 1.0), max(_area(box), 1.0)
    score += geometry_weight * 0.6 * math.exp(-abs(math.log(current_area / previous_area)))
    if track_id == int(state["track_id"]):
        score += 4.0
    return score


def _best_assignment(
    detections: list[dict[str, Any]],
    states: dict[int, dict[str, Any]],
    raw_owner: dict[int, tuple[int, int]],
    frame_index: int,
    expected_fighters: int,
    width: int,
    height: int,
    owner_memory_frames: int,
) -> dict[int, int]:
    if not detections:
        return {}
    ranked_indices = sorted(
        range(len(detections)),
        key=lambda index: _quality(detections[index], width, height),
        reverse=True,
    )[: max(expected_fighters + 2, expected_fighters)]
    if not states and len(ranked_indices) >= expected_fighters:
        selected = ranked_indices[:expected_fighters]
        selected.sort(key=lambda index: float(_center(list(detections[index]["bbox_xyxy"]))[0]))
        return {fighter: detection_index for fighter, detection_index in enumerate(selected)}

    choices: list[int | None] = [*ranked_indices, None]
    best_score = -float("inf")
    best: dict[int, int] = {}
    for assignment in itertools.product(choices, repeat=expected_fighters):
        used = [index for index in assignment if index is not None]
        if len(used) != len(set(used)):
            continue
        score = 0.0
        mapped: dict[int, int] = {}
        for fighter, detection_index in enumerate(assignment):
            if detection_index is None:
                score -= 0.75 if fighter in states else 0.15
                continue
            value = _match_score(
                fighter,
                detections[detection_index],
                states.get(fighter),
                raw_owner,
                frame_index,
                width,
                height,
                owner_memory_frames,
            )
            score += value
            mapped[fighter] = detection_index
        if score > best_score:
            best_score, best = score, mapped
    return best


def _segments(
    assignments: dict[int, list[tuple[int, float, int]]],
    fps: float,
    expected_fighters: int,
    maximum_gap_frames: int = 6,
) -> list[dict[str, Any]]:
    fighters: list[dict[str, Any]] = []
    half_frame = 0.5 / max(fps, 1e-6)
    for fighter in range(expected_fighters):
        rows = assignments.get(fighter, [])
        segments: list[dict[str, Any]] = []
        for frame_index, timestamp, track_id in rows:
            if (
                segments
                and int(segments[-1]["track_id"]) == track_id
                and frame_index - int(segments[-1]["end_frame"]) <= maximum_gap_frames
            ):
                segments[-1]["end_frame"] = frame_index
                segments[-1]["end_sec"] = timestamp + half_frame
            else:
                segments.append({
                    "track_id": track_id,
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "start_sec": max(0.0, timestamp - half_frame),
                    "end_sec": timestamp + half_frame,
                })
        fighters.append({
            "fighter_id": fighter,
            "name": f"fighter_{fighter}",
            "track_segments": segments,
        })
    return fighters


def stitch_fighter_tracks(
    records: list[dict[str, Any]],
    width: int,
    height: int,
    fps: float,
    expected_fighters: int = 2,
    owner_memory_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate selected detections with canonical IDs and return track segments.

    Raw tracker IDs remain in the records. Canonical IDs are assigned using raw
    ID persistence plus predicted spatial continuity, allowing a fighter to be
    represented by many raw track fragments without treating every fragment as
    a different person.
    """
    states: dict[int, dict[str, Any]] = {}
    raw_owner: dict[int, tuple[int, int]] = {}
    assignments: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    owner_memory_frames = max(1, int(round(owner_memory_seconds * fps)))
    assigned_detections = 0
    total_detections = 0

    for row in records:
        frame_index = int(row["frame_index"])
        timestamp = float(row["timestamp"])
        detections = list(row.get("detections", []))
        total_detections += len(detections)
        for detection in detections:
            detection.pop("fighter_id", None)
            detection.pop("identity_score", None)
        chosen = _best_assignment(
            detections,
            states,
            raw_owner,
            frame_index,
            expected_fighters,
            width,
            height,
            owner_memory_frames,
        )
        for fighter, detection_index in chosen.items():
            detection = detections[detection_index]
            box = list(detection["bbox_xyxy"])
            center = _center(box)
            previous = states.get(fighter)
            velocity = np.zeros(2, dtype=np.float32)
            if previous is not None:
                gap = max(1, frame_index - int(previous["last_frame"]))
                measured = (center - previous["center"]) / gap
                velocity = 0.65 * previous["velocity"] + 0.35 * measured
            track_id = int(detection["track_id"])
            identity_score = _match_score(
                fighter,
                detection,
                previous,
                raw_owner,
                frame_index,
                width,
                height,
                owner_memory_frames,
            )
            detection["fighter_id"] = fighter
            detection["identity_score"] = round(float(identity_score), 6)
            states[fighter] = {
                "box": box,
                "center": center,
                "velocity": velocity,
                "last_frame": frame_index,
                "track_id": track_id,
            }
            raw_owner[track_id] = (fighter, frame_index)
            assignments[fighter].append((frame_index, timestamp, track_id))
            assigned_detections += 1

    fighters = _segments(assignments, fps, expected_fighters)
    counts = {str(fighter): len(assignments.get(fighter, [])) for fighter in range(expected_fighters)}
    segment_counts = {str(row["fighter_id"]): len(row["track_segments"]) for row in fighters}
    raw_ids = {
        str(fighter): sorted({track_id for _, _, track_id in assignments.get(fighter, [])})
        for fighter in range(expected_fighters)
    }
    report = {
        "method": "raw_id_plus_spatial_continuity_v1",
        "frames": len(records),
        "total_detections": total_detections,
        "assigned_fighter_detections": assigned_detections,
        "unassigned_detections": total_detections - assigned_detections,
        "assigned_frames_by_fighter": counts,
        "segments_by_fighter": segment_counts,
        "raw_track_ids_by_fighter": raw_ids,
        "manual_review_required": True,
    }
    return fighters, report
