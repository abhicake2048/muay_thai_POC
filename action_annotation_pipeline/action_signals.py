"""Pose-timeline construction and high-recall Muay Thai actionness signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from common import PipelineError, canonical_fighter_for_track, merge_intervals


EFFECTORS: dict[str, dict[str, Any]] = {
    "left_hand": {"side": "left", "limb": "hand", "proximal": 5, "middle": 7, "distal": 9},
    "right_hand": {"side": "right", "limb": "hand", "proximal": 6, "middle": 8, "distal": 10},
    "left_leg": {"side": "left", "limb": "leg", "proximal": 11, "middle": 13, "distal": 15},
    "right_leg": {"side": "right", "limb": "leg", "proximal": 12, "middle": 14, "distal": 16},
}


def fighter_map_coverage(
    fighter_map: dict[str, Any],
    times: np.ndarray,
    raw_tracks: np.ndarray,
    expected_fighters: int,
    tolerance: float = 0.20,
    canonical_fighters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure nominal map span and actual pose-to-fighter assignment coverage."""
    pose_start = float(np.min(times)) if len(times) else 0.0
    pose_end = float(np.max(times)) if len(times) else 0.0
    intervals_by_fighter: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for fighter in fighter_map.get("fighters", []):
        fighter_id = int(fighter.get("fighter_id", -1))
        for segment in fighter.get("track_segments", []):
            intervals_by_fighter[fighter_id].append(
                (float(segment.get("start_sec", 0.0)), float(segment.get("end_sec", 0.0)))
            )

    fighters: list[dict[str, Any]] = []
    for fighter_id in range(expected_fighters):
        merged = merge_intervals(intervals_by_fighter.get(fighter_id, []))
        start = merged[0][0] if merged else None
        end = merged[-1][1] if merged else None
        gaps = [
            {"start_sec": left[1], "end_sec": right[0], "duration_sec": right[0] - left[1]}
            for left, right in zip(merged[:-1], merged[1:])
            if right[0] - left[1] > tolerance
        ]
        fighters.append({
            "fighter_id": fighter_id,
            "coverage_start_sec": start,
            "coverage_end_sec": end,
            "gaps": gaps,
            "covers_observed_pose_span": False,
        })

    unmapped = 0
    mapped_by_fighter: dict[int, int] = defaultdict(int)
    observed_times: dict[int, list[float]] = defaultdict(list)
    for index, (timestamp, raw_track) in enumerate(zip(times, raw_tracks)):
        fighter = None
        if canonical_fighters is not None and index < len(canonical_fighters):
            canonical_value = int(canonical_fighters[index])
            fighter = canonical_value if canonical_value >= 0 else None
        if fighter is None:
            fighter = canonical_fighter_for_track(fighter_map, int(raw_track), float(timestamp))
        if fighter is None:
            unmapped += 1
        else:
            mapped_by_fighter[int(fighter)] += 1
            observed_times[int(fighter)].append(float(timestamp))
    for row in fighters:
        observed = observed_times.get(int(row["fighter_id"]), [])
        row["observed_start_sec"] = min(observed) if observed else None
        row["observed_end_sec"] = max(observed) if observed else None
        row["mapped_pose_instances"] = len(observed)
        row["covers_observed_pose_span"] = bool(
            observed
            and row["coverage_start_sec"] is not None
            and row["coverage_end_sec"] is not None
            and float(row["coverage_start_sec"]) <= float(row["observed_start_sec"]) + tolerance
            and float(row["coverage_end_sec"]) >= float(row["observed_end_sec"]) - tolerance
        )
    mapped = max(0, int(len(times)) - unmapped)
    mapped_ratio = mapped / max(1, int(len(times)))
    all_intervals = [interval for intervals in intervals_by_fighter.values() for interval in intervals]
    derived_processed_start = min((start for start, _ in all_intervals), default=None)
    derived_processed_end = max((end for _, end in all_intervals), default=None)
    map_processed_start = fighter_map.get("processed_start_sec", derived_processed_start)
    map_processed_end = fighter_map.get("processed_end_sec", derived_processed_end)
    processed_covers_pose_span = bool(
        map_processed_start is not None
        and map_processed_end is not None
        and float(map_processed_start) <= pose_start + tolerance
        and float(map_processed_end) >= pose_end - tolerance
    )
    complete = bool(
        mapped_ratio >= 0.95
        and processed_covers_pose_span
        and all(item["covers_observed_pose_span"] and item["mapped_pose_instances"] > 0 for item in fighters)
    )
    return {
        "pose_start_sec": pose_start,
        "pose_end_sec": pose_end,
        "pose_instances": int(len(times)),
        "mapped_pose_instances": mapped,
        "unmapped_pose_instances": int(unmapped),
        "mapped_pose_ratio": round(mapped_ratio, 6),
        "unmapped_pose_ratio": round(1.0 - mapped_ratio, 6),
        "mapped_instances_by_fighter": dict(sorted(mapped_by_fighter.items())),
        "fighters": fighters,
        "complete": complete,
        "processed_covers_pose_span": processed_covers_pose_span,
        "map_review_status": fighter_map.get("review_status", "legacy_unknown"),
        "map_processed_start_sec": map_processed_start,
        "map_processed_end_sec": map_processed_end,
    }


def require_fighter_map_coverage(report: dict[str, Any]) -> None:
    if report.get("complete"):
        return
    details = ", ".join(
        f"fighter {row['fighter_id']}: map={row['coverage_start_sec']}..{row['coverage_end_sec']}, "
        f"observed={row.get('observed_start_sec')}..{row.get('observed_end_sec')}"
        for row in report.get("fighters", [])
    )
    raise PipelineError(
        "fighter_map.json does not cover the pose timeline "
        f"{report.get('pose_start_sec', 0):.3f}..{report.get('pose_end_sec', 0):.3f}s "
        f"({details}); mapped pose ratio={float(report.get('mapped_pose_ratio', 0.0)):.3f}, "
        f"unmapped instances={int(report.get('unmapped_pose_instances', 0))}. "
        "Re-run tracking and RTMPose so canonical fighter stitching is applied, "
        "or review and extend a manually verified map."
    )


def _interpolate_short_gaps(
    points: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    max_gap: int,
) -> int:
    present = np.flatnonzero(np.max(scores, axis=1) > 0)
    filled = 0
    for left, right in zip(present[:-1], present[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > max_gap:
            continue
        for frame in range(left + 1, right):
            alpha = (frame - left) / (right - left)
            points[frame] = (1.0 - alpha) * points[left] + alpha * points[right]
            scores[frame] = np.minimum(scores[left], scores[right]) * 0.75
            boxes[frame] = (1.0 - alpha) * boxes[left] + alpha * boxes[right]
            filled += 1
    return filled


def _smooth(points: np.ndarray, scores: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1:
        return points.copy(), scores.copy()
    half = window // 2
    output = points.copy()
    output_scores = scores.copy()
    present = np.max(scores, axis=1) > 0
    for frame in np.flatnonzero(present):
        start, end = max(0, frame - half), min(len(points), frame + half + 1)
        weights = scores[start:end, :, None]
        valid = np.isfinite(points[start:end]).all(axis=-1, keepdims=True)
        weights = np.where(valid, weights, 0.0)
        denominator = np.sum(weights, axis=0)
        numerator = np.nansum(points[start:end] * weights, axis=0)
        output[frame] = np.divide(
            numerator,
            denominator,
            out=points[frame].copy(),
            where=denominator > 1e-6,
        )
        output_scores[frame] = np.max(scores[start:end], axis=0)
    return output, output_scores


def build_fighter_timelines(
    data: Any,
    fighter_map: dict[str, Any],
    expected_fighters: int,
    fps: float,
    max_gap_frames: int = 6,
    smoothing_window: int = 5,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Convert sparse raw-track pose rows into continuous canonical fighters."""
    frames = np.asarray(data["frame_indices"], dtype=np.int32)
    times = np.asarray(data["timestamps"], dtype=np.float64)
    raw_tracks = np.asarray(data["track_ids"], dtype=np.int32)
    data_keys = set(data.files) if hasattr(data, "files") else set(data.keys())
    canonical_fighters = np.asarray(data["fighter_ids"], dtype=np.int16) if "fighter_ids" in data_keys else None
    points = np.asarray(data["keypoints"], dtype=np.float32)
    scores = np.asarray(data["keypoint_scores"], dtype=np.float32)
    boxes = np.asarray(data["bboxes_xyxy"], dtype=np.float32)
    bbox_scores = np.asarray(data["bbox_scores"], dtype=np.float32)
    frame_count = int(np.max(frames)) + 1 if len(frames) else 0
    if frame_count < 2:
        raise PipelineError("At least two pose frames are required for action proposals")

    timelines: dict[int, dict[str, np.ndarray]] = {}
    for fighter in range(expected_fighters):
        timelines[fighter] = {
            "points": np.full((frame_count, 17, 2), np.nan, dtype=np.float32),
            "scores": np.zeros((frame_count, 17), dtype=np.float32),
            "boxes": np.full((frame_count, 4), np.nan, dtype=np.float32),
            "bbox_scores": np.zeros(frame_count, dtype=np.float32),
            "raw_tracks": np.full(frame_count, -1, dtype=np.int32),
            "times": np.arange(frame_count, dtype=np.float64) / max(fps, 1e-6),
        }

    discarded = 0
    for index, frame in enumerate(frames):
        fighter = int(canonical_fighters[index]) if canonical_fighters is not None else canonical_fighter_for_track(
            fighter_map, int(raw_tracks[index]), float(times[index])
        )
        if fighter is None or fighter not in timelines:
            discarded += 1
            continue
        timeline = timelines[fighter]
        if bbox_scores[index] < timeline["bbox_scores"][frame]:
            continue
        timeline["points"][frame] = points[index]
        timeline["scores"][frame] = scores[index]
        timeline["boxes"][frame] = boxes[index]
        timeline["bbox_scores"][frame] = bbox_scores[index]
        timeline["raw_tracks"][frame] = raw_tracks[index]

    interpolated: dict[int, int] = {}
    for fighter, timeline in timelines.items():
        interpolated[fighter] = _interpolate_short_gaps(
            timeline["points"], timeline["scores"], timeline["boxes"], max_gap_frames
        )
        timeline["points"], timeline["scores"] = _smooth(
            timeline["points"], timeline["scores"], smoothing_window
        )
        normalized = np.full_like(timeline["points"], np.nan)
        scales = np.full(frame_count, np.nan, dtype=np.float32)
        centers = np.full((frame_count, 2), np.nan, dtype=np.float32)
        for frame in range(frame_count):
            frame_points = timeline["points"][frame]
            frame_scores = timeline["scores"][frame]
            if np.max(frame_scores) <= 0:
                continue
            hips_valid = np.all(frame_scores[[11, 12]] >= 0.10)
            shoulders_valid = np.all(frame_scores[[5, 6]] >= 0.10)
            if hips_valid:
                center = np.mean(frame_points[[11, 12]], axis=0)
            else:
                box = timeline["boxes"][frame]
                center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
            scale = 0.0
            if hips_valid and shoulders_valid:
                scale = float(np.linalg.norm(np.mean(frame_points[[5, 6]], axis=0) - center))
            if not np.isfinite(scale) or scale < 8.0:
                box = timeline["boxes"][frame]
                scale = max(8.0, float(np.hypot(box[2] - box[0], box[3] - box[1])) * 0.25)
            centers[frame], scales[frame] = center, scale
            normalized[frame] = (frame_points - center) / scale
        timeline["normalized"] = normalized
        timeline["centers"] = centers
        timeline["scales"] = scales

    return timelines, {
        "frame_count": frame_count,
        "fps": fps,
        "discarded_unmapped_instances": discarded,
        "interpolated_frames_by_fighter": interpolated,
    }


def _point_speed(points: np.ndarray, scores: np.ndarray, joint: int, fps: float) -> np.ndarray:
    output = np.zeros(len(points), dtype=np.float32)
    valid = (
        (scores[:-1, joint] >= 0.10)
        & (scores[1:, joint] >= 0.10)
        & np.isfinite(points[:-1, joint]).all(axis=-1)
        & np.isfinite(points[1:, joint]).all(axis=-1)
    )
    delta = np.linalg.norm(points[1:, joint] - points[:-1, joint], axis=-1) * fps
    output[1:] = np.where(valid, delta, 0.0)
    return output


def _joint_angle(points: np.ndarray, scores: np.ndarray, proximal: int, middle: int, distal: int) -> np.ndarray:
    first = points[:, proximal] - points[:, middle]
    second = points[:, distal] - points[:, middle]
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cosine = np.divide(np.sum(first * second, axis=-1), denominator, out=np.zeros(len(points), dtype=np.float32), where=denominator > 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    valid = np.all(scores[:, [proximal, middle, distal]] >= 0.10, axis=1)
    return np.where(valid, angle, np.nan).astype(np.float32)


def _angle_speed(angle: np.ndarray, fps: float) -> np.ndarray:
    output = np.zeros(len(angle), dtype=np.float32)
    valid = np.isfinite(angle[:-1]) & np.isfinite(angle[1:])
    output[1:] = np.where(valid, np.abs(angle[1:] - angle[:-1]) * fps / np.pi, 0.0)
    return output


def _target_convergence(
    actor: dict[str, np.ndarray],
    opponent: dict[str, np.ndarray] | None,
    distal: int,
    fps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(actor["points"])
    by_zone = {zone: np.zeros(count, dtype=np.float32) for zone in ("head", "body", "general_leg")}
    if opponent is None:
        return np.zeros(count, dtype=np.float32), by_zone
    regions = {"head": (0, 1, 2, 3, 4), "body": (5, 6, 11, 12), "general_leg": (13, 14, 15, 16)}
    actor_point = actor["points"][:, distal]
    actor_valid = actor["scores"][:, distal] >= 0.10
    scales = actor["scales"]
    for zone, joints in regions.items():
        distances = np.full(count, np.nan, dtype=np.float32)
        for frame in range(count):
            valid_joints = [joint for joint in joints if opponent["scores"][frame, joint] >= 0.10]
            if not actor_valid[frame] or not valid_joints or not np.isfinite(scales[frame]):
                continue
            distance = np.min(np.linalg.norm(opponent["points"][frame, valid_joints] - actor_point[frame], axis=-1))
            distances[frame] = distance / max(float(scales[frame]), 1e-6)
        valid = np.isfinite(distances[:-1]) & np.isfinite(distances[1:])
        convergence = np.zeros(count, dtype=np.float32)
        convergence[1:] = np.where(valid, np.maximum(0.0, distances[:-1] - distances[1:]) * fps, 0.0)
        by_zone[zone] = convergence
    combined = np.maximum.reduce(list(by_zone.values()))
    return combined, by_zone


def robust_actionness(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values), {"median": 0.0, "mad": 0.0, "scale": 1.0}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale < 1e-4:
        scale = max(float(np.percentile(finite, 90)) - median, 0.05)
    score = np.maximum(0.0, (values - median) / scale)
    return score.astype(np.float32), {"median": median, "mad": mad, "scale": scale}


def compute_action_signals(
    timeline: dict[str, np.ndarray],
    opponent: dict[str, np.ndarray] | None,
    fps: float,
) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    for name, spec in EFFECTORS.items():
        proximal, middle, distal = spec["proximal"], spec["middle"], spec["distal"]
        distal_speed = _point_speed(timeline["normalized"], timeline["scores"], distal, fps)
        middle_speed = _point_speed(timeline["normalized"], timeline["scores"], middle, fps)
        acceleration = np.zeros_like(distal_speed)
        acceleration[1:] = np.abs(np.diff(distal_speed)) * fps
        angle = _joint_angle(timeline["normalized"], timeline["scores"], proximal, middle, distal)
        extension_speed = _angle_speed(angle, fps)
        convergence, zone_convergence = _target_convergence(timeline, opponent, distal, fps)
        confidence = np.sqrt(np.maximum(0.0, timeline["scores"][:, distal] * timeline["scores"][:, middle]))
        raw = (
            0.52 * np.clip(distal_speed, 0.0, 8.0)
            + 0.16 * np.clip(middle_speed, 0.0, 8.0)
            + 0.08 * np.clip(acceleration / 10.0, 0.0, 4.0)
            + 0.12 * np.clip(extension_speed, 0.0, 5.0)
            + 0.12 * np.clip(convergence, 0.0, 8.0)
        ) * np.maximum(0.25, confidence)
        actionness, statistics = robust_actionness(raw)
        signals[name] = {
            **spec,
            "raw": raw,
            "actionness": actionness,
            "distal_speed": distal_speed,
            "middle_speed": middle_speed,
            "acceleration": acceleration,
            "extension_speed": extension_speed,
            "convergence": convergence,
            "zone_convergence": zone_convergence,
            "confidence": confidence,
            "statistics": statistics,
        }
    return signals


def local_peaks(values: np.ndarray, threshold: float, separation_frames: int) -> list[int]:
    candidates = [
        index for index in range(1, len(values) - 1)
        if values[index] >= threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]
    ]
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda index: float(values[index]), reverse=True):
        if all(abs(candidate - other) >= separation_frames for other in selected):
            selected.append(candidate)
    return sorted(selected)


def hysteresis_bounds(values: np.ndarray, peak: int, low: float, below_frames: int = 2) -> tuple[int, int]:
    left, below = peak, 0
    while left > 0:
        left -= 1
        below = below + 1 if values[left] < low else 0
        if below >= below_frames:
            left += below_frames
            break
    right, below = peak, 0
    while right < len(values) - 1:
        right += 1
        below = below + 1 if values[right] < low else 0
        if below >= below_frames:
            right -= below_frames
            break
    return max(0, left), min(len(values) - 1, right)


def action_components(
    values: np.ndarray,
    high: float,
    low: float,
    separation_frames: int,
    *,
    below_frames: int = 2,
    split_valley: float = 1.10,
) -> list[tuple[int, int, int]]:
    """Return one ``(peak, onset, offset)`` tuple per physical action.

    The previous implementation expanded every local peak independently.  If
    several peaks belonged to the same hysteresis component it consequently
    emitted the exact same interval two, three, or more times.  Components are
    now formed first.  A component is split only when two sufficiently
    separated peaks also have a sustained valley between them, which preserves
    real combinations without multiplying noisy peaks from one strike.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3:
        return []
    peaks = local_peaks(values, high, separation_frames)
    if not peaks:
        return []

    bounds = sorted(hysteresis_bounds(values, peak, low, below_frames) for peak in peaks)
    merged: list[list[int]] = []
    for onset, offset in bounds:
        if not merged or onset > merged[-1][1] + 1:
            merged.append([onset, offset])
        else:
            merged[-1][1] = max(merged[-1][1], offset)

    output: list[tuple[int, int, int]] = []
    for component_onset, component_offset in merged:
        candidates = [peak for peak in peaks if component_onset <= peak <= component_offset]
        if not candidates:
            candidates = [component_onset + int(np.argmax(values[component_onset:component_offset + 1]))]
        candidates.sort()

        selected = [candidates[0]]
        for candidate in candidates[1:]:
            previous = selected[-1]
            valley = values[previous + 1:candidate]
            sustained_valley = (
                candidate - previous >= separation_frames
                and len(valley) >= 2
                and any(
                    valley[index] <= split_valley and valley[index + 1] <= split_valley
                    for index in range(len(valley) - 1)
                )
            )
            if sustained_valley:
                selected.append(candidate)
            elif values[candidate] > values[previous]:
                selected[-1] = candidate

        split_frames = [
            left + int(np.argmin(values[left:right + 1]))
            for left, right in zip(selected[:-1], selected[1:])
        ]
        for index, peak in enumerate(selected):
            onset = component_onset if index == 0 else split_frames[index - 1] + 1
            offset = component_offset if index == len(selected) - 1 else split_frames[index]
            output.append((peak, onset, offset))
    return output


def family_candidates(signal: dict[str, Any], peak: int) -> tuple[list[dict[str, Any]], str, str]:
    distal = float(signal["distal_speed"][peak])
    middle = float(signal["middle_speed"][peak])
    if signal["limb"] == "hand":
        elbow_like = middle >= 0.82 * max(distal, 1e-6)
        labels = [("elbow", 0.58), ("punch", 0.42)] if elbow_like else [("punch", 0.72), ("elbow", 0.28)]
        limb = "elbow" if elbow_like else "hand"
        hint = "punch_or_elbow"
    else:
        knee_like = middle >= 0.82 * max(distal, 1e-6)
        labels = [("knee", 0.58), ("kick", 0.42)] if knee_like else [("kick", 0.72), ("knee", 0.28)]
        limb = "knee" if knee_like else "leg"
        hint = "kick_or_knee"
    return ([{"label": label, "confidence": confidence} for label, confidence in labels], limb, hint)
