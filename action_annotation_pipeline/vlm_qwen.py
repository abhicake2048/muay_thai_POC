"""Local Qwen3-VL discovery and labeling for actor-centric fight windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    PipelineError,
    canonical_fighter_for_track,
    load_fighter_map,
    load_manifest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from schema import DEFAULT_LIMB, DEFAULT_SURFACE, effector_id, technique_family


PROMPT = """You are proposing Muay Thai temporal annotations for human review.
The ordered composite frames cover {window_start:.3f} to {window_end:.3f} seconds at {fps:.1f} FPS.
Each frame shows the full scene above, acting fighter {actor_id} below-left, and the opponent below-right.
Find ZERO OR MORE committed actions by fighter {actor_id}. Include combinations as separate events.
Include discrete attacks, defenses, clinch changes, and balance events represented by the allowed taxonomy;
do not label ordinary guard holding, circling, feints, or stance adjustment as a committed action.
Use the fighter's anatomical left/right, never screen-left/screen-right. Use unknown when hidden.
Allowed techniques: {techniques}.
Return exactly one JSON object and no prose:
{{"contains_action":true,"window_summary":"Actor throws one visible lead-hand punch.","events":[{{"start_offset_sec":0.2,"peak_offset_sec":0.45,"end_offset_sec":0.75,
"technique":"jab","technique_candidates":[{{"technique":"jab","confidence":0.7}}],
"anatomical_side":"left","effector_limb":"hand","stance_at_start":"orthodox",
"stance_role":"lead","target_zone":"head","outcome":"attempted","contact_surface":"fist",
"visibility":"visible","confidence":0.7,"uncertainty_flags":[],
"classification_reason":"Lead hand extends toward the opponent's head and retracts.",
"boundary_reason":"Starts at extension onset, peaks at maximum reach, ends after retraction.",
"visual_evidence":["lead wrist advances toward opponent head","elbow extends then flexes"]}}],
"no_action_confidence":0.0,"no_action_reason":""}}
Offsets are relative to the start of this window, in seconds. Boundaries must be ordered and inside the window.
If no committed action occurs, return contains_action=false, events=[], a confidence from 0 to 1,
and a concise no_action_reason describing the observable normal movement. Never omit the window assessment.
Reasons must cite short observable visual evidence, not hidden reasoning or speculation.
Do not invent contact or anatomical side. Predictions are proposals, not verified truth."""

TECHNIQUE_ALIASES = {
    "roundhouse": "roundhouse_kick",
    "round_kick": "roundhouse_kick",
    "low_kick": "roundhouse_kick",
    "body_kick": "roundhouse_kick",
    "head_kick": "roundhouse_kick",
    "front_kick": "teep",
    "push_kick": "teep",
    "straight_punch": "cross",
    "rear_straight": "cross",
    "spinning_back_fist": "spinning_backfist",
}


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(value, dict):
            continue
        if not isinstance(value.get("events", []), list):
            raise ValueError("model output events must be an array")
        return value
    detail = f" ({errors[-1]})" if errors else ""
    raise ValueError(f"model output contained no valid JSON object{detail}")


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_vlm_payload(
    payload: dict[str, Any],
    window_duration: float,
    allowed_techniques: set[str],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the semantic VLM contract, not merely JSON syntax."""
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be an array")
    contains_action = payload.get("contains_action")
    if not isinstance(contains_action, bool):
        raise ValueError("contains_action must be true or false")
    if contains_action != bool(events):
        raise ValueError("contains_action must be true exactly when events is non-empty")
    if not str(payload.get("window_summary", "")).strip():
        raise ValueError("window_summary must be a non-empty observable summary")
    no_action_confidence = _finite_number(payload.get("no_action_confidence", 0.0), "no_action_confidence")
    if not 0.0 <= no_action_confidence <= 1.0:
        raise ValueError("no_action_confidence must be between 0 and 1")
    if not events and not str(payload.get("no_action_reason", "")).strip():
        raise ValueError("no_action_reason is required when events is empty")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{index}] must be an object")
        start = _finite_number(event.get("start_offset_sec"), f"events[{index}].start_offset_sec")
        peak = _finite_number(event.get("peak_offset_sec"), f"events[{index}].peak_offset_sec")
        end = _finite_number(event.get("end_offset_sec"), f"events[{index}].end_offset_sec")
        if not (0.0 <= start <= peak <= end <= window_duration + 1e-6):
            raise ValueError(f"events[{index}] boundaries must satisfy 0 <= start <= peak <= end <= window duration")
        if end - start < 0.03:
            raise ValueError(f"events[{index}] is shorter than 0.03 seconds")
        confidence = _finite_number(event.get("confidence"), f"events[{index}].confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"events[{index}].confidence must be between 0 and 1")
        technique = _normalize_technique(event.get("technique"))
        if technique not in allowed_techniques:
            raise ValueError(f"events[{index}].technique is not in the allowed taxonomy: {technique}")
        candidates = event.get("technique_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"events[{index}].technique_candidates must be a non-empty array")
        for candidate_index, candidate in enumerate(candidates[:3]):
            if not isinstance(candidate, dict):
                raise ValueError(f"events[{index}].technique_candidates[{candidate_index}] must be an object")
            candidate_label = _normalize_technique(candidate.get("technique"))
            if candidate_label not in allowed_techniques:
                raise ValueError(f"events[{index}] has an invalid technique candidate: {candidate_label}")
            candidate_confidence = _finite_number(
                candidate.get("confidence"), f"events[{index}].technique_candidates[{candidate_index}].confidence"
            )
            if not 0.0 <= candidate_confidence <= 1.0:
                raise ValueError(f"events[{index}] technique candidate confidence must be between 0 and 1")
        if schema:
            categorical = {
                "anatomical_side": "anatomical_sides",
                "effector_limb": "effector_limbs",
                "stance_at_start": "stance_classes",
                "stance_role": "stance_roles",
                "target_zone": "target_zones",
                "outcome": "outcomes",
                "contact_surface": "contact_surfaces",
                "visibility": "visibility_levels",
            }
            for field, schema_field in categorical.items():
                if event.get(field) not in schema.get(schema_field, []):
                    raise ValueError(f"events[{index}].{field} is missing or outside the schema")
        if not isinstance(event.get("uncertainty_flags"), list):
            raise ValueError(f"events[{index}].uncertainty_flags must be an array")
        if not str(event.get("classification_reason", "")).strip():
            raise ValueError(f"events[{index}].classification_reason is required")
        if not str(event.get("boundary_reason", "")).strip():
            raise ValueError(f"events[{index}].boundary_reason is required")
        evidence = event.get("visual_evidence")
        if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
            raise ValueError(f"events[{index}].visual_evidence must contain at least one observation")
    return payload


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _crop(frame: np.ndarray, box: list[float] | None, context: float = 0.35) -> np.ndarray:
    if box is None:
        return np.zeros((max(1, frame.shape[0] // 2), max(1, frame.shape[1] // 2), 3), dtype=np.uint8)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    dx, dy = (x2 - x1) * context, (y2 - y1) * context
    left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
    crop = frame[top:bottom, left:right]
    return crop if crop.size else frame


def _boxes_for_time(
    tracks_by_frame: dict[int, list[dict[str, Any]]],
    fighter_map: dict[str, Any],
    timestamp: float,
    video_fps: float,
) -> dict[int, list[float]]:
    frame_index = int(round(timestamp * video_fps))
    result: dict[int, list[float]] = {}
    detections: list[dict[str, Any]] = []
    for radius in range(0, 7):
        candidate_frames = [frame_index] if radius == 0 else [frame_index - radius, frame_index + radius]
        for candidate_frame in candidate_frames:
            if tracks_by_frame.get(candidate_frame):
                detections = tracks_by_frame[candidate_frame]
                break
        if detections:
            break
    for detection in detections:
        fighter_value = detection.get("fighter_id")
        fighter = int(fighter_value) if fighter_value is not None else canonical_fighter_for_track(
            fighter_map, int(detection["track_id"]), timestamp
        )
        if fighter is not None:
            result[int(fighter)] = list(detection["bbox_xyxy"])
    return result


def _composite_frames(
    video: Path,
    start: float,
    end: float,
    sample_fps: float,
    actor_id: int,
    video_fps: float,
    tracks_by_frame: dict[int, list[dict[str, Any]]],
    fighter_map: dict[str, Any],
    maximum: int = 24,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video: {video}")
    count = min(maximum, max(1, int(np.ceil((end - start) * sample_fps))))
    # Preserve the declared temporal cadence. Linspace including both window
    # endpoints would turn a requested 8 FPS clip into 7.67 FPS over 3 s and
    # shift the model's timestamp grounding.
    timestamps = np.minimum(end, start + np.arange(count, dtype=np.float64) / sample_fps)
    frames: list[np.ndarray] = []
    requested_frames = len(timestamps)
    decoded_frames = 0
    actor_box_frames = 0
    opponent_box_frames = 0
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            decoded_frames += 1
            boxes = _boxes_for_time(tracks_by_frame, fighter_map, float(timestamp), video_fps)
            if actor_id in boxes:
                actor_box_frames += 1
            scene = frame.copy()
            for fighter, box in boxes.items():
                color = (60, 210, 150) if fighter == actor_id else (60, 150, 250)
                x1, y1, x2, y2 = [round(value) for value in box]
                cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
                cv2.putText(scene, f"fighter {fighter}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(scene, f"t={float(timestamp):.3f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            opponent_id = 1 - actor_id if actor_id in (0, 1) else -1
            if opponent_id in boxes:
                opponent_box_frames += 1
            composite = np.zeros((448, 448, 3), dtype=np.uint8)
            composite[:252] = _letterbox(scene, 448, 252)
            composite[252:] = np.concatenate([
                # A missing actor remains a black crop at its original
                # timestamp. Dropping the whole frame would compress time and
                # make model offsets disagree with proxy frames. The window is
                # still rejected below when crop coverage is insufficient.
                _letterbox(_crop(frame, boxes.get(actor_id)), 224, 196),
                _letterbox(_crop(frame, boxes.get(opponent_id)), 224, 196),
            ], axis=1)
            cv2.putText(composite, f"ACTOR {actor_id}", (8, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 210, 150), 2)
            cv2.putText(composite, "OPPONENT", (232, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 150, 250), 2)
            frames.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    denominator = max(decoded_frames, 1)
    return frames, {
        "requested_frames": requested_frames,
        "decoded_frames": decoded_frames,
        "composite_frames": len(frames),
        "actor_box_frames": actor_box_frames,
        "opponent_box_frames": opponent_box_frames,
        "actor_crop_coverage": round(actor_box_frames / denominator, 6),
        "opponent_crop_coverage": round(opponent_box_frames / denominator, 6),
    }


def build_scan_tasks(
    rows: list[dict[str, Any]],
    duration: float,
    expected_fighters: int,
    window_seconds: float,
    hop_seconds: float,
    scan_mode: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    covered: dict[int, list[tuple[float, float]]] = {fighter: [] for fighter in range(expected_fighters)}
    if scan_mode in {"candidates", "hybrid"}:
        candidate_tasks: dict[int, list[dict[str, Any]]] = {fighter: [] for fighter in range(expected_fighters)}
        for row in sorted(rows, key=lambda item: (int(item.get("actor_id", -1)), float(item.get("peak_sec", 0.0)))):
            actor = int(row.get("actor_id", -1))
            if actor not in covered:
                continue
            peak = float(row.get("peak_sec", 0.0))
            margin = min(0.40, window_seconds * 0.15)
            existing = next(
                (
                    task for task in candidate_tasks[actor]
                    if float(task["start_sec"]) + margin <= peak <= float(task["end_sec"]) - margin
                ),
                None,
            )
            if existing is not None:
                existing["source_event_ids"].append(str(row.get("event_id", "")))
                if float(row.get("proposal_confidence", row.get("actionness_score", 0.0))) > float(
                    existing["source"].get("proposal_confidence", existing["source"].get("actionness_score", 0.0))
                ):
                    existing["source"] = row
                continue
            start = max(0.0, min(duration - min(window_seconds, duration), peak - window_seconds / 2))
            end = min(duration, start + window_seconds)
            covered[actor].append((start, end))
            task = {
                "task_id": "", "kind": "candidate", "actor_id": actor,
                "start_sec": start, "end_sec": end, "source": row,
                "source_event_ids": [str(row.get("event_id", ""))],
            }
            candidate_tasks[actor].append(task)
            tasks.append(task)
        for index, task in enumerate(tasks, start=1):
            task["task_id"] = f"candidate_{index:06d}"
    if scan_mode in {"dense", "hybrid"}:
        for actor in range(expected_fighters):
            center = min(window_seconds / 2, duration / 2)
            while center < duration + 1e-6:
                start = max(0.0, min(duration - min(window_seconds, duration), center - window_seconds / 2))
                end = min(duration, start + window_seconds)
                midpoint = (start + end) / 2
                if not any(existing_start <= midpoint <= existing_end for existing_start, existing_end in covered[actor]):
                    tasks.append({
                        "task_id": f"dense_{len(tasks) + 1:06d}", "kind": "dense", "actor_id": actor,
                        "start_sec": start, "end_sec": end, "source": None,
                    })
                if end >= duration:
                    break
                center += hop_seconds
    return sorted(tasks, key=lambda item: (float(item["start_sec"]), int(item["actor_id"]), item["kind"]))


def _candidate_list(value: Any, technique: str, confidence: float, allowed: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                label, score = item, 0.0
            elif isinstance(item, dict):
                label = str(item.get("technique", item.get("label", "")))
                score = float(item.get("confidence", 0.0))
            else:
                continue
            label = _normalize_technique(label)
            if label in allowed:
                candidates.append({"technique": label, "confidence": round(max(0.0, min(1.0, score)), 6), "source": "qwen_actions"})
    if technique in allowed and not any(item["technique"] == technique for item in candidates):
        candidates.append({"technique": technique, "confidence": round(confidence, 6), "source": "qwen_actions"})
    candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return candidates[:3]


def _normalize_technique(value: Any) -> str:
    label = str(value or "unknown_action").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return TECHNIQUE_ALIASES.get(label, label)


def normalize_events(
    payload: dict[str, Any],
    task: dict[str, Any],
    manifest: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_techniques = set(schema.get("trainable_event_classes", schema.get("primary_action_classes", [])))
    allowed_sides = set(schema.get("anatomical_sides", []))
    allowed_limbs = set(schema.get("effector_limbs", []))
    window_start, window_end = float(task["start_sec"]), float(task["end_sec"])
    window_duration = window_end - window_start
    source = task.get("source")
    results: list[dict[str, Any]] = []
    for index, prediction in enumerate(payload.get("events", []), start=1):
        if not isinstance(prediction, dict):
            continue
        start_offset = max(0.0, min(window_duration, float(prediction.get("start_offset_sec", 0.0))))
        end_offset = max(start_offset, min(window_duration, float(prediction.get("end_offset_sec", window_duration))))
        peak_offset = max(start_offset, min(end_offset, float(prediction.get("peak_offset_sec", (start_offset + end_offset) / 2))))
        if end_offset - start_offset < 0.03:
            continue
        confidence = max(0.0, min(1.0, float(prediction.get("confidence", 0.0))))
        technique = _normalize_technique(prediction.get("technique", "unknown_action"))
        if technique not in allowed_techniques:
            technique = "unknown_action"
        candidates = _candidate_list(prediction.get("technique_candidates"), technique, confidence, allowed_techniques)
        side = str(prediction.get("anatomical_side", "unknown"))
        if side not in allowed_sides:
            side = "unknown"
        family = technique_family(schema, technique) or str(prediction.get("action_family", "unknown"))
        if family not in {*schema.get("action_families", {}), "balance"}:
            family = "unknown"
        limb = str(prediction.get("effector_limb", DEFAULT_LIMB.get(family, "unknown")))
        if limb not in allowed_limbs:
            limb = DEFAULT_LIMB.get(family, "unknown")
        base = dict(source) if isinstance(source, dict) else {}
        event_id = f"qwen_{task['task_id']}_{index:02d}"
        proxy_fps = float(manifest.get("proxy_video", {}).get("fps", base.get("proxy_fps", 30.0)))
        value = {
            **base,
            "event_id": event_id,
            "source_event_id": source.get("event_id") if isinstance(source, dict) else None,
            "video_id": manifest["video_id"],
            "event_type": "balance" if family == "balance" else "action",
            "actor_id": int(task["actor_id"]),
            "target_id": -1 if family == "balance" else (1 - int(task["actor_id"]) if int(task["actor_id"]) in (0, 1) else -1),
            "start_sec": round(window_start + start_offset, 6),
            "peak_sec": round(window_start + peak_offset, 6),
            "end_sec": round(window_start + end_offset, 6),
            "proxy_fps": proxy_fps,
            "proxy_start_frame": int(round((window_start + start_offset) * proxy_fps)),
            "proxy_peak_frame": int(round((window_start + peak_offset) * proxy_fps)),
            "proxy_end_frame": int(round((window_start + end_offset) * proxy_fps)),
            "technique": technique,
            "label": technique,
            "technique_candidates": candidates,
            "action_family": family,
            "anatomical_side": side,
            "side": side,
            "effector_limb": limb,
            "effector_id": effector_id(side, limb),
            "stance_at_start": prediction.get("stance_at_start", "unknown"),
            "stance_role": prediction.get("stance_role", "unknown"),
            "target_zone": prediction.get("target_zone", "unknown"),
            "outcome": prediction.get("outcome", "uncertain"),
            "contact_surface": prediction.get("contact_surface", DEFAULT_SURFACE.get(family, "unknown")),
            "visibility": prediction.get("visibility", "unknown"),
            "occluded": prediction.get("visibility") in {"partially_occluded", "heavily_occluded"},
            "uncertainty_flags": sorted(set(prediction.get("uncertainty_flags", []))),
            "review_status": "proposed",
            "annotator": "automatic_qwen3_vl",
            "source": "proposal",
            "proposal_branch": "qwen_actions",
            "proposal_confidence": round(confidence, 6),
            "attribute_confidence": round(confidence, 6),
            "classification_reason": str(prediction.get("classification_reason", "")).strip(),
            "boundary_reason": str(prediction.get("boundary_reason", "")).strip(),
            "visual_evidence": [
                str(item).strip() for item in prediction.get("visual_evidence", [])
                if str(item).strip()
            ],
            "window_summary": str(payload.get("window_summary", "")).strip(),
            "annotation_revision": 1,
            "provenance": {
                "source": "qwen_actions", "task_id": task["task_id"], "scan_kind": task["kind"],
                "window_start_sec": window_start, "window_end_sec": window_end,
                "source_pose_event_id": source.get("event_id") if isinstance(source, dict) else None,
                "source_pose_event_ids": list(task.get("source_event_ids", [])),
            },
        }
        results.append(value)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing local Qwen3-VL model directory")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--scan-mode", choices=("candidates", "hybrid", "dense"), default="hybrid")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--hop-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-mode", choices=("video", "frames"), default="video")
    parser.add_argument("--minimum-actor-crop-coverage", type=float, default=0.60)
    parser.add_argument("--debug-previews", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _model_inputs(processor: Any, images: list[np.ndarray], prompt: str, vision_mode: str, sample_fps: float) -> tuple[Any, str]:
    """Prepare temporally ordered Qwen input, with an explicit frame fallback."""
    if vision_mode == "video":
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "video", "video": [Image.fromarray(image) for image in images], "sample_fps": sample_fps},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if video_inputs:
            video_inputs, video_metadata = zip(*video_inputs)
            video_inputs, video_metadata = list(video_inputs), list(video_metadata)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            do_resize=False,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs, "video"

    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": image} for image in images],
        {"type": "text", "text": prompt},
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text_prompt], images=images, padding=True, return_tensors="pt"), "frames"


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        project = args.project.expanduser().resolve()
        model_path = args.model.expanduser().resolve()
        if not model_path.is_dir():
            raise PipelineError(f"Local Qwen model directory not found: {model_path}")
        if args.window_seconds <= 0 or args.hop_seconds <= 0 or args.fps <= 0 or args.max_new_tokens <= 0:
            raise PipelineError("window, hop, FPS and max-new-tokens must be positive")
        if not 0.0 <= args.minimum_actor_crop_coverage <= 1.0:
            raise PipelineError("--minimum-actor-crop-coverage must be between 0 and 1")
        source_path = args.input.expanduser().resolve() if args.input else project / "proposals" / "branches" / "pose_attributes.jsonl"
        output = project / "proposals" / "branches" / "qwen_actions.jsonl"
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Output exists: {output}; use --overwrite")
        manifest = load_manifest(project)
        schema = read_json(project / "annotations" / "classes.json")
        video = Path(manifest["paths"]["proxy_video"])
        video_fps = float(manifest["proxy_video"]["fps"])
        tracks_by_frame = {int(row["frame_index"]): list(row.get("detections", [])) for row in read_jsonl(project / "tracks" / "tracks.jsonl")}
        fighter_map = load_fighter_map(project)
        rows = read_jsonl(source_path) if source_path.is_file() and source_path.stat().st_size else []
        tasks = build_scan_tasks(
            rows, float(manifest["proxy_video"]["duration_sec"]), int(manifest.get("expected_fighters", 2)),
            args.window_seconds, args.hop_seconds, args.scan_mode,
        )
        if args.limit:
            tasks = tasks[: args.limit]

        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = 448 * 448
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
            device_map="auto" if args.device.startswith("cuda") else None,
            attn_implementation="sdpa",
        ).eval()
        if not args.device.startswith("cuda"):
            model = model.to(args.device)

        allowed_techniques = set(schema.get("trainable_event_classes", schema["primary_action_classes"]))
        techniques = ", ".join(schema.get("trainable_event_classes", schema["primary_action_classes"]))
        raw_dir = project / "proposals" / "raw" / "qwen"
        raw_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = raw_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[dict[str, Any]] = []
        assessments: list[dict[str, Any]] = []
        failures = 0
        parse_failures = 0
        crop_failures = 0
        valid_zero_event_tasks = 0
        model_calls = 0
        used_modes: dict[str, int] = {"video": 0, "frames": 0}
        previews_written = 0
        for task_index, task in enumerate(tasks, start=1):
            images, crop_stats = _composite_frames(
                video, float(task["start_sec"]), float(task["end_sec"]), args.fps,
                int(task["actor_id"]), video_fps, tracks_by_frame, fighter_map,
            )
            record: dict[str, Any] = {
                "task": {key: value for key, value in task.items() if key != "source"},
                "source_event_id": task.get("source", {}).get("event_id") if isinstance(task.get("source"), dict) else None,
                "frame_count": len(images),
                "crop_stats": crop_stats,
            }
            assessment = {
                "task_id": task["task_id"],
                "scan_kind": task["kind"],
                "actor_id": int(task["actor_id"]),
                "start_sec": float(task["start_sec"]),
                "end_sec": float(task["end_sec"]),
                "source_event_ids": list(task.get("source_event_ids", [])),
                "crop_stats": crop_stats,
                "raw_record": f"proposals/raw/qwen/{task['task_id']}.json",
            }
            if not images:
                record["error"] = "no actor-targeted frames decoded"
                assessment.update({"status": "failed", "error": record["error"], "contains_action": None})
                assessments.append(assessment)
                failures += 1
                crop_failures += 1
                write_json(raw_dir / f"{task['task_id']}.json", record)
                continue
            if float(crop_stats["actor_crop_coverage"]) < args.minimum_actor_crop_coverage:
                record["error"] = "insufficient_actor_crop_coverage"
                assessment.update({"status": "failed", "error": record["error"], "contains_action": None})
                assessments.append(assessment)
                failures += 1
                crop_failures += 1
                write_json(raw_dir / f"{task['task_id']}.json", record)
                continue
            if previews_written < max(0, args.debug_previews):
                preview = np.concatenate([cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images[:4]], axis=1)
                cv2.imwrite(str(preview_dir / f"{task['task_id']}.jpg"), preview)
                previews_written += 1
            base_prompt = PROMPT.format(
                window_start=float(task["start_sec"]), window_end=float(task["end_sec"]),
                fps=args.fps, actor_id=int(task["actor_id"]), techniques=techniques,
            )
            source = task.get("source")
            if isinstance(source, dict):
                base_prompt += (
                    "\nPose evidence (a hint, not truth): "
                    f"peak={source.get('peak_sec')}s; effector={source.get('pose_effector_id', source.get('effector_id'))}; "
                    f"family_candidates={source.get('family_candidates', [])}."
                )
            raw = ""
            payload: dict[str, Any] | None = None
            parse_error = ""
            input_errors: list[str] = []
            prompt_history: list[str] = []
            mode_used = args.vision_mode
            for attempt in range(2):
                prompt = base_prompt if attempt == 0 else (
                    base_prompt + f"\nYour previous response could not be parsed ({parse_error}). "
                    "Return only one corrected JSON object beginning with { and ending with }."
                )
                prompt_history.append(prompt)
                try:
                    inputs, mode_used = _model_inputs(processor, images, prompt, args.vision_mode, args.fps)
                except Exception as exc:
                    if args.vision_mode != "video":
                        raise
                    input_errors.append(f"video input failed: {type(exc).__name__}: {exc}")
                    inputs, mode_used = _model_inputs(processor, images, prompt, "frames", args.fps)
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                model_calls += 1
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                trimmed = generated[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                try:
                    payload = _json_object(raw)
                    validate_vlm_payload(
                        payload,
                        float(task["end_sec"]) - float(task["start_sec"]),
                        allowed_techniques,
                        schema,
                    )
                    parse_error = ""
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
                    payload = None
            used_modes[mode_used] = used_modes.get(mode_used, 0) + 1
            record.update({
                "raw_output": raw,
                "prompt_history": prompt_history,
                "parse_error": parse_error,
                "input_warnings": input_errors,
                "vision_mode": mode_used,
                "parsed": payload,
            })
            if payload is None:
                failures += 1
                parse_failures += 1
                assessment.update({
                    "status": "failed", "error": parse_error or "invalid model output",
                    "contains_action": None, "vision_mode": mode_used,
                })
            else:
                events = normalize_events(payload, task, manifest, schema)
                if not events:
                    valid_zero_event_tasks += 1
                record["normalized_event_ids"] = [event["event_id"] for event in events]
                assessment.update({
                    "status": "action" if events else "no_action",
                    "contains_action": bool(events),
                    "window_summary": str(payload.get("window_summary", "")).strip(),
                    "no_action_confidence": float(payload.get("no_action_confidence", 0.0)),
                    "no_action_reason": str(payload.get("no_action_reason", "")).strip(),
                    "normalized_event_ids": [event["event_id"] for event in events],
                    "vision_mode": mode_used,
                })
                outputs.extend(events)
            assessments.append(assessment)
            write_json(raw_dir / f"{task['task_id']}.json", record)
            if task_index % 10 == 0:
                print(f"Qwen action scan {task_index}/{len(tasks)}; events={len(outputs)} failures={failures}", flush=True)

        outputs.sort(key=lambda row: (float(row["start_sec"]), int(row["actor_id"]), float(row["peak_sec"])))
        write_jsonl(output, outputs)
        assessments_output = project / "proposals" / "branches" / "qwen_window_assessments.jsonl"
        write_jsonl(assessments_output, assessments)
        summary = {
            "tasks": len(tasks),
            "events": len(outputs),
            "failures": failures,
            "parse_failures": parse_failures,
            "insufficient_actor_crop_tasks": crop_failures,
            "valid_zero_event_tasks": valid_zero_event_tasks,
            "action_windows": sum(item.get("status") == "action" for item in assessments),
            "no_action_windows": sum(item.get("status") == "no_action" for item in assessments),
            "failed_windows": sum(item.get("status") == "failed" for item in assessments),
            "model_calls": model_calls,
            "vision_modes": used_modes,
            "debug_previews": previews_written,
            "scan_mode": args.scan_mode,
            "model": str(model_path),
            "local_only": True,
            "max_new_tokens": args.max_new_tokens,
            "window_assessments": str(assessments_output),
        }
        write_json(raw_dir / "summary.json", summary)
        print(json.dumps({**summary, "output": str(output), "raw_dir": str(raw_dir)}, indent=2))
        return 0
    except ImportError as exc:
        print(f"Error: install the action-image dependencies: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
