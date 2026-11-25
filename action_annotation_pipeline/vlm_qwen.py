"""Local Qwen3-VL discovery and labeling for actor-centric fight windows."""

from __future__ import annotations

import argparse
import json
import re
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
Use the fighter's anatomical left/right, never screen-left/screen-right. Use unknown when hidden.
Allowed techniques: {techniques}.
Return exactly one JSON object and no prose:
{{"events":[{{"start_offset_sec":0.0,"peak_offset_sec":0.0,"end_offset_sec":0.0,
"technique":"jab","technique_candidates":[{{"technique":"jab","confidence":0.7}}],
"anatomical_side":"left","effector_limb":"hand","stance_at_start":"orthodox",
"stance_role":"lead","target_zone":"head","outcome":"attempted","contact_surface":"fist",
"visibility":"visible","confidence":0.7,"uncertainty_flags":[]}}],"no_action_confidence":0.0}}
Offsets are relative to the start of this window, in seconds. Boundaries must be ordered and inside the window.
Do not invent contact or anatomical side. Predictions are proposals, not verified truth."""


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model output contained no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output JSON was not an object")
    if not isinstance(value.get("events", []), list):
        raise ValueError("model output events must be an array")
    return value


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
        return frame
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
    for detection in tracks_by_frame.get(frame_index, []):
        fighter = canonical_fighter_for_track(fighter_map, int(detection["track_id"]), timestamp)
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
    maximum: int = 18,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video: {video}")
    count = min(maximum, max(1, int(np.floor((end - start) * sample_fps)) + 1))
    timestamps = np.linspace(start, end, count, endpoint=True)
    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            boxes = _boxes_for_time(tracks_by_frame, fighter_map, float(timestamp), video_fps)
            scene = frame.copy()
            for fighter, box in boxes.items():
                color = (60, 210, 150) if fighter == actor_id else (60, 150, 250)
                x1, y1, x2, y2 = [round(value) for value in box]
                cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
                cv2.putText(scene, f"fighter {fighter}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(scene, f"t={float(timestamp):.3f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            opponent_id = 1 - actor_id if actor_id in (0, 1) else -1
            composite = np.zeros((448, 448, 3), dtype=np.uint8)
            composite[:252] = _letterbox(scene, 448, 252)
            composite[252:] = np.concatenate([
                _letterbox(_crop(frame, boxes.get(actor_id)), 224, 196),
                _letterbox(_crop(frame, boxes.get(opponent_id)), 224, 196),
            ], axis=1)
            cv2.putText(composite, f"ACTOR {actor_id}", (8, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 210, 150), 2)
            cv2.putText(composite, "OPPONENT", (232, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 150, 250), 2)
            frames.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


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
        for row in rows:
            actor = int(row.get("actor_id", -1))
            if actor not in covered:
                continue
            peak = float(row.get("peak_sec", 0.0))
            start = max(0.0, min(duration - min(window_seconds, duration), peak - window_seconds / 2))
            end = min(duration, start + window_seconds)
            covered[actor].append((start, end))
            tasks.append({
                "task_id": f"candidate_{len(tasks) + 1:06d}", "kind": "candidate", "actor_id": actor,
                "start_sec": start, "end_sec": end, "source": row,
            })
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
            if label in allowed:
                candidates.append({"technique": label, "confidence": round(max(0.0, min(1.0, score)), 6), "source": "qwen_actions"})
    if technique in allowed and not any(item["technique"] == technique for item in candidates):
        candidates.append({"technique": technique, "confidence": round(confidence, 6), "source": "qwen_actions"})
    candidates.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return candidates[:3]


def normalize_events(
    payload: dict[str, Any],
    task: dict[str, Any],
    manifest: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    allowed_techniques = set(schema.get("primary_action_classes", []))
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
        technique = str(prediction.get("technique", "unknown_action"))
        if technique not in allowed_techniques:
            technique = "unknown_action"
        candidates = _candidate_list(prediction.get("technique_candidates"), technique, confidence, allowed_techniques)
        side = str(prediction.get("anatomical_side", "unknown"))
        if side not in allowed_sides:
            side = "unknown"
        family = technique_family(schema, technique) or str(prediction.get("action_family", "unknown"))
        if family not in schema.get("action_families", {}):
            family = "unknown"
        limb = str(prediction.get("effector_limb", DEFAULT_LIMB.get(family, "unknown")))
        if limb not in allowed_limbs:
            limb = DEFAULT_LIMB.get(family, "unknown")
        base = dict(source) if isinstance(source, dict) else {}
        event_id = f"qwen_{task['task_id']}_{index:02d}"
        value = {
            **base,
            "event_id": event_id,
            "source_event_id": source.get("event_id") if isinstance(source, dict) else None,
            "video_id": manifest["video_id"],
            "event_type": "action",
            "actor_id": int(task["actor_id"]),
            "target_id": 1 - int(task["actor_id"]) if int(task["actor_id"]) in (0, 1) else -1,
            "start_sec": round(window_start + start_offset, 6),
            "peak_sec": round(window_start + peak_offset, 6),
            "end_sec": round(window_start + end_offset, 6),
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
            "annotation_revision": 1,
            "provenance": {
                "source": "qwen_actions", "task_id": task["task_id"], "scan_kind": task["kind"],
                "window_start_sec": window_start, "window_end_sec": window_end,
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
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        project = args.project.expanduser().resolve()
        model_path = args.model.expanduser().resolve()
        if not model_path.is_dir():
            raise PipelineError(f"Local Qwen model directory not found: {model_path}")
        if args.window_seconds <= 0 or args.hop_seconds <= 0 or args.fps <= 0:
            raise PipelineError("window, hop and FPS must be positive")
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

        techniques = ", ".join(schema["primary_action_classes"])
        raw_dir = project / "proposals" / "raw" / "qwen"
        raw_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[dict[str, Any]] = []
        failures = 0
        for task_index, task in enumerate(tasks, start=1):
            images = _composite_frames(
                video, float(task["start_sec"]), float(task["end_sec"]), args.fps,
                int(task["actor_id"]), video_fps, tracks_by_frame, fighter_map,
            )
            record: dict[str, Any] = {"task": {key: value for key, value in task.items() if key != "source"}, "source_event_id": task.get("source", {}).get("event_id") if isinstance(task.get("source"), dict) else None, "frame_count": len(images)}
            if not images:
                record["error"] = "no frames decoded"
                failures += 1
                write_json(raw_dir / f"{task['task_id']}.json", record)
                continue
            base_prompt = PROMPT.format(
                window_start=float(task["start_sec"]), window_end=float(task["end_sec"]),
                fps=args.fps, actor_id=int(task["actor_id"]), techniques=techniques,
            )
            raw = ""
            payload: dict[str, Any] | None = None
            parse_error = ""
            for attempt in range(2):
                prompt = base_prompt if attempt == 0 else base_prompt + "\nYour previous response was invalid JSON. Return only the corrected object."
                messages = [{"role": "user", "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ]}]
                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text_prompt], images=images, padding=True, return_tensors="pt")
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=768, do_sample=False)
                trimmed = generated[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                try:
                    payload = _json_object(raw)
                    parse_error = ""
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
            record.update({"raw_output": raw, "parse_error": parse_error, "parsed": payload})
            if payload is None:
                failures += 1
            else:
                events = normalize_events(payload, task, manifest, schema)
                record["normalized_event_ids"] = [event["event_id"] for event in events]
                outputs.extend(events)
            write_json(raw_dir / f"{task['task_id']}.json", record)
            if task_index % 10 == 0:
                print(f"Qwen action scan {task_index}/{len(tasks)}; events={len(outputs)} failures={failures}", flush=True)

        outputs.sort(key=lambda row: (float(row["start_sec"]), int(row["actor_id"]), float(row["peak_sec"])))
        write_jsonl(output, outputs)
        summary = {"tasks": len(tasks), "events": len(outputs), "failures": failures, "scan_mode": args.scan_mode, "model": str(model_path), "local_only": True}
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
