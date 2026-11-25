"""Optional local-only Qwen3-VL proposal labeler for short candidate windows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2

from common import PipelineError, canonical_fighter_for_track, load_fighter_map, load_manifest, read_jsonl, write_jsonl


PROMPT = """You are labeling a Muay Thai action candidate. Return exactly one JSON object.
Use anatomical left/right of the acting fighter, never screen direction. Do not guess hidden limbs.
Allowed fields: technique, action_family, anatomical_side, effector_limb, stance_at_start,
stance_role, target_zone, outcome, contact_surface, visibility, confidence, uncertainty_flags.
The clip actor is fighter {actor_id}; each full-scene frame is followed by an actor crop.
The candidate peak is near the center of the supplied frames.
If no committed action is visible, set technique to unknown_action and confidence below 0.4.
No prose and no markdown."""


def _frames(video: Path, start: float, end: float, fps: float, actor_id: int, video_fps: float, tracks_by_frame: dict[int, list[dict]], fighter_map: dict, maximum: int = 16) -> list[Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video: {video}")
    frames = []
    step = 1.0 / max(fps, 0.1)
    timestamps = [start + index * step for index in range(maximum) if start + index * step <= end + 1e-6]
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_index = int(round(timestamp * video_fps))
                actor_box = None
                for detection in tracks_by_frame.get(frame_index, []):
                    fighter = canonical_fighter_for_track(fighter_map, int(detection["track_id"]), timestamp)
                    if fighter == actor_id:
                        actor_box = detection["bbox_xyxy"]
                        break
                if actor_box is not None:
                    height, width = frame.shape[:2]
                    x1, y1, x2, y2 = [float(value) for value in actor_box]
                    dx, dy = (x2 - x1) * 0.30, (y2 - y1) * 0.30
                    left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
                    right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
                    crop = frame[top:bottom, left:right]
                    if crop.size:
                        frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model output contained no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output JSON was not an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing local model directory; downloads are disabled")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
        source = args.input.expanduser().resolve() if args.input else project / "proposals" / "branches" / "pose_attributes.jsonl"
        output = project / "proposals" / "branches" / "qwen3_vl.jsonl"
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Output exists: {output}; use --overwrite")
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        video_fps = float(manifest["proxy_video"]["fps"])
        tracks_by_frame = {int(row["frame_index"]): list(row.get("detections", [])) for row in read_jsonl(project / "tracks" / "tracks.jsonl")}
        fighter_map = load_fighter_map(project)
        rows = read_jsonl(source)
        if args.limit:
            rows = rows[: args.limit]
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
            device_map="auto" if args.device.startswith("cuda") else None,
        ).eval()
        outputs = []
        for row in rows:
            start = max(0.0, float(row["peak_sec"]) - 1.5)
            end = min(float(manifest["proxy_video"]["duration_sec"]), float(row["peak_sec"]) + 1.5)
            images = _frames(video, start, end, args.fps, int(row.get("actor_id", -1)), video_fps, tracks_by_frame, fighter_map)
            if not images:
                continue
            messages = [{"role": "user", "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": PROMPT.format(actor_id=row.get("actor_id", -1))},
            ]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=images, padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            result = dict(row)
            try:
                prediction = _json_object(raw)
                result.update(prediction)
                result["proposal_confidence"] = float(prediction.get("confidence", 0.0))
            except (ValueError, json.JSONDecodeError) as exc:
                result.setdefault("uncertainty_flags", []).append("vlm_json_parse_failed")
                result["proposal_confidence"] = 0.0
                result["vlm_parse_error"] = str(exc)
            result["proposal_branch"] = "qwen3_vl"
            result["source_event_id"] = row.get("event_id")
            result["vlm_raw_output"] = raw
            result.setdefault("provenance", {})["vlm"] = {"model": str(model_path), "local_only": True, "sample_fps": args.fps}
            outputs.append(result)
        write_jsonl(output, outputs)
        print(json.dumps({"proposals": len(outputs), "output": str(output), "local_only": True}, indent=2))
        return 0
    except ImportError as exc:
        print(f"Error: install the optional VLM dependencies: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
