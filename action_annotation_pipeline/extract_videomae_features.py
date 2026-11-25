"""Extract local Hugging Face VideoMAE/VideoMAEv2 actor-conditioned features."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PipelineError, canonical_fighter_for_track, load_fighter_map, load_manifest, read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing local VideoMAE/VideoMAEv2 model directory")
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--crop-context", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _crop(frame: np.ndarray, box: list[float] | None, context: float) -> np.ndarray:
    if box is None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    dx, dy = (x2 - x1) * context, (y2 - y1) * context
    left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
    crop = frame[top:bottom, left:right]
    return cv2.cvtColor(crop if crop.size else frame, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        project = args.project.expanduser().resolve()
        model_path = args.model.expanduser().resolve()
        if not model_path.is_dir():
            raise PipelineError(f"Local feature model not found: {model_path}")
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        tracks = read_jsonl(project / "tracks" / "tracks.jsonl")
        fighter_map = load_fighter_map(project)
        output_dir = project / "exports" / "tridet" / "features"
        expected = int(manifest.get("expected_fighters", 2))
        outputs = [output_dir / f"{manifest['video_id']}__fighter{fighter}.npy" for fighter in range(expected)]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise PipelineError("Feature outputs exist; use --overwrite")
        tracks_by_frame = {int(row["frame_index"]): list(row.get("detections", [])) for row in tracks}
        processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        ).to(args.device).eval()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise PipelineError(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        buffers: dict[int, list[np.ndarray]] = defaultdict(list)
        features: dict[int, list[np.ndarray]] = defaultdict(list)
        timestamps: list[float] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes: dict[int, list[float]] = {}
                for detection in tracks_by_frame.get(frame_index, []):
                    raw = int(detection["track_id"])
                    fighter = canonical_fighter_for_track(fighter_map, raw, frame_index / fps)
                    if fighter is not None:
                        boxes[fighter] = list(detection["bbox_xyxy"])
                for fighter in range(expected):
                    buffers[fighter].append(_crop(frame, boxes.get(fighter), args.crop_context))
                    if len(buffers[fighter]) > args.clip_frames:
                        buffers[fighter].pop(0)
                if frame_index + 1 >= args.clip_frames and (frame_index + 1 - args.clip_frames) % args.stride == 0:
                    for fighter in range(expected):
                        inputs = processor(list(buffers[fighter]), return_tensors="pt")
                        inputs = {key: value.to(args.device) for key, value in inputs.items()}
                        with torch.inference_mode():
                            hidden = model(**inputs).last_hidden_state
                        features[fighter].append(hidden.float().mean(dim=tuple(range(1, hidden.ndim))).cpu().numpy()[0])
                    timestamps.append(frame_index / fps)
                frame_index += 1
        finally:
            capture.release()
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_manifest = read_json(project / "exports" / "tridet" / "feature_manifest.json") if (project / "exports" / "tridet" / "feature_manifest.json").is_file() else {}
        for fighter, path in enumerate(outputs):
            array = np.asarray(features[fighter], dtype=np.float16)
            np.save(path, array)
            stream = f"{manifest['video_id']}__fighter{fighter}"
            feature_manifest[stream] = {"feature_file": str(path.relative_to(project / "exports" / "tridet")), "status": "complete", "shape": list(array.shape), "dtype": "float16", "actor_id": fighter, "feat_stride_frames": args.stride, "clip_frames": args.clip_frames, "model": str(model_path), "local_only": True}
        np.save(output_dir / "timestamps.npy", np.asarray(timestamps, dtype=np.float32))
        write_json(project / "exports" / "tridet" / "feature_manifest.json", feature_manifest)
        summary = {"streams": expected, "feature_steps": len(timestamps), "stride_frames": args.stride, "outputs": [str(path) for path in outputs]}
        print(json.dumps(summary, indent=2))
        return 0
    except ImportError as exc:
        print(f"Error: install transformers and torch: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
