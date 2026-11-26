"""Extract local Hugging Face VideoMAE/VideoMAEv2 actor-conditioned features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PipelineError, canonical_fighter_for_track, load_fighter_map, load_manifest, read_json, read_jsonl, write_json
from training_contracts import validate_feature_array


def pool_video_tokens(hidden: Any) -> np.ndarray:
    """Pool only the patch/tubelet token axis of a [1,tokens,channels] tensor."""
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise PipelineError(f"Unexpected VideoMAE hidden state shape: {tuple(hidden.shape)}")
    vector = hidden.float().mean(dim=1).cpu().numpy()[0]
    if vector.ndim != 1 or vector.shape[0] < 2:
        raise PipelineError(f"Invalid pooled VideoMAE feature shape: {vector.shape}")
    return vector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                    fighter_value = detection.get("fighter_id")
                    fighter = int(fighter_value) if fighter_value is not None else canonical_fighter_for_track(
                        fighter_map, raw, frame_index / fps
                    )
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
                        # Pool patch/tubelet tokens only. Averaging the hidden
                        # dimension as well would collapse every temporal step
                        # to one scalar and make it unusable by TriDet.
                        features[fighter].append(pool_video_tokens(hidden))
                    timestamps.append(frame_index / fps)
                frame_index += 1
        finally:
            capture.release()
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_manifest = read_json(project / "exports" / "tridet" / "feature_manifest.json") if (project / "exports" / "tridet" / "feature_manifest.json").is_file() else {}
        for fighter, path in enumerate(outputs):
            array = np.asarray(features[fighter], dtype=np.float16)
            feature_errors = validate_feature_array(array, len(timestamps))
            if feature_errors:
                raise PipelineError(f"Invalid TriDet features for fighter {fighter}: {'; '.join(feature_errors)}")
            np.save(path, array)
            stream = f"{manifest['video_id']}__fighter{fighter}"
            feature_manifest[stream] = {"feature_file": str(path.relative_to(project / "exports" / "tridet")), "status": "complete", "shape": list(array.shape), "dtype": "float16", "sha256": _sha256(path), "actor_id": fighter, "feat_stride_frames": args.stride, "clip_frames": args.clip_frames, "feature_dim": int(array.shape[1]), "fps": fps, "model": str(model_path), "local_only": True}
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
