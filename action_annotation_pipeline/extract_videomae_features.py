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


def slice_stream_features(
    full_features: np.ndarray,
    timestamps: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    source_start: float,
    source_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only feature clips wholly contained in one trainable source range."""
    selection = (window_starts >= source_start - 1e-6) & (window_ends <= source_end + 1e-6)
    return full_features[selection], timestamps[selection] - source_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing local VideoMAE/VideoMAEv2 model directory")
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--crop-context", type=float, default=0.50)
    parser.add_argument("--max-box-gap-frames", type=int, default=6)
    parser.add_argument("--minimum-actor-crop-coverage", type=float, default=0.80)
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
        poses_path = project / "poses" / "poses.npz"
        tridet_annotations = project / "exports" / "tridet" / "annotations.json"
        if not tridet_annotations.is_file():
            raise PipelineError("TriDet annotations are missing; run mtdata export before mtdata features")
        tridet_payload = read_json(tridet_annotations)
        streams = dict(tridet_payload.get("database", {}))
        if not streams:
            raise PipelineError("TriDet annotations contain no actor streams")
        output_dir = project / "exports" / "tridet" / "features"
        expected = int(manifest.get("expected_fighters", 2))
        outputs = [output_dir / f"{stream_id}.npy" for stream_id in streams]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise PipelineError("Feature outputs exist; use --overwrite")
        if args.clip_frames <= 0 or args.stride <= 0 or args.max_box_gap_frames < 0:
            raise PipelineError("clip-frames/stride must be positive and max-box-gap-frames non-negative")
        if not 0.0 <= args.minimum_actor_crop_coverage <= 1.0:
            raise PipelineError("minimum-actor-crop-coverage must be between 0 and 1")
        tracks_by_frame = {int(row["frame_index"]): list(row.get("detections", [])) for row in tracks}
        pose_boxes_by_frame: dict[int, dict[int, list[float]]] = defaultdict(dict)
        if poses_path.is_file():
            pose_data = np.load(poses_path)
            for index, frame_value in enumerate(pose_data["frame_indices"]):
                frame_value = int(frame_value)
                if "fighter_ids" in pose_data.files:
                    fighter = int(pose_data["fighter_ids"][index])
                else:
                    fighter_value = canonical_fighter_for_track(
                        fighter_map, int(pose_data["track_ids"][index]), float(pose_data["timestamps"][index])
                    )
                    fighter = int(fighter_value) if fighter_value is not None else -1
                if 0 <= fighter < expected:
                    pose_boxes_by_frame[frame_value][fighter] = list(map(float, pose_data["bboxes_xyxy"][index]))
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
        box_buffers: dict[int, list[bool]] = defaultdict(list)
        features: dict[int, list[np.ndarray]] = defaultdict(list)
        feature_crop_coverage: dict[int, list[float]] = defaultdict(list)
        last_boxes: dict[int, list[float]] = {}
        last_box_frames: dict[int, int] = {}
        timestamps: list[float] = []
        window_starts: list[float] = []
        window_ends: list[float] = []
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
                for fighter, box in pose_boxes_by_frame.get(frame_index, {}).items():
                    boxes.setdefault(fighter, box)
                for fighter in range(expected):
                    box = boxes.get(fighter)
                    if box is not None:
                        last_boxes[fighter] = box
                        last_box_frames[fighter] = frame_index
                    elif fighter in last_boxes and frame_index - last_box_frames[fighter] <= args.max_box_gap_frames:
                        box = last_boxes[fighter]
                    buffers[fighter].append(_crop(frame, box, args.crop_context))
                    box_buffers[fighter].append(box is not None)
                    if len(buffers[fighter]) > args.clip_frames:
                        buffers[fighter].pop(0)
                        box_buffers[fighter].pop(0)
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
                        feature_crop_coverage[fighter].append(float(np.mean(box_buffers[fighter])))
                    start_time = (frame_index + 1 - args.clip_frames) / fps
                    end_time = frame_index / fps
                    window_starts.append(start_time)
                    window_ends.append(end_time)
                    timestamps.append((start_time + end_time) / 2)
                frame_index += 1
        finally:
            capture.release()
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_manifest = read_json(project / "exports" / "tridet" / "feature_manifest.json") if (project / "exports" / "tridet" / "feature_manifest.json").is_file() else {}
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        start_array = np.asarray(window_starts, dtype=np.float64)
        end_array = np.asarray(window_ends, dtype=np.float64)
        full_features = {fighter: np.asarray(features[fighter], dtype=np.float16) for fighter in range(expected)}
        full_crop_coverage = {fighter: np.asarray(feature_crop_coverage[fighter], dtype=np.float32) for fighter in range(expected)}
        timestamps_dir = output_dir / "timestamps"
        timestamps_dir.mkdir(parents=True, exist_ok=True)
        for stream_id, record in streams.items():
            fighter = int(record.get("actor_id", -1))
            if fighter not in full_features:
                raise PipelineError(f"TriDet stream {stream_id} has invalid actor_id {fighter}")
            source_start = float(record.get("source_start_sec", 0.0))
            source_end = float(record.get("source_end_sec", manifest["proxy_video"]["duration_sec"]))
            array, local_timestamps = slice_stream_features(
                full_features[fighter], timestamp_array, start_array, end_array, source_start, source_end
            )
            stream_coverage, _ = slice_stream_features(
                full_crop_coverage[fighter][:, None], timestamp_array, start_array, end_array, source_start, source_end
            )
            mean_crop_coverage = float(stream_coverage.mean()) if stream_coverage.size else 0.0
            feature_errors = validate_feature_array(array, len(local_timestamps))
            if feature_errors:
                raise PipelineError(f"Invalid TriDet features for stream {stream_id}: {'; '.join(feature_errors)}")
            if mean_crop_coverage < args.minimum_actor_crop_coverage:
                raise PipelineError(
                    f"Actor crop coverage for {stream_id} is {mean_crop_coverage:.3f}, below "
                    f"{args.minimum_actor_crop_coverage:.3f}; correct tracks/poses or mark the interval ignored"
                )
            path = output_dir / f"{stream_id}.npy"
            timestamp_path = timestamps_dir / f"{stream_id}.npy"
            np.save(path, array)
            np.save(timestamp_path, local_timestamps.astype(np.float32))
            feature_manifest[stream_id] = {"feature_file": str(path.relative_to(project / "exports" / "tridet")), "timestamps_file": str(timestamp_path.relative_to(project / "exports" / "tridet")), "status": "complete", "shape": list(array.shape), "dtype": "float16", "sha256": _sha256(path), "actor_id": fighter, "source_start_sec": source_start, "source_end_sec": source_end, "actor_crop_coverage": round(mean_crop_coverage, 6), "minimum_actor_crop_coverage": args.minimum_actor_crop_coverage, "max_box_gap_frames": args.max_box_gap_frames, "feat_stride_frames": args.stride, "clip_frames": args.clip_frames, "feature_dim": int(array.shape[1]), "fps": fps, "model": str(model_path), "local_only": True}
        write_json(project / "exports" / "tridet" / "feature_manifest.json", feature_manifest)
        summary = {"streams": len(streams), "full_video_feature_steps": len(timestamps), "stride_frames": args.stride, "outputs": [str(path) for path in outputs]}
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
