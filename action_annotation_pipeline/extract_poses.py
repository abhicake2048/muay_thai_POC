"""Run the local RTMPose model on tracked fighter boxes and save dense poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from common import (
    PipelineError,
    canonical_fighter_for_track,
    load_fighter_map,
    load_manifest,
    read_jsonl,
    write_json,
)


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

COCO_LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = [(52, 211, 153), (251, 146, 60), (96, 165, 250), (244, 114, 182)]


@contextmanager
def trusted_checkpoint_loading() -> Iterator[None]:
    """Let legacy MMEngine load a trusted checkpoint under PyTorch 2.6+.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    MMEngine 0.10.7 does not pass that argument when loading a local file, and
    older MMPose checkpoints commonly contain NumPy metadata that the restricted
    unpickler rejects.  PyTorch documents this environment variable for legacy
    call sites that cannot pass ``weights_only=False`` themselves.

    Unrestricted pickle loading can execute code, so keep the override scoped
    to model initialization and use only checkpoints from a trusted source.
    """
    variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(variable)
    os.environ[variable] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--bbox-padding", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=0, help="Testing only; 0 processes all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, help="Override manifest pose config (teacher models)")
    parser.add_argument("--checkpoint", type=Path, help="Override manifest checkpoint (teacher models)")
    parser.add_argument("--output", type=Path, help="Override poses.npz destination")
    parser.add_argument("--overlay", type=Path, help="Override overlay destination")
    parser.add_argument("--metadata", type=Path, help="Override metadata destination")
    return parser.parse_args()


def padded_box(box: list[float], width: int, height: int, padding: float) -> list[float]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width - 1), x2 + dx), min(float(height - 1), y2 + dy)]


def draw_pose(frame: np.ndarray, fighter_id: int, track_id: int, box: np.ndarray, points: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    color = COLORS[abs(fighter_id) % len(COLORS)]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"fighter {fighter_id} / raw {track_id}", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    for start, end in COCO_LINKS:
        if scores[start] >= threshold and scores[end] >= threshold:
            cv2.line(frame, tuple(np.rint(points[start]).astype(int)), tuple(np.rint(points[end]).astype(int)), color, 2, cv2.LINE_AA)
    for point, score in zip(points, scores):
        if score >= threshold:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    try:
        import torch
        from mmpose.apis import inference_topdown, init_model

        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        config = args.config.expanduser().resolve() if args.config else Path(manifest["paths"]["rtmpose_config"])
        checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else Path(manifest["paths"]["rtmpose_checkpoint"])
        tracks_path = project / "tracks" / "tracks.jsonl"
        fighter_map = load_fighter_map(project)
        output_path = args.output.expanduser().resolve() if args.output else project / "poses" / "poses.npz"
        overlay_path = args.overlay.expanduser().resolve() if args.overlay else project / "qa" / "poses_overlay.mp4"
        metadata_path = args.metadata.expanduser().resolve() if args.metadata else project / "poses" / "metadata.json"
        for required in (video, config, checkpoint, tracks_path):
            if not required.is_file():
                raise PipelineError(f"Required input not found: {required}")
        if output_path.exists() and not args.overwrite:
            raise PipelineError(f"Pose output already exists: {output_path}; pass --overwrite to replace")
        if not 0 <= args.bbox_padding <= 0.5:
            raise PipelineError("--bbox-padding must be between 0 and 0.5")

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise PipelineError("CUDA requested but unavailable")
        print(f"Loading trusted RTMPose checkpoint: {checkpoint}", flush=True)
        with trusted_checkpoint_loading():
            model = init_model(str(config), str(checkpoint), device=device)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        track_rows = read_jsonl(tracks_path)
        tracks_by_frame = {int(row["frame_index"]): list(row.get("detections", [])) for row in track_rows}
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise PipelineError(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise PipelineError(f"Cannot create overlay: {overlay_path}")

        instance_frames: list[int] = []
        timestamps: list[float] = []
        track_ids: list[int] = []
        fighter_ids: list[int] = []
        bboxes: list[np.ndarray] = []
        bbox_scores: list[float] = []
        keypoints: list[np.ndarray] = []
        keypoint_scores: list[np.ndarray] = []
        frame_index = 0
        threshold = float(manifest.get("thresholds", {}).get("keypoint_confidence", 0.2))
        discarded_unmapped_detections = 0
        discarded_duplicate_fighter_detections = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if args.max_frames and frame_index >= args.max_frames:
                    break
                raw_detections = tracks_by_frame.get(frame_index, [])
                timestamp = frame_index / fps
                selected: dict[int, dict[str, Any]] = {}
                for detection in raw_detections:
                    fighter = detection.get("fighter_id")
                    if fighter is None:
                        fighter = canonical_fighter_for_track(
                            fighter_map,
                            int(detection["track_id"]),
                            timestamp,
                        )
                    if fighter is None or int(fighter) < 0:
                        discarded_unmapped_detections += 1
                        continue
                    fighter = int(fighter)
                    previous = selected.get(fighter)
                    if previous is not None:
                        discarded_duplicate_fighter_detections += 1
                    if previous is None or float(detection.get("confidence", 0.0)) > float(previous.get("confidence", 0.0)):
                        selected[fighter] = detection
                detections = [selected[fighter] for fighter in sorted(selected)]
                boxes = [padded_box(list(det["bbox_xyxy"]), width, height, args.bbox_padding) for det in detections]
                if boxes:
                    samples = inference_topdown(model, frame, bboxes=np.asarray(boxes, dtype=np.float32), bbox_format="xyxy")
                    for detection, box, sample in zip(detections, boxes, samples):
                        instances = sample.pred_instances
                        points = np.asarray(instances.keypoints, dtype=np.float32)[0]
                        if hasattr(instances, "keypoint_scores"):
                            scores = np.asarray(instances.keypoint_scores, dtype=np.float32)[0]
                        else:
                            scores = np.ones(points.shape[0], dtype=np.float32)
                        track_id = int(detection["track_id"])
                        fighter_value = detection.get("fighter_id")
                        if fighter_value is None:
                            fighter_value = canonical_fighter_for_track(fighter_map, track_id, timestamp)
                        if fighter_value is None:
                            continue
                        fighter_id = int(fighter_value)
                        instance_frames.append(frame_index)
                        timestamps.append(timestamp)
                        track_ids.append(track_id)
                        fighter_ids.append(fighter_id)
                        bboxes.append(np.asarray(box, dtype=np.float32))
                        bbox_scores.append(float(detection.get("confidence", 1.0)))
                        keypoints.append(points)
                        keypoint_scores.append(scores)
                        draw_pose(frame, fighter_id, track_id, np.asarray(box), points, scores, threshold)
                cv2.putText(frame, f"frame {frame_index}  {frame_index/fps:.3f}s", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                frame_index += 1
                if frame_index % 300 == 0:
                    print(f"Posed {frame_index}/{total or '?'} frames; {len(instance_frames)} instances", flush=True)
        finally:
            capture.release()
            writer.release()

        if not keypoints:
            raise PipelineError("RTMPose produced no fighter poses")
        np.savez_compressed(
            output_path,
            frame_indices=np.asarray(instance_frames, dtype=np.int32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            track_ids=np.asarray(track_ids, dtype=np.int32),
            fighter_ids=np.asarray(fighter_ids, dtype=np.int16),
            bboxes_xyxy=np.asarray(bboxes, dtype=np.float32),
            bbox_scores=np.asarray(bbox_scores, dtype=np.float32),
            keypoints=np.asarray(keypoints, dtype=np.float32),
            keypoint_scores=np.asarray(keypoint_scores, dtype=np.float32),
        )
        write_json(metadata_path, {
            "instances": len(keypoints), "processed_frames": frame_index, "fps": fps,
            "width": width, "height": height, "device": str(device),
            "config": str(config), "checkpoint": str(checkpoint), "bbox_padding": args.bbox_padding,
            "canonical_fighters_only": True,
            "discarded_unmapped_detections": discarded_unmapped_detections,
            "discarded_duplicate_fighter_detections": discarded_duplicate_fighter_detections,
        })
        print(json.dumps({"instances": len(keypoints), "frames": frame_index, "poses": str(output_path), "overlay": str(overlay_path)}, indent=2))
        return 0
    except (ImportError, OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
