"""Create actor-conditioned RGB videos for later VideoMAEv2/TriDet features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import PipelineError, load_fighter_map, load_manifest, raw_track_for_fighter, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--padding", type=float, default=1.8, help="Crop size relative to fighter box")
    parser.add_argument("--smoothing", type=float, default=0.82, help="Previous crop weight")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def square_crop(frame: np.ndarray, center_x: float, center_y: float, side: float, output_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = max(32.0, side)
    x1, y1 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x2, y2 = int(round(center_x + side / 2)), int(round(center_y + side / 2))
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - width), max(0, y2 - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1, x2 = x1 + pad_left, x2 + pad_left
        y1, y2 = y1 + pad_top, y2 + pad_top
    crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    try:
        if args.size < 112 or not 1.0 <= args.padding <= 4.0 or not 0 <= args.smoothing < 1:
            raise PipelineError("Invalid crop size, padding, or smoothing")
        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        fighter_map = load_fighter_map(project)
        tracks = read_jsonl(project / "tracks" / "tracks.jsonl")
        video = Path(manifest["paths"]["proxy_video"])
        output_dir = project / "exports" / "tridet" / "actor_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = int(manifest.get("expected_fighters", 2))
        outputs = [output_dir / f"{manifest['video_id']}__fighter{fighter_id}.mp4" for fighter_id in range(expected)]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise PipelineError("Actor videos already exist; pass --overwrite to replace")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise PipelineError(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        writers = [cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.size, args.size)) for path in outputs]
        if not all(writer.isOpened() for writer in writers):
            capture.release()
            for writer in writers:
                writer.release()
            raise PipelineError("Cannot create actor crop videos")
        rows_by_frame = {int(row["frame_index"]): row for row in tracks}
        states: list[np.ndarray | None] = [None] * expected
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                detections = rows_by_frame.get(frame_index, {}).get("detections", [])
                by_track = {int(det["track_id"]): det for det in detections}
                for fighter_id, writer in enumerate(writers):
                    raw_track = raw_track_for_fighter(fighter_map, fighter_id, timestamp)
                    detection = by_track.get(raw_track) if raw_track is not None else None
                    if detection is not None:
                        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
                        proposed = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * args.padding], dtype=np.float32)
                        states[fighter_id] = proposed if states[fighter_id] is None else args.smoothing * states[fighter_id] + (1 - args.smoothing) * proposed
                    if states[fighter_id] is None:
                        states[fighter_id] = np.asarray([frame.shape[1] / 2, frame.shape[0] / 2, max(frame.shape[:2])], dtype=np.float32)
                    center_x, center_y, side = states[fighter_id]
                    writer.write(square_crop(frame, float(center_x), float(center_y), float(side), args.size))
                frame_index += 1
                if frame_index % 300 == 0:
                    print(f"Cropped {frame_index} frames", flush=True)
        finally:
            capture.release()
            for writer in writers:
                writer.release()
        metadata = {
            "videos": [str(path) for path in outputs], "frames": frame_index, "fps": fps,
            "size": args.size, "padding": args.padding, "smoothing": args.smoothing,
            "warning": "These actor-conditioned videos require identity-map review before feature extraction.",
        }
        write_json(output_dir / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
