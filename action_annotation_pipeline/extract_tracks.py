"""Track fighters in an annotation proxy using the project's YOLO11 weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PipelineError, load_manifest, write_json, write_jsonl


COLORS = [(52, 211, 153), (251, 146, 60), (96, 165, 250), (244, 114, 182)]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class SimpleFighterTracker:
    """Small dependency-free tracker for a two-fighter, mostly fixed camera scene."""

    def __init__(self, width: int, height: int, max_missed: int = 30, minimum_score: float = 0.12):
        self.width = width
        self.height = height
        self.max_missed = max_missed
        self.minimum_score = minimum_score
        self.next_id = 0
        self.tracks: dict[int, dict[str, Any]] = {}

    @staticmethod
    def center(box: np.ndarray) -> np.ndarray:
        return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)

    def update(self, boxes: np.ndarray, confidences: np.ndarray) -> list[tuple[int, np.ndarray, float]]:
        for track in self.tracks.values():
            track["missed"] += 1
        pairs: list[tuple[float, int, int]] = []
        diagonal = max(1.0, float(np.hypot(self.width, self.height)))
        for track_id, track in self.tracks.items():
            predicted_center = track["center"] + track["velocity"]
            for detection_index, box in enumerate(boxes):
                distance_score = max(0.0, 1.0 - float(np.linalg.norm(self.center(box) - predicted_center)) / (0.35 * diagonal))
                score = 0.62 * box_iou(track["box"], box) + 0.38 * distance_score
                pairs.append((score, track_id, detection_index))
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        output: list[tuple[int, np.ndarray, float]] = []
        for score, track_id, detection_index in sorted(pairs, reverse=True):
            if score < self.minimum_score or track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            track = self.tracks[track_id]
            box = boxes[detection_index]
            new_center = self.center(box)
            measured_velocity = new_center - track["center"]
            track["velocity"] = 0.65 * track["velocity"] + 0.35 * measured_velocity
            track["center"] = new_center
            track["box"] = box.copy()
            track["missed"] = 0
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            output.append((track_id, box, float(confidences[detection_index])))
        for detection_index, box in enumerate(boxes):
            if detection_index in assigned_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "box": box.copy(), "center": self.center(box),
                "velocity": np.zeros(2, dtype=np.float32), "missed": 0,
            }
            output.append((track_id, box, float(confidences[detection_index])))
        stale = [track_id for track_id, track in self.tracks.items() if track["missed"] > self.max_missed]
        for track_id in stale:
            del self.tracks[track_id]
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ...")
    parser.add_argument(
        "--tracker",
        default="simple",
        help="'simple' requires no extra package; or use an Ultralytics YAML such as botsort.yaml",
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--max-detections", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=0, help="Testing only; 0 processes all frames")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def color_for(track_id: int) -> tuple[int, int, int]:
    return COLORS[abs(int(track_id)) % len(COLORS)]


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO

        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        weights = Path(manifest["paths"]["yolo_weights"])
        if not video.is_file() or not weights.is_file():
            raise PipelineError(f"Missing proxy or YOLO weights: {video}, {weights}")
        tracks_path = project / "tracks" / "tracks.jsonl"
        overlay_path = project / "qa" / "tracks_overlay.mp4"
        if tracks_path.exists() and not args.overwrite:
            raise PipelineError(f"Tracks already exist: {tracks_path}; pass --overwrite to replace")

        model = YOLO(str(weights))
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise PipelineError(f"Cannot open proxy video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise PipelineError(f"Cannot create overlay: {overlay_path}")
        simple_tracker = SimpleFighterTracker(width, height) if args.tracker == "simple" else None

        confidence = args.confidence
        if confidence is None:
            confidence = float(manifest.get("thresholds", {}).get("detector_confidence", 0.25))
        device: str | int = args.device
        if args.device == "auto":
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"
        elif args.device.isdigit():
            device = int(args.device)

        records: list[dict[str, Any]] = []
        stats: dict[int, dict[str, float]] = defaultdict(lambda: {
            "frames": 0.0, "confidence_sum": 0.0, "area_sum": 0.0, "centrality_sum": 0.0
        })
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if args.max_frames and frame_index >= args.max_frames:
                    break
                detections: list[dict[str, Any]] = []
                if simple_tracker is not None:
                    result = model.predict(
                        frame, conf=confidence, iou=args.iou, max_det=args.max_detections,
                        device=device, verbose=False,
                    )[0]
                    if result.boxes is not None and len(result.boxes):
                        xyxy = result.boxes.xyxy.detach().cpu().numpy()
                        confs = result.boxes.conf.detach().cpu().numpy()
                    else:
                        xyxy = np.zeros((0, 4), dtype=np.float32)
                        confs = np.zeros((0,), dtype=np.float32)
                    tracked = simple_tracker.update(xyxy, confs)
                else:
                    result = model.track(
                        frame, persist=True, tracker=args.tracker, conf=confidence,
                        iou=args.iou, max_det=args.max_detections, device=device, verbose=False,
                    )[0]
                    tracked = []
                    if result.boxes is not None and len(result.boxes):
                        xyxy = result.boxes.xyxy.detach().cpu().numpy()
                        confs = result.boxes.conf.detach().cpu().numpy()
                        ids = result.boxes.id
                        raw_ids = ids.detach().cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy), dtype=int) + 1000000
                        tracked = [(int(track_id), box, float(score)) for box, score, track_id in zip(xyxy, confs, raw_ids)]
                for track_id, box, score in tracked:
                    x1, y1, x2, y2 = [float(value) for value in box]
                    record = {
                        "track_id": int(track_id),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "confidence": float(score),
                    }
                    detections.append(record)
                    box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
                    center_x, center_y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                    centrality = 1.0 - min(1.0, math.hypot(center_x - 0.5, center_y - 0.52) / 0.72)
                    item = stats[int(track_id)]
                    item["frames"] += 1
                    item["confidence_sum"] += float(score)
                    item["area_sum"] += box_width * box_height / (width * height)
                    item["centrality_sum"] += centrality
                    color = color_for(int(track_id))
                    cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
                    cv2.putText(frame, f"raw track {track_id} {score:.2f}", (round(x1), max(22, round(y1) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
                timestamp = frame_index / fps
                cv2.putText(frame, f"frame {frame_index}  {timestamp:.3f}s", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                records.append({"frame_index": frame_index, "timestamp": timestamp, "detections": detections})
                frame_index += 1
                if frame_index % 300 == 0:
                    print(f"Tracked {frame_index}/{total or '?'} frames", flush=True)
        finally:
            capture.release()
            writer.release()

        if not records:
            raise PipelineError("No frames were decoded")
        write_jsonl(tracks_path, records)

        ranked: list[dict[str, Any]] = []
        for track_id, item in stats.items():
            count = max(1.0, item["frames"])
            mean_conf = item["confidence_sum"] / count
            mean_area = item["area_sum"] / count
            mean_centrality = item["centrality_sum"] / count
            coverage = item["frames"] / len(records)
            score = coverage * (0.55 * mean_conf + 0.25 * min(1.0, mean_area / 0.12) + 0.20 * mean_centrality)
            ranked.append({
                "track_id": track_id,
                "detected_frames": int(item["frames"]),
                "coverage": coverage,
                "mean_confidence": mean_conf,
                "mean_area_ratio": mean_area,
                "mean_centrality": mean_centrality,
                "selection_score": score,
            })
        ranked.sort(key=lambda value: value["selection_score"], reverse=True)
        write_json(project / "tracks" / "track_summary.json", {"tracks": ranked})
        chosen = ranked[: int(manifest.get("expected_fighters", 2))]
        fighter_map = {
            "schema_version": "1.0",
            "instructions": "Verify these suggested raw track IDs in tracks_overlay.mp4. If an ID switches, split track_segments at the switch time.",
            "fighters": [
                {
                    "fighter_id": index,
                    "name": f"fighter_{index}",
                    "track_segments": [{"track_id": int(value["track_id"]), "start_sec": 0.0, "end_sec": float(records[-1]["timestamp"])}],
                }
                for index, value in enumerate(chosen)
            ],
        }
        suggested_path = project / "tracks" / "fighter_map.suggested.json"
        write_json(suggested_path, fighter_map)
        final_map = project / "tracks" / "fighter_map.json"
        if not final_map.exists():
            write_json(final_map, fighter_map)
        print(json.dumps({"frames": len(records), "tracks": len(ranked), "suggested_fighters": chosen, "tracks_file": str(tracks_path), "overlay": str(overlay_path), "fighter_map": str(final_map)}, indent=2))
        return 0
    except (ImportError, OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
