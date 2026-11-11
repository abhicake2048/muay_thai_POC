"""Run YOLO11 fighter detection followed by RTMPose on an MP4 video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from custom_combat_skeleton import KEYPOINT_NAMES, SKELETON_LINKS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = PROJECT_ROOT / "weights" / "yolo_fighter_detector" / "best.pt"
DEFAULT_POSE_CONFIG = PROJECT_ROOT / "configs" / "rtmpose_s_combat.py"
SKELETON_INDICES = [
    (KEYPOINT_NAMES.index(start), KEYPOINT_NAMES.index(end))
    for start, end in SKELETON_LINKS
]
COLORS = [(52, 211, 153), (251, 146, 60)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input MP4 video")
    parser.add_argument("output", type=Path, help="Annotated output MP4")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--pose-config", type=Path, default=DEFAULT_POSE_CONFIG)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--keypoint-conf", type=float, default=0.30)
    parser.add_argument("--max-fighters", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 <= args.det_conf <= 1.0 or not 0.0 <= args.keypoint_conf <= 1.0:
        parser.error("confidence thresholds must be between zero and one")
    if args.max_fighters <= 0:
        parser.error("--max-fighters must be greater than zero")
    return args


def resolve_devices(requested: str) -> tuple[str, str]:
    import torch

    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU")
    pose_device = requested
    if requested == "cpu":
        yolo_device: str | int = "cpu"
    elif requested == "cuda":
        yolo_device = 0
        pose_device = "cuda:0"
    elif requested.startswith("cuda:"):
        yolo_device = int(requested.split(":", maxsplit=1)[1])
    else:
        raise RuntimeError(f"Unsupported device: {requested}")
    return str(yolo_device), pose_device


def draw_pose(
    frame: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    fighter_index: int,
    threshold: float,
) -> None:
    color = COLORS[fighter_index % len(COLORS)]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"fighter {fighter_index + 1}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    for start, end in SKELETON_INDICES:
        if scores[start] >= threshold and scores[end] >= threshold:
            p1 = tuple(np.rint(keypoints[start]).astype(int))
            p2 = tuple(np.rint(keypoints[end]).astype(int))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for point, score in zip(keypoints, scores):
        if score >= threshold:
            center = tuple(np.rint(point).astype(int))
            cv2.circle(frame, center, 3, (0, 255, 255), -1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    try:
        from mmpose.apis import inference_topdown, init_model
        from ultralytics import YOLO

        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        detector_path = args.detector.expanduser().resolve()
        pose_config = args.pose_config.expanduser().resolve()
        pose_checkpoint = args.pose_checkpoint.expanduser().resolve()
        for path in (input_path, detector_path, pose_config, pose_checkpoint):
            if not path.is_file():
                raise RuntimeError(f"Required file not found: {path}")

        yolo_device, pose_device = resolve_devices(args.device)
        detector = YOLO(str(detector_path))
        pose_model = init_model(str(pose_config), str(pose_checkpoint), device=pose_device)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid dimensions")
        if fps <= 0:
            fps = 30.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        written = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detection = detector.predict(
                    frame,
                    conf=args.det_conf,
                    classes=[0],
                    device=yolo_device,
                    verbose=False,
                )[0]
                boxes: list[np.ndarray] = []
                if detection.boxes is not None and len(detection.boxes):
                    xyxy = detection.boxes.xyxy.detach().cpu().numpy()
                    confidences = detection.boxes.conf.detach().cpu().numpy()
                    order = np.argsort(-confidences)[: args.max_fighters]
                    boxes = [xyxy[index] for index in order]

                if boxes:
                    box_array = np.asarray(boxes, dtype=np.float32)
                    pose_results = inference_topdown(
                        pose_model, frame, bboxes=box_array, bbox_format="xyxy"
                    )
                    for fighter_index, (bbox, sample) in enumerate(zip(box_array, pose_results)):
                        instances = sample.pred_instances
                        keypoints = np.asarray(instances.keypoints)[0]
                        if hasattr(instances, "keypoint_scores"):
                            scores = np.asarray(instances.keypoint_scores)[0]
                        else:
                            scores = np.ones(17, dtype=np.float32)
                        draw_pose(
                            frame,
                            bbox,
                            keypoints,
                            scores,
                            fighter_index,
                            args.keypoint_conf,
                        )
                writer.write(frame)
                written += 1
                if written % 100 == 0:
                    print(f"Processed {written}/{source_frames or '?'} frames")
        finally:
            capture.release()
            writer.release()

        if written == 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("No frames were read from the input video")
        if source_frames > 0 and written != source_frames:
            print(
                f"Warning: source reported {source_frames} frames, but {written} were decoded",
                file=sys.stderr,
            )
        print(f"Annotated video: {output_path}")
        print(f"Frames: {written}; resolution: {width}x{height}; FPS: {fps:.3f}")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
