"""Mine review-required YOLO and RTMPose feedback samples from processed videos."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import PipelineError, load_manifest, read_jsonl, write_json, write_jsonl


COCO_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--max-yolo", type=int, default=200)
    parser.add_argument("--max-rtmpose", type=int, default=200)
    parser.add_argument("--minimum-gap", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _spaced(rows: list[tuple[float, int, Any]], limit: int, gap_frames: int) -> list[tuple[float, int, Any]]:
    chosen: list[tuple[float, int, Any]] = []
    for row in sorted(rows, key=lambda item: item[0], reverse=True):
        if all(abs(row[1] - other[1]) >= gap_frames for other in chosen):
            chosen.append(row)
            if len(chosen) >= limit:
                break
    return sorted(chosen, key=lambda item: item[1])


def _extract_frames(video: Path, requests: dict[int, list[Path]]) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video: {video}")
    try:
        for frame_index in sorted(requests):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for path in requests[frame_index]:
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        capture.release()


def main() -> int:
    args = parse_args()
    try:
        project = args.project.expanduser().resolve()
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        tracks = read_jsonl(project / "tracks" / "tracks.jsonl")
        poses_path = project / "poses" / "poses.npz"
        if not poses_path.is_file():
            raise PipelineError(f"Pose file not found: {poses_path}")
        yolo_root = project / "exports" / "yolo_feedback"
        pose_root = project / "exports" / "rtmpose_feedback"
        outputs = [yolo_root / "review_manifest.jsonl", pose_root / "annotations_preliminary.json"]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise PipelineError("Feedback outputs exist; use --overwrite")
        fps = float(manifest["proxy_video"]["fps"])
        width, height = int(manifest["proxy_video"]["width"]), int(manifest["proxy_video"]["height"])
        gap_frames = max(1, int(round(args.minimum_gap * fps)))

        detection_by_frame: dict[int, list[dict[str, Any]]] = {}
        yolo_candidates = []
        for row in tracks:
            frame = int(row["frame_index"])
            detections = list(row.get("detections", []))
            detection_by_frame[frame] = detections
            confidences = [float(item.get("confidence", 0.0)) for item in detections]
            count_penalty = abs(len(detections) - int(manifest.get("expected_fighters", 2)))
            overlap = max((_iou(a["bbox_xyxy"], b["bbox_xyxy"]) for index, a in enumerate(detections) for b in detections[index + 1:]), default=0.0)
            score = count_penalty * 2.0 + (1.0 - min(confidences, default=0.0)) + overlap
            yolo_candidates.append((score, frame, {"detections": detections, "reasons": {"count_mismatch": count_penalty, "low_confidence": round(1.0 - min(confidences, default=0.0), 4), "overlap": round(overlap, 4)}}))
        selected_yolo = _spaced(yolo_candidates, args.max_yolo, gap_frames)

        pose_data = np.load(poses_path)
        teacher_sets = [np.load(path) for path in sorted((project / "poses" / "teachers").glob("*.npz"))]
        teacher_lookups = [{
            (int(frame), int(track)): index
            for index, (frame, track) in enumerate(zip(data["frame_indices"], data["track_ids"]))
        } for data in teacher_sets]
        pose_by_frame: dict[int, list[int]] = {}
        for index, frame in enumerate(pose_data["frame_indices"]):
            pose_by_frame.setdefault(int(frame), []).append(index)
        peak_frames = set()
        events_path = project / "annotations" / "events.jsonl"
        if events_path.is_file() and events_path.stat().st_size:
            for event in read_jsonl(events_path):
                if event.get("event_type") == "action":
                    peak_frames.add(int(round(float(event.get("peak_sec", 0.0)) * fps)))
        proposal_path = project / "proposals" / "fused_proposals.jsonl"
        if proposal_path.is_file() and proposal_path.stat().st_size:
            for event in read_jsonl(proposal_path):
                peak_frames.add(int(round(float(event.get("peak_sec", 0.0)) * fps)))
        pose_candidates = []
        for frame, indices in pose_by_frame.items():
            mean_confidence = float(np.mean(pose_data["keypoint_scores"][indices]))
            low_joint = float(np.mean(pose_data["keypoint_scores"][indices] < 0.2))
            disagreements = []
            for primary_index in indices:
                track = int(pose_data["track_ids"][primary_index])
                box = pose_data["bboxes_xyxy"][primary_index]
                diagonal = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
                for teacher_data, lookup in zip(teacher_sets, teacher_lookups):
                    teacher_index = lookup.get((frame, track))
                    if teacher_index is None:
                        continue
                    valid = (pose_data["keypoint_scores"][primary_index] >= 0.2) & (teacher_data["keypoint_scores"][teacher_index] >= 0.2)
                    if np.any(valid):
                        disagreements.append(float(np.mean(np.linalg.norm(pose_data["keypoints"][primary_index][valid] - teacher_data["keypoints"][teacher_index][valid], axis=1)) / diagonal))
            disagreement = max(disagreements, default=0.0)
            score = (1.0 - mean_confidence) + low_joint + min(1.0, disagreement * 5.0) + (1.0 if frame in peak_frames else 0.0)
            pose_candidates.append((score, frame, {"indices": indices, "action_peak": frame in peak_frames, "mean_keypoint_confidence": round(mean_confidence, 4), "low_joint_fraction": round(low_joint, 4), "teacher_disagreement": round(disagreement, 6)}))
        selected_pose = _spaced(pose_candidates, args.max_rtmpose, gap_frames)

        requests: dict[int, list[Path]] = {}
        yolo_manifest = []
        for score, frame, details in selected_yolo:
            name = f"{manifest['video_id']}__frame_{frame:08d}.jpg"
            image_path = yolo_root / "images" / name
            label_path = yolo_root / "labels_preliminary" / f"{Path(name).stem}.txt"
            requests.setdefault(frame, []).append(image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            labels = []
            for detection in details["detections"]:
                x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
                labels.append(f"0 {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            yolo_manifest.append({"image": str(image_path), "preliminary_label": str(label_path), "frame_index": frame, "timestamp_sec": frame / fps, "selection_score": round(score, 6), "selection_reasons": details["reasons"], "review_status": "preannotation", "source_group": manifest["video_id"]})
        write_jsonl(yolo_root / "review_manifest.jsonl", yolo_manifest)
        write_json(yolo_root / "data_preliminary.yaml.json", {"names": {"0": "fighter"}, "review_required": True, "source_group": manifest["video_id"]})

        coco_images, coco_annotations = [], []
        annotation_id = 1
        pose_manifest = []
        for image_id, (score, frame, details) in enumerate(selected_pose, start=1):
            name = f"{manifest['video_id']}__frame_{frame:08d}.jpg"
            image_path = pose_root / "images" / name
            requests.setdefault(frame, []).append(image_path)
            coco_images.append({"id": image_id, "file_name": f"images/{name}", "width": width, "height": height, "source_group": manifest["video_id"], "frame_index": frame})
            for index in details["indices"]:
                points = pose_data["keypoints"][index]
                keypoint_scores = pose_data["keypoint_scores"][index]
                box = pose_data["bboxes_xyxy"][index]
                keypoints = []
                for point, confidence in zip(points, keypoint_scores):
                    visibility = 2 if confidence >= 0.2 else 1 if confidence > 0 else 0
                    keypoints.extend([round(float(point[0]), 3), round(float(point[1]), 3), visibility])
                coco_annotations.append({
                    "id": annotation_id, "image_id": image_id, "category_id": 1,
                    "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    "area": float((box[2]-box[0])*(box[3]-box[1])), "iscrowd": 0,
                    "keypoints": keypoints, "num_keypoints": int(np.sum(keypoint_scores >= 0.2)),
                    "track_id": int(pose_data["track_ids"][index]), "review_status": "preannotation",
                })
                annotation_id += 1
            pose_manifest.append({"image": str(image_path), "frame_index": frame, "timestamp_sec": frame / fps, "selection_score": round(score, 6), "action_peak": details["action_peak"], "teacher_disagreement": details["teacher_disagreement"], "review_status": "preannotation", "source_group": manifest["video_id"]})
        write_json(pose_root / "annotations_preliminary.json", {
            "info": {"description": "RTMPose preannotations; human review required", "schema_version": "2.0"},
            "images": coco_images, "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "fighter", "supercategory": "person", "keypoints": COCO_KEYPOINTS, "skeleton": COCO_SKELETON}],
        })
        write_jsonl(pose_root / "review_manifest.jsonl", pose_manifest)
        _extract_frames(video, requests)
        summary = {"yolo_feedback_frames": len(selected_yolo), "rtmpose_feedback_frames": len(selected_pose), "human_review_required": True, "source_group": manifest["video_id"]}
        write_json(project / "qa" / "feedback_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
