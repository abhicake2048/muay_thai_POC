"""Create the sparring-video temporal annotation workspace and proxy video."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from common import PipelineError, load_config, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inspect_video(path: Path) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise PipelineError(f"Invalid video metadata: {path}")
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_sec": frames / fps,
    }


def create_proxy(source: Path, destination: Path, fps: float, width: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise PipelineError("ffmpeg is required to create the annotation proxy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"fps={fps},scale={width}:-2:flags=lanczos",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 or not destination.is_file():
        raise PipelineError("ffmpeg failed to create the annotation proxy")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        source = Path(config["source_video_resolved"])
        project_dir = Path(config["output_dir_resolved"])
        if not source.is_file():
            raise PipelineError(f"Source video not found: {source}")
        for key in ("class_schema_resolved", "yolo_weights_resolved", "rtmpose_config_resolved", "rtmpose_checkpoint_resolved"):
            if not Path(config[key]).is_file():
                raise PipelineError(f"Required file not found: {config[key]}")

        if project_dir.exists() and any(project_dir.iterdir()):
            if not args.overwrite:
                raise PipelineError(
                    f"Project already exists: {project_dir}. Use --overwrite only to rebuild it."
                )
            shutil.rmtree(project_dir)

        for relative in (
            "video",
            "tracks",
            "masks",
            "poses",
            "poses/teachers",
            "proposals",
            "proposals/branches",
            "annotations",
            "exports/posec3d",
            "exports/tridet",
            "exports/yolo_feedback",
            "exports/rtmpose_feedback",
            "features",
            "review",
            "qa",
            "logs",
        ):
            (project_dir / relative).mkdir(parents=True, exist_ok=True)

        source_meta = inspect_video(source)
        proxy = project_dir / "video" / f"{config['video_id']}_30fps.mp4"
        if args.skip_proxy:
            proxy = source
        else:
            create_proxy(
                source,
                proxy,
                float(config.get("proxy_fps", 30.0)),
                int(config.get("proxy_width", source_meta["width"])),
            )
        proxy_meta = inspect_video(proxy)

        timestamp_map = {
            "version": "2.0",
            "mapping": "constant_rate_formula",
            "source_fps": float(source_meta["fps"]),
            "source_frame_count": int(source_meta["frame_count"]),
            "proxy_fps": float(proxy_meta["fps"]),
            "proxy_frame_count": int(proxy_meta["frame_count"]),
            "proxy_to_source": "source_frame = round(proxy_frame * source_fps / proxy_fps)",
            "source_to_proxy": "proxy_frame = round(source_frame * proxy_fps / source_fps)",
            "timestamp_rule": "timestamp_sec = frame_index / stream_fps",
            "maximum_rounding_error_sec": 0.5 / min(float(source_meta["fps"]), float(proxy_meta["fps"])),
        }
        write_json(project_dir / "video" / "timestamp_map.json", timestamp_map)

        class_schema = read_json(Path(config["class_schema_resolved"]))
        write_json(project_dir / "annotations" / "classes.json", class_schema)
        shutil.copy2(Path(__file__).parent / "event.schema.json", project_dir / "annotations" / "event.schema.json")
        shutil.copy2(Path(__file__).parent / "ANNOTATION_GUIDE.md", project_dir / "annotations" / "ANNOTATION_GUIDE.md")
        (project_dir / "annotations" / "events.jsonl").write_text("", encoding="utf-8")
        (project_dir / "proposals" / "action_proposals.jsonl").write_text("", encoding="utf-8")
        (project_dir / "proposals" / "fused_proposals.jsonl").write_text("", encoding="utf-8")
        shutil.copy2(Path(__file__).parent / "annotator" / "index.html", project_dir / "review" / "annotate.html")

        manifest = {
            "schema_version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": config["project_id"],
            "video_id": config["video_id"],
            "subset": config.get("subset", "training"),
            "expected_fighters": int(config.get("expected_fighters", 2)),
            "source_video": source_meta,
            "proxy_video": proxy_meta,
            "paths": {
                "project_dir": str(project_dir),
                "source_video": str(source),
                "proxy_video": str(proxy),
                "class_schema": str(project_dir / "annotations" / "classes.json"),
                "event_schema": str(project_dir / "annotations" / "event.schema.json"),
                "events": str(project_dir / "annotations" / "events.jsonl"),
                "timestamp_map": str(project_dir / "video" / "timestamp_map.json"),
                "yolo_weights": config["yolo_weights_resolved"],
                "rtmpose_config": config["rtmpose_config_resolved"],
                "rtmpose_checkpoint": config["rtmpose_checkpoint_resolved"],
            },
            "thresholds": {
                "detector_confidence": float(config.get("detector_confidence", 0.25)),
                "keypoint_confidence": float(config.get("keypoint_confidence", 0.2)),
            },
            "notes": config.get("notes", ""),
            "pipeline": {
                "proxy_fps": float(proxy_meta["fps"]),
                "feature_stride_frames": int(config.get("feature_stride_frames", 4)),
                "vlm_local_only": True,
                "human_verification_required": True
            },
        }
        write_json(project_dir / "manifest.json", manifest)
        shutil.copy2(args.config.resolve(), project_dir / "project_config.json")

        project_readme = f"""# {config['project_id']}

Source: `{source}`

Annotation proxy: `{proxy}`

1. Run `python mtdata.py infer {project_dir}` from the repository root.
2. Run `python mtdata.py review {project_dir}` and review every proposal.
3. Export the browser/FiftyOne result and run `mtdata sync-review`.
4. Run `mtdata export` after validation.

Predictions are proposals, not ground truth. Verify technique, anatomical side,
effector, stance role, target, outcome, actor and temporal boundaries.
"""
        (project_dir / "README.md").write_text(project_readme, encoding="utf-8")
        print(json.dumps({"project_dir": str(project_dir), "proxy": str(proxy), "manifest": str(project_dir / 'manifest.json')}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
