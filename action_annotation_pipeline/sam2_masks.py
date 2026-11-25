"""Optional SAM2.1 box-prompt propagation for fighter identity masks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from common import PipelineError, canonical_fighter_for_track, load_fighter_map, load_manifest, read_jsonl, write_jsonl


def _rle(mask: np.ndarray) -> dict:
    pixels = mask.astype(np.uint8).flatten(order="F")
    counts, previous, run = [], 0, 0
    for pixel in pixels:
        value = int(pixel)
        if value == previous:
            run += 1
        else:
            counts.append(run)
            run, previous = 1, value
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        project = args.project.expanduser().resolve()
        output = project / "masks" / "sam2_masks.jsonl"
        if output.exists() and output.stat().st_size and not args.overwrite:
            raise PipelineError(f"Output exists: {output}; use --overwrite")
        manifest = load_manifest(project)
        video = Path(manifest["paths"]["proxy_video"])
        tracks = read_jsonl(project / "tracks" / "tracks.jsonl")
        fighter_map = load_fighter_map(project)
        fps = float(manifest["proxy_video"]["fps"])
        prompts = {}
        for row in tracks:
            frame = int(row["frame_index"])
            for detection in row.get("detections", []):
                fighter = canonical_fighter_for_track(fighter_map, int(detection["track_id"]), frame / fps)
                if fighter is not None and fighter not in prompts:
                    prompts[fighter] = (frame, np.asarray(detection["bbox_xyxy"], dtype=np.float32))
            if len(prompts) >= int(manifest.get("expected_fighters", 2)):
                break
        if not prompts:
            raise PipelineError("No fighter prompt boxes were found")
        with tempfile.TemporaryDirectory(prefix="mtdata_sam2_") as temporary:
            frames_dir = Path(temporary)
            capture = cv2.VideoCapture(str(video))
            frame_index = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok or (args.max_frames and frame_index >= args.max_frames):
                        break
                    cv2.imwrite(str(frames_dir / f"{frame_index:08d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    frame_index += 1
            finally:
                capture.release()
            predictor = build_sam2_video_predictor(str(args.config.resolve()), str(args.checkpoint.resolve()), device=args.device)
            state = predictor.init_state(video_path=str(frames_dir))
            for fighter, (prompt_frame, box) in prompts.items():
                if prompt_frame < frame_index:
                    predictor.add_new_points_or_box(inference_state=state, frame_idx=prompt_frame, obj_id=int(fighter), box=box)
            rows = []
            context = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else torch.inference_mode()
            with torch.inference_mode(), context:
                for out_frame, object_ids, mask_logits in predictor.propagate_in_video(state):
                    objects = []
                    for index, object_id in enumerate(object_ids):
                        logits = mask_logits[index]
                        mask = (logits > 0.0).detach().cpu().numpy().squeeze()
                        objects.append({"fighter_id": int(object_id), "segmentation": _rle(mask), "mask_score": float(torch.sigmoid(logits).mean().item())})
                    rows.append({"frame_index": int(out_frame), "timestamp_sec": int(out_frame) / fps, "objects": objects})
            write_jsonl(output, rows)
        print(json.dumps({"frames": len(rows), "fighters": sorted(prompts), "output": str(output)}, indent=2))
        return 0
    except ImportError as exc:
        print(f"Error: install SAM2: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
