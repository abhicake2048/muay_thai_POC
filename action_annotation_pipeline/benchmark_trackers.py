"""Compare tracker JSONL outputs against a small reviewed fighter-ID gold set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import PipelineError, read_jsonl, write_json


def iou(left: list[float], right: list[float]) -> float:
    x1, y1, x2, y2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-6)


def score_tracker(gold_rows: list[dict[str, Any]], predicted_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    predictions = {int(row["frame_index"]): list(row.get("detections", [])) for row in predicted_rows}
    matches: list[tuple[int, int, int, float]] = []
    total_gold = total_predictions = 0
    for row in gold_rows:
        frame = int(row["frame_index"])
        fighters = list(row.get("fighters", []))
        detections = predictions.get(frame, [])
        total_gold += len(fighters)
        total_predictions += len(detections)
        candidates = sorted(((iou(gold["bbox_xyxy"], prediction["bbox_xyxy"]), gold, prediction) for gold in fighters for prediction in detections), key=lambda item: item[0], reverse=True)
        used_gold, used_tracks = set(), set()
        for overlap, gold, prediction in candidates:
            fighter, track = int(gold["fighter_id"]), int(prediction["track_id"])
            if overlap < threshold or fighter in used_gold or track in used_tracks:
                continue
            used_gold.add(fighter); used_tracks.add(track)
            matches.append((frame, fighter, track, overlap))
    votes: dict[int, Counter[int]] = defaultdict(Counter)
    for _, fighter, track, _ in matches:
        votes[track][fighter] += 1
    mapping = {track: counts.most_common(1)[0][0] for track, counts in votes.items()}
    idtp = sum(1 for _, fighter, track, _ in matches if mapping.get(track) == fighter)
    idfp, idfn = max(0, total_predictions - idtp), max(0, total_gold - idtp)
    idf1 = 2 * idtp / max(1, 2 * idtp + idfp + idfn)
    switches = 0
    for fighter in sorted({item[1] for item in matches}):
        sequence = [track for _, _, track, _ in sorted((item for item in matches if item[1] == fighter))]
        switches += sum(left != right for left, right in zip(sequence[:-1], sequence[1:]))
    return {
        "gold_instances": total_gold, "predicted_instances": total_predictions,
        "matched_instances": len(matches), "idf1": idf1,
        "identity_switches": switches,
        "mean_iou": sum(item[3] for item in matches) / len(matches) if matches else 0.0,
        "coverage": len(matches) / total_gold if total_gold else 0.0,
        "track_to_fighter": {str(track): fighter for track, fighter in mapping.items()},
        "selection_score": idf1 - min(0.25, switches * 0.01),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=tracks.jsonl; repeat for each tracker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        gold = read_jsonl(args.gold.resolve())
        results = {}
        for value in args.candidate:
            if "=" not in value:
                raise PipelineError("--candidate must be NAME=PATH")
            name, raw_path = value.split("=", 1)
            results[name] = score_tracker(gold, read_jsonl(Path(raw_path).resolve()), args.iou)
        selected = max(results, key=lambda name: results[name]["selection_score"])
        report = {"iou_threshold": args.iou, "selected_tracker": selected, "results": results, "gold": str(args.gold.resolve())}
        write_json(args.output.resolve(), report)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
