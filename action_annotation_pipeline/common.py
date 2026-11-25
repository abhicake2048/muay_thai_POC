"""Shared helpers for the Muay Thai temporal-action annotation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent


class PipelineError(RuntimeError):
    """Raised when a project artifact is missing or inconsistent."""


def resolve_project_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise PipelineError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = read_json(config_path)
    required = (
        "project_id",
        "video_id",
        "source_video",
        "output_dir",
        "class_schema",
        "yolo_weights",
        "rtmpose_config",
        "rtmpose_checkpoint",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise PipelineError(f"Config is missing: {', '.join(missing)}")
    for key in (
        "source_video",
        "output_dir",
        "class_schema",
        "yolo_weights",
        "rtmpose_config",
        "rtmpose_checkpoint",
    ):
        config[f"{key}_resolved"] = str(resolve_project_path(config[key]))
    config["config_path"] = str(config_path)
    return config


def load_manifest(project_dir: Path) -> dict[str, Any]:
    manifest_path = project_dir.expanduser().resolve() / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["project_dir"] = str(project_dir.expanduser().resolve())
    return manifest


def load_fighter_map(project_dir: Path) -> dict[str, Any]:
    path = project_dir.expanduser().resolve() / "tracks" / "fighter_map.json"
    payload = read_json(path)
    if not isinstance(payload.get("fighters"), list):
        raise PipelineError(f"Invalid fighter map: {path}")
    return payload


def raw_track_for_fighter(fighter_map: dict[str, Any], fighter_id: int, timestamp: float) -> int | None:
    for fighter in fighter_map.get("fighters", []):
        if int(fighter.get("fighter_id", -1)) != int(fighter_id):
            continue
        for segment in fighter.get("track_segments", []):
            start = float(segment.get("start_sec", 0.0))
            end = float(segment.get("end_sec", float("inf")))
            if start <= timestamp <= end:
                return int(segment["track_id"])
    return None


def canonical_fighter_for_track(
    fighter_map: dict[str, Any], raw_track_id: int, timestamp: float
) -> int | None:
    for fighter in fighter_map.get("fighters", []):
        for segment in fighter.get("track_segments", []):
            if int(segment.get("track_id", -1)) != int(raw_track_id):
                continue
            start = float(segment.get("start_sec", 0.0))
            end = float(segment.get("end_sec", float("inf")))
            if start <= timestamp <= end:
                return int(fighter["fighter_id"])
    return None


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_covered(start: float, end: float, coverage: Iterable[tuple[float, float]], tolerance: float = 0.05) -> bool:
    return any(a - tolerance <= start and end <= b + tolerance for a, b in coverage)

