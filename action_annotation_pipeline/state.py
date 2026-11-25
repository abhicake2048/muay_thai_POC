"""SQLite-backed resumability and content signatures for Data Factory stages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_signature(inputs: Sequence[Path], parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(parameters, sort_keys=True, default=str).encode("utf-8"))
    for path in sorted((item.resolve() for item in inputs), key=str):
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            stat = path.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(hash_file(path).encode("ascii"))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


class StageState:
    """Track stage status, signature, logs and outputs for one project."""

    def __init__(self, project: Path):
        self.project = project.expanduser().resolve()
        self.path = self.project / "pipeline_state.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stages (
                    stage TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    command TEXT,
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    message TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def record(self, stage: str, signature: str, status: str, *, command: str = "", outputs: Sequence[Path] = (), message: str = "") -> None:
        now = utc_now()
        with closing(self.connect()) as connection:
            previous = connection.execute("SELECT started_at FROM stages WHERE stage = ?", (stage,)).fetchone()
            started = now if status == "running" or previous is None else previous["started_at"]
            finished = now if status in {"complete", "failed", "skipped"} else None
            connection.execute(
                """
                INSERT INTO stages(stage, signature, status, started_at, finished_at, command, outputs_json, message)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stage) DO UPDATE SET
                    signature=excluded.signature, status=excluded.status,
                    started_at=excluded.started_at, finished_at=excluded.finished_at,
                    command=excluded.command, outputs_json=excluded.outputs_json, message=excluded.message
                """,
                (stage, signature, status, started, finished, command,
                 json.dumps([str(path) for path in outputs]), message),
            )
            connection.execute(
                "INSERT INTO events(stage, created_at, level, message) VALUES(?, ?, ?, ?)",
                (stage, now, "error" if status == "failed" else "info", message or status),
            )
            connection.commit()

    def is_current(self, stage: str, signature: str, outputs: Sequence[Path]) -> bool:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT signature, status FROM stages WHERE stage = ?", (stage,)).fetchone()
        return bool(
            row and row["status"] == "complete" and row["signature"] == signature
            and all(path.exists() for path in outputs)
        )

    def rows(self) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM stages ORDER BY started_at, stage")]

    @contextmanager
    def stage(self, stage: str, signature: str, *, command: str = "", outputs: Sequence[Path] = ()) -> Iterator[None]:
        self.record(stage, signature, "running", command=command, outputs=outputs)
        try:
            yield
        except Exception as exc:
            self.record(stage, signature, "failed", command=command, outputs=outputs, message=str(exc))
            raise
        self.record(stage, signature, "complete", command=command, outputs=outputs)
