"""Migrate reviewed v1 JSONL annotations to the factorized v2 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .common import PipelineError, read_json, read_jsonl, write_jsonl
    from .schema import canonicalize_event
except ImportError:
    from common import PipelineError, read_json, read_jsonl, write_jsonl
    from schema import canonicalize_event


def migrate_rows(rows: list[dict], schema: dict) -> list[dict]:
    return [canonicalize_event(row, schema) for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("classes.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source, destination = args.input.resolve(), args.output.resolve()
        if destination.exists() and not args.overwrite:
            raise PipelineError(f"Output exists: {destination}; use --overwrite")
        rows = migrate_rows(read_jsonl(source), read_json(args.schema.resolve()))
        write_jsonl(destination, rows)
        print(json.dumps({"input": str(source), "output": str(destination), "events": len(rows)}, indent=2))
        return 0
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
