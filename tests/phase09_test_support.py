"""Read-only and copied-artifact helpers for Phase 09 tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any


def copy_run(source_run: Path, root: Path) -> Path:
    """Copy a run while preserving path identity expected by persisted records."""
    destination = root / source_run.parent.name / source_run.name
    destination.parent.mkdir(parents=True)
    shutil.copytree(source_run, destination, copy_function=os.link)
    return destination


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    write_bytes(path, payload.encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    write_bytes(path, payload.encode("utf-8"))


def write_bytes(path: Path, payload: bytes) -> None:
    """Detach a hard-linked copied fixture before replacing its bytes."""
    path.unlink()
    path.write_bytes(payload)


def artifact_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
