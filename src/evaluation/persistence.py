"""Atomic deterministic publication of Phase 09 evaluation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from src.evaluation.errors import (
    EvaluationConflictError,
    EvaluationError,
    EvaluationPersistenceError,
)
from src.evaluation.schemas import (
    BCRExportRecord,
    MRAEvaluationResult,
    RunEvaluationSummary,
)


EVALUATION_FILENAMES = (
    "mra_records.jsonl",
    "bcr_export.jsonl",
    "evaluation_summary.json",
)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _jsonl_payload(records: tuple[object, ...], expected_type: type[object]) -> bytes:
    lines: list[str] = []
    for record in records:
        if not isinstance(record, expected_type):
            raise EvaluationPersistenceError(
                f"expected {expected_type.__name__}, got {type(record).__name__}"
            )
        value = record.to_dict()
        try:
            line = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationPersistenceError(
                f"evaluation record is not strict JSON: {exc}"
            ) from exc
        lines.append(line + "\n")
    if not lines:
        raise EvaluationPersistenceError("evaluation JSONL output must not be empty")
    return "".join(lines).encode("utf-8")


def _summary_payload(summary: RunEvaluationSummary) -> bytes:
    if not isinstance(summary, RunEvaluationSummary):
        raise EvaluationPersistenceError(
            "summary must be RunEvaluationSummary"
        )
    try:
        return (
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationPersistenceError(
            f"evaluation summary is not strict JSON: {exc}"
        ) from exc


def _validate_mra_summary_eligibility(
    records: tuple[MRAEvaluationResult, ...],
    summary: RunEvaluationSummary,
) -> None:
    expected = (
        summary.evaluation_mode,
        summary.primary_eligible,
        summary.included_in_final_research_aggregation,
    )
    for record in records:
        actual = (
            record.evaluation_mode,
            record.primary_eligible,
            record.included_in_final_research_aggregation,
        )
        if actual != expected:
            raise EvaluationPersistenceError(
                "MRA record eligibility metadata does not match evaluation summary"
            )


def _write_new_file(path: Path, payload: bytes) -> None:
    """Write and sync one new file; exposed privately for failure-boundary tests."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_jsonl_file(path: Path, expected: bytes) -> None:
    actual = path.read_bytes()
    if actual != expected:
        raise EvaluationPersistenceError(
            f"staged {path.name} bytes differ from deterministic payload"
        )
    try:
        text = actual.decode("utf-8")
        lines = text.splitlines(keepends=True)
        if not lines or any(not line.endswith("\n") or line.endswith("\r\n") for line in lines):
            raise ValueError("records must be LF-terminated")
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("each JSONL line must contain an object")
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationPersistenceError(
            f"staged {path.name} failed strict validation: {exc}"
        ) from exc


def _validate_summary_file(path: Path, expected: bytes) -> None:
    actual = path.read_bytes()
    if actual != expected:
        raise EvaluationPersistenceError(
            "staged evaluation_summary.json bytes differ from deterministic payload"
        )
    try:
        value: Any = json.loads(actual.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationPersistenceError(
            f"staged evaluation_summary.json is invalid: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise EvaluationPersistenceError(
            "staged evaluation_summary.json must contain an object"
        )


def publish_evaluation(
    *,
    output_dir: str | Path,
    mra_records: tuple[MRAEvaluationResult, ...],
    bcr_records: tuple[BCRExportRecord, ...],
    summary: RunEvaluationSummary,
) -> Path:
    """Stage, validate, and atomically publish one fresh evaluation directory."""
    try:
        requested = Path(output_dir)
    except (TypeError, ValueError) as exc:
        raise EvaluationPersistenceError(f"invalid output_dir: {output_dir!r}") from exc
    if not requested.name:
        raise EvaluationPersistenceError("output_dir must name a new directory")
    parent = requested.parent
    try:
        if parent.is_symlink():
            raise EvaluationPersistenceError(
                f"evaluation output parent must not be a symlink: {parent}"
            )
        parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = parent.resolve(strict=True)
    except EvaluationPersistenceError:
        raise
    except (OSError, ValueError) as exc:
        raise EvaluationPersistenceError(
            f"could not prepare evaluation output parent {parent}: {exc}"
        ) from exc
    final_dir = resolved_parent / requested.name
    staging_dir = resolved_parent / f".{requested.name}.staging"
    if _path_exists(final_dir):
        raise EvaluationConflictError(
            f"evaluation output already exists: {final_dir}"
        )
    if _path_exists(staging_dir):
        raise EvaluationConflictError(
            f"evaluation staging output already exists: {staging_dir}"
        )

    mra_payload = _jsonl_payload(mra_records, MRAEvaluationResult)
    bcr_payload = _jsonl_payload(bcr_records, BCRExportRecord)
    summary_payload = _summary_payload(summary)
    _validate_mra_summary_eligibility(mra_records, summary)
    payloads = {
        "mra_records.jsonl": mra_payload,
        "bcr_export.jsonl": bcr_payload,
        "evaluation_summary.json": summary_payload,
    }

    staging_created = False
    try:
        staging_dir.mkdir(mode=0o700)
        staging_created = True
        for filename in EVALUATION_FILENAMES:
            _write_new_file(staging_dir / filename, payloads[filename])
        actual_files = tuple(sorted(path.name for path in staging_dir.iterdir()))
        if actual_files != tuple(sorted(EVALUATION_FILENAMES)):
            raise EvaluationPersistenceError(
                "staged evaluation directory has an unexpected file layout"
            )
        _validate_jsonl_file(staging_dir / "mra_records.jsonl", mra_payload)
        _validate_jsonl_file(staging_dir / "bcr_export.jsonl", bcr_payload)
        _validate_summary_file(
            staging_dir / "evaluation_summary.json",
            summary_payload,
        )
        if _path_exists(final_dir):
            raise EvaluationConflictError(
                f"evaluation output already exists: {final_dir}"
            )
        staging_dir.rename(final_dir)
    except BaseException as exc:
        cleanup_error: OSError | None = None
        if staging_created and _path_exists(staging_dir):
            try:
                shutil.rmtree(staging_dir)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(f"staging cleanup also failed: {cleanup_error}")
        if isinstance(exc, EvaluationError):
            raise
        if isinstance(exc, OSError):
            raise EvaluationPersistenceError(
                f"evaluation publication failed: {exc}"
            ) from exc
        raise
    return final_dir


__all__ = ["EVALUATION_FILENAMES", "publish_evaluation"]
