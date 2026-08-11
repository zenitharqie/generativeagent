"""Fresh, separate, atomic publication of Phase 10 BCR judgments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

from src.evaluation.errors import (
    BCRJudgmentConflictError,
    BCRJudgmentError,
    BCRJudgmentPersistenceError,
)
from src.evaluation.judged_schemas import BCRJudgedRecord, BCRJudgmentSummary


BCR_JUDGMENT_FILENAMES = (
    "bcr_judged_records.jsonl",
    "bcr_judgment_summary.json",
)


@dataclass(frozen=True)
class JudgmentOutputPaths:
    final_directory: Path
    staging_directory: Path


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise BCRJudgmentPersistenceError(
                "judgment output path must not contain symlink components"
            )


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def prepare_judgment_output(
    output_dir: str | Path,
    *,
    source_directory: Path,
) -> JudgmentOutputPaths:
    """Reject conflicts and any output path nested in the read-only source export."""
    try:
        source_requested = Path(source_directory)
        if source_requested.is_symlink():
            raise BCRJudgmentPersistenceError(
                "source evaluation directory must not be a symlink"
            )
        source = source_requested.resolve(strict=True)
        if not source.is_dir():
            raise BCRJudgmentPersistenceError(
                "source evaluation path must be a directory"
            )
    except BCRJudgmentError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise BCRJudgmentPersistenceError(
            "could not resolve source evaluation directory"
        ) from exc

    try:
        requested = Path(output_dir)
    except (TypeError, ValueError) as exc:
        raise BCRJudgmentPersistenceError("invalid judgment output directory") from exc
    if not requested.name:
        raise BCRJudgmentPersistenceError("judgment output_dir must name a directory")
    if ".." in requested.parts:
        raise BCRJudgmentPersistenceError(
            "judgment output path traversal is not permitted"
        )

    try:
        absolute_requested = (
            requested if requested.is_absolute() else Path.cwd() / requested
        )
        _reject_symlink_components(absolute_requested.parent)
        final = absolute_requested.resolve(strict=False)
    except BCRJudgmentError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise BCRJudgmentPersistenceError(
            "could not resolve judgment output path"
        ) from exc

    parent = final.parent
    staging = parent / f".{final.name}.staging"
    if _inside(final, source) or _inside(staging, source):
        raise BCRJudgmentPersistenceError(
            "judgment output must be separate from the read-only source evaluation directory"
        )
    if _exists(absolute_requested) or _exists(final):
        raise BCRJudgmentConflictError(f"judgment output already exists: {final}")
    if _exists(staging):
        raise BCRJudgmentConflictError(f"judgment staging output already exists: {staging}")

    try:
        parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise BCRJudgmentPersistenceError(
            "could not prepare judgment output parent"
        ) from exc
    if resolved_parent != parent or _inside(resolved_parent, source):
        raise BCRJudgmentPersistenceError(
            "judgment output parent changed during validation"
        )
    if _exists(final):
        raise BCRJudgmentConflictError(f"judgment output already exists: {final}")
    if _exists(staging):
        raise BCRJudgmentConflictError(f"judgment staging output already exists: {staging}")
    return JudgmentOutputPaths(final, staging)


def _records_payload(records: tuple[BCRJudgedRecord, ...]) -> bytes:
    if not records or any(not isinstance(item, BCRJudgedRecord) for item in records):
        raise BCRJudgmentPersistenceError("judged records must be a non-empty typed tuple")
    try:
        return "".join(
            json.dumps(
                item.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in records
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BCRJudgmentPersistenceError("judged records are not strict JSON") from exc


def _summary_payload(summary: BCRJudgmentSummary) -> bytes:
    if not isinstance(summary, BCRJudgmentSummary):
        raise BCRJudgmentPersistenceError("summary must be BCRJudgmentSummary")
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
        raise BCRJudgmentPersistenceError("judgment summary is not strict JSON") from exc


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def publish_bcr_judgments(
    *,
    paths: JudgmentOutputPaths,
    records: tuple[BCRJudgedRecord, ...],
    summary: BCRJudgmentSummary,
) -> Path:
    """Write all bytes to staging, validate, and publish by one directory rename."""
    if not isinstance(paths, JudgmentOutputPaths):
        raise BCRJudgmentPersistenceError("paths must be JudgmentOutputPaths")
    if summary.records != records:
        raise BCRJudgmentPersistenceError("summary records must equal published records")
    if _exists(paths.final_directory):
        raise BCRJudgmentConflictError(
            f"judgment output already exists: {paths.final_directory}"
        )
    if _exists(paths.staging_directory):
        raise BCRJudgmentConflictError(
            f"judgment staging output already exists: {paths.staging_directory}"
        )
    payloads = {
        "bcr_judged_records.jsonl": _records_payload(records),
        "bcr_judgment_summary.json": _summary_payload(summary),
    }
    staging_created = False
    try:
        paths.staging_directory.mkdir(mode=0o700)
        staging_created = True
        for filename in BCR_JUDGMENT_FILENAMES:
            _write_new(paths.staging_directory / filename, payloads[filename])
        actual = tuple(sorted(item.name for item in paths.staging_directory.iterdir()))
        if actual != tuple(sorted(BCR_JUDGMENT_FILENAMES)):
            raise BCRJudgmentPersistenceError("staged judgment layout is invalid")
        for filename, expected in payloads.items():
            if (paths.staging_directory / filename).read_bytes() != expected:
                raise BCRJudgmentPersistenceError(
                    f"staged {filename} differs from deterministic payload"
                )
            json_text = expected.decode("utf-8")
            if not json_text.endswith("\n") or "\r" in json_text:
                raise BCRJudgmentPersistenceError(f"staged {filename} has invalid line endings")
        if _exists(paths.final_directory):
            raise BCRJudgmentConflictError(
                f"judgment output already exists: {paths.final_directory}"
            )
        paths.staging_directory.rename(paths.final_directory)
    except BaseException as exc:
        cleanup_error: OSError | None = None
        if staging_created and _exists(paths.staging_directory):
            try:
                shutil.rmtree(paths.staging_directory)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(f"staging cleanup also failed: {cleanup_error}")
        if isinstance(exc, BCRJudgmentError):
            raise
        if isinstance(exc, OSError):
            raise BCRJudgmentPersistenceError("judgment publication failed") from exc
        raise
    return paths.final_directory


__all__ = [
    "BCR_JUDGMENT_FILENAMES",
    "JudgmentOutputPaths",
    "prepare_judgment_output",
    "publish_bcr_judgments",
]
