"""Conflict-safe Phase 13 experiment directories and durable publications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from src.storage.run_directory import (
    InvalidIdentifierError,
    InvalidStoragePathError,
    RunConflictError,
    StorageError,
    validate_identifier,
)


EXPERIMENT_DIRECTORY_SCHEMA_VERSION = "1.0"
EVALUATIONS_DIRECTORY_NAME = "evaluations"
EVIDENCE_DIRECTORY_NAME = "experiment_evidence"
PLAN_FILENAME = "experiment_plan.json"
ATTEMPTS_FILENAME = "provider_attempts.jsonl"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
MANIFEST_FILENAME = "experiment_manifest.json"


class ExperimentStorageError(StorageError):
    """Raised when Phase 13 persistence cannot remain safe and auditable."""


class ExperimentPathError(InvalidStoragePathError):
    """Raised when a Phase 13 path is unsafe."""


@dataclass(frozen=True)
class ExperimentPaths:
    output_root: Path
    experiment_dir: Path
    evaluations_dir: Path
    evidence_dir: Path
    plan_file: Path
    attempts_file: Path
    terminal_report_file: Path
    manifest_file: Path
    manifest_staging_file: Path

    def source_run_dir(self, run_id: str) -> Path:
        validate_identifier(run_id, field_name="run_id")
        return self.experiment_dir / run_id

    def evaluation_dir(self, run_id: str) -> Path:
        validate_identifier(run_id, field_name="run_id")
        return self.evaluations_dir / run_id


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_existing_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if _lexists(current) and current.is_symlink():
            raise ExperimentPathError("experiment path components must not be symlinks")


def build_experiment_paths(
    output_root: str | Path,
    *,
    experiment_id: str,
) -> ExperimentPaths:
    """Resolve but do not create one fresh Phase 13 output namespace."""
    try:
        experiment_id = validate_identifier(experiment_id, field_name="experiment_id")
    except InvalidIdentifierError:
        raise ExperimentPathError("experiment_id is not a safe path segment") from None
    try:
        requested = Path(output_root)
    except (TypeError, ValueError) as exc:
        raise ExperimentPathError("invalid output_root") from exc
    if not str(requested) or ".." in requested.parts:
        raise ExperimentPathError("output_root must not be empty or contain traversal")
    absolute = requested if requested.is_absolute() else Path.cwd() / requested
    _reject_existing_symlink_components(absolute)
    try:
        root = absolute.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentPathError("could not resolve output_root") from exc
    experiment = root / experiment_id
    paths = ExperimentPaths(
        output_root=root,
        experiment_dir=experiment,
        evaluations_dir=experiment / EVALUATIONS_DIRECTORY_NAME,
        evidence_dir=experiment / EVIDENCE_DIRECTORY_NAME,
        plan_file=experiment / EVIDENCE_DIRECTORY_NAME / PLAN_FILENAME,
        attempts_file=experiment / EVIDENCE_DIRECTORY_NAME / ATTEMPTS_FILENAME,
        terminal_report_file=(
            experiment / EVIDENCE_DIRECTORY_NAME / TERMINAL_REPORT_FILENAME
        ),
        manifest_file=experiment / MANIFEST_FILENAME,
        manifest_staging_file=experiment / f".{MANIFEST_FILENAME}.staging",
    )
    if len(set(paths.__dict__.values())) != len(paths.__dict__):
        raise ExperimentPathError("experiment artifact paths must be distinct")
    return paths


def validate_fresh_experiment_paths(paths: ExperimentPaths) -> None:
    """Reject output reuse, resume, overwrite, and staging conflicts."""
    if not isinstance(paths, ExperimentPaths):
        raise ExperimentPathError("paths must be ExperimentPaths")
    _reject_existing_symlink_components(paths.output_root)
    if _lexists(paths.experiment_dir):
        raise RunConflictError(
            f"experiment directory already exists: {paths.experiment_dir}"
        )
    sibling_staging = paths.output_root / f".{paths.experiment_dir.name}.staging"
    if _lexists(sibling_staging):
        raise RunConflictError(f"experiment staging path already exists: {sibling_staging}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_experiment_directory(paths: ExperimentPaths) -> ExperimentPaths:
    """Create the fresh final namespace only after all real preflights pass."""
    validate_fresh_experiment_paths(paths)
    created_experiment = False
    try:
        paths.output_root.mkdir(parents=True, exist_ok=True)
        _reject_existing_symlink_components(paths.output_root)
        resolved_root = paths.output_root.resolve(strict=True)
        if not resolved_root.is_dir() or resolved_root != paths.output_root:
            raise ExperimentPathError("resolved output_root identity changed")
        paths.experiment_dir.mkdir(mode=0o700)
        created_experiment = True
        paths.evaluations_dir.mkdir(mode=0o700)
        paths.evidence_dir.mkdir(mode=0o700)
        _create_empty_durable(paths.attempts_file)
        _fsync_directory(paths.evidence_dir)
        _fsync_directory(paths.experiment_dir)
        _fsync_directory(paths.output_root)
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        if created_experiment and _lexists(paths.experiment_dir):
            try:
                if paths.experiment_dir.is_symlink():
                    paths.experiment_dir.unlink()
                else:
                    shutil.rmtree(paths.experiment_dir)
            except BaseException as cleanup_exc:  # retain originating exception
                cleanup_error = cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(
                f"experiment directory cleanup also failed: {type(cleanup_error).__name__}"
            )
        if isinstance(exc, (ExperimentPathError, RunConflictError)):
            raise
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise ExperimentStorageError("could not create fresh experiment directory") from exc
    return paths


def _no_follow_flags(base: int) -> int:
    return base | getattr(os, "O_NOFOLLOW", 0)


def _create_empty_durable(path: Path) -> None:
    descriptor = os.open(
        path,
        _no_follow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
        0o600,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while publishing experiment artifact")
        offset += written


_FileIdentity = tuple[int, int]


def _read_regular_file_with_identity(path: Path) -> tuple[bytes, _FileIdentity]:
    descriptor = os.open(path, _no_follow_flags(os.O_RDONLY))
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ExperimentStorageError("experiment artifact is not one regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (details.st_dev, details.st_ino)
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path) -> bytes:
    payload, _ = _read_regular_file_with_identity(path)
    return payload


def _remove_owned_regular_file(path: Path, expected: _FileIdentity) -> bool:
    """Remove only the regular file whose inode this invocation published."""
    try:
        details = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or (details.st_dev, details.st_ino) != expected
    ):
        raise ExperimentStorageError(
            "published manifest ownership changed before rollback"
        )
    os.unlink(path)
    if _lexists(path):
        raise ExperimentStorageError("published manifest rollback was incomplete")
    return True


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def strict_json_mapping(payload: bytes, *, artifact_name: str) -> Mapping[str, Any]:
    try:
        if not payload.endswith(b"\n") or b"\r" in payload:
            raise ValueError("JSON must be LF-terminated")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise ExperimentStorageError(f"{artifact_name} is malformed strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ExperimentStorageError(f"{artifact_name} must contain a JSON object")
    return value


def publish_new_json(
    path: Path,
    payload: bytes,
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    """Atomically publish one new deterministic JSON evidence file."""
    if _lexists(path):
        raise RunConflictError(f"experiment artifact already exists: {path}")
    staging = path.parent / f".{path.name}.staging"
    if _lexists(staging):
        raise RunConflictError(f"experiment staging artifact already exists: {staging}")
    created = False
    try:
        descriptor = os.open(
            staging,
            _no_follow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        created = True
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        staged = _read_regular_file(staging)
        if staged != payload:
            raise ExperimentStorageError("staged experiment artifact bytes changed")
        validator(strict_json_mapping(staged, artifact_name=path.name))
        if _lexists(path):
            raise RunConflictError(f"experiment artifact already exists: {path}")
        os.replace(staging, path)
        created = False
        _fsync_directory(path.parent)
        if _read_regular_file(path) != staged:
            raise ExperimentStorageError("published experiment artifact bytes changed")
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        if created and _lexists(staging):
            try:
                staging.unlink()
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(
                f"staging cleanup also failed: {type(cleanup_error).__name__}"
            )
        if isinstance(exc, (ExperimentStorageError, RunConflictError, KeyboardInterrupt)):
            raise
        raise ExperimentStorageError("experiment artifact publication failed") from exc


def append_attempt(path: Path, payload: bytes) -> None:
    """Append one canonical attempt line durably without following links."""
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ExperimentStorageError("attempt payload must be LF-terminated")
    strict_json_mapping(payload, artifact_name=ATTEMPTS_FILENAME)
    descriptor = os.open(path, _no_follow_flags(os.O_WRONLY | os.O_APPEND))
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ExperimentStorageError("attempt stream is not one regular file")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_experiment_manifest(
    paths: ExperimentPaths,
    payload: bytes,
    *,
    validator: Callable[[Mapping[str, Any]], None],
    source_recheck: Callable[[], None],
) -> Path:
    """Read back, validate, recheck sources, and atomically publish the manifest."""
    final = paths.manifest_file
    staging = paths.manifest_staging_file
    if _lexists(final) or _lexists(staging):
        raise RunConflictError("experiment manifest or staging artifact already exists")
    staging_created = False
    owned_final_identity: _FileIdentity | None = None
    publication_acknowledged = False
    try:
        descriptor = os.open(
            staging,
            _no_follow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        staging_created = True
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        staged, staged_identity = _read_regular_file_with_identity(staging)
        if staged != payload:
            raise ExperimentStorageError("staged manifest differs from canonical bytes")
        parsed = strict_json_mapping(staged, artifact_name=MANIFEST_FILENAME)
        validator(parsed)
        source_recheck()
        if _lexists(final):
            raise RunConflictError("experiment manifest already exists")
        # Mark the validated inode as rollback-owned before entering the rename
        # syscall so an interrupt immediately after a successful rename cannot
        # leave an untracked completed manifest.
        owned_final_identity = staged_identity
        os.replace(staging, final)
        staging_created = False
        _fsync_directory(paths.experiment_dir)
        published, published_identity = _read_regular_file_with_identity(final)
        if published_identity != owned_final_identity:
            raise ExperimentStorageError("published manifest ownership changed")
        if published != staged:
            raise ExperimentStorageError("published manifest differs from validated staging")
        validator(strict_json_mapping(published, artifact_name=MANIFEST_FILENAME))
        publication_acknowledged = True
    except BaseException as exc:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if owned_final_identity is not None and not publication_acknowledged:
            try:
                removed = _remove_owned_regular_file(final, owned_final_identity)
                if removed:
                    _fsync_directory(paths.experiment_dir)
            except BaseException as cleanup_exc:
                cleanup_errors.append(("owned final rollback", cleanup_exc))
        if staging_created and _lexists(staging):
            try:
                staging.unlink()
            except BaseException as cleanup_exc:
                cleanup_errors.append(("manifest staging cleanup", cleanup_exc))

        cleanup_notes = tuple(
            f"{label} also failed: {type(error).__name__}"
            for label, error in cleanup_errors
        )
        if isinstance(exc, (ExperimentStorageError, RunConflictError, KeyboardInterrupt)):
            if hasattr(exc, "add_note"):
                for note in cleanup_notes:
                    exc.add_note(note)
            raise
        error = ExperimentStorageError("experiment manifest publication failed")
        if hasattr(error, "add_note"):
            for note in cleanup_notes:
                error.add_note(note)
        raise error from exc
    return final


__all__ = [
    "ATTEMPTS_FILENAME",
    "EVALUATIONS_DIRECTORY_NAME",
    "EVIDENCE_DIRECTORY_NAME",
    "EXPERIMENT_DIRECTORY_SCHEMA_VERSION",
    "ExperimentPathError",
    "ExperimentPaths",
    "ExperimentStorageError",
    "MANIFEST_FILENAME",
    "PLAN_FILENAME",
    "TERMINAL_REPORT_FILENAME",
    "append_attempt",
    "build_experiment_paths",
    "create_experiment_directory",
    "publish_experiment_manifest",
    "publish_new_json",
    "strict_json_mapping",
    "validate_fresh_experiment_paths",
]
