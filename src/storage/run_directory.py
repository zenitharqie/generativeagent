"""Safe creation of isolated research run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Final

from src.config_loader import loader as config_loader
from src.config_loader.validators import ConfigValidationError
from src.logging_system.run_summary import (
    RunSummary,
    RunSummaryError,
    read_run_summary,
    write_run_summary,
)


_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
)

_JSONL_FILENAMES: Final[tuple[str, ...]] = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)


class StorageError(RuntimeError):
    """Base error for run storage operations."""


class InvalidStoragePathError(StorageError):
    """Raised when a storage path is invalid or unsafe."""


class InvalidIdentifierError(StorageError, ValueError):
    """Raised when a caller-supplied identifier is not a safe path segment."""


class RunConflictError(StorageError):
    """Raised when a requested run or its staging directory already exists."""


@dataclass(frozen=True)
class RunPaths:
    """All canonical paths belonging to one isolated run."""

    output_root: Path
    experiment_dir: Path
    run_dir: Path
    config_snapshot: Path
    run_summary: Path
    retrieval_logs: Path
    behavior_logs: Path
    memory_logs: Path
    reflection_logs: Path
    errors: Path
    event_summaries: Path
    natural_block_summaries: Path
    api_usage_logs: Path

    @property
    def jsonl_files(self) -> tuple[Path, ...]:
        """Return every append-only log path in canonical creation order."""
        return (
            self.retrieval_logs,
            self.behavior_logs,
            self.memory_logs,
            self.reflection_logs,
            self.errors,
            self.event_summaries,
            self.natural_block_summaries,
            self.api_usage_logs,
        )


def validate_identifier(value: str, *, field_name: str) -> str:
    """Validate that an identifier is one conservative, portable path segment."""
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise InvalidIdentifierError(
            f"{field_name} must match {_IDENTIFIER_PATTERN.pattern!r}"
        )
    return value


def create_run_directory(
    output_root: str | Path,
    experiment_id: str,
    run_id: str,
    *,
    config_directory: str | Path = config_loader.DEFAULT_CONFIG_DIRECTORY,
    retrieval_config_id: str = "balanced",
    repetition_index: int = 0,
) -> RunPaths:
    """Initialize and atomically publish a complete fresh run directory.

    The caller owns all identifiers. Existing final or staging paths are always
    conflicts: this function never overwrites, resumes, repairs, or truncates a
    run. The final path remains absent until the validated config snapshot,
    every required JSONL stream, and an initialized run summary are complete.
    """
    experiment_id = validate_identifier(experiment_id, field_name="experiment_id")
    run_id = validate_identifier(run_id, field_name="run_id")
    initialized_summary = RunSummary(
        experiment_id=experiment_id,
        run_id=run_id,
        retrieval_config_id=retrieval_config_id,
        repetition_index=repetition_index,
    )
    root = _prepare_output_root(output_root)
    experiment_dir = _prepare_experiment_directory(root, experiment_id)

    run_dir = experiment_dir / run_id
    staging_dir = experiment_dir / f".{run_id}.staging"
    if _path_exists(run_dir):
        raise RunConflictError(f"run directory already exists: {run_dir}")
    if _path_exists(staging_dir):
        raise RunConflictError(f"run staging directory already exists: {staging_dir}")

    staging_created = False
    try:
        staging_dir.mkdir(mode=0o700)
        staging_created = True
        staging_paths = _build_run_paths(root, experiment_dir, staging_dir)

        # Local import avoids a module cycle: config_snapshot's public error
        # type derives from StorageError in this module.
        from src.storage.config_snapshot import copy_config_snapshot

        copy_config_snapshot(config_directory, staging_paths.config_snapshot)
        for path in staging_paths.jsonl_files:
            _create_empty_file(path)

        write_run_summary(staging_paths.run_summary, initialized_summary)
        _validate_staged_run(staging_paths, initialized_summary)

        # Concurrent writers are out of scope, but this second check prevents
        # an ordinary pre-existing target from ever being replaced.
        if _path_exists(run_dir):
            raise RunConflictError(f"run directory already exists: {run_dir}")
        _publish_staging_directory(staging_dir, run_dir)
    except BaseException as exc:
        cleanup_error = (
            _cleanup_staging_directory(staging_dir) if staging_created else None
        )
        if cleanup_error is not None:
            raise StorageError(
                f"run initialization failed for {run_dir}: {exc}; "
                f"staging cleanup also failed: {cleanup_error}"
            ) from exc
        if isinstance(exc, OSError):
            raise StorageError(
                f"could not initialize run directory {run_dir}: {exc}"
            ) from exc
        raise

    return _build_run_paths(root, experiment_dir, run_dir)


def _prepare_output_root(output_root: str | Path) -> Path:
    if isinstance(output_root, str) and not output_root:
        raise InvalidStoragePathError("output_root must not be empty")
    try:
        path = Path(output_root)
    except (TypeError, ValueError) as exc:
        raise InvalidStoragePathError(f"invalid output_root: {output_root!r}") from exc

    try:
        if path.is_symlink():
            raise InvalidStoragePathError(
                f"output_root must not be a symlink: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
    except InvalidStoragePathError:
        raise
    except (OSError, ValueError) as exc:
        raise InvalidStoragePathError(f"could not create output_root {path}: {exc}") from exc
    if not resolved.is_dir():
        raise InvalidStoragePathError(f"output_root is not a directory: {resolved}")
    return resolved


def _prepare_experiment_directory(root: Path, experiment_id: str) -> Path:
    path = root / experiment_id
    if path.is_symlink():
        raise InvalidStoragePathError(
            f"experiment directory must not be a symlink: {path}"
        )
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise InvalidStoragePathError(
            f"could not create experiment directory {path}: {exc}"
        ) from exc
    if not resolved.is_dir() or resolved.parent != root:
        raise InvalidStoragePathError(
            f"experiment directory escapes output_root: {path}"
        )
    return resolved


def _create_empty_file(path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n"):
        pass


def _validate_staged_run(paths: RunPaths, expected_summary: RunSummary) -> None:
    """Reject publication unless every staged initialization artifact is valid."""
    snapshot = paths.config_snapshot
    expected_config_names = tuple(config_loader.CONFIG_FILES.values())
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise StorageError("staged config_snapshot is missing or invalid")
    try:
        actual_config_names = tuple(sorted(path.name for path in snapshot.iterdir()))
    except OSError as exc:
        raise StorageError(f"could not inspect staged config_snapshot: {exc}") from exc
    if actual_config_names != tuple(sorted(expected_config_names)):
        raise StorageError(
            "staged config_snapshot does not contain the exact canonical config files"
        )
    if any(
        not (snapshot / filename).is_file() or (snapshot / filename).is_symlink()
        for filename in expected_config_names
    ):
        raise StorageError("staged config_snapshot contains a missing or invalid file")
    try:
        config = config_loader.load_experiment_config(snapshot)
    except (config_loader.ConfigLoadError, ConfigValidationError) as exc:
        raise StorageError(f"staged config_snapshot is not valid: {exc}") from exc
    if expected_summary.retrieval_config_id not in {
        retrieval_config.config_id for retrieval_config in config.retrieval_configs
    }:
        raise StorageError(
            "initialized run summary references an unknown retrieval configuration"
        )

    for path in paths.jsonl_files:
        if path.is_symlink() or not path.is_file():
            raise StorageError(
                f"required staged JSONL file is missing or invalid: {path.name}"
            )
        try:
            if path.stat().st_size != 0:
                raise StorageError(
                    f"required staged JSONL file is not empty: {path.name}"
                )
        except OSError as exc:
            raise StorageError(
                f"could not inspect required staged JSONL file {path.name}: {exc}"
            ) from exc

    try:
        persisted_summary = read_run_summary(paths.run_summary)
    except RunSummaryError as exc:
        raise StorageError(f"staged run_summary.json is invalid: {exc}") from exc
    if (
        persisted_summary != expected_summary
        or persisted_summary.status != "initialized"
    ):
        raise StorageError(
            "staged run_summary.json is not the expected initialized summary"
        )


def _publish_staging_directory(staging_dir: Path, run_dir: Path) -> None:
    """Atomically publish one fully validated sibling staging directory."""
    staging_dir.rename(run_dir)


def _cleanup_staging_directory(path: Path) -> OSError | None:
    if not _path_exists(path):
        return None
    try:
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)
    except OSError as exc:
        return exc
    return None


def _path_exists(path: Path) -> bool:
    """Include broken symlinks in conflict detection."""
    return path.exists() or path.is_symlink()


def _build_run_paths(root: Path, experiment_dir: Path, run_dir: Path) -> RunPaths:
    return RunPaths(
        output_root=root,
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        config_snapshot=run_dir / "config_snapshot",
        run_summary=run_dir / "run_summary.json",
        retrieval_logs=run_dir / "retrieval_logs.jsonl",
        behavior_logs=run_dir / "behavior_logs.jsonl",
        memory_logs=run_dir / "memory_logs.jsonl",
        reflection_logs=run_dir / "reflection_logs.jsonl",
        errors=run_dir / "errors.jsonl",
        event_summaries=run_dir / "event_summaries.jsonl",
        natural_block_summaries=run_dir / "natural_block_summaries.jsonl",
        api_usage_logs=run_dir / "api_usage_logs.jsonl",
    )
