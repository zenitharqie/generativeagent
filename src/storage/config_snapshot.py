"""Validated, byte-preserving configuration snapshots for research runs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from src.config_loader import loader as config_loader
from src.config_loader.validators import ConfigValidationError
from src.storage.run_directory import StorageError


class ConfigSnapshotError(StorageError):
    """Raised when a complete safe configuration snapshot cannot be published."""


def copy_config_snapshot(
    config_directory: str | Path,
    snapshot_directory: str | Path,
) -> tuple[Path, ...]:
    """Validate and atomically publish the canonical configuration files.

    Only filenames named by ``config_loader.CONFIG_FILES`` are copied. Source
    bytes are preserved exactly; unrelated files such as ``.env`` are never
    discovered or copied.
    """
    destination_path = _coerce_destination(snapshot_directory)
    _ensure_run_is_writable(destination_path)
    source_root = _resolve_source_root(config_directory)
    canonical_sources = _canonical_source_files(source_root)
    _validate_configuration(source_root)

    destination = _prepare_destination(destination_path)
    staging = destination.parent / f".{destination.name}.staging"
    if _path_exists(staging):
        raise ConfigSnapshotError(
            f"config snapshot staging directory already exists: {staging}"
        )

    try:
        staging.mkdir(mode=0o700)
        for filename, source in canonical_sources:
            _copy_file_durably(source, staging / filename)
        os.replace(staging, destination)
    except OSError as exc:
        cleanup_error = _cleanup_staging_directory(staging)
        detail = f"; staging cleanup also failed: {cleanup_error}" if cleanup_error else ""
        raise ConfigSnapshotError(
            f"could not create config snapshot at {destination}: {exc}{detail}"
        ) from exc

    return tuple(destination / filename for filename, _ in canonical_sources)


def _coerce_destination(snapshot_directory: str | Path) -> Path:
    try:
        destination = Path(snapshot_directory)
    except (TypeError, ValueError) as exc:
        raise ConfigSnapshotError(
            f"invalid config snapshot directory: {snapshot_directory!r}"
        ) from exc
    if not destination.name or destination.name in {".", ".."}:
        raise ConfigSnapshotError(
            f"invalid config snapshot directory: {snapshot_directory!r}"
        )
    return destination


def _ensure_run_is_writable(destination: Path) -> None:
    """Reject snapshot mutation when the destination belongs to a terminal run."""
    if any(
        ancestor.is_symlink()
        for ancestor in (
            destination.parent,
            destination.parent.parent,
            destination.parent.parent.parent,
        )
    ):
        raise ConfigSnapshotError(
            "config snapshot output, experiment, and run directories must not be symlinks"
        )
    summary_path = destination.parent / "run_summary.json"
    if not summary_path.exists() and not summary_path.is_symlink():
        return

    from src.logging_system.run_summary import (
        TERMINAL_RUN_STATUSES,
        RunSummaryError,
        read_run_summary,
    )

    try:
        summary = read_run_summary(summary_path)
    except RunSummaryError as exc:
        raise ConfigSnapshotError(
            f"cannot verify config snapshot lifecycle: {exc}"
        ) from exc
    if summary.status in TERMINAL_RUN_STATUSES:
        raise ConfigSnapshotError(
            f"run summary status {summary.status!r} is terminal and read-only"
        )


def _resolve_source_root(config_directory: str | Path) -> Path:
    try:
        path = Path(config_directory)
        resolved = path.resolve(strict=True)
    except (TypeError, ValueError, OSError) as exc:
        raise ConfigSnapshotError(
            f"invalid configuration directory: {config_directory!r}"
        ) from exc
    if not resolved.is_dir():
        raise ConfigSnapshotError(f"configuration path is not a directory: {resolved}")
    return resolved


def _canonical_source_files(source_root: Path) -> tuple[tuple[str, Path], ...]:
    canonical: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for filename in config_loader.CONFIG_FILES.values():
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise ConfigSnapshotError(
                f"unsafe canonical configuration filename: {filename!r}"
            )
        if filename in seen:
            raise ConfigSnapshotError(
                f"duplicate canonical configuration filename: {filename!r}"
            )
        seen.add(filename)

        source = source_root / filename
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConfigSnapshotError(
                f"missing or unreadable configuration file: {source}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(source_root):
            raise ConfigSnapshotError(
                f"configuration file escapes its source directory: {source}"
            )
        canonical.append((filename, resolved))
    return tuple(canonical)


def _validate_configuration(source_root: Path) -> None:
    try:
        config_loader.load_experiment_config(source_root)
    except (config_loader.ConfigLoadError, ConfigValidationError) as exc:
        raise ConfigSnapshotError(f"configuration validation failed: {exc}") from exc


def _prepare_destination(snapshot_directory: str | Path) -> Path:
    destination = Path(snapshot_directory)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        raise ConfigSnapshotError(
            f"could not create snapshot parent {destination.parent}: {exc}"
        ) from exc
    if destination.is_symlink():
        raise ConfigSnapshotError(
            f"config snapshot directory must not be a symlink: {destination}"
        )
    if destination.exists():
        if not destination.is_dir():
            raise ConfigSnapshotError(
                f"config snapshot destination is not a directory: {destination}"
            )
        try:
            if any(destination.iterdir()):
                raise ConfigSnapshotError(
                    f"config snapshot destination is not empty: {destination}"
                )
        except OSError as exc:
            raise ConfigSnapshotError(
                f"could not inspect config snapshot destination {destination}: {exc}"
            ) from exc
    return destination


def _copy_file_durably(source: Path, destination: Path) -> None:
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        shutil.copyfileobj(source_file, destination_file)
        destination_file.flush()
        os.fsync(destination_file.fileno())


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
    return path.exists() or path.is_symlink()
