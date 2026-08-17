"""Conflict-only durable publication for Ground Truth methodology artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Callable, Mapping

from .errors import GroundTruthPersistenceError
from .finalizer import GroundTruthFinalizationBundle
from .schemas import MAX_INPUT_BYTES


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if _lexists(current) and current.is_symlink():
            raise GroundTruthPersistenceError("output path components must not be symlinks")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short write")
            written += count
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise GroundTruthPersistenceError("staging artifact is not a regular file")
        return info.st_dev, info.st_ino
    finally:
        os.close(descriptor)


def _read_regular(path: Path, *, expected_identity: tuple[int, int] | None = None) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise GroundTruthPersistenceError("published artifact is not a regular file")
        if expected_identity is not None and (info.st_dev, info.st_ino) != expected_identity:
            raise GroundTruthPersistenceError("published artifact identity changed")
        if info.st_size > MAX_INPUT_BYTES:
            raise GroundTruthPersistenceError("published artifact exceeds size limit")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                raise GroundTruthPersistenceError("published artifact read was incomplete")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_parse(payload: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject(_: str) -> None:
        raise ValueError("non-finite number")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise GroundTruthPersistenceError("published artifact is not strict JSON") from exc


def _same_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return not stat.S_ISLNK(info.st_mode) and (info.st_dev, info.st_ino) == identity


def publish_json_file(path: str | Path, payload: bytes) -> Path:
    """Publish one deterministic JSON file using a fresh hard-link boundary."""
    final = Path(path)
    if ".." in final.parts:
        raise GroundTruthPersistenceError("output path must not contain traversal")
    _reject_symlink_components(final.parent)
    if not final.parent.is_dir() or _lexists(final):
        raise GroundTruthPersistenceError("output file must be fresh in an existing safe directory")
    staging = final.parent / f".{final.name}.staging"
    if _lexists(staging):
        raise GroundTruthPersistenceError("staging file already exists")
    identity: tuple[int, int] | None = None
    published = False
    try:
        identity = _write_new(staging, payload)
        staged = _read_regular(staging, expected_identity=identity)
        if staged != payload:
            raise GroundTruthPersistenceError("staged bytes differ from canonical bytes")
        _strict_parse(staged)
        os.link(staging, final, follow_symlinks=False)
        published = True
        staging.unlink()
        _fsync_directory(final.parent)
        final_payload = _read_regular(final, expected_identity=identity)
        if final_payload != payload:
            raise GroundTruthPersistenceError("final bytes differ from validated staging bytes")
        return final
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        for candidate, owned in ((staging, identity), (final, identity if published else None)):
            if owned is None or not _same_identity(candidate, owned):
                continue
            try:
                candidate.unlink()
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_error or cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(f"publication cleanup also failed: {type(cleanup_error).__name__}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, GroundTruthPersistenceError):
            raise
        raise GroundTruthPersistenceError("could not publish Ground Truth artifact") from exc


def fingerprint_source_files(paths: tuple[Path, ...]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for path in paths:
        if ".." in path.parts:
            raise GroundTruthPersistenceError("methodology source paths must not contain traversal")
        _reject_symlink_components(path)
        if path.is_symlink() or not path.is_file():
            raise GroundTruthPersistenceError("methodology source input is missing or unsafe")
        try:
            payload = _read_regular(path)
        except OSError as exc:
            raise GroundTruthPersistenceError("could not fingerprint methodology source input") from exc
        result.append((str(path.resolve(strict=True)), sha256(payload).hexdigest()))
    return tuple(result)


def publish_finalization_bundle(
    output_directory: str | Path,
    bundle: GroundTruthFinalizationBundle,
    *,
    source_fingerprints: tuple[tuple[str, str], ...],
    recheck_sources: Callable[[], tuple[tuple[str, str], ...]],
) -> Path:
    """Atomically publish a fully read-back-validated immutable evidence directory."""
    final = Path(output_directory)
    if ".." in final.parts:
        raise GroundTruthPersistenceError("output path must not contain traversal")
    _reject_symlink_components(final.parent)
    if not final.parent.is_dir() or _lexists(final):
        raise GroundTruthPersistenceError("finalization output must be a fresh directory")
    staging = final.parent / f".{final.name}.staging"
    if _lexists(staging):
        raise GroundTruthPersistenceError("finalization staging path already exists")
    staging_identity: tuple[int, int] | None = None
    final_identity: tuple[int, int] | None = None
    published = False
    try:
        staging.mkdir(mode=0o700)
        info = staging.lstat()
        staging_identity = (info.st_dev, info.st_ino)
        directories = sorted(
            {Path(name).parent for name in bundle.files if Path(name).parent != Path(".")},
            key=lambda item: (len(item.parts), item.as_posix()),
        )
        for relative in directories:
            (staging / relative).mkdir(mode=0o700)
        identities: dict[str, tuple[int, int]] = {}
        for name in sorted(bundle.files):
            path = staging / name
            identities[name] = _write_new(path, bundle.files[name])
        for relative in sorted(directories, key=lambda item: (-len(item.parts), item.as_posix())):
            _fsync_directory(staging / relative)
        _fsync_directory(staging)
        for name in sorted(bundle.files):
            payload = _read_regular(staging / name, expected_identity=identities[name])
            if payload != bundle.files[name]:
                raise GroundTruthPersistenceError("staged finalization bytes changed")
            _strict_parse(payload)
        if recheck_sources() != source_fingerprints:
            raise GroundTruthPersistenceError("methodology source inputs changed before publication")
        if _lexists(final):
            raise GroundTruthPersistenceError("finalization output conflict appeared")
        os.rename(staging, final)
        published = True
        final_identity = staging_identity
        _fsync_directory(final.parent)
        if final_identity is None or not _same_identity(final, final_identity):
            raise GroundTruthPersistenceError("finalization directory identity changed")
        for name in sorted(bundle.files):
            payload = _read_regular(final / name, expected_identity=identities[name])
            if payload != bundle.files[name]:
                raise GroundTruthPersistenceError("final finalization bytes changed")
            _strict_parse(payload)
        if recheck_sources() != source_fingerprints:
            raise GroundTruthPersistenceError("methodology source inputs changed after publication")
        return final
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        owned_path: Path | None = None
        owned_identity: tuple[int, int] | None = None
        if published:
            owned_path, owned_identity = final, final_identity
        else:
            owned_path, owned_identity = staging, staging_identity
        if owned_path is not None and owned_identity is not None and _same_identity(owned_path, owned_identity):
            try:
                shutil.rmtree(owned_path)
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None and hasattr(exc, "add_note"):
            exc.add_note(f"finalization cleanup also failed: {type(cleanup_error).__name__}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, GroundTruthPersistenceError):
            raise
        raise GroundTruthPersistenceError("could not publish Ground Truth finalization") from exc


__all__ = [
    "fingerprint_source_files",
    "publish_finalization_bundle",
    "publish_json_file",
]
