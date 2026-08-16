"""Conflict-safe deterministic exporter for the read-only Phase 12 replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from src.visualizer.errors import (
    ReplayConflictError,
    ReplayExportError,
    ReplayInputError,
    ReplaySecurityError,
)
from src.visualizer.readers import (
    InputSnapshot,
    LoadedOverlay,
    attach_judgment_overlay,
    load_evaluation_overlay,
    load_pilot_overlay,
    load_run_artifacts,
)
from src.visualizer.schemas import REPLAY_SCHEMA_VERSION, ReplayDocument
from src.visualizer.timeline import reconstruct_replay


MAX_REPLAY_OUTPUT_BYTES = 256 * 1024 * 1024

_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "embedding",
        "environment",
        "headers",
        "prompt",
        "query_embedding",
        "raw_prompt",
        "raw_provider_response",
        "request_body",
        "response_body",
        "traceback",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)\b(?:openai[_-]?)?api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@"),
)


@dataclass(frozen=True)
class ReplayBuild:
    document: ReplayDocument
    input_snapshots: tuple[InputSnapshot, ...]


@dataclass(frozen=True)
class ReplayExportResult:
    output_path: Path
    output_sha256: str
    output_bytes: int
    document: ReplayDocument


def _with_source_hash(value: Mapping[str, Any], snapshot: InputSnapshot) -> dict[str, Any]:
    result = dict(value)
    result["source_aggregate_sha256"] = snapshot.aggregate_fingerprint
    return result


def build_replay(
    *,
    run_dir: str | Path,
    evaluation_dir: str | Path | None = None,
    judgment_dir: str | Path | None = None,
    pilot_evidence_dir: str | Path | None = None,
    pilot_review_dir: str | Path | None = None,
) -> ReplayBuild:
    """Load and project persisted artifacts without writing or executing research code."""
    if judgment_dir is not None and evaluation_dir is None:
        raise ReplayInputError("a judgment overlay requires an explicit evaluation overlay")
    if pilot_review_dir is not None and pilot_evidence_dir is None:
        raise ReplayInputError("a pilot review overlay requires explicit pilot evidence")

    run = load_run_artifacts(run_dir)
    snapshots: list[InputSnapshot] = [run.snapshot]
    evaluation_value: Mapping[str, Any] | None = None
    if evaluation_dir is not None:
        evaluation = load_evaluation_overlay(evaluation_dir, run_summary=run.summary)
        snapshots.append(evaluation.snapshot)
        if judgment_dir is not None:
            evaluation, judgment = attach_judgment_overlay(
                evaluation,
                judgment_dir,
                run_summary=run.summary,
            )
            snapshots.append(judgment.snapshot)
            evaluation_value = _with_source_hash(evaluation.value, evaluation.snapshot)
            evaluation_value["judgment_source_aggregate_sha256"] = (
                judgment.snapshot.aggregate_fingerprint
            )
        else:
            evaluation_value = _with_source_hash(evaluation.value, evaluation.snapshot)

    pilot_value: Mapping[str, Any] | None = None
    if pilot_evidence_dir is not None:
        pilot, review = load_pilot_overlay(
            pilot_evidence_dir,
            run=run,
            review_dir=pilot_review_dir,
        )
        snapshots.append(pilot.snapshot)
        pilot_value = _with_source_hash(pilot.value, pilot.snapshot)
        if review is not None:
            snapshots.append(review.snapshot)
            pilot_value["review_source_aggregate_sha256"] = (
                review.snapshot.aggregate_fingerprint
            )

    for snapshot in snapshots:
        snapshot.verify_unchanged()
    document = reconstruct_replay(
        run,
        evaluation_overlay=evaluation_value,
        pilot_overlay=pilot_value,
    )
    for snapshot in snapshots:
        snapshot.verify_unchanged()
    return ReplayBuild(document=document, input_snapshots=tuple(snapshots))


def _walk_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.casefold() in _FORBIDDEN_KEYS:
                raise ReplaySecurityError(
                    f"replay projection contains forbidden field {key!r}"
                )
            _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_keys(item)


def serialize_replay(document: ReplayDocument) -> bytes:
    """Serialize one replay canonically, with no time-, path-, or run-dependent salt."""
    if not isinstance(document, ReplayDocument):
        raise ReplayExportError("document must be a ReplayDocument")
    value = document.to_dict()
    _walk_keys(value)
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReplayExportError("replay projection is not strict JSON") from exc
    if len(payload) > MAX_REPLAY_OUTPUT_BYTES:
        raise ReplaySecurityError("replay projection exceeds the output size limit")
    decoded = payload.decode("utf-8")
    if any(pattern.search(decoded) for pattern in _SECRET_PATTERNS):
        raise ReplaySecurityError("replay projection contains secret-like material")
    return payload


def _reject_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                raise ReplaySecurityError("output paths must not traverse symbolic links")
        except OSError as exc:
            raise ReplayInputError("could not inspect the output path") from exc


def _resolve_output_path(
    output: str | Path,
    *,
    snapshots: Sequence[InputSnapshot],
) -> Path:
    try:
        requested = Path(output)
    except (TypeError, ValueError) as exc:
        raise ReplayInputError("invalid replay output path") from exc
    if ".." in requested.parts:
        raise ReplaySecurityError("output path traversal is not permitted")
    if requested.name in {"", ".", ".."} or requested.suffix.casefold() != ".json":
        raise ReplayInputError("replay output must be a named .json file")
    _reject_symlink_components(requested.parent)
    try:
        parent = requested.parent.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReplayInputError("replay output parent directory does not exist") from exc
    if not parent.is_dir():
        raise ReplayInputError("replay output parent must be a directory")
    output_path = parent / requested.name
    if os.path.lexists(output_path):
        raise ReplayConflictError("replay output already exists; overwrite is forbidden")
    for snapshot in snapshots:
        try:
            output_path.relative_to(snapshot.directory)
        except ValueError:
            pass
        else:
            raise ReplaySecurityError("replay output must remain outside every input directory")
    return output_path


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ReplayExportError("could not durably publish replay output") from exc


class _DuplicateReplayKeyError(ValueError):
    pass


def _strict_replay_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateReplayKeyError("duplicate replay JSON key")
        result[key] = value
    return result


def _reject_replay_constant(value: str) -> None:
    raise ValueError("non-finite replay JSON number")


def _validate_staged_replay_payload(
    payload: bytes,
    *,
    expected_payload: bytes,
    document: ReplayDocument,
) -> None:
    """Validate the exact bytes read back from disk against the immutable document."""
    if len(payload) != len(expected_payload) or sha256(payload).digest() != sha256(
        expected_payload
    ).digest() or payload != expected_payload:
        raise ReplaySecurityError("replay staging bytes changed before publication")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_replay_object,
            parse_constant=_reject_replay_constant,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateReplayKeyError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ReplayExportError("staged replay is not strict JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayExportError("staged replay schema version is invalid")
    # ReplayDocument is the immutable, already-validated domain contract. Exact
    # equality validates every nested field while prohibiting repair or migration.
    if value != document.to_dict():
        raise ReplayExportError("staged replay does not match the ReplayDocument schema")
    _walk_keys(value)


def _read_and_validate_replay_file(
    path: Path,
    *,
    expected_stat: os.stat_result,
    expected_payload: bytes,
    document: ReplayDocument,
) -> bytes:
    """Reopen a staged/published file without following links and validate its bytes."""
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReplaySecurityError("replay publication file could not be safely inspected") from exc
    expected_identity = (expected_stat.st_dev, expected_stat.st_ino)
    if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != expected_identity:
        raise ReplaySecurityError("replay publication file identity changed")
    if before.st_size != len(expected_payload):
        raise ReplaySecurityError("replay publication file length changed")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    read_error: BaseException | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != expected_identity:
            raise ReplaySecurityError("replay publication file changed during safe open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_REPLAY_OUTPUT_BYTES:
                raise ReplaySecurityError("replay publication file exceeds the output limit")
            chunks.append(chunk)
        payload = b"".join(chunks)
    except ReplaySecurityError as exc:
        read_error = exc
        raise
    except OSError as exc:
        read_error = ReplaySecurityError(
            "replay publication file could not be safely read"
        )
        raise read_error from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                if read_error is not None:
                    read_error.add_note("cleanup could not close the replay read-back descriptor")
                else:
                    raise ReplayExportError("could not close replay read-back descriptor") from exc

    try:
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReplaySecurityError("replay publication file disappeared during read-back") from exc
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ):
        raise ReplaySecurityError("replay publication file changed during read-back")
    _validate_staged_replay_payload(
        payload,
        expected_payload=expected_payload,
        document=document,
    )
    return payload


def _unlink_if_same_inode(path: Path, expected: os.stat_result) -> OSError | None:
    try:
        actual = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    if stat.S_ISREG(actual.st_mode) and (actual.st_dev, actual.st_ino) == (
        expected.st_dev,
        expected.st_ino,
    ):
        try:
            path.unlink()
        except OSError as exc:
            return exc
    return None


def export_visualizer_data(
    *,
    run_dir: str | Path,
    output: str | Path,
    evaluation_dir: str | Path | None = None,
    judgment_dir: str | Path | None = None,
    pilot_evidence_dir: str | Path | None = None,
    pilot_review_dir: str | Path | None = None,
) -> ReplayExportResult:
    """Atomically publish one new replay JSON while inputs remain unchanged."""
    build = build_replay(
        run_dir=run_dir,
        evaluation_dir=evaluation_dir,
        judgment_dir=judgment_dir,
        pilot_evidence_dir=pilot_evidence_dir,
        pilot_review_dir=pilot_review_dir,
    )
    payload = serialize_replay(build.document)
    output_path = _resolve_output_path(output, snapshots=build.input_snapshots)
    for snapshot in build.input_snapshots:
        snapshot.verify_unchanged()

    descriptor: int | None = None
    staging_path: Path | None = None
    linked = False
    staging_stat: os.stat_result | None = None
    originating_error: BaseException | None = None
    try:
        descriptor, raw_staging = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".staging",
            dir=output_path.parent,
        )
        staging_path = Path(raw_staging)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ReplayExportError("could not write the replay staging file")
            view = view[written:]
        os.fsync(descriptor)
        staging_stat = os.fstat(descriptor)
        if not stat.S_ISREG(staging_stat.st_mode) or staging_stat.st_size != len(payload):
            raise ReplaySecurityError("replay staging descriptor is not the expected regular file")
        os.close(descriptor)
        descriptor = None
        staged_payload = _read_and_validate_replay_file(
            staging_path,
            expected_stat=staging_stat,
            expected_payload=payload,
            document=build.document,
        )
        for snapshot in build.input_snapshots:
            snapshot.verify_unchanged()
        try:
            os.link(staging_path, output_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ReplayConflictError(
                "replay output appeared during publication; overwrite is forbidden"
            ) from exc
        except OSError as exc:
            raise ReplayExportError("could not atomically publish replay output") from exc
        linked = True
        published_payload = _read_and_validate_replay_file(
            output_path,
            expected_stat=staging_stat,
            expected_payload=staged_payload,
            document=build.document,
        )
        if published_payload != staged_payload:
            raise ReplaySecurityError("published replay differs from validated staging bytes")
        for snapshot in build.input_snapshots:
            snapshot.verify_unchanged()
        _fsync_directory(output_path.parent)
        for snapshot in build.input_snapshots:
            snapshot.verify_unchanged()
    except BaseException as exc:
        originating_error = exc
        if linked and staging_stat is not None:
            cleanup_error = _unlink_if_same_inode(output_path, staging_stat)
            if cleanup_error is not None:
                exc.add_note("cleanup could not remove the unpublished replay output")
        raise
    finally:
        cleanup_errors: list[OSError] = []
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                cleanup_errors.append(exc)
        if staging_path is not None:
            try:
                staging_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if originating_error is not None:
                for cleanup_error in cleanup_errors:
                    originating_error.add_note(
                        f"replay cleanup also failed ({type(cleanup_error).__name__})"
                    )
            else:
                if linked and staging_stat is not None:
                    output_cleanup_error = _unlink_if_same_inode(output_path, staging_stat)
                    if output_cleanup_error is not None:
                        cleanup_errors[0].add_note(
                            "cleanup also could not remove the published output"
                        )
                cleanup_failure = ReplayExportError(
                    "could not completely clean replay publication staging"
                )
                for extra_error in cleanup_errors[1:]:
                    cleanup_failure.add_note(
                        f"additional cleanup failure: {type(extra_error).__name__}"
                    )
                raise cleanup_failure from cleanup_errors[0]

    return ReplayExportResult(
        output_path=output_path,
        output_sha256=sha256(payload).hexdigest(),
        output_bytes=len(payload),
        document=build.document,
    )


__all__ = [
    "MAX_REPLAY_OUTPUT_BYTES",
    "ReplayBuild",
    "ReplayExportResult",
    "build_replay",
    "export_visualizer_data",
    "serialize_replay",
]
