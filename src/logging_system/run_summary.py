"""Validated run lifecycle summaries with atomic persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar, Mapping

from src.logging_system.schemas import LoggingError, RunLogMetadata, SCHEMA_VERSION


RUN_STATUSES = frozenset(
    {"initialized", "completed", "failed", "interrupted", "excluded"}
)
TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "interrupted", "excluded"}
)

_EXPECTED_COMPLETED_COUNTS = {
    "completed_natural_blocks": 7,
    "completed_target_events": 5,
    "completed_main_turns": 104,
}

__all__ = [
    "CorruptRunSummaryError",
    "RUN_STATUSES",
    "RunSummary",
    "RunSummaryError",
    "RunSummaryStateError",
    "RunSummaryValidationError",
    "RunSummaryWriteError",
    "TERMINAL_RUN_STATUSES",
    "read_run_summary",
    "write_run_summary",
]


class RunSummaryError(LoggingError):
    """Base error for run-summary validation and persistence."""


class RunSummaryValidationError(RunSummaryError):
    """Raised when a run summary contains invalid research metadata."""


class RunSummaryStateError(RunSummaryError):
    """Raised when a write would violate the run lifecycle."""


class CorruptRunSummaryError(RunSummaryError):
    """Raised when an existing run summary cannot be trusted."""


class RunSummaryWriteError(RunSummaryError):
    """Raised when a run summary cannot be durably persisted."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _require_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunSummaryValidationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True)
class RunSummary:
    """One immutable snapshot of a run's lifecycle and aggregate counts."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    status: str = "initialized"
    completed_natural_blocks: int = 0
    completed_target_events: int = 0
    completed_main_turns: int = 0
    error_count: int = 0
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "repetition_index",
            "status",
            "completed_natural_blocks",
            "completed_target_events",
            "completed_main_turns",
            "error_count",
        }
    )

    def __post_init__(self) -> None:
        try:
            RunLogMetadata(
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                retrieval_config_id=self.retrieval_config_id,
                repetition_index=self.repetition_index,
            )
        except LoggingError as exc:
            raise RunSummaryValidationError(str(exc)) from exc
        if not isinstance(self.status, str) or self.status not in RUN_STATUSES:
            raise RunSummaryValidationError(
                f"unsupported run status: {self.status!r}"
            )
        for field_name in (
            "completed_natural_blocks",
            "completed_target_events",
            "completed_main_turns",
            "error_count",
        ):
            _require_count(getattr(self, field_name), field_name)
        for field_name, maximum in _EXPECTED_COMPLETED_COUNTS.items():
            if getattr(self, field_name) > maximum:
                raise RunSummaryValidationError(
                    f"{field_name} must not exceed the fixed run total {maximum}"
                )
        if self.status == "initialized" and any(
            getattr(self, field_name) != 0
            for field_name in (
                "completed_natural_blocks",
                "completed_target_events",
                "completed_main_turns",
                "error_count",
            )
        ):
            raise RunSummaryValidationError(
                "initialized run summaries must have zero aggregate counts"
            )
        if self.status == "completed":
            mismatched = [
                field_name
                for field_name, expected in _EXPECTED_COMPLETED_COUNTS.items()
                if getattr(self, field_name) != expected
            ]
            if mismatched:
                raise RunSummaryValidationError(
                    "completed run summary must contain the fixed completed totals: "
                    + ", ".join(mismatched)
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh canonical JSON object."""
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "status": self.status,
            "completed_natural_blocks": self.completed_natural_blocks,
            "completed_target_events": self.completed_target_events,
            "completed_main_turns": self.completed_main_turns,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunSummary:
        """Validate an existing persisted summary without ignoring fields."""
        if not isinstance(value, Mapping):
            raise RunSummaryValidationError("run summary must be a JSON object")
        if any(not isinstance(key, str) for key in value):
            raise RunSummaryValidationError("run summary keys must be strings")
        keys = set(value)
        if keys != cls._FIELDS:
            missing = sorted(cls._FIELDS - keys)
            extra = sorted(keys - cls._FIELDS)
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected fields: {', '.join(extra)}")
            raise RunSummaryValidationError("invalid run summary; " + "; ".join(details))
        if value["schema_version"] != SCHEMA_VERSION:
            raise RunSummaryValidationError(
                f"run summary schema_version must be {SCHEMA_VERSION!r}"
            )
        return cls(
            experiment_id=value["experiment_id"],
            run_id=value["run_id"],
            retrieval_config_id=value["retrieval_config_id"],
            repetition_index=value["repetition_index"],
            status=value["status"],
            completed_natural_blocks=value["completed_natural_blocks"],
            completed_target_events=value["completed_target_events"],
            completed_main_turns=value["completed_main_turns"],
            error_count=value["error_count"],
        )


def _serialize_summary(summary: RunSummary) -> str:
    if not isinstance(summary, RunSummary):
        raise RunSummaryValidationError("summary must be a RunSummary")
    try:
        return json.dumps(
            summary.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise RunSummaryValidationError(
            f"run summary is not JSON serializable: {exc}"
        ) from exc


def _validate_summary_path_identity(path: Path, summary: RunSummary) -> None:
    if path.name != "run_summary.json":
        return
    if any(
        ancestor.is_symlink()
        for ancestor in (path.parent, path.parent.parent, path.parent.parent.parent)
    ):
        raise RunSummaryValidationError(
            "run summary output, experiment, and run directories must not be symlinks"
        )
    valid_run_directories = {summary.run_id, f".{summary.run_id}.staging"}
    if (
        path.parent.name not in valid_run_directories
        or path.parent.parent.name != summary.experiment_id
    ):
        raise RunSummaryValidationError(
            "run_summary.json path does not match summary experiment_id/run_id"
        )


def read_run_summary(path: str | Path) -> RunSummary:
    """Read and strictly validate one existing run summary."""
    summary_path = Path(path)
    if summary_path.is_symlink():
        raise CorruptRunSummaryError(
            f"refusing to read a run summary through a symbolic link: {summary_path}"
        )
    if not summary_path.is_file():
        raise CorruptRunSummaryError(f"run summary does not exist: {summary_path}")
    try:
        with summary_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_strict_json_object)
    except _DuplicateJsonKeyError as exc:
        raise CorruptRunSummaryError(
            f"invalid JSON in run summary {summary_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CorruptRunSummaryError(
            f"invalid JSON in run summary {summary_path}: {exc.msg}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise RunSummaryWriteError(
            f"could not read run summary {summary_path}: {exc}"
        ) from exc
    try:
        summary = RunSummary.from_dict(value)
    except RunSummaryValidationError as exc:
        raise CorruptRunSummaryError(
            f"invalid existing run summary {summary_path}: {exc}"
        ) from exc
    try:
        _validate_summary_path_identity(summary_path, summary)
    except RunSummaryValidationError as exc:
        raise CorruptRunSummaryError(
            f"invalid existing run summary {summary_path}: {exc}"
        ) from exc
    return summary


def _validate_transition(previous: RunSummary | None, new: RunSummary) -> None:
    if previous is None:
        if new.status != "initialized":
            raise RunSummaryStateError(
                "a run summary must be created with initialized status"
            )
        return

    if previous.status in TERMINAL_RUN_STATUSES:
        raise RunSummaryStateError(
            f"run summary status {previous.status!r} is terminal and read-only"
        )
    if new.status not in TERMINAL_RUN_STATUSES:
        raise RunSummaryStateError(
            "initialized run summary may transition only to a terminal status"
        )
    identity_fields = (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    )
    mismatched = [
        field_name
        for field_name in identity_fields
        if getattr(previous, field_name) != getattr(new, field_name)
    ]
    if mismatched:
        raise RunSummaryStateError(
            "run summary identity cannot change: " + ", ".join(mismatched)
        )


def _atomic_replace(path: Path, payload: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        cleanup_error: OSError | None = None
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
        cleanup_detail = (
            f"; staging cleanup also failed: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        raise RunSummaryWriteError(
            f"could not atomically write run summary {path}: {exc}{cleanup_detail}"
        ) from exc


def write_run_summary(path: str | Path, summary: RunSummary) -> None:
    """Atomically create or advance a run summary.

    A new summary must be ``initialized``.  It may be replaced once by a
    terminal summary.  All terminal states are permanently read-only.
    """
    payload = _serialize_summary(summary)
    summary_path = Path(path)
    if summary_path.is_symlink():
        raise RunSummaryStateError(
            f"refusing to write a run summary through a symbolic link: {summary_path}"
        )

    previous = read_run_summary(summary_path) if summary_path.exists() else None
    _validate_transition(previous, summary)
    _validate_summary_path_identity(summary_path, summary)

    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunSummaryWriteError(
            f"could not create run summary directory {summary_path.parent}: {exc}"
        ) from exc
    _atomic_replace(summary_path, payload)
