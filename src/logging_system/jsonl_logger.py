"""Deterministic, append-only JSON Lines persistence.

The writer deliberately owns no simulation state. Callers provide an already
validated record. Canonical research streams enforce their typed schema,
physical run identity, and stable duplicate policy; custom streams may add a
caller-defined duplicate key.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from copy import deepcopy
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, TypeAlias

from src.logging_system.schemas import (
    APIUsageLogRecord,
    BehaviorLogRecord,
    ErrorLogRecord,
    LoggingError,
    MemoryLogRecord,
    ReflectionLogRecord,
    RetrievalCandidateRecord,
    RetrievalLogRecord,
    RetrievalWeightsRecord,
    RunLogMetadata,
    SCHEMA_VERSION,
    StepLogMetadata,
    TurnLogMetadata,
)


JsonObject: TypeAlias = dict[str, Any]
DuplicateKey: TypeAlias = Callable[[Mapping[str, Any]], Hashable]

_RUN_FIELDS = {
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
}
_STEP_FIELDS = {
    "step_id",
    "step_type",
    "is_target_event",
    "evaluation_enabled",
    "day",
    "period",
    "timestamp",
    "event_id",
}
_TURN_FIELDS = {"turn_id", "turn_index", "agent_id"}
_SHARED_FIELDS = _RUN_FIELDS | _STEP_FIELDS | _TURN_FIELDS | {"schema_version"}
_CANDIDATE_FIELDS = {
    "memory_id",
    "category",
    "raw_recency",
    "raw_importance",
    "raw_relevance",
    "normalized_recency",
    "normalized_importance",
    "normalized_relevance",
    "final_score",
    "selected",
}

_CANONICAL_RECORD_TYPES: dict[str, type[object]] = {
    "retrieval_logs.jsonl": RetrievalLogRecord,
    "behavior_logs.jsonl": BehaviorLogRecord,
    "memory_logs.jsonl": MemoryLogRecord,
    "reflection_logs.jsonl": ReflectionLogRecord,
    "api_usage_logs.jsonl": APIUsageLogRecord,
    "errors.jsonl": ErrorLogRecord,
}
_CANONICAL_JSONL_FILENAMES = frozenset(
    {
        *_CANONICAL_RECORD_TYPES,
        "event_summaries.jsonl",
        "natural_block_summaries.jsonl",
    }
)
_CONTROLLER_RESERVED_FILENAMES = frozenset(
    {"event_summaries.jsonl", "natural_block_summaries.jsonl"}
)
_CANONICAL_FIELDS = {
    "retrieval_logs.jsonl": _SHARED_FIELDS
    | {
        "query",
        "query_embedding",
        "weights",
        "top_k",
        "candidate_count",
        "candidates",
        "selected_memory_ids",
        "accessed_memory_ids",
    },
    "behavior_logs.jsonl": _SHARED_FIELDS
    | {"response", "retrieved_memory_ids", "plan_summary"},
    "memory_logs.jsonl": _SHARED_FIELDS
    | {
        "memory_id",
        "memory_type",
        "category",
        "text",
        "importance",
        "created_at",
        "last_accessed_at",
        "embedding",
        "metadata",
    },
    "reflection_logs.jsonl": _SHARED_FIELDS
    | {"reflection", "category", "memory_id"},
    "api_usage_logs.jsonl": _SHARED_FIELDS
    | {
        "call_type",
        "model",
        "prompt_template",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "status",
        "error_type",
    },
    "errors.jsonl": _SHARED_FIELDS
    | {"error_type", "message", "recoverable", "retry_count"},
}
_CANONICAL_DUPLICATE_FIELDS: dict[str, tuple[str, ...]] = {
    "retrieval_logs.jsonl": ("experiment_id", "run_id", "turn_id"),
    "behavior_logs.jsonl": ("experiment_id", "run_id", "turn_id"),
    "memory_logs.jsonl": ("experiment_id", "run_id", "memory_id"),
    "reflection_logs.jsonl": ("experiment_id", "run_id", "turn_id"),
}

__all__ = [
    "CorruptJsonlError",
    "DuplicateKey",
    "DuplicateRecordError",
    "JsonlLogger",
    "JsonlLoggerError",
    "JsonlValidationError",
    "JsonlWriteError",
    "append_jsonl",
]


class JsonlLoggerError(LoggingError):
    """Base error for JSONL validation and persistence failures."""


class JsonlValidationError(JsonlLoggerError):
    """Raised before writing when a record cannot be serialized safely."""


class CorruptJsonlError(JsonlLoggerError):
    """Raised when an existing JSONL file is not a complete valid record stream."""


class DuplicateRecordError(JsonlLoggerError):
    """Raised when a caller-defined record key is already present."""


class JsonlWriteError(JsonlLoggerError):
    """Raised when a JSONL path cannot be created, read, or durably appended."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _validate_json_value(value: Any, field_path: str) -> None:
    """Reject values that JSON would coerce or encode non-portably."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise JsonlValidationError(f"{field_path} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JsonlValidationError(
                    f"{field_path} keys must be strings, got {type(key).__name__}"
                )
            _validate_json_value(item, f"{field_path}.{key}")
        return
    raise JsonlValidationError(
        f"{field_path} contains unsupported value type {type(value).__name__}"
    )


def _record_to_mapping(record: object) -> JsonObject:
    """Return a defensive mapping from an explicit record serializer."""
    if isinstance(record, Mapping):
        value: object = record
    else:
        to_dict = getattr(record, "to_dict", None)
        if not callable(to_dict):
            raise JsonlValidationError(
                "record must be a mapping or expose an explicit to_dict() serializer"
            )
        try:
            value = to_dict()
        except LoggingError:
            raise
        except Exception as exc:
            raise JsonlValidationError(f"record serialization failed: {exc}") from exc

    if not isinstance(value, Mapping):
        raise JsonlValidationError("record serializer must return a mapping")
    try:
        mapping = deepcopy(dict(value))
        _validate_json_value(mapping, "record")
    except JsonlValidationError:
        raise
    except Exception as exc:
        raise JsonlValidationError(
            f"record could not be defensively copied and validated: {exc}"
        ) from exc
    if mapping.get("schema_version") != SCHEMA_VERSION:
        raise JsonlValidationError(
            f"record schema_version must be {SCHEMA_VERSION!r}"
        )
    return mapping


def _run_metadata_from_mapping(value: Mapping[str, Any]) -> RunLogMetadata:
    return RunLogMetadata(
        experiment_id=value["experiment_id"],
        run_id=value["run_id"],
        retrieval_config_id=value["retrieval_config_id"],
        repetition_index=value["repetition_index"],
    )


def _step_metadata_from_mapping(
    value: Mapping[str, Any],
    run: RunLogMetadata,
) -> StepLogMetadata:
    return StepLogMetadata(
        run=run,
        step_id=value["step_id"],
        step_type=value["step_type"],
        is_target_event=value["is_target_event"],
        evaluation_enabled=value["evaluation_enabled"],
        day=value["day"],
        period=value["period"],
        timestamp=value["timestamp"],
        event_id=value["event_id"],
    )


def _turn_metadata_from_mapping(
    value: Mapping[str, Any],
    run: RunLogMetadata,
) -> TurnLogMetadata:
    return TurnLogMetadata(
        step=_step_metadata_from_mapping(value, run),
        turn_id=value["turn_id"],
        turn_index=value["turn_index"],
        agent_id=value["agent_id"],
    )


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected fields: {', '.join(extra)}")
    raise JsonlValidationError(f"invalid {context}; " + "; ".join(details))


def _validate_canonical_mapping(path: Path, value: Mapping[str, Any]) -> None:
    """Reconstruct a typed canonical record to reject semantic corruption."""
    filename = path.name
    expected_fields = _CANONICAL_FIELDS.get(filename)
    if expected_fields is None:
        if filename in _CANONICAL_JSONL_FILENAMES:
            try:
                _run_metadata_from_mapping(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise JsonlValidationError(
                    f"invalid {filename} shared run metadata: {exc}"
                ) from exc
        return
    _require_exact_fields(value, expected_fields, f"{filename} record")

    try:
        run = _run_metadata_from_mapping(value)
        rebuilt: object
        if filename == "retrieval_logs.jsonl":
            turn = _turn_metadata_from_mapping(value, run)
            weights_value = value["weights"]
            if not isinstance(weights_value, Mapping):
                raise JsonlValidationError("retrieval weights must be an object")
            _require_exact_fields(
                weights_value,
                {"recency", "importance", "relevance"},
                "retrieval weights",
            )
            candidate_values = value["candidates"]
            if not isinstance(candidate_values, list):
                raise JsonlValidationError("retrieval candidates must be an array")
            candidates: list[RetrievalCandidateRecord] = []
            for index, candidate_value in enumerate(candidate_values):
                if not isinstance(candidate_value, Mapping):
                    raise JsonlValidationError(
                        f"retrieval candidate {index} must be an object"
                    )
                _require_exact_fields(
                    candidate_value,
                    _CANDIDATE_FIELDS,
                    f"retrieval candidate {index}",
                )
                candidates.append(RetrievalCandidateRecord(**candidate_value))
            query_embedding = value["query_embedding"]
            if query_embedding is not None and not isinstance(query_embedding, list):
                raise JsonlValidationError("query_embedding must be an array or null")
            if not isinstance(value["selected_memory_ids"], list) or not isinstance(
                value["accessed_memory_ids"], list
            ):
                raise JsonlValidationError(
                    "retrieval selected/accessed memory IDs must be arrays"
                )
            rebuilt = RetrievalLogRecord(
                turn=turn,
                query=value["query"],
                query_embedding=(
                    None if query_embedding is None else tuple(query_embedding)
                ),
                weights=RetrievalWeightsRecord(**weights_value),
                top_k=value["top_k"],
                candidate_count=value["candidate_count"],
                candidates=tuple(candidates),
                selected_memory_ids=tuple(value["selected_memory_ids"]),
                accessed_memory_ids=tuple(value["accessed_memory_ids"]),
            )
        elif filename == "behavior_logs.jsonl":
            if not isinstance(value["retrieved_memory_ids"], list):
                raise JsonlValidationError("retrieved_memory_ids must be an array")
            rebuilt = BehaviorLogRecord(
                turn=_turn_metadata_from_mapping(value, run),
                response=value["response"],
                retrieved_memory_ids=tuple(value["retrieved_memory_ids"]),
                plan_summary=value["plan_summary"],
            )
        elif filename == "memory_logs.jsonl":
            embedding = value["embedding"]
            if embedding is not None and not isinstance(embedding, list):
                raise JsonlValidationError("memory embedding must be an array or null")
            if value["turn_id"] is None:
                null_fields = {"day", "period", "event_id", "turn_index"}
                if any(value[field_name] is not None for field_name in null_fields) or (
                    value["is_target_event"] is not False
                    or value["evaluation_enabled"] is not False
                    or value["timestamp"] != value["created_at"]
                ):
                    raise JsonlValidationError(
                        "initialization memory has inconsistent shared metadata"
                    )
                turn = None
            else:
                turn = _turn_metadata_from_mapping(value, run)
            rebuilt = MemoryLogRecord(
                run=run,
                memory_id=value["memory_id"],
                agent_id=value["agent_id"],
                memory_type=value["memory_type"],
                category=value["category"],
                text=value["text"],
                importance=value["importance"],
                created_at=value["created_at"],
                last_accessed_at=value["last_accessed_at"],
                step_id=value["step_id"],
                step_type=value["step_type"],
                embedding=None if embedding is None else tuple(embedding),
                metadata=value["metadata"],
                turn=turn,
            )
        elif filename == "reflection_logs.jsonl":
            rebuilt = ReflectionLogRecord(
                turn=_turn_metadata_from_mapping(value, run),
                reflection=value["reflection"],
                category=value["category"],
                memory_id=value["memory_id"],
            )
        elif filename == "api_usage_logs.jsonl":
            rebuilt = APIUsageLogRecord(
                turn=_turn_metadata_from_mapping(value, run),
                call_type=value["call_type"],
                model=value["model"],
                prompt_template=value["prompt_template"],
                input_tokens=value["input_tokens"],
                output_tokens=value["output_tokens"],
                total_tokens=value["total_tokens"],
                status=value["status"],
                error_type=value["error_type"],
            )
        else:
            if value["step_id"] is None:
                context_fields = _STEP_FIELDS | _TURN_FIELDS
                if any(value[field_name] is not None for field_name in context_fields):
                    raise JsonlValidationError(
                        "run-level error context fields must all be null"
                    )
                context = None
            elif value["turn_id"] is None:
                if value["turn_index"] is not None or value["agent_id"] is not None:
                    raise JsonlValidationError(
                        "step-level error turn fields must all be null"
                    )
                context = _step_metadata_from_mapping(value, run)
            else:
                context = _turn_metadata_from_mapping(value, run)
            rebuilt = ErrorLogRecord(
                run=run,
                error_type=value["error_type"],
                message=value["message"],
                recoverable=value["recoverable"],
                retry_count=value["retry_count"],
                context=context,
            )
    except JsonlValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise JsonlValidationError(
            f"invalid {filename} record: {exc}"
        ) from exc

    if getattr(rebuilt, "to_dict")() != dict(value):
        raise JsonlValidationError(
            f"{filename} record does not match its canonical serialized form"
        )


def _validate_record_type_for_path(path: Path, record: object) -> None:
    if path.name in _CONTROLLER_RESERVED_FILENAMES:
        raise JsonlValidationError(
            f"{path.name} is reserved until the simulation controller defines "
            "its persisted record schema"
        )
    expected = _CANONICAL_RECORD_TYPES.get(path.name)
    if expected is not None and not isinstance(record, expected):
        raise JsonlValidationError(
            f"{path.name} requires {expected.__name__}, got {type(record).__name__}"
        )


def _validate_path_identity(path: Path, value: Mapping[str, Any]) -> None:
    if path.name not in _CANONICAL_JSONL_FILENAMES:
        return
    if any(
        ancestor.is_symlink()
        for ancestor in (path.parent, path.parent.parent, path.parent.parent.parent)
    ):
        raise JsonlValidationError(
            "canonical JSONL output, experiment, and run directories must not be symlinks"
        )
    try:
        run = _run_metadata_from_mapping(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise JsonlValidationError(
            f"invalid {path.name} shared run metadata: {exc}"
        ) from exc
    if path.parent.name != run.run_id or path.parent.parent.name != run.experiment_id:
        raise JsonlValidationError(
            "canonical JSONL path does not match record experiment_id/run_id"
        )


def _serialize_record(record: object) -> tuple[JsonObject, str]:
    mapping = _record_to_mapping(record)
    try:
        payload = json.dumps(
            mapping,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise JsonlValidationError(f"record is not JSON serializable: {exc}") from exc
    return mapping, payload + "\n"


def _read_existing_records(path: Path) -> tuple[JsonObject, ...]:
    if path.is_symlink():
        raise CorruptJsonlError(f"refusing to append through a symbolic link: {path}")
    if not path.exists():
        return ()
    if not path.is_file():
        raise CorruptJsonlError(f"JSONL path is not a regular file: {path}")

    records: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n") or line.endswith("\r\n"):
                    raise CorruptJsonlError(
                        f"{path} line {line_number} is not LF-terminated"
                    )
                if not line.strip():
                    raise CorruptJsonlError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                try:
                    value = json.loads(line, object_pairs_hook=_strict_json_object)
                except _DuplicateJsonKeyError as exc:
                    raise CorruptJsonlError(
                        f"invalid JSON in {path} line {line_number}: {exc}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise CorruptJsonlError(
                        f"invalid JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise CorruptJsonlError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                if value.get("schema_version") != SCHEMA_VERSION:
                    raise CorruptJsonlError(
                        f"{path} line {line_number} has an unsupported schema_version"
                    )
                try:
                    _validate_json_value(value, f"existing line {line_number}")
                    _validate_canonical_mapping(path, value)
                    _validate_path_identity(path, value)
                except JsonlValidationError as exc:
                    raise CorruptJsonlError(
                        f"invalid value in {path} line {line_number}: {exc}"
                    ) from exc
                records.append(value)
    except CorruptJsonlError:
        raise
    except UnicodeError as exc:
        raise CorruptJsonlError(
            f"existing JSONL file is not valid UTF-8: {path}"
        ) from exc
    except OSError as exc:
        raise JsonlWriteError(
            f"could not inspect existing JSONL file {path}: {exc}"
        ) from exc
    return tuple(records)


def _extract_key(
    strategy: DuplicateKey,
    record: Mapping[str, Any],
    *,
    existing: bool,
) -> Hashable:
    try:
        key = strategy(record)
        hash(key)
    except Exception as exc:
        error_type = CorruptJsonlError if existing else JsonlValidationError
        location = "existing record" if existing else "new record"
        raise error_type(f"duplicate-key strategy failed for {location}: {exc}") from exc
    return key


def _field_key(fields: tuple[str, ...]) -> DuplicateKey:
    def extract(record: Mapping[str, Any]) -> Hashable:
        return tuple(record[field_name] for field_name in fields)

    return extract


def _cross_validate_persisted_memory_categories(
    retrieval_path: Path,
    retrieval: RetrievalLogRecord,
) -> None:
    memory_path = retrieval_path.parent / "memory_logs.jsonl"
    memory_records = _read_existing_records(memory_path)
    memory_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for memory in memory_records:
        key = (
            memory["experiment_id"],
            memory["run_id"],
            memory["memory_id"],
        )
        if key in memory_index:
            raise CorruptJsonlError(
                f"duplicate persisted memory identity {key!r} in {memory_path}"
            )
        memory_index[key] = memory

    run = retrieval.turn.run
    for candidate in retrieval.candidates:
        memory = memory_index.get(
            (run.experiment_id, run.run_id, candidate.memory_id)
        )
        if memory is None:
            continue
        persisted_run = _run_metadata_from_mapping(memory)
        if persisted_run != run:
            raise JsonlValidationError(
                f"candidate {candidate.memory_id!r} has different persisted run metadata"
            )
        if memory["agent_id"] != retrieval.turn.agent_id:
            raise JsonlValidationError(
                f"candidate {candidate.memory_id!r} resolves to a foreign agent"
            )
        if memory["category"] != candidate.category:
            raise JsonlValidationError(
                f"candidate {candidate.memory_id!r} category does not match memory_logs.jsonl"
            )


def _cross_validate_memory_with_persisted_retrievals(
    memory_path: Path,
    memory: MemoryLogRecord,
) -> None:
    retrieval_path = memory_path.parent / "retrieval_logs.jsonl"
    retrieval_records = _read_existing_records(retrieval_path)
    for retrieval in retrieval_records:
        for candidate in retrieval["candidates"]:
            if candidate["memory_id"] != memory.memory_id:
                continue
            persisted_run = _run_metadata_from_mapping(retrieval)
            if persisted_run != memory.run:
                raise JsonlValidationError(
                    f"memory {memory.memory_id!r} has different retrieval run metadata"
                )
            if retrieval["agent_id"] != memory.agent_id:
                raise JsonlValidationError(
                    f"memory {memory.memory_id!r} resolves to a foreign retrieval agent"
                )
            if candidate["category"] != memory.category:
                raise JsonlValidationError(
                    f"memory {memory.memory_id!r} category does not match retrieval_logs.jsonl"
                )


def _cross_validate_seed_memory_with_snapshot(
    memory_path: Path,
    memory: MemoryLogRecord,
) -> None:
    """Validate one realized seed against the persisted canonical seed spec."""
    if memory.memory_type != "seed_persona":
        return

    from src.config_loader import (
        ConfigLoadError,
        ConfigValidationError,
        load_experiment_config,
    )

    snapshot = memory_path.parent / "config_snapshot"
    try:
        config = load_experiment_config(snapshot)
    except (ConfigLoadError, ConfigValidationError) as exc:
        raise JsonlValidationError(
            f"cannot validate seed_persona against config_snapshot/agents.json: {exc}"
        ) from exc

    seed_index = memory.metadata.get("seed_index")
    if (
        isinstance(seed_index, bool)
        or not isinstance(seed_index, int)
        or seed_index < 1
    ):
        raise JsonlValidationError(
            "seed_persona metadata.seed_index must be a one-based positive integer"
        )

    agents_by_id = {agent.agent_id: agent for agent in config.agents}
    agent = agents_by_id.get(memory.agent_id)
    if agent is None:
        raise JsonlValidationError(
            f"seed_persona agent_id {memory.agent_id!r} is absent from agents.json"
        )
    if seed_index > len(agent.seed_memories):
        raise JsonlValidationError(
            f"seed_persona seed_index {seed_index} is out of range for "
            f"agent_id {memory.agent_id!r}"
        )

    expected_memory_id = f"mem_{memory.agent_id}_{seed_index:06d}"
    if memory.memory_id != expected_memory_id:
        raise JsonlValidationError(
            "seed_persona memory_id does not match agent_id and deterministic seed order"
        )

    expected = agent.seed_memories[seed_index - 1]
    for field_name in ("text", "category", "importance"):
        if getattr(memory, field_name) != getattr(expected, field_name):
            raise JsonlValidationError(
                f"seed_persona {field_name} does not match "
                "config_snapshot/agents.json"
            )


def _require_matching_turn_metadata(
    persisted: Mapping[str, Any],
    current: TurnLogMetadata,
    context: str,
) -> None:
    current_fields = current.to_dict()
    mismatched = sorted(
        field_name
        for field_name in (_RUN_FIELDS | _STEP_FIELDS | _TURN_FIELDS)
        if persisted[field_name] != current_fields[field_name]
    )
    if mismatched:
        raise JsonlValidationError(
            f"{context} has inconsistent turn metadata: {', '.join(mismatched)}"
        )


def _cross_validate_retrieval_with_persisted_behavior(
    retrieval_path: Path,
    retrieval: RetrievalLogRecord,
) -> None:
    behavior_path = retrieval_path.parent / "behavior_logs.jsonl"
    behavior_records = _read_existing_records(behavior_path)
    for behavior in behavior_records:
        if behavior["turn_id"] != retrieval.turn.turn_id:
            continue
        _require_matching_turn_metadata(
            behavior,
            retrieval.turn,
            "behavior/retrieval records",
        )
        if behavior["retrieved_memory_ids"] != list(retrieval.selected_memory_ids):
            raise JsonlValidationError(
                "behavior retrieved_memory_ids do not match retrieval selected_memory_ids"
            )


def _cross_validate_behavior_with_persisted_retrieval(
    behavior_path: Path,
    behavior: BehaviorLogRecord,
) -> None:
    retrieval_path = behavior_path.parent / "retrieval_logs.jsonl"
    retrieval_records = _read_existing_records(retrieval_path)
    for retrieval in retrieval_records:
        if retrieval["turn_id"] != behavior.turn.turn_id:
            continue
        _require_matching_turn_metadata(
            retrieval,
            behavior.turn,
            "retrieval/behavior records",
        )
        if retrieval["selected_memory_ids"] != list(behavior.retrieved_memory_ids):
            raise JsonlValidationError(
                "behavior retrieved_memory_ids do not match retrieval selected_memory_ids"
            )


class JsonlLogger:
    """Append complete deterministic records to one JSONL stream.

    ``duplicate_key`` supplements, but cannot replace, the fixed duplicate
    policy for canonical research streams.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        duplicate_key: DuplicateKey | None = None,
    ) -> None:
        self._path = Path(path)
        if duplicate_key is not None and not callable(duplicate_key):
            raise TypeError("duplicate_key must be callable or None")
        self._duplicate_key = duplicate_key

    @property
    def path(self) -> Path:
        """Return the target path without creating it."""
        return self._path

    def append(self, record: object) -> None:
        """Validate and durably append one LF-terminated record.

        Existing data is inspected before the file is opened in append mode.
        A malformed or incomplete existing line is never repaired or truncated.
        """
        _validate_record_type_for_path(self._path, record)
        mapping, serialized = _serialize_record(record)
        _validate_canonical_mapping(self._path, mapping)
        _validate_path_identity(self._path, mapping)

        duplicate_strategies: list[DuplicateKey] = []
        canonical_fields = _CANONICAL_DUPLICATE_FIELDS.get(self._path.name)
        if canonical_fields is not None:
            duplicate_strategies.append(_field_key(canonical_fields))
        if self._duplicate_key is not None:
            duplicate_strategies.append(self._duplicate_key)
        new_keys = tuple(
            _extract_key(strategy, mapping, existing=False)
            for strategy in duplicate_strategies
        )

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JsonlWriteError(
                f"could not create JSONL parent directory {self._path.parent}: {exc}"
            ) from exc

        _ensure_sibling_run_is_writable(self._path, mapping)
        if (
            self._path.name == "retrieval_logs.jsonl"
            and isinstance(record, RetrievalLogRecord)
        ):
            _cross_validate_persisted_memory_categories(self._path, record)
            _cross_validate_retrieval_with_persisted_behavior(self._path, record)
        elif (
            self._path.name == "behavior_logs.jsonl"
            and isinstance(record, BehaviorLogRecord)
        ):
            _cross_validate_behavior_with_persisted_retrieval(self._path, record)
        elif (
            self._path.name == "memory_logs.jsonl"
            and isinstance(record, MemoryLogRecord)
        ):
            _cross_validate_seed_memory_with_snapshot(self._path, record)
            _cross_validate_memory_with_persisted_retrievals(self._path, record)
        existing_records = _read_existing_records(self._path)
        if duplicate_strategies:
            for existing_record in existing_records:
                for strategy, new_key in zip(duplicate_strategies, new_keys):
                    existing_key = _extract_key(
                        strategy,
                        existing_record,
                        existing=True,
                    )
                    if existing_key == new_key:
                        raise DuplicateRecordError(
                            f"duplicate JSONL record key {new_key!r} in {self._path}"
                        )

        try:
            with self._path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise JsonlWriteError(f"could not append JSONL record to {self._path}: {exc}") from exc


def append_jsonl(
    path: str | Path,
    record: object,
    *,
    duplicate_key: DuplicateKey | None = None,
) -> None:
    """Append one record without retaining writer state."""
    JsonlLogger(path, duplicate_key=duplicate_key).append(record)


def _ensure_sibling_run_is_writable(
    jsonl_path: Path,
    record: Mapping[str, Any],
) -> None:
    """Protect canonical run logs once their sibling summary is terminal."""
    summary_path = jsonl_path.parent / "run_summary.json"
    if not summary_path.exists() and not summary_path.is_symlink():
        return

    # Local import keeps the low-level JSONL serializer independent while
    # enforcing the published run lifecycle when a summary is present.
    from src.logging_system.run_summary import (
        TERMINAL_RUN_STATUSES,
        RunSummaryStateError,
        read_run_summary,
    )

    run_summary = read_run_summary(summary_path)
    if run_summary.status in TERMINAL_RUN_STATUSES:
        raise RunSummaryStateError(
            f"run summary status {run_summary.status!r} is terminal and read-only"
        )
    identity_fields = (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    )
    try:
        mismatched = [
            field_name
            for field_name in identity_fields
            if record[field_name] != getattr(run_summary, field_name)
        ]
    except KeyError as exc:
        raise JsonlValidationError(
            "record beside run_summary.json is missing shared run metadata"
        ) from exc
    if mismatched:
        raise JsonlValidationError(
            "record identity does not match sibling run summary: "
            + ", ".join(mismatched)
        )
