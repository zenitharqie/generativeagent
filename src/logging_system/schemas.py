"""Immutable Phase 07 log records and explicit serialization adapters.

The cognitive pipeline remains unaware of logging.  Callers supply run/turn
metadata and the query embedding that is not retained by ``CognitiveResult``;
these adapters only project already-computed immutable results into passive
records.  No retrieval, embedding, or model operation is performed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
from math import isclose, isfinite
import re
from types import MappingProxyType
from typing import Any

from src.cognition.pipeline import CognitiveResult
from src.config_loader.validators import FIXED_EVENT_IDS, FIXED_RETRIEVAL_WEIGHTS
from src.domain.agents import FIXED_AGENT_IDS
from src.domain.memory import MEMORY_TYPES, SOURCE_STEP_TYPES, MemoryEntry
from src.domain.retrieval import WEIGHT_SUM_TOLERANCE
from src.domain.simulation import STEP_TYPES
from src.llm.base import LLMCallType
from src.retrieval.result import RetrievalCandidate, RetrievalResult


SCHEMA_VERSION = "1.0"
FIXED_TOP_K = 5

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class LoggingError(ValueError):
    """Base error for invalid or unserializable Phase 07 log data."""


class LogSchemaError(LoggingError):
    """Raised when a log record violates its persisted schema."""


class SerializationError(LoggingError):
    """Raised when a supported record cannot be serialized as strict JSON."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LogSchemaError(f"{field_name} must be a non-empty string")
    return value


def _require_identifier(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    if _IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise LogSchemaError(
            f"{field_name} must be a single safe identifier segment"
        )
    return text


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise LogSchemaError(f"{field_name} must be boolean")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogSchemaError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LogSchemaError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(value: object, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise LogSchemaError(f"{field_name} must be a finite number")
    return value


def _normalize_embedding(
    value: object,
    field_name: str,
) -> tuple[int | float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or not value:
        raise LogSchemaError(
            f"{field_name} must be a non-empty tuple of finite numbers or None"
        )
    return tuple(
        _require_finite_number(component, f"{field_name} component")
        for component in value
    )


def _embedding_from_sequence(
    value: Sequence[float] | None,
    field_name: str,
) -> tuple[int | float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LogSchemaError(f"{field_name} must be a numeric sequence or None")
    return _normalize_embedding(tuple(value), field_name)


def _normalize_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise LogSchemaError(f"{field_name} must be a tuple of strings")
    normalized = tuple(_require_identifier(item, f"{field_name} item") for item in value)
    if len(normalized) != len(set(normalized)):
        raise LogSchemaError(f"{field_name} must not contain duplicate values")
    return normalized


def _freeze_json(value: object, field_name: str) -> object:
    """Validate JSON data while converting mutable containers to immutable ones."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _require_finite_number(value, field_name)
    if isinstance(value, Enum):
        return _freeze_json(value.value, field_name)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LogSchemaError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, f"{field_name} item") for item in value
        )
    raise LogSchemaError(
        f"{field_name} contains unsupported value type {type(value).__name__}"
    )


def _thaw_json(value: object) -> Any:
    """Return a fresh JSON-compatible copy of an immutable schema value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Enum):
        return _thaw_json(value.value)
    return value


@dataclass(frozen=True)
class RunLogMetadata:
    """Caller-supplied identity shared by all records in one run."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "run_id", "retrieval_config_id"):
            object.__setattr__(
                self,
                field_name,
                _require_identifier(getattr(self, field_name), field_name),
            )
        if self.retrieval_config_id not in FIXED_RETRIEVAL_WEIGHTS:
            raise LogSchemaError(
                f"unsupported retrieval_config_id: {self.retrieval_config_id!r}"
            )
        _require_non_negative_int(self.repetition_index, "repetition_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
        }


@dataclass(frozen=True)
class StepLogMetadata:
    """Validated logical simulation metadata shared by records for one step."""

    run: RunLogMetadata
    step_id: str
    step_type: str
    is_target_event: bool
    evaluation_enabled: bool
    day: int
    period: str
    timestamp: int
    event_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunLogMetadata):
            raise LogSchemaError("run must be RunLogMetadata")
        object.__setattr__(
            self, "step_id", _require_identifier(self.step_id, "step_id")
        )
        if self.step_type not in STEP_TYPES:
            raise LogSchemaError(f"unsupported step_type: {self.step_type!r}")
        _require_bool(self.is_target_event, "is_target_event")
        _require_bool(self.evaluation_enabled, "evaluation_enabled")
        expected_target = self.step_type == "target_event"
        if self.is_target_event != expected_target:
            raise LogSchemaError("is_target_event must match step_type")
        if self.evaluation_enabled != expected_target:
            raise LogSchemaError(
                "evaluation_enabled must be true only for target-event steps"
            )
        _require_positive_int(self.day, "day")
        if self.day > 6:
            raise LogSchemaError("day must be within the fixed 6-day timeline")
        object.__setattr__(self, "period", _require_text(self.period, "period"))
        _require_non_negative_int(self.timestamp, "timestamp")
        if expected_target:
            object.__setattr__(
                self,
                "event_id",
                _require_identifier(self.event_id, "event_id"),
            )
            if self.event_id not in FIXED_EVENT_IDS:
                raise LogSchemaError(f"unsupported event_id: {self.event_id!r}")
        elif self.event_id is not None:
            raise LogSchemaError("natural-interaction metadata must not have event_id")

    def to_dict(self) -> dict[str, Any]:
        data = self.run.to_dict()
        data.update(
            {
                "step_id": self.step_id,
                "step_type": self.step_type,
                "is_target_event": self.is_target_event,
                "evaluation_enabled": self.evaluation_enabled,
                "day": self.day,
                "period": self.period,
                "timestamp": self.timestamp,
                "event_id": self.event_id,
            }
        )
        return data


@dataclass(frozen=True)
class TurnLogMetadata:
    """Validated caller-supplied identity for one agent's logical turn."""

    step: StepLogMetadata
    turn_id: str
    turn_index: int
    agent_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.step, StepLogMetadata):
            raise LogSchemaError("step must be StepLogMetadata")
        object.__setattr__(
            self, "turn_id", _require_identifier(self.turn_id, "turn_id")
        )
        _require_non_negative_int(self.turn_index, "turn_index")
        if self.agent_id not in FIXED_AGENT_IDS:
            raise LogSchemaError(f"unsupported agent_id: {self.agent_id!r}")

    @property
    def run(self) -> RunLogMetadata:
        return self.step.run

    def to_dict(self) -> dict[str, Any]:
        data = self.step.to_dict()
        data.update(
            {
                "turn_id": self.turn_id,
                "turn_index": self.turn_index,
                "agent_id": self.agent_id,
            }
        )
        return data


@dataclass(frozen=True)
class RetrievalWeightsRecord:
    """Explicit JSON projection of the three research-critical weights."""

    recency: float
    importance: float
    relevance: float

    def __post_init__(self) -> None:
        for field_name in ("recency", "importance", "relevance"):
            value = _require_finite_number(getattr(self, field_name), field_name)
            if value < 0:
                raise LogSchemaError(f"{field_name} weight must be non-negative")
        if not isclose(
            self.recency + self.importance + self.relevance,
            1.0,
            rel_tol=0.0,
            abs_tol=WEIGHT_SUM_TOLERANCE,
        ):
            raise LogSchemaError("retrieval weights must approximately sum to 1.0")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "recency": self.recency,
            "importance": self.importance,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class RetrievalCandidateRecord:
    """One ranked candidate without duplicated memory text or embedding."""

    memory_id: str
    category: str
    raw_recency: float
    raw_importance: float
    raw_relevance: float
    normalized_recency: float
    normalized_importance: float
    normalized_relevance: float
    final_score: float
    selected: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "memory_id", _require_identifier(self.memory_id, "memory_id")
        )
        object.__setattr__(self, "category", _require_text(self.category, "category"))
        for field_name in (
            "raw_recency",
            "raw_importance",
            "raw_relevance",
            "final_score",
        ):
            _require_finite_number(getattr(self, field_name), field_name)
        for field_name in (
            "normalized_recency",
            "normalized_importance",
            "normalized_relevance",
        ):
            value = _require_finite_number(getattr(self, field_name), field_name)
            if not 0 <= value <= 1:
                raise LogSchemaError(f"{field_name} must be between 0 and 1")
        _require_bool(self.selected, "selected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "raw_recency": self.raw_recency,
            "raw_importance": self.raw_importance,
            "raw_relevance": self.raw_relevance,
            "normalized_recency": self.normalized_recency,
            "normalized_importance": self.normalized_importance,
            "normalized_relevance": self.normalized_relevance,
            "final_score": self.final_score,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class RetrievalLogRecord:
    """Complete pre-access retrieval audit for one cognitive turn.

    ``RetrievalResult.applied_weights`` is persisted as ``weights`` because the
    Phase 07 log contract names that field ``weights``. All component names and
    numeric values remain unchanged.
    """

    turn: TurnLogMetadata
    query: str
    query_embedding: tuple[int | float, ...] | None
    weights: RetrievalWeightsRecord
    top_k: int
    candidate_count: int
    candidates: tuple[RetrievalCandidateRecord, ...]
    selected_memory_ids: tuple[str, ...]
    accessed_memory_ids: tuple[str, ...]
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnLogMetadata):
            raise LogSchemaError("turn must be TurnLogMetadata")
        object.__setattr__(self, "query", _require_text(self.query, "query"))
        object.__setattr__(
            self,
            "query_embedding",
            _normalize_embedding(self.query_embedding, "query_embedding"),
        )
        if not isinstance(self.weights, RetrievalWeightsRecord):
            raise LogSchemaError("weights must be RetrievalWeightsRecord")
        actual_weights = (
            self.weights.recency,
            self.weights.importance,
            self.weights.relevance,
        )
        expected_weights = FIXED_RETRIEVAL_WEIGHTS[
            self.turn.run.retrieval_config_id
        ]
        if actual_weights != expected_weights:
            raise LogSchemaError(
                "weights must match the fixed retrieval_config_id configuration"
            )
        _require_positive_int(self.top_k, "top_k")
        if self.top_k != FIXED_TOP_K:
            raise LogSchemaError(f"top_k must equal the fixed research value {FIXED_TOP_K}")
        _require_non_negative_int(self.candidate_count, "candidate_count")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, RetrievalCandidateRecord)
            for candidate in self.candidates
        ):
            raise LogSchemaError(
                "candidates must be a tuple of RetrievalCandidateRecord values"
            )
        if self.candidate_count != len(self.candidates):
            raise LogSchemaError("candidate_count must equal candidates length")
        object.__setattr__(
            self,
            "selected_memory_ids",
            _normalize_string_tuple(
                self.selected_memory_ids, "selected_memory_ids"
            ),
        )
        object.__setattr__(
            self,
            "accessed_memory_ids",
            _normalize_string_tuple(
                self.accessed_memory_ids, "accessed_memory_ids"
            ),
        )
        candidate_ids = tuple(candidate.memory_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise LogSchemaError("retrieval candidates must have unique memory IDs")
        expected_selected = candidate_ids[: min(self.top_k, self.candidate_count)]
        flagged_selected = tuple(
            candidate.memory_id for candidate in self.candidates if candidate.selected
        )
        if self.selected_memory_ids != expected_selected:
            raise LogSchemaError("selected_memory_ids must be the ranked Top-K IDs")
        if flagged_selected != expected_selected:
            raise LogSchemaError("selected flags must identify the ranked Top-K")
        if self.accessed_memory_ids != self.selected_memory_ids:
            raise LogSchemaError(
                "accessed_memory_ids must equal selected_memory_ids in order"
            )

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": self.schema_version}
        data.update(self.turn.to_dict())
        data.update(
            {
                "query": self.query,
                "query_embedding": (
                    None
                    if self.query_embedding is None
                    else list(self.query_embedding)
                ),
                "weights": self.weights.to_dict(),
                "top_k": self.top_k,
                "candidate_count": self.candidate_count,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "selected_memory_ids": list(self.selected_memory_ids),
                "accessed_memory_ids": list(self.accessed_memory_ids),
            }
        )
        return data


@dataclass(frozen=True)
class BehaviorLogRecord:
    """Outward action projection with no unselected retrieval audit data."""

    turn: TurnLogMetadata
    response: str
    retrieved_memory_ids: tuple[str, ...]
    plan_summary: str | None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnLogMetadata):
            raise LogSchemaError("turn must be TurnLogMetadata")
        object.__setattr__(self, "response", _require_text(self.response, "response"))
        object.__setattr__(
            self,
            "retrieved_memory_ids",
            _normalize_string_tuple(
                self.retrieved_memory_ids, "retrieved_memory_ids"
            ),
        )
        if self.plan_summary is not None:
            object.__setattr__(
                self,
                "plan_summary",
                _require_text(self.plan_summary, "plan_summary"),
            )

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": self.schema_version}
        data.update(self.turn.to_dict())
        data.update(
            {
                "response": self.response,
                "retrieved_memory_ids": list(self.retrieved_memory_ids),
                "plan_summary": self.plan_summary,
            }
        )
        return data


@dataclass(frozen=True)
class MemoryLogRecord:
    """One complete memory, with source fields mapped to shared step fields.

    ``MemoryEntry.source_step_id`` and ``source_step_type`` are persisted as
    ``step_id`` and ``step_type`` to satisfy the Phase 07 shared metadata
    contract.  Seed memories may omit turn context while retaining their
    ``agent_initialization`` source type.
    """

    run: RunLogMetadata
    memory_id: str
    agent_id: str
    memory_type: str
    category: str
    text: str
    importance: float
    created_at: int
    last_accessed_at: int
    step_id: str
    step_type: str
    embedding: tuple[int | float, ...] | None
    metadata: Mapping[str, Any]
    turn: TurnLogMetadata | None = None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunLogMetadata):
            raise LogSchemaError("run must be RunLogMetadata")
        object.__setattr__(
            self, "memory_id", _require_identifier(self.memory_id, "memory_id")
        )
        if self.agent_id not in FIXED_AGENT_IDS:
            raise LogSchemaError(f"unsupported agent_id: {self.agent_id!r}")
        if self.memory_type not in MEMORY_TYPES:
            raise LogSchemaError(f"unsupported memory_type: {self.memory_type!r}")
        object.__setattr__(self, "category", _require_text(self.category, "category"))
        object.__setattr__(self, "text", _require_text(self.text, "text"))
        importance = _require_finite_number(self.importance, "importance")
        if not 1 <= importance <= 10:
            raise LogSchemaError("importance must be between 1 and 10")
        _require_non_negative_int(self.created_at, "created_at")
        _require_non_negative_int(self.last_accessed_at, "last_accessed_at")
        if self.last_accessed_at < self.created_at:
            raise LogSchemaError("last_accessed_at must not precede created_at")
        object.__setattr__(
            self, "step_id", _require_identifier(self.step_id, "step_id")
        )
        if self.step_type not in SOURCE_STEP_TYPES:
            raise LogSchemaError(f"unsupported memory step_type: {self.step_type!r}")
        if self.step_type == "agent_initialization" and self.memory_type != "seed_persona":
            raise LogSchemaError(
                "agent_initialization source is valid only for seed_persona memories"
            )
        object.__setattr__(
            self,
            "embedding",
            _normalize_embedding(self.embedding, "embedding"),
        )
        if not isinstance(self.metadata, Mapping):
            raise LogSchemaError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))
        if self.turn is None:
            if self.step_type != "agent_initialization":
                raise LogSchemaError(
                    "simulation memory records require source turn metadata"
                )
        else:
            if not isinstance(self.turn, TurnLogMetadata):
                raise LogSchemaError("turn must be TurnLogMetadata or None")
            if self.turn.run != self.run:
                raise LogSchemaError("memory turn metadata belongs to a different run")
            if (
                self.turn.agent_id != self.agent_id
                or self.turn.step.step_id != self.step_id
                or self.turn.step.step_type != self.step_type
                or self.turn.step.timestamp != self.created_at
            ):
                raise LogSchemaError(
                    "memory identity does not match supplied turn metadata"
                )

    def to_dict(self) -> dict[str, Any]:
        if self.turn is None:
            data = self.run.to_dict()
            data.update(
                {
                    "step_id": self.step_id,
                    "step_type": self.step_type,
                    "is_target_event": False,
                    "evaluation_enabled": False,
                    "day": None,
                    "period": None,
                    "timestamp": self.created_at,
                    "event_id": None,
                    "turn_id": None,
                    "turn_index": None,
                    "agent_id": self.agent_id,
                }
            )
        else:
            data = self.turn.to_dict()
        data = {"schema_version": self.schema_version, **data}
        data.update(
            {
                "memory_id": self.memory_id,
                "memory_type": self.memory_type,
                "category": self.category,
                "text": self.text,
                "importance": self.importance,
                "created_at": self.created_at,
                "last_accessed_at": self.last_accessed_at,
                "embedding": None if self.embedding is None else list(self.embedding),
                "metadata": _thaw_json(self.metadata),
            }
        )
        return data


@dataclass(frozen=True)
class ReflectionLogRecord:
    """One explicitly requested reflection and its optional stored memory ID."""

    turn: TurnLogMetadata
    reflection: str
    category: str
    memory_id: str | None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnLogMetadata):
            raise LogSchemaError("turn must be TurnLogMetadata")
        object.__setattr__(
            self, "reflection", _require_text(self.reflection, "reflection")
        )
        object.__setattr__(self, "category", _require_text(self.category, "category"))
        if self.memory_id is not None:
            object.__setattr__(
                self,
                "memory_id",
                _require_identifier(self.memory_id, "memory_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": self.schema_version}
        data.update(self.turn.to_dict())
        data.update(
            {
                "reflection": self.reflection,
                "category": self.category,
                "memory_id": self.memory_id,
            }
        )
        return data


@dataclass(frozen=True)
class APIUsageLogRecord:
    """Passive future-provider usage metadata; ``prompt_template`` is an ID."""

    turn: TurnLogMetadata
    call_type: LLMCallType | str
    model: str
    prompt_template: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    status: str
    error_type: str | None = None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnLogMetadata):
            raise LogSchemaError("turn must be TurnLogMetadata")
        try:
            call_type = (
                self.call_type
                if isinstance(self.call_type, LLMCallType)
                else LLMCallType(self.call_type)
            )
        except (TypeError, ValueError) as exc:
            raise LogSchemaError(f"unsupported call_type: {self.call_type!r}") from exc
        object.__setattr__(self, "call_type", call_type)
        for field_name in ("model", "status"):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "prompt_template",
            _require_identifier(self.prompt_template, "prompt_template"),
        )
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            _require_non_negative_int(getattr(self, field_name), field_name)
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise LogSchemaError("total_tokens must equal input_tokens + output_tokens")
        if self.error_type is not None:
            object.__setattr__(
                self,
                "error_type",
                _require_identifier(self.error_type, "error_type"),
            )

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": self.schema_version}
        data.update(self.turn.to_dict())
        data.update(
            {
                "call_type": self.call_type.value,
                "model": self.model,
                "prompt_template": self.prompt_template,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "status": self.status,
                "error_type": self.error_type,
            }
        )
        return data


@dataclass(frozen=True)
class ErrorLogRecord:
    """Sanitized explicit failure metadata with optional step/turn context."""

    run: RunLogMetadata
    error_type: str
    message: str
    recoverable: bool
    retry_count: int | None = None
    context: StepLogMetadata | TurnLogMetadata | None = None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunLogMetadata):
            raise LogSchemaError("run must be RunLogMetadata")
        object.__setattr__(
            self,
            "error_type",
            _require_identifier(self.error_type, "error_type"),
        )
        object.__setattr__(self, "message", _require_text(self.message, "message"))
        _require_bool(self.recoverable, "recoverable")
        if self.retry_count is not None:
            _require_non_negative_int(self.retry_count, "retry_count")
        if self.context is not None:
            if not isinstance(self.context, (StepLogMetadata, TurnLogMetadata)):
                raise LogSchemaError(
                    "context must be StepLogMetadata, TurnLogMetadata, or None"
                )
            context_run = (
                self.context.run
                if isinstance(self.context, TurnLogMetadata)
                else self.context.run
            )
            if context_run != self.run:
                raise LogSchemaError("error context belongs to a different run")

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": self.schema_version}
        data.update(self.run.to_dict())
        data.update(
            {
                "step_id": None,
                "step_type": None,
                "is_target_event": None,
                "evaluation_enabled": None,
                "day": None,
                "period": None,
                "timestamp": None,
                "event_id": None,
                "turn_id": None,
                "turn_index": None,
                "agent_id": None,
            }
        )
        if isinstance(self.context, StepLogMetadata):
            data.update(self.context.to_dict())
        elif isinstance(self.context, TurnLogMetadata):
            data.update(self.context.to_dict())
        data.update(
            {
                "error_type": self.error_type,
                "message": self.message,
                "recoverable": self.recoverable,
                "retry_count": self.retry_count,
            }
        )
        return data


@dataclass(frozen=True)
class CognitiveLogRecords:
    """Non-persisted bundle of specialized records for one successful turn."""

    retrieval: RetrievalLogRecord
    behavior: BehaviorLogRecord
    memories: tuple[MemoryLogRecord, ...]
    reflection: ReflectionLogRecord | None


PersistedLogRecord = (
    RetrievalLogRecord
    | BehaviorLogRecord
    | MemoryLogRecord
    | ReflectionLogRecord
    | APIUsageLogRecord
    | ErrorLogRecord
)


def _validate_result_identity(
    result: CognitiveResult,
    turn: TurnLogMetadata,
) -> None:
    if not isinstance(result, CognitiveResult):
        raise LogSchemaError("result must be a CognitiveResult")
    if not isinstance(turn, TurnLogMetadata):
        raise LogSchemaError("turn must be TurnLogMetadata")
    if (
        turn.turn_id != result.turn_id
        or turn.agent_id != result.agent_id
        or turn.step.step_id != result.source_step_id
        or turn.step.step_type != result.source_step_type
        or turn.step.timestamp != result.logical_timestamp
        or turn.step.event_id != result.event_id
        or turn.run.retrieval_config_id != result.retrieval_config_id
    ):
        raise LogSchemaError("turn metadata identity does not match CognitiveResult")
    perception = result.perception
    if (
        turn.step.day != perception.day
        or turn.step.period != perception.period
        or turn.step.is_target_event
        != (result.source_step_type == "target_event")
    ):
        raise LogSchemaError("turn metadata does not match result perception")

    memories = tuple(
        candidate.memory for candidate in result.retrieval_result.candidates
    ) + result.created_memories
    for memory in memories:
        if memory.agent_id != result.agent_id:
            raise LogSchemaError("cognitive result contains foreign-agent memory")
        if memory.run_id != turn.run.run_id:
            raise LogSchemaError("cognitive result contains cross-run memory")


def turn_metadata_from_result(
    result: CognitiveResult,
    *,
    run: RunLogMetadata,
    turn_index: int,
    evaluation_enabled: bool,
) -> TurnLogMetadata:
    """Build and cross-check caller-owned metadata for a completed result."""
    if not isinstance(result, CognitiveResult):
        raise LogSchemaError("result must be a CognitiveResult")
    turn = TurnLogMetadata(
        step=StepLogMetadata(
            run=run,
            step_id=result.source_step_id,
            step_type=result.source_step_type,
            is_target_event=result.source_step_type == "target_event",
            evaluation_enabled=evaluation_enabled,
            day=result.perception.day,
            period=result.perception.period,
            timestamp=result.logical_timestamp,
            event_id=result.event_id,
        ),
        turn_id=result.turn_id,
        turn_index=turn_index,
        agent_id=result.agent_id,
    )
    _validate_result_identity(result, turn)
    return turn


def _candidate_record(candidate: RetrievalCandidate) -> RetrievalCandidateRecord:
    if not isinstance(candidate, RetrievalCandidate):
        raise LogSchemaError("candidate must be RetrievalCandidate")
    return RetrievalCandidateRecord(
        memory_id=candidate.memory_id,
        category=candidate.memory.category,
        raw_recency=candidate.raw_recency,
        raw_importance=candidate.raw_importance,
        raw_relevance=candidate.raw_relevance,
        normalized_recency=candidate.normalized_recency,
        normalized_importance=candidate.normalized_importance,
        normalized_relevance=candidate.normalized_relevance,
        final_score=candidate.final_score,
        selected=candidate.selected,
    )


def retrieval_record_from_result(
    result: CognitiveResult,
    turn: TurnLogMetadata,
    *,
    query_embedding: Sequence[float] | None,
) -> RetrievalLogRecord:
    """Project a complete pre-access retrieval result without rescoring it."""
    _validate_result_identity(result, turn)
    retrieval: RetrievalResult = result.retrieval_result
    embedding = _embedding_from_sequence(query_embedding, "query_embedding")
    if embedding is not None:
        for candidate in retrieval.candidates:
            memory_embedding = candidate.memory.embedding
            if memory_embedding is not None and len(memory_embedding) != len(embedding):
                raise LogSchemaError(
                    "query_embedding dimension does not match candidate memory embedding"
                )
    weights = retrieval.applied_weights
    return RetrievalLogRecord(
        turn=turn,
        query=retrieval.query,
        query_embedding=embedding,
        weights=RetrievalWeightsRecord(
            recency=weights.recency,
            importance=weights.importance,
            relevance=weights.relevance,
        ),
        top_k=retrieval.top_k,
        candidate_count=retrieval.candidate_count,
        candidates=tuple(_candidate_record(item) for item in retrieval.candidates),
        selected_memory_ids=retrieval.selected_memory_ids,
        accessed_memory_ids=result.accessed_memory_ids,
    )


def behavior_record_from_result(
    result: CognitiveResult,
    turn: TurnLogMetadata,
) -> BehaviorLogRecord:
    """Project only outward behavior and selected IDs from a completed turn."""
    _validate_result_identity(result, turn)
    return BehaviorLogRecord(
        turn=turn,
        response=result.action.response,
        retrieved_memory_ids=result.selected_memory_ids,
        plan_summary=result.plan.plan,
    )


def memory_record_from_entry(
    memory: MemoryEntry,
    *,
    run: RunLogMetadata,
    turn: TurnLogMetadata | None = None,
) -> MemoryLogRecord:
    """Serialize one existing memory without generating or altering an embedding."""
    if not isinstance(memory, MemoryEntry):
        raise LogSchemaError("memory must be a MemoryEntry")
    if memory.run_id != run.run_id:
        raise LogSchemaError("memory belongs to a different or unspecified run")
    return MemoryLogRecord(
        run=run,
        memory_id=memory.memory_id,
        agent_id=memory.agent_id,
        memory_type=memory.memory_type,
        category=memory.category,
        text=memory.text,
        importance=memory.importance,
        created_at=memory.created_at,
        last_accessed_at=memory.last_accessed_at,
        step_id=memory.source_step_id,
        step_type=memory.source_step_type,
        embedding=memory.embedding,
        metadata=memory.metadata,
        turn=turn,
    )


def memory_records_from_result(
    result: CognitiveResult,
    turn: TurnLogMetadata,
) -> tuple[MemoryLogRecord, ...]:
    """Project newly created memories in the pipeline's canonical order."""
    _validate_result_identity(result, turn)
    return tuple(
        memory_record_from_entry(memory, run=turn.run, turn=turn)
        for memory in result.created_memories
    )


def reflection_record_from_result(
    result: CognitiveResult,
    turn: TurnLogMetadata,
) -> ReflectionLogRecord | None:
    """Return one reflection record when reflection was explicitly produced."""
    _validate_result_identity(result, turn)
    if result.reflection is None:
        return None
    reflection_memory_id = next(
        (
            memory.memory_id
            for memory in result.created_memories
            if memory.memory_type == "reflection"
        ),
        None,
    )
    return ReflectionLogRecord(
        turn=turn,
        reflection=result.reflection.reflection,
        category=result.reflection.category,
        memory_id=reflection_memory_id,
    )


def cognitive_records_from_result(
    result: CognitiveResult,
    turn: TurnLogMetadata,
    *,
    query_embedding: Sequence[float] | None,
) -> CognitiveLogRecords:
    """Build all specialized success records without rerunning any stage."""
    return CognitiveLogRecords(
        retrieval=retrieval_record_from_result(
            result,
            turn,
            query_embedding=query_embedding,
        ),
        behavior=behavior_record_from_result(result, turn),
        memories=memory_records_from_result(result, turn),
        reflection=reflection_record_from_result(result, turn),
    )


def index_memory_records(
    records: Iterable[MemoryLogRecord],
) -> Mapping[tuple[str, str, str], MemoryLogRecord]:
    """Build a unique cross-run memory index for audit cross-validation."""
    index: dict[tuple[str, str, str], MemoryLogRecord] = {}
    for record in records:
        if not isinstance(record, MemoryLogRecord):
            raise LogSchemaError("memory record index accepts MemoryLogRecord values")
        key = (
            record.run.experiment_id,
            record.run.run_id,
            record.memory_id,
        )
        if key in index:
            raise LogSchemaError(
                "duplicate memory record for "
                f"experiment_id={key[0]!r}, run_id={key[1]!r}, memory_id={key[2]!r}"
            )
        index[key] = record
    return MappingProxyType(index)


def validate_candidate_memory_categories(
    retrieval: RetrievalLogRecord,
    memories: Iterable[MemoryLogRecord]
    | Mapping[tuple[str, str, str], MemoryLogRecord],
) -> None:
    """Cross-check candidate categories wherever a memory record is available."""
    if not isinstance(retrieval, RetrievalLogRecord):
        raise LogSchemaError("retrieval must be RetrievalLogRecord")
    if isinstance(memories, Mapping):
        memory_index = memories
        if any(
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(record, MemoryLogRecord)
            or key
            != (
                record.run.experiment_id,
                record.run.run_id,
                record.memory_id,
            )
            for key, record in memory_index.items()
        ):
            raise LogSchemaError("memory index has an invalid key or record")
    else:
        memory_index = index_memory_records(memories)
    experiment_id = retrieval.turn.run.experiment_id
    run_id = retrieval.turn.run.run_id
    for candidate in retrieval.candidates:
        memory = memory_index.get((experiment_id, run_id, candidate.memory_id))
        if memory is None:
            continue
        if memory.run != retrieval.turn.run:
            raise LogSchemaError(
                f"candidate {candidate.memory_id!r} resolves to different run metadata"
            )
        if memory.agent_id != retrieval.turn.agent_id:
            raise LogSchemaError(
                f"candidate {candidate.memory_id!r} resolves to a foreign agent"
            )
        if memory.category != candidate.category:
            raise LogSchemaError(
                f"candidate {candidate.memory_id!r} category does not match memory log"
            )


def record_to_dict(record: PersistedLogRecord) -> dict[str, Any]:
    """Return a defensive JSON-compatible mapping for a supported log record."""
    supported = (
        RetrievalLogRecord,
        BehaviorLogRecord,
        MemoryLogRecord,
        ReflectionLogRecord,
        APIUsageLogRecord,
        ErrorLogRecord,
    )
    if not isinstance(record, supported):
        raise SerializationError(
            f"unsupported persisted record type {type(record).__name__}"
        )
    data = record.to_dict()
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SerializationError("record has an unsupported schema_version")
    return _thaw_json(data)


def serialize_record(record: PersistedLogRecord) -> str:
    """Serialize a record deterministically as strict compact JSON, without LF."""
    try:
        return json.dumps(
            record_to_dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SerializationError):
            raise
        raise SerializationError("record is not strict JSON serializable") from exc


__all__ = [
    "APIUsageLogRecord",
    "BehaviorLogRecord",
    "CognitiveLogRecords",
    "ErrorLogRecord",
    "FIXED_TOP_K",
    "LogSchemaError",
    "LoggingError",
    "MemoryLogRecord",
    "PersistedLogRecord",
    "ReflectionLogRecord",
    "RetrievalCandidateRecord",
    "RetrievalLogRecord",
    "RetrievalWeightsRecord",
    "RunLogMetadata",
    "SCHEMA_VERSION",
    "SerializationError",
    "StepLogMetadata",
    "TurnLogMetadata",
    "behavior_record_from_result",
    "cognitive_records_from_result",
    "index_memory_records",
    "memory_record_from_entry",
    "memory_records_from_result",
    "record_to_dict",
    "reflection_record_from_result",
    "retrieval_record_from_result",
    "serialize_record",
    "turn_metadata_from_result",
    "validate_candidate_memory_categories",
]
