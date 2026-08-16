"""Immutable Phase 12 replay-domain and serialization contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from src.visualizer.errors import ReplaySchemaError


REPLAY_SCHEMA_VERSION = "1.0"
SOURCE_SCHEMA_VERSION = "1.0"
SUPPORTED_RUN_STATUSES = frozenset({"completed", "excluded", "failed", "interrupted"})
STEP_STATES = frozenset({"complete", "partial", "unobserved", "out_of_scope"})
TURN_STATES = frozenset({"complete", "partial"})
FRAME_KINDS = frozenset({"initialization", "main_turn", "reflection", "failure"})

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _text(value: object, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReplaySchemaError(f"{field_name} must be a non-empty string")
    return value


def _non_negative(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplaySchemaError(f"{field_name} must be a non-negative integer")
    return value


def _positive(value: object, field_name: str) -> int:
    value = _non_negative(value, field_name)
    if value == 0:
        raise ReplaySchemaError(f"{field_name} must be positive")
    return value


def _number(value: object, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ReplaySchemaError(f"{field_name} must be a finite number")
    return value


def _string_tuple(value: object, field_name: str, *, empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReplaySchemaError(f"{field_name} must be a tuple of non-empty strings")
    if not empty and not value:
        raise ReplaySchemaError(f"{field_name} must not be empty")
    return value


def _freeze_json(value: Any, field_name: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{field_name}[]") for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ReplaySchemaError(f"{field_name} keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{field_name}.{key}") for key, item in value.items()}
        )
    raise ReplaySchemaError(f"{field_name} contains an unsupported value")


def thaw_json(value: Any) -> Any:
    """Return a detached JSON-compatible value from deeply frozen data."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class ReplayIssue:
    code: str
    severity: str
    scope: str
    message: str

    def __post_init__(self) -> None:
        for name in ("code", "scope", "message"):
            _text(getattr(self, name), name)
        if self.severity not in {"warning", "incomplete"}:
            raise ReplaySchemaError("issue severity must be warning or incomplete")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "scope": self.scope,
            "message": self.message,
        }


@dataclass(frozen=True)
class ReplayAgent:
    agent_id: str
    display_name: str
    role: str
    archetype: str
    persona_summary: str
    traits: tuple[str, ...]
    goals: tuple[str, ...]
    communication_style: str
    motivations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "agent_id",
            "display_name",
            "role",
            "archetype",
            "persona_summary",
            "communication_style",
        ):
            _text(getattr(self, name), name)
        for name in ("traits", "goals", "motivations"):
            _string_tuple(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "role": self.role,
            "archetype": self.archetype,
            "persona_summary": self.persona_summary,
            "traits": list(self.traits),
            "goals": list(self.goals),
            "communication_style": self.communication_style,
            "motivations": list(self.motivations),
        }


@dataclass(frozen=True)
class ReplayMemory:
    memory_id: str
    owner_agent_id: str
    memory_type: str
    category: str
    text: str
    importance: int | float
    created_at: int
    last_accessed_at_at_persistence: int
    source_step_id: str
    source_step_type: str
    source_turn_id: str | None
    source_turn_index: int | None
    seed_index: int | None = None
    persisted_access_timestamps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "memory_id",
            "owner_agent_id",
            "memory_type",
            "category",
            "text",
            "source_step_id",
            "source_step_type",
        ):
            _text(getattr(self, name), name)
        _number(self.importance, "importance")
        _non_negative(self.created_at, "created_at")
        _non_negative(self.last_accessed_at_at_persistence, "last_accessed_at_at_persistence")
        if self.source_turn_id is not None:
            _text(self.source_turn_id, "source_turn_id")
            _positive(self.source_turn_index, "source_turn_index")
        elif self.source_turn_index is not None:
            raise ReplaySchemaError("turn index requires a source turn ID")
        if self.seed_index is not None:
            _positive(self.seed_index, "seed_index")
        if not isinstance(self.persisted_access_timestamps, tuple):
            raise ReplaySchemaError("persisted_access_timestamps must be a tuple")
        for timestamp in self.persisted_access_timestamps:
            _positive(timestamp, "persisted access timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "owner_agent_id": self.owner_agent_id,
            "memory_type": self.memory_type,
            "category": self.category,
            "text": self.text,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed_at_at_persistence": self.last_accessed_at_at_persistence,
            "source_step_id": self.source_step_id,
            "source_step_type": self.source_step_type,
            "source_turn_id": self.source_turn_id,
            "source_turn_index": self.source_turn_index,
            "seed_index": self.seed_index,
            "persisted_access_timestamps": list(self.persisted_access_timestamps),
        }


@dataclass(frozen=True)
class ReplayCandidate:
    rank: int
    memory_id: str
    category: str
    raw_recency: int | float
    raw_importance: int | float
    raw_relevance: int | float
    normalized_recency: int | float
    normalized_importance: int | float
    normalized_relevance: int | float
    final_score: int | float
    selected: bool

    def __post_init__(self) -> None:
        _positive(self.rank, "candidate rank")
        _text(self.memory_id, "memory_id")
        _text(self.category, "category")
        for name in (
            "raw_recency",
            "raw_importance",
            "raw_relevance",
            "normalized_recency",
            "normalized_importance",
            "normalized_relevance",
            "final_score",
        ):
            _number(getattr(self, name), name)
        if not isinstance(self.selected, bool):
            raise ReplaySchemaError("selected must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
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
class ReplayRetrieval:
    query: str
    weights: Mapping[str, int | float]
    top_k: int
    candidate_count: int
    candidates: tuple[ReplayCandidate, ...]
    selected_memory_ids: tuple[str, ...]
    accessed_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.query, "query")
        object.__setattr__(self, "weights", _freeze_json(self.weights, "weights"))
        if set(self.weights) != {"recency", "importance", "relevance"}:
            raise ReplaySchemaError("retrieval weights must have exactly three components")
        for name, value in self.weights.items():
            _number(value, f"weights.{name}")
        _positive(self.top_k, "top_k")
        _non_negative(self.candidate_count, "candidate_count")
        if not isinstance(self.candidates, tuple) or self.candidate_count != len(self.candidates):
            raise ReplaySchemaError("candidate_count must match candidate tuple")
        _string_tuple(self.selected_memory_ids, "selected_memory_ids", empty=True)
        _string_tuple(self.accessed_memory_ids, "accessed_memory_ids", empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "weights": thaw_json(self.weights),
            "top_k": self.top_k,
            "candidate_count": self.candidate_count,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_memory_ids": list(self.selected_memory_ids),
            "accessed_memory_ids": list(self.accessed_memory_ids),
        }


@dataclass(frozen=True)
class ReplayUsageRecord:
    turn_id: str
    call_type: str
    model: str
    prompt_template_id: str
    status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    error_type: str | None

    def __post_init__(self) -> None:
        for name in ("turn_id", "call_type", "model", "prompt_template_id", "status"):
            _text(getattr(self, name), name)
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            _non_negative(getattr(self, name), name)
        if self.error_type is not None:
            _text(self.error_type, "error_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "call_type": self.call_type,
            "model": self.model,
            "prompt_template_id": self.prompt_template_id,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class ReplayTurn:
    turn_id: str
    turn_index: int
    agent_id: str
    logical_timestamp: int
    completeness: str
    response: str | None
    plan_summary: str | None
    retrieval: ReplayRetrieval | None
    action_memory_id: str | None
    usage_records: tuple[ReplayUsageRecord, ...] = ()

    def __post_init__(self) -> None:
        _text(self.turn_id, "turn_id")
        _positive(self.turn_index, "turn_index")
        _text(self.agent_id, "agent_id")
        _positive(self.logical_timestamp, "logical_timestamp")
        if self.completeness not in TURN_STATES:
            raise ReplaySchemaError("turn completeness is invalid")
        for name in ("response", "plan_summary", "action_memory_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.retrieval is not None and not isinstance(self.retrieval, ReplayRetrieval):
            raise ReplaySchemaError("retrieval must be ReplayRetrieval or None")
        if not isinstance(self.usage_records, tuple):
            raise ReplaySchemaError("usage_records must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "agent_id": self.agent_id,
            "logical_timestamp": self.logical_timestamp,
            "completeness": self.completeness,
            "response": self.response,
            "plan_summary": self.plan_summary,
            "retrieval": None if self.retrieval is None else self.retrieval.to_dict(),
            "action_memory_id": self.action_memory_id,
            "usage_records": [item.to_dict() for item in self.usage_records],
        }


@dataclass(frozen=True)
class ReplayReflection:
    operation_id: str
    participant_index: int
    agent_id: str
    logical_timestamp: int
    completeness: str
    text: str | None
    category: str | None
    memory_id: str | None

    def __post_init__(self) -> None:
        _text(self.operation_id, "operation_id")
        _positive(self.participant_index, "participant_index")
        _text(self.agent_id, "agent_id")
        _positive(self.logical_timestamp, "logical_timestamp")
        if self.completeness not in TURN_STATES:
            raise ReplaySchemaError("reflection completeness is invalid")
        for name in ("text", "category", "memory_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "participant_index": self.participant_index,
            "agent_id": self.agent_id,
            "logical_timestamp": self.logical_timestamp,
            "completeness": self.completeness,
            "text": self.text,
            "category": self.category,
            "memory_id": self.memory_id,
        }


@dataclass(frozen=True)
class ReplayStep:
    step_id: str
    step_type: str
    day: int
    period: str
    title: str
    description: str
    participants: tuple[str, ...]
    event_id: str | None
    evaluation_enabled: bool
    state: str
    turns: tuple[ReplayTurn, ...]
    reflections: tuple[ReplayReflection, ...]
    final_logical_timestamp: int | None
    structural_summary: Mapping[str, Any] | None
    issues: tuple[ReplayIssue, ...] = ()

    def __post_init__(self) -> None:
        for name in ("step_id", "step_type", "period", "title", "description"):
            _text(getattr(self, name), name)
        _positive(self.day, "day")
        _string_tuple(self.participants, "participants")
        if self.event_id is not None:
            _text(self.event_id, "event_id")
        if not isinstance(self.evaluation_enabled, bool):
            raise ReplaySchemaError("evaluation_enabled must be boolean")
        if self.state not in STEP_STATES:
            raise ReplaySchemaError("step state is invalid")
        if not isinstance(self.turns, tuple) or not isinstance(self.reflections, tuple):
            raise ReplaySchemaError("step turns/reflections must be tuples")
        if self.final_logical_timestamp is not None:
            _positive(self.final_logical_timestamp, "final_logical_timestamp")
        if self.structural_summary is not None:
            object.__setattr__(
                self,
                "structural_summary",
                _freeze_json(self.structural_summary, "structural_summary"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "day": self.day,
            "period": self.period,
            "title": self.title,
            "description": self.description,
            "participants": list(self.participants),
            "event_id": self.event_id,
            "evaluation_enabled": self.evaluation_enabled,
            "state": self.state,
            "turns": [item.to_dict() for item in self.turns],
            "reflections": [item.to_dict() for item in self.reflections],
            "final_logical_timestamp": self.final_logical_timestamp,
            "structural_summary": (
                None if self.structural_summary is None else thaw_json(self.structural_summary)
            ),
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class ReplayFailure:
    error_type: str
    safe_message: str
    recoverable: bool
    retry_count: int | None
    step_id: str | None
    turn_id: str | None
    logical_timestamp: int | None

    def __post_init__(self) -> None:
        _text(self.error_type, "error_type")
        _text(self.safe_message, "safe_message")
        if not isinstance(self.recoverable, bool):
            raise ReplaySchemaError("recoverable must be boolean")
        if self.retry_count is not None:
            _non_negative(self.retry_count, "retry_count")
        for name in ("step_id", "turn_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.logical_timestamp is not None:
            _non_negative(self.logical_timestamp, "logical_timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "safe_message": self.safe_message,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "step_id": self.step_id,
            "turn_id": self.turn_id,
            "logical_timestamp": self.logical_timestamp,
        }


@dataclass(frozen=True)
class ReplayFrame:
    playback_index: int
    frame_kind: str
    logical_timestamp: int | None
    step_id: str | None
    entity_id: str

    def __post_init__(self) -> None:
        _non_negative(self.playback_index, "playback_index")
        if self.frame_kind not in FRAME_KINDS:
            raise ReplaySchemaError("unsupported frame kind")
        if self.logical_timestamp is not None:
            _non_negative(self.logical_timestamp, "logical_timestamp")
        if self.step_id is not None:
            _text(self.step_id, "step_id")
        _text(self.entity_id, "entity_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "playback_index": self.playback_index,
            "frame_kind": self.frame_kind,
            "logical_timestamp": self.logical_timestamp,
            "step_id": self.step_id,
            "entity_id": self.entity_id,
        }


@dataclass(frozen=True)
class ReplayRun:
    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    source_status: str
    replay_completeness: str
    completed_natural_blocks: int
    completed_target_events: int
    completed_main_turns: int
    error_count: int
    intended_scope: str
    primary_research_data: bool

    def __post_init__(self) -> None:
        for name in ("experiment_id", "run_id", "retrieval_config_id", "replay_completeness", "intended_scope"):
            _text(getattr(self, name), name)
        _positive(self.repetition_index, "repetition_index")
        if self.source_status not in SUPPORTED_RUN_STATUSES:
            raise ReplaySchemaError("unsupported source run status")
        for name in (
            "completed_natural_blocks",
            "completed_target_events",
            "completed_main_turns",
            "error_count",
        ):
            _non_negative(getattr(self, name), name)
        if not isinstance(self.primary_research_data, bool):
            raise ReplaySchemaError("primary_research_data must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "source_status": self.source_status,
            "replay_completeness": self.replay_completeness,
            "completed_natural_blocks": self.completed_natural_blocks,
            "completed_target_events": self.completed_target_events,
            "completed_main_turns": self.completed_main_turns,
            "error_count": self.error_count,
            "intended_scope": self.intended_scope,
            "primary_research_data": self.primary_research_data,
        }


@dataclass(frozen=True)
class ReplayDocument:
    source: Mapping[str, Any]
    run: ReplayRun
    agents: tuple[ReplayAgent, ...]
    steps: tuple[ReplayStep, ...]
    memories: tuple[ReplayMemory, ...]
    playback: tuple[ReplayFrame, ...]
    usage: Mapping[str, Any]
    failure: ReplayFailure | None
    evaluation_overlay: Mapping[str, Any] | None
    pilot_overlay: Mapping[str, Any] | None
    issues: tuple[ReplayIssue, ...]
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ReplaySchemaError("unsupported replay schema version")
        object.__setattr__(self, "source", _freeze_json(self.source, "source"))
        object.__setattr__(self, "usage", _freeze_json(self.usage, "usage"))
        if self.evaluation_overlay is not None:
            object.__setattr__(
                self,
                "evaluation_overlay",
                _freeze_json(self.evaluation_overlay, "evaluation_overlay"),
            )
        if self.pilot_overlay is not None:
            object.__setattr__(
                self,
                "pilot_overlay",
                _freeze_json(self.pilot_overlay, "pilot_overlay"),
            )
        if not isinstance(self.run, ReplayRun):
            raise ReplaySchemaError("run must be ReplayRun")
        for name in ("agents", "steps", "memories", "playback", "issues"):
            if not isinstance(getattr(self, name), tuple):
                raise ReplaySchemaError(f"{name} must be a tuple")
        if self.failure is not None and not isinstance(self.failure, ReplayFailure):
            raise ReplaySchemaError("failure must be ReplayFailure or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": thaw_json(self.source),
            "run": self.run.to_dict(),
            "agents": [item.to_dict() for item in self.agents],
            "steps": [item.to_dict() for item in self.steps],
            "memories": [item.to_dict() for item in self.memories],
            "playback": [item.to_dict() for item in self.playback],
            "usage": thaw_json(self.usage),
            "failure": None if self.failure is None else self.failure.to_dict(),
            "evaluation_overlay": (
                None if self.evaluation_overlay is None else thaw_json(self.evaluation_overlay)
            ),
            "pilot_overlay": (
                None if self.pilot_overlay is None else thaw_json(self.pilot_overlay)
            ),
            "issues": [item.to_dict() for item in self.issues],
        }


__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "ReplayAgent",
    "ReplayCandidate",
    "ReplayDocument",
    "ReplayFailure",
    "ReplayFrame",
    "ReplayIssue",
    "ReplayMemory",
    "ReplayReflection",
    "ReplayRetrieval",
    "ReplayRun",
    "ReplayStep",
    "ReplayTurn",
    "ReplayUsageRecord",
    "SOURCE_SCHEMA_VERSION",
    "thaw_json",
]
