"""Immutable Phase 09 result, export, and future-judge schema contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import re
from typing import Any, ClassVar

from src.config_loader.validators import FIXED_EVENT_IDS, FIXED_RETRIEVAL_WEIGHTS
from src.domain.agents import FIXED_AGENT_IDS
from src.logging_system import FIXED_TOP_K, SCHEMA_VERSION

from src.evaluation.errors import EvaluationSchemaError


EVALUATION_SCHEMA_VERSION = SCHEMA_VERSION
EVALUATION_MODES = frozenset({"primary", "diagnostic"})
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _validate_evaluation_eligibility(
    evaluation_mode: object,
    primary_eligible: object,
    included_in_final_research_aggregation: object,
) -> None:
    """Enforce the shared Phase 09 mode and eligibility-field contract."""
    if not isinstance(evaluation_mode, str) or evaluation_mode not in EVALUATION_MODES:
        raise EvaluationSchemaError(
            f"unsupported evaluation_mode: {evaluation_mode!r}"
        )
    if not isinstance(primary_eligible, bool) or not isinstance(
        included_in_final_research_aggregation, bool
    ):
        raise EvaluationSchemaError("evaluation eligibility fields must be boolean")
    expected_flags = evaluation_mode == "primary"
    if (
        primary_eligible != expected_flags
        or included_in_final_research_aggregation != expected_flags
    ):
        raise EvaluationSchemaError(
            "primary eligibility flags must match evaluation_mode"
        )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSchemaError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: object, field_name: str) -> str:
    value = _text(value, field_name)
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise EvaluationSchemaError(
            f"{field_name} must be a single safe identifier segment"
        )
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationSchemaError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationSchemaError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _text_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
    identifiers: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise EvaluationSchemaError(f"{field_name} must be a tuple")
    validate = _identifier if identifiers else _text
    result = tuple(validate(item, f"{field_name} item") for item in value)
    if not allow_empty and not result:
        raise EvaluationSchemaError(f"{field_name} must not be empty")
    if unique and len(result) != len(set(result)):
        raise EvaluationSchemaError(f"{field_name} must not contain duplicates")
    return result


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _binary_judgment(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise EvaluationSchemaError(f"{field_name} must be exactly 0 or 1")
    return value


def _rubric_assessment(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value not in {"consistent", "inconsistent"}:
        raise EvaluationSchemaError(
            f"{field_name} must be 'consistent' or 'inconsistent'"
        )
    return value


@dataclass(frozen=True)
class EvaluationUnitIdentity:
    """Exact, non-fuzzy identity of one target-event agent response."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    event_id: str
    step_id: str
    turn_id: str
    turn_index: int
    timestamp: int
    agent_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "event_id",
            "step_id",
            "turn_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        if self.retrieval_config_id not in FIXED_RETRIEVAL_WEIGHTS:
            raise EvaluationSchemaError(
                f"unsupported retrieval_config_id: {self.retrieval_config_id!r}"
            )
        if self.event_id not in FIXED_EVENT_IDS:
            raise EvaluationSchemaError(f"unsupported event_id: {self.event_id!r}")
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or not 1 <= self.repetition_index <= 3
        ):
            raise EvaluationSchemaError("repetition_index must be an integer from 1 to 3")
        _positive_int(self.turn_index, "turn_index")
        _positive_int(self.timestamp, "timestamp")
        if self.agent_id not in FIXED_AGENT_IDS:
            raise EvaluationSchemaError(f"unsupported agent_id: {self.agent_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "event_id": self.event_id,
            "step_id": self.step_id,
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
        }


@dataclass(frozen=True)
class SelectedMemoryEvidence:
    """One selected memory with audit labels and judge-visible text separated."""

    retrieval_rank: int
    memory_id: str
    category: str
    text: str

    def __post_init__(self) -> None:
        _positive_int(self.retrieval_rank, "retrieval_rank")
        if self.retrieval_rank > FIXED_TOP_K:
            raise EvaluationSchemaError(
                f"retrieval_rank must not exceed fixed Top-K {FIXED_TOP_K}"
            )
        object.__setattr__(
            self, "memory_id", _identifier(self.memory_id, "memory_id")
        )
        object.__setattr__(self, "category", _text(self.category, "category"))
        object.__setattr__(self, "text", _text(self.text, "text"))

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "retrieval_rank": self.retrieval_rank,
            "memory_id": self.memory_id,
            "category": self.category,
        }


@dataclass(frozen=True)
class PriorBehaviorEvidence:
    """One earlier same-agent target response with split audit/judge projections."""

    event_id: str
    step_id: str
    turn_id: str
    turn_index: int
    timestamp: int
    event_title: str
    event_description: str
    response: str

    def __post_init__(self) -> None:
        for field_name in ("event_id", "step_id", "turn_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        _positive_int(self.turn_index, "turn_index")
        _positive_int(self.timestamp, "timestamp")
        for field_name in ("event_title", "event_description", "response"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "step_id": self.step_id,
            "turn_id": self.turn_id,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }

    def to_judge_dict(self) -> dict[str, Any]:
        return {
            "event_title": self.event_title,
            "event_description": self.event_description,
            "response": self.response,
        }


@dataclass(frozen=True)
class PersonaJudgeInput:
    """Persona attributes approved for future BCR judgment."""

    display_name: str
    role: str
    archetype: str
    persona_summary: str
    traits: tuple[str, ...]
    goals: tuple[str, ...]
    communication_style: str
    motivations: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "display_name",
            "role",
            "archetype",
            "persona_summary",
            "communication_style",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        for field_name in ("traits", "goals", "motivations"):
            object.__setattr__(
                self,
                field_name,
                _text_tuple(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
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
class TargetEventJudgeInput:
    """Controlled target stimulus visible to the future evaluator."""

    title: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "description": self.description}


@dataclass(frozen=True)
class BCRJudgeInput:
    """Condition-blind payload to be judged only in Phase 10."""

    persona: PersonaJudgeInput
    target_event: TargetEventJudgeInput
    response: str
    prior_target_behaviors: tuple[PriorBehaviorEvidence, ...]
    selected_memory_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.persona, PersonaJudgeInput):
            raise EvaluationSchemaError("persona must be PersonaJudgeInput")
        if not isinstance(self.target_event, TargetEventJudgeInput):
            raise EvaluationSchemaError("target_event must be TargetEventJudgeInput")
        object.__setattr__(self, "response", _text(self.response, "response"))
        if not isinstance(self.prior_target_behaviors, tuple) or any(
            not isinstance(item, PriorBehaviorEvidence)
            for item in self.prior_target_behaviors
        ):
            raise EvaluationSchemaError(
                "prior_target_behaviors must be a tuple of PriorBehaviorEvidence values"
            )
        timestamps = tuple(item.timestamp for item in self.prior_target_behaviors)
        if timestamps != tuple(sorted(timestamps)) or len(timestamps) != len(
            set(timestamps)
        ):
            raise EvaluationSchemaError(
                "prior_target_behaviors must be uniquely chronological"
            )
        object.__setattr__(
            self,
            "selected_memory_texts",
            _text_tuple(
                self.selected_memory_texts,
                "selected_memory_texts",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona.to_dict(),
            "target_event": self.target_event.to_dict(),
            "response": self.response,
            "prior_target_behaviors": [
                item.to_judge_dict() for item in self.prior_target_behaviors
            ],
            "selected_memory_texts": list(self.selected_memory_texts),
        }


@dataclass(frozen=True)
class BCRModelMetadata:
    """Outer metadata slots reserved for Phase 10 independence auditing."""

    simulator_model_identity: str | None = None
    evaluator_model_identity: str | None = None
    evaluator_prompt_version: str | None = None
    evaluator_rubric_version: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "simulator_model_identity",
            "evaluator_model_identity",
            "evaluator_prompt_version",
            "evaluator_rubric_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "simulator_model_identity": self.simulator_model_identity,
            "evaluator_model_identity": self.evaluator_model_identity,
            "evaluator_prompt_version": self.evaluator_prompt_version,
            "evaluator_rubric_version": self.evaluator_rubric_version,
        }


@dataclass(frozen=True)
class FutureBCRJudgment:
    """Binary judged-value contract for Phase 10; never populated by Phase 09."""

    communication_style_assessment: str
    focus_consistency_assessment: str
    temporal_consistency_assessment: str
    consistent: int
    reason: str

    def __post_init__(self) -> None:
        for field_name in (
            "communication_style_assessment",
            "focus_consistency_assessment",
            "temporal_consistency_assessment",
        ):
            object.__setattr__(
                self,
                field_name,
                _rubric_assessment(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "consistent",
            _binary_judgment(self.consistent, "consistent"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "communication_style_assessment": self.communication_style_assessment,
            "focus_consistency_assessment": self.focus_consistency_assessment,
            "temporal_consistency_assessment": self.temporal_consistency_assessment,
            "consistent": self.consistent,
            "reason": self.reason,
        }


def future_bcr_contract_dict() -> dict[str, Any]:
    """Return the unpopulated, frozen binary Phase 10 judgment contract."""
    return {
        "status": "not_judged_phase_09",
        "decision_type": "binary",
        "allowed_final_values": [0, 1],
        "rubric_dimensions": [
            "communication_style",
            "focus_consistency",
            "temporal_consistency",
        ],
        "rubric_assessment_values": ["consistent", "inconsistent"],
        "final_judgment_field": "consistent",
        "final_metric": "proportion_of_consistent_responses",
    }


@dataclass(frozen=True)
class MRAEvaluationResult:
    """Auditable fixed-denominator MRA for one selected Top-K result."""

    identity: EvaluationUnitIdentity
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    top_k: int
    candidate_count: int
    selected_memories: tuple[SelectedMemoryEvidence, ...]
    expected_categories: tuple[str, ...]
    matched_memory_ids: tuple[str, ...]
    unmatched_memory_ids: tuple[str, ...]
    matched_count: int
    denominator: int
    mra: float
    schema_version: str = field(default=EVALUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationUnitIdentity):
            raise EvaluationSchemaError("identity must be EvaluationUnitIdentity")
        _validate_evaluation_eligibility(
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        if self.top_k != FIXED_TOP_K or self.denominator != FIXED_TOP_K:
            raise EvaluationSchemaError(
                f"top_k and denominator must equal fixed value {FIXED_TOP_K}"
            )
        _non_negative_int(self.candidate_count, "candidate_count")
        if not isinstance(self.selected_memories, tuple) or any(
            not isinstance(item, SelectedMemoryEvidence)
            for item in self.selected_memories
        ):
            raise EvaluationSchemaError(
                "selected_memories must be a tuple of SelectedMemoryEvidence values"
            )
        if len(self.selected_memories) > FIXED_TOP_K:
            raise EvaluationSchemaError("selected_memories exceeds fixed Top-K")
        if self.candidate_count < len(self.selected_memories):
            raise EvaluationSchemaError(
                "candidate_count must not be smaller than retrieved_count"
            )
        ranks = tuple(item.retrieval_rank for item in self.selected_memories)
        if ranks != tuple(range(1, len(self.selected_memories) + 1)):
            raise EvaluationSchemaError(
                "selected memory retrieval ranks must be consecutive and one-based"
            )
        selected_ids = self.selected_memory_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise EvaluationSchemaError("selected memory IDs must be unique")
        object.__setattr__(
            self,
            "expected_categories",
            _text_tuple(
                self.expected_categories,
                "expected_categories",
                unique=True,
            ),
        )
        for field_name in ("matched_memory_ids", "unmatched_memory_ids"):
            object.__setattr__(
                self,
                field_name,
                _text_tuple(
                    getattr(self, field_name),
                    field_name,
                    allow_empty=True,
                    identifiers=True,
                    unique=True,
                ),
            )
        expected_matches = tuple(
            item.memory_id
            for item in self.selected_memories
            if item.category in self.expected_categories
        )
        expected_unmatched = tuple(
            item.memory_id
            for item in self.selected_memories
            if item.category not in self.expected_categories
        )
        if self.matched_memory_ids != expected_matches:
            raise EvaluationSchemaError(
                "matched_memory_ids do not match selected-memory categories"
            )
        if self.unmatched_memory_ids != expected_unmatched:
            raise EvaluationSchemaError(
                "unmatched_memory_ids do not match selected-memory categories"
            )
        _non_negative_int(self.matched_count, "matched_count")
        if self.matched_count != len(self.matched_memory_ids):
            raise EvaluationSchemaError(
                "matched_count must equal matched_memory_ids length"
            )
        if (
            isinstance(self.mra, bool)
            or not isinstance(self.mra, (int, float))
            or not isfinite(float(self.mra))
            or not 0 <= self.mra <= 1
        ):
            raise EvaluationSchemaError("mra must be a finite number in [0, 1]")
        if self.mra != self.matched_count / FIXED_TOP_K:
            raise EvaluationSchemaError("mra must equal matched_count divided by 5")

    @property
    def retrieved_count(self) -> int:
        return len(self.selected_memories)

    @property
    def selected_memory_ids(self) -> tuple[str, ...]:
        return tuple(item.memory_id for item in self.selected_memories)

    @property
    def retrieved_categories(self) -> tuple[str, ...]:
        return tuple(item.category for item in self.selected_memories)

    @property
    def matched_categories(self) -> tuple[str, ...]:
        matched = set(self.matched_memory_ids)
        return tuple(
            item.category for item in self.selected_memories if item.memory_id in matched
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_aggregation": (
                self.included_in_final_research_aggregation
            ),
            **self.identity.to_dict(),
        }
        value.update(
            {
                "top_k": self.top_k,
                "retrieved_count": self.retrieved_count,
                "candidate_count": self.candidate_count,
                "selected_memory_ids": list(self.selected_memory_ids),
                "retrieved_categories": list(self.retrieved_categories),
                "expected_categories": list(self.expected_categories),
                "matched_memory_ids": list(self.matched_memory_ids),
                "unmatched_memory_ids": list(self.unmatched_memory_ids),
                "matched_categories": list(self.matched_categories),
                "matched_count": self.matched_count,
                "denominator": self.denominator,
                "mra": self.mra,
            }
        )
        return value


@dataclass(frozen=True)
class BCRExportRecord:
    """One audited export row with a condition-blind nested judge payload."""

    identity: EvaluationUnitIdentity
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    selected_memories: tuple[SelectedMemoryEvidence, ...]
    prior_behaviors: tuple[PriorBehaviorEvidence, ...]
    judge_input: BCRJudgeInput
    model_metadata: BCRModelMetadata = BCRModelMetadata()
    schema_version: str = field(default=EVALUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationUnitIdentity):
            raise EvaluationSchemaError("identity must be EvaluationUnitIdentity")
        _validate_evaluation_eligibility(
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        if not isinstance(self.selected_memories, tuple) or any(
            not isinstance(item, SelectedMemoryEvidence)
            for item in self.selected_memories
        ):
            raise EvaluationSchemaError("selected_memories has invalid values")
        if len(self.selected_memories) > FIXED_TOP_K:
            raise EvaluationSchemaError("selected_memories exceeds fixed Top-K")
        ranks = tuple(item.retrieval_rank for item in self.selected_memories)
        if ranks != tuple(range(1, len(self.selected_memories) + 1)):
            raise EvaluationSchemaError(
                "selected memory ranks must preserve one-based retrieval order"
            )
        memory_ids = tuple(item.memory_id for item in self.selected_memories)
        if len(memory_ids) != len(set(memory_ids)):
            raise EvaluationSchemaError("selected memory IDs must be unique")
        if not isinstance(self.prior_behaviors, tuple) or any(
            not isinstance(item, PriorBehaviorEvidence) for item in self.prior_behaviors
        ):
            raise EvaluationSchemaError("prior_behaviors has invalid values")
        if not isinstance(self.judge_input, BCRJudgeInput):
            raise EvaluationSchemaError("judge_input must be BCRJudgeInput")
        if not isinstance(self.model_metadata, BCRModelMetadata):
            raise EvaluationSchemaError("model_metadata must be BCRModelMetadata")
        if self.judge_input.selected_memory_texts != tuple(
            item.text for item in self.selected_memories
        ):
            raise EvaluationSchemaError(
                "judge selected memory text must match outer selected-memory evidence"
            )
        if self.judge_input.prior_target_behaviors != self.prior_behaviors:
            raise EvaluationSchemaError(
                "judge prior behavior must match outer prior-behavior evidence"
            )
        if any(item.timestamp >= self.identity.timestamp for item in self.prior_behaviors):
            raise EvaluationSchemaError(
                "prior behavior must be strictly earlier than the current response"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_aggregation": (
                self.included_in_final_research_aggregation
            ),
            "audit_metadata": self.identity.to_dict(),
            "selected_memory_evidence": [
                item.to_audit_dict() for item in self.selected_memories
            ],
            "prior_behavior_evidence": [
                item.to_audit_dict() for item in self.prior_behaviors
            ],
            "model_metadata": self.model_metadata.to_dict(),
            "judge_input": self.judge_input.to_dict(),
        }


@dataclass(frozen=True)
class MRAGroupAggregate:
    """One deterministic arithmetic-mean breakdown."""

    dimension: str
    key: str
    unit_count: int
    mean_mra: float

    _DIMENSIONS: ClassVar[frozenset[str]] = frozenset(
        {"run", "configuration", "event", "agent"}
    )

    def __post_init__(self) -> None:
        if self.dimension not in self._DIMENSIONS:
            raise EvaluationSchemaError(
                f"unsupported aggregate dimension: {self.dimension!r}"
            )
        object.__setattr__(self, "key", _identifier(self.key, "aggregate key"))
        _positive_int(self.unit_count, "unit_count")
        if (
            isinstance(self.mean_mra, bool)
            or not isinstance(self.mean_mra, (int, float))
            or not isfinite(float(self.mean_mra))
            or not 0 <= self.mean_mra <= 1
        ):
            raise EvaluationSchemaError("mean_mra must be a finite number in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "key": self.key,
            "unit_count": self.unit_count,
            "mean_mra": self.mean_mra,
        }


@dataclass(frozen=True)
class RunMRAAggregates:
    """Per-run MRA mean plus deterministic event and agent breakdowns."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    evaluation_mode: str
    unit_count: int
    run: MRAGroupAggregate
    by_event: tuple[MRAGroupAggregate, ...]
    by_agent: tuple[MRAGroupAggregate, ...]

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "run_id", "retrieval_config_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name),
            )
        if self.retrieval_config_id not in FIXED_RETRIEVAL_WEIGHTS:
            raise EvaluationSchemaError("unsupported retrieval_config_id")
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or not 1 <= self.repetition_index <= 3
        ):
            raise EvaluationSchemaError("repetition_index must be from 1 through 3")
        if self.evaluation_mode not in EVALUATION_MODES:
            raise EvaluationSchemaError("unsupported evaluation_mode")
        _positive_int(self.unit_count, "unit_count")
        if self.evaluation_mode == "primary" and self.unit_count != 20:
            raise EvaluationSchemaError("primary run aggregate must contain 20 units")
        if not isinstance(self.run, MRAGroupAggregate) or self.run.dimension != "run":
            raise EvaluationSchemaError("run must be a run MRAGroupAggregate")
        if self.run.key != self.run_id or self.run.unit_count != self.unit_count:
            raise EvaluationSchemaError("run aggregate identity/count is inconsistent")
        for values, dimension in (
            (self.by_event, "event"),
            (self.by_agent, "agent"),
        ):
            if not isinstance(values, tuple) or not values or any(
                not isinstance(item, MRAGroupAggregate)
                or item.dimension != dimension
                for item in values
            ):
                raise EvaluationSchemaError(
                    f"by_{dimension} must contain {dimension} aggregates"
                )
            if sum(item.unit_count for item in values) != self.unit_count:
                raise EvaluationSchemaError(
                    f"by_{dimension} counts must sum to unit_count"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "evaluation_mode": self.evaluation_mode,
            "unit_count": self.unit_count,
            "run": self.run.to_dict(),
            "by_event": [item.to_dict() for item in self.by_event],
            "by_agent": [item.to_dict() for item in self.by_agent],
        }


@dataclass(frozen=True)
class ConfigurationMRAAggregate:
    """Macro-average of exactly three eligible repetition/run means."""

    experiment_id: str
    retrieval_config_id: str
    repetition_indices: tuple[int, ...]
    run_ids: tuple[str, ...]
    run_mra_values: tuple[float, ...]
    mean_mra: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, "experiment_id")
        )
        object.__setattr__(
            self,
            "retrieval_config_id",
            _identifier(self.retrieval_config_id, "retrieval_config_id"),
        )
        if self.retrieval_config_id not in FIXED_RETRIEVAL_WEIGHTS:
            raise EvaluationSchemaError("unsupported retrieval_config_id")
        if (
            not isinstance(self.repetition_indices, tuple)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.repetition_indices
            )
            or self.repetition_indices != (1, 2, 3)
        ):
            raise EvaluationSchemaError("repetition_indices must be exactly (1, 2, 3)")
        object.__setattr__(
            self,
            "run_ids",
            _text_tuple(self.run_ids, "run_ids", identifiers=True, unique=True),
        )
        if not isinstance(self.run_mra_values, tuple):
            raise EvaluationSchemaError("run_mra_values must be a tuple")
        if len(self.run_ids) != 3 or len(self.run_mra_values) != 3:
            raise EvaluationSchemaError(
                "configuration aggregation requires exactly three runs"
            )
        for value in self.run_mra_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise EvaluationSchemaError("run MRA values must be in [0, 1]")
        if self.mean_mra != sum(self.run_mra_values) / 3:
            raise EvaluationSchemaError(
                "configuration mean_mra must macro-average three run means"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_indices": list(self.repetition_indices),
            "run_ids": list(self.run_ids),
            "run_mra_values": list(self.run_mra_values),
            "mean_mra": self.mean_mra,
            "aggregation": "macro_average_of_three_repetition_run_means",
        }


@dataclass(frozen=True)
class RunEvaluationSummary:
    """One complete deterministic Phase 09 publication summary."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    source_run_status: str
    source_schema_version: str
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    mra_aggregates: RunMRAAggregates
    bcr_export_count: int
    model_metadata: BCRModelMetadata
    schema_version: str = field(default=EVALUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "source_run_status",
            "source_schema_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if self.retrieval_config_id not in FIXED_RETRIEVAL_WEIGHTS:
            raise EvaluationSchemaError("unsupported retrieval_config_id")
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or not 1 <= self.repetition_index <= 3
        ):
            raise EvaluationSchemaError("repetition_index must be from 1 through 3")
        if self.source_schema_version != SCHEMA_VERSION:
            raise EvaluationSchemaError(
                f"source_schema_version must be {SCHEMA_VERSION!r}"
            )
        _validate_evaluation_eligibility(
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        expected = self.evaluation_mode == "primary"
        expected_status = "completed" if expected else "excluded"
        if self.source_run_status != expected_status:
            raise EvaluationSchemaError(
                "source_run_status must match primary/diagnostic evaluation mode"
            )
        if not isinstance(self.mra_aggregates, RunMRAAggregates):
            raise EvaluationSchemaError("mra_aggregates must be RunMRAAggregates")
        expected_identity = (
            self.experiment_id,
            self.run_id,
            self.retrieval_config_id,
            self.repetition_index,
            self.evaluation_mode,
        )
        actual_identity = (
            self.mra_aggregates.experiment_id,
            self.mra_aggregates.run_id,
            self.mra_aggregates.retrieval_config_id,
            self.mra_aggregates.repetition_index,
            self.mra_aggregates.evaluation_mode,
        )
        if actual_identity != expected_identity:
            raise EvaluationSchemaError("summary/MRA aggregate identity mismatch")
        _positive_int(self.bcr_export_count, "bcr_export_count")
        if self.bcr_export_count != self.mra_aggregates.unit_count:
            raise EvaluationSchemaError(
                "BCR export and MRA unit counts must be identical"
            )
        if not isinstance(self.model_metadata, BCRModelMetadata):
            raise EvaluationSchemaError("model_metadata must be BCRModelMetadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "source_run_status": self.source_run_status,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_aggregation": (
                self.included_in_final_research_aggregation
            ),
            "mra_formula": "matched_selected_memories_divided_by_fixed_top_k_5",
            "mra_aggregates": self.mra_aggregates.to_dict(),
            "bcr_export_count": self.bcr_export_count,
            "bcr_judgment": future_bcr_contract_dict(),
            "model_metadata": self.model_metadata.to_dict(),
            "output_files": [
                "mra_records.jsonl",
                "bcr_export.jsonl",
                "evaluation_summary.json",
            ],
        }


__all__ = [
    "BCRExportRecord",
    "BCRJudgeInput",
    "BCRModelMetadata",
    "EVALUATION_MODES",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationUnitIdentity",
    "FutureBCRJudgment",
    "MRAEvaluationResult",
    "MRAGroupAggregate",
    "ConfigurationMRAAggregate",
    "PersonaJudgeInput",
    "PriorBehaviorEvidence",
    "RunEvaluationSummary",
    "RunMRAAggregates",
    "SelectedMemoryEvidence",
    "TargetEventJudgeInput",
    "future_bcr_contract_dict",
]
