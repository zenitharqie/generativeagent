"""Immutable Phase 13 full-experiment contracts and deterministic planning."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from src.config_loader import (
    DEFAULT_CONFIG_DIRECTORY,
    ExperimentConfig,
    load_experiment_config,
)
from src.config_loader.validators import FIXED_RETRIEVAL_WEIGHTS
from src.domain import NaturalInteractionBlockStep, TargetEventStep
from src.llm.settings import MAX_PROVIDER_ATTEMPTS
from src.llm.usage import ProviderAttemptRecord
from src.storage.run_directory import InvalidIdentifierError, validate_identifier


EXPERIMENT_SCHEMA_VERSION = "1.0"
FULL_EXPERIMENT_CONFIRMATION = "RUN_12_CONFIGURED_RUNS"
FULL_EXPERIMENT_MODES = frozenset({"dry-run", "dummy", "real"})
CONFIGURATION_MAJOR_ORDER = (
    "balanced",
    "recency_heavy",
    "relevance_heavy",
    "importance_heavy",
)
RUNS_PER_CONFIGURATION = 3
EXPECTED_RUN_COUNT = 12


class FullExperimentError(RuntimeError):
    """Base error for isolated Phase 13 orchestration."""


class ExperimentContractError(FullExperimentError, ValueError):
    """Raised when the frozen 4x3 experiment contract is not satisfied."""


class ExperimentAuthorizationError(FullExperimentError):
    """Raised before credential inspection when explicit authorization is absent."""


class ExperimentPreflightError(FullExperimentError):
    """Raised when a safe, offline real-execution preflight fails."""


class ExperimentUsageError(FullExperimentError):
    """Raised when attempt or token accounting cannot be trusted."""


class ExperimentBudgetError(ExperimentUsageError):
    """Raised when a call or token ceiling prevents further execution."""


class ExperimentExecutionError(FullExperimentError):
    """Safe terminal Phase 13 failure without provider-originated text."""

    def __init__(
        self,
        error_type: str,
        *,
        experiment_dir: Path | None,
        terminal_state: str,
        failed_run_id: str | None,
        evidence_published: bool,
    ) -> None:
        super().__init__("full experiment did not complete; inspect safe experiment evidence")
        self.error_type = _safe_identifier(error_type, "error_type")
        self.experiment_dir = experiment_dir
        if terminal_state not in {"failed", "interrupted"}:
            raise ExperimentContractError("terminal_state must be failed or interrupted")
        self.terminal_state = terminal_state
        self.failed_run_id = failed_run_id
        self.evidence_published = evidence_published


def _safe_identifier(value: object, field_name: str) -> str:
    try:
        return validate_identifier(value, field_name=field_name)  # type: ignore[arg-type]
    except InvalidIdentifierError:
        raise ExperimentContractError(f"{field_name} is not a safe identifier") from None


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentContractError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    result = _non_negative_int(value, field_name)
    if result == 0:
        raise ExperimentContractError(f"{field_name} must be positive")
    return result


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize deterministic UTF-8 JSON without time, UUID, or host metadata."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def content_hash(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class FullRunScope:
    simulation_days: int
    step_count: int
    natural_block_count: int
    target_event_count: int
    main_turn_count: int
    post_step_reflection_count: int
    seed_memory_count: int
    target_evaluation_unit_count: int

    def __post_init__(self) -> None:
        actual = (
            self.simulation_days,
            self.step_count,
            self.natural_block_count,
            self.target_event_count,
            self.main_turn_count,
            self.post_step_reflection_count,
            self.seed_memory_count,
            self.target_evaluation_unit_count,
        )
        if actual != (6, 12, 7, 5, 104, 48, 12, 20):
            raise ExperimentContractError(
                f"derived full-run structure {actual} differs from the frozen contract"
            )

    @property
    def expected_memory_record_count(self) -> int:
        return (
            self.seed_memory_count
            + self.main_turn_count
            + self.post_step_reflection_count
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "simulation_days": self.simulation_days,
            "step_count": self.step_count,
            "natural_block_count": self.natural_block_count,
            "target_event_count": self.target_event_count,
            "main_turn_count": self.main_turn_count,
            "post_step_reflection_count": self.post_step_reflection_count,
            "seed_memory_count": self.seed_memory_count,
            "memory_record_count": self.expected_memory_record_count,
            "target_evaluation_unit_count": self.target_evaluation_unit_count,
        }


@dataclass(frozen=True)
class ProviderCallLimits:
    simulator_logical_calls: int
    embedding_logical_calls: int
    evaluator_logical_calls: int
    maximum_attempts_per_logical_call: int

    def __post_init__(self) -> None:
        for field_name in (
            "simulator_logical_calls",
            "embedding_logical_calls",
            "evaluator_logical_calls",
        ):
            _non_negative_int(getattr(self, field_name), field_name)
        if self.maximum_attempts_per_logical_call != 2:
            raise ExperimentContractError(
                "provider retry policy no longer permits exactly two attempts"
            )
        if (
            self.simulator_logical_calls,
            self.embedding_logical_calls,
            self.evaluator_logical_calls,
        ) != (408, 268, 0):
            raise ExperimentContractError(
                "derived provider call graph differs from the frozen Phase 13 contract"
            )

    @property
    def total_logical_calls(self) -> int:
        return (
            self.simulator_logical_calls
            + self.embedding_logical_calls
            + self.evaluator_logical_calls
        )

    @property
    def maximum_physical_attempts(self) -> int:
        return self.total_logical_calls * self.maximum_attempts_per_logical_call

    def logical_limit(self, service_role: str) -> int:
        if service_role not in {"simulator", "embedding", "evaluator"}:
            raise ExperimentUsageError("unsupported provider service role")
        return getattr(self, f"{service_role}_logical_calls")

    def physical_limit(self, service_role: str) -> int:
        return self.logical_limit(service_role) * self.maximum_attempts_per_logical_call

    def to_dict(self) -> dict[str, int]:
        return {
            "simulator_logical_calls": self.simulator_logical_calls,
            "embedding_logical_calls": self.embedding_logical_calls,
            "evaluator_logical_calls": self.evaluator_logical_calls,
            "total_logical_calls": self.total_logical_calls,
            "maximum_attempts_per_logical_call": (
                self.maximum_attempts_per_logical_call
            ),
            "maximum_physical_attempts": self.maximum_physical_attempts,
        }


def derive_full_run_contract(config: ExperimentConfig) -> tuple[FullRunScope, ProviderCallLimits]:
    """Trace the frozen execution graph instead of trusting Phase 13 constants."""
    if not isinstance(config, ExperimentConfig):
        raise ExperimentContractError("config must be a validated ExperimentConfig")
    natural = tuple(
        step for step in config.steps if isinstance(step, NaturalInteractionBlockStep)
    )
    targets = tuple(step for step in config.steps if isinstance(step, TargetEventStep))
    main_turns = sum(step.total_turns for step in natural) + sum(
        len(step.participants) for step in targets
    )
    reflections = sum(len(step.participants) for step in config.steps)
    seed_memories = sum(len(agent.seed_memories) for agent in config.agents)
    scope = FullRunScope(
        simulation_days=config.settings.simulation_days,
        step_count=len(config.steps),
        natural_block_count=len(natural),
        target_event_count=len(targets),
        main_turn_count=main_turns,
        post_step_reflection_count=reflections,
        seed_memory_count=seed_memories,
        target_evaluation_unit_count=sum(len(step.participants) for step in targets),
    )
    # Every main turn generates plan, action, and importance. Every post-step
    # reflection generates reflection and importance. Embeddings cover seeds,
    # one query and action memory per main turn, and one reflection memory.
    calls = ProviderCallLimits(
        simulator_logical_calls=main_turns * 3 + reflections * 2,
        embedding_logical_calls=seed_memories + main_turns * 2 + reflections,
        evaluator_logical_calls=0,
        maximum_attempts_per_logical_call=MAX_PROVIDER_ATTEMPTS,
    )
    return scope, calls


@dataclass(frozen=True)
class ExperimentRunSpec:
    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    recency_weight: float
    importance_weight: float
    relevance_weight: float

    def __post_init__(self) -> None:
        _safe_identifier(self.experiment_id, "experiment_id")
        _safe_identifier(self.run_id, "run_id")
        if self.retrieval_config_id not in CONFIGURATION_MAJOR_ORDER:
            raise ExperimentContractError("run uses an unsupported retrieval config")
        if self.run_id != f"{self.retrieval_config_id}_rep{self.repetition_index:02d}":
            raise ExperimentContractError("run_id does not match config/repetition identity")
        if self.repetition_index not in (1, 2, 3):
            raise ExperimentContractError("repetition_index must be 1, 2, or 3")
        expected = FIXED_RETRIEVAL_WEIGHTS[self.retrieval_config_id]
        actual = (self.recency_weight, self.importance_weight, self.relevance_weight)
        if actual != expected:
            raise ExperimentContractError("run retrieval weights differ from fixed config")

    @property
    def source_directory_name(self) -> str:
        return self.run_id

    @property
    def evaluation_directory_name(self) -> str:
        return self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "weights": {
                "recency": self.recency_weight,
                "importance": self.importance_weight,
                "relevance": self.relevance_weight,
            },
            "source_directory_name": self.source_directory_name,
            "evaluation_directory_name": self.evaluation_directory_name,
        }


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    mode: str
    runs: tuple[ExperimentRunSpec, ...]
    run_scope: FullRunScope
    per_run_call_limits: ProviderCallLimits
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _safe_identifier(self.experiment_id, "experiment_id")
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ExperimentContractError("unsupported experiment plan schema")
        if self.mode not in FULL_EXPERIMENT_MODES:
            raise ExperimentContractError("unsupported full-experiment mode")
        expected_pairs = tuple(
            (config_id, repetition)
            for config_id in CONFIGURATION_MAJOR_ORDER
            for repetition in range(1, RUNS_PER_CONFIGURATION + 1)
        )
        actual_pairs = tuple(
            (run.retrieval_config_id, run.repetition_index) for run in self.runs
        )
        if len(self.runs) != EXPECTED_RUN_COUNT or actual_pairs != expected_pairs:
            raise ExperimentContractError("experiment plan must be exact configuration-major 4x3")
        if len({run.run_id for run in self.runs}) != EXPECTED_RUN_COUNT:
            raise ExperimentContractError("experiment run IDs must be unique")
        if any(run.experiment_id != self.experiment_id for run in self.runs):
            raise ExperimentContractError("run/experiment identity mismatch")
        if not isinstance(self.run_scope, FullRunScope) or not isinstance(
            self.per_run_call_limits, ProviderCallLimits
        ):
            raise ExperimentContractError("experiment plan uses invalid derived contracts")

    @property
    def total_main_turns(self) -> int:
        return self.run_scope.main_turn_count * len(self.runs)

    @property
    def total_reflections(self) -> int:
        return self.run_scope.post_step_reflection_count * len(self.runs)

    @property
    def total_steps(self) -> int:
        return self.run_scope.step_count * len(self.runs)

    @property
    def total_target_units(self) -> int:
        return self.run_scope.target_evaluation_unit_count * len(self.runs)

    @property
    def total_simulator_logical_calls(self) -> int:
        return self.per_run_call_limits.simulator_logical_calls * len(self.runs)

    @property
    def total_embedding_logical_calls(self) -> int:
        return self.per_run_call_limits.embedding_logical_calls * len(self.runs)

    @property
    def total_evaluator_logical_calls(self) -> int:
        return 0

    @property
    def total_logical_calls(self) -> int:
        return self.per_run_call_limits.total_logical_calls * len(self.runs)

    @property
    def maximum_physical_attempts(self) -> int:
        return self.per_run_call_limits.maximum_physical_attempts * len(self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "completion_contract": "source_plus_phase09_evaluation_ready",
            "run_count": len(self.runs),
            "runs": [run.to_dict() for run in self.runs],
            "per_run_structure": self.run_scope.to_dict(),
            "per_run_call_limits": self.per_run_call_limits.to_dict(),
            "experiment_totals": {
                "step_count": self.total_steps,
                "main_turn_count": self.total_main_turns,
                "post_step_reflection_count": self.total_reflections,
                "mra_record_count": self.total_target_units,
                "bcr_export_record_count": self.total_target_units,
                "simulator_logical_calls": self.total_simulator_logical_calls,
                "embedding_logical_calls": self.total_embedding_logical_calls,
                "evaluator_logical_calls": self.total_evaluator_logical_calls,
                "total_logical_calls": self.total_logical_calls,
                "maximum_physical_attempts": self.maximum_physical_attempts,
            },
        }

    @property
    def plan_hash(self) -> str:
        return content_hash(self.to_dict())


def build_full_experiment_plan(
    *,
    experiment_id: str,
    mode: str,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> ExperimentPlan:
    config = load_experiment_config(config_directory)
    scope, calls = derive_full_run_contract(config)
    configured = {item.config_id: item for item in config.retrieval_configs}
    if set(configured) != set(CONFIGURATION_MAJOR_ORDER):
        raise ExperimentContractError("retrieval configuration inventory has drifted")
    runs: list[ExperimentRunSpec] = []
    for config_id in CONFIGURATION_MAJOR_ORDER:
        value = configured[config_id]
        for repetition in range(1, config.settings.repetitions + 1):
            runs.append(
                ExperimentRunSpec(
                    experiment_id=experiment_id,
                    run_id=f"{config_id}_rep{repetition:02d}",
                    retrieval_config_id=config_id,
                    repetition_index=repetition,
                    recency_weight=value.weights.recency,
                    importance_weight=value.weights.importance,
                    relevance_weight=value.weights.relevance,
                )
            )
    return ExperimentPlan(
        experiment_id=experiment_id,
        mode=mode,
        runs=tuple(runs),
        run_scope=scope,
        per_run_call_limits=calls,
    )


@dataclass(frozen=True)
class ExperimentAttempt:
    """One run-qualified physical provider attempt."""

    run_id: str
    attempt: ProviderAttemptRecord

    def __post_init__(self) -> None:
        _safe_identifier(self.run_id, "run_id")
        if not isinstance(self.attempt, ProviderAttemptRecord):
            raise ExperimentUsageError("attempt must be ProviderAttemptRecord")
        if self.attempt.service_role not in {"simulator", "embedding"}:
            raise ExperimentUsageError("Phase 13 permits no evaluator attempts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            **self.attempt.to_dict(),
        }


@dataclass(frozen=True)
class RoleUsageSummary:
    service_role: str
    logical_call_count: int
    physical_attempt_count: int
    retry_count: int
    known_input_tokens: int
    known_output_tokens: int
    known_total_tokens: int
    missing_usage_attempt_count: int
    success_count: int
    failed_count: int
    invalid_output_count: int

    def __post_init__(self) -> None:
        if self.service_role not in {"simulator", "embedding"}:
            raise ExperimentUsageError("invalid Phase 13 usage service role")
        for name in (
            "logical_call_count",
            "physical_attempt_count",
            "retry_count",
            "known_input_tokens",
            "known_output_tokens",
            "known_total_tokens",
            "missing_usage_attempt_count",
            "success_count",
            "failed_count",
            "invalid_output_count",
        ):
            _non_negative_int(getattr(self, name), name)
        retry_arithmetic_valid = (
            self.retry_count == 0
            if self.physical_attempt_count == 0
            else self.retry_count
            == self.physical_attempt_count - self.logical_call_count
        )
        if (
            not retry_arithmetic_valid
            or self.success_count + self.failed_count + self.invalid_output_count
            != self.physical_attempt_count
        ):
            raise ExperimentUsageError("role usage arithmetic is inconsistent")

    @property
    def usage_complete(self) -> bool:
        return self.missing_usage_attempt_count == 0

    @property
    def actual_total_tokens(self) -> int | None:
        return self.known_total_tokens if self.usage_complete else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_role": self.service_role,
            "logical_call_count": self.logical_call_count,
            "physical_attempt_count": self.physical_attempt_count,
            "retry_count": self.retry_count,
            "known_input_tokens": self.known_input_tokens,
            "known_output_tokens": self.known_output_tokens,
            "known_total_tokens": self.known_total_tokens,
            "missing_usage_attempt_count": self.missing_usage_attempt_count,
            "usage_complete": self.usage_complete,
            "actual_total_tokens": self.actual_total_tokens,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "invalid_output_count": self.invalid_output_count,
        }


@dataclass(frozen=True)
class ExperimentUsageSummary:
    max_total_tokens: int | None
    logical_call_count: int
    physical_attempt_count: int
    retry_count: int
    known_input_tokens: int
    known_output_tokens: int
    known_total_tokens: int
    missing_usage_attempt_count: int
    budget_state: str
    role_usage: tuple[RoleUsageSummary, ...]

    def __post_init__(self) -> None:
        if self.max_total_tokens is not None:
            _positive_int(self.max_total_tokens, "max_total_tokens")
        if self.budget_state not in {"not_applicable", "admitted", "exhausted", "exceeded"}:
            raise ExperimentUsageError("unsupported budget state")
        for name in (
            "logical_call_count",
            "physical_attempt_count",
            "retry_count",
            "known_input_tokens",
            "known_output_tokens",
            "known_total_tokens",
            "missing_usage_attempt_count",
        ):
            _non_negative_int(getattr(self, name), name)
        if tuple(item.service_role for item in self.role_usage) != (
            "simulator",
            "embedding",
        ):
            raise ExperimentUsageError("usage role inventory must be simulator/embedding")
        if (
            self.logical_call_count != sum(item.logical_call_count for item in self.role_usage)
            or self.physical_attempt_count
            != sum(item.physical_attempt_count for item in self.role_usage)
            or (
                self.retry_count != 0
                if self.physical_attempt_count == 0
                else self.retry_count
                != self.physical_attempt_count - self.logical_call_count
            )
            or self.known_input_tokens
            != sum(item.known_input_tokens for item in self.role_usage)
            or self.known_output_tokens
            != sum(item.known_output_tokens for item in self.role_usage)
            or self.known_total_tokens
            != sum(item.known_total_tokens for item in self.role_usage)
            or self.missing_usage_attempt_count
            != sum(item.missing_usage_attempt_count for item in self.role_usage)
        ):
            raise ExperimentUsageError("experiment usage arithmetic is inconsistent")

    @property
    def usage_complete(self) -> bool:
        return self.missing_usage_attempt_count == 0

    @property
    def actual_total_tokens(self) -> int | None:
        return self.known_total_tokens if self.usage_complete else None

    @classmethod
    def from_attempts(
        cls,
        attempts: tuple[ExperimentAttempt, ...],
        *,
        max_total_tokens: int | None,
        budget_state: str,
        dummy_logical_counts: tuple[int, int] | None = None,
    ) -> "ExperimentUsageSummary":
        if dummy_logical_counts is not None and attempts:
            raise ExperimentUsageError("dummy logical counts cannot accompany attempts")
        roles: list[RoleUsageSummary] = []
        for role_index, role in enumerate(("simulator", "embedding")):
            selected = tuple(item for item in attempts if item.attempt.service_role == role)
            logical = {
                (item.run_id, item.attempt.logical_call_id) for item in selected
            }
            logical_count = (
                dummy_logical_counts[role_index]
                if dummy_logical_counts is not None
                else len(logical)
            )
            roles.append(
                RoleUsageSummary(
                    service_role=role,
                    logical_call_count=logical_count,
                    physical_attempt_count=len(selected),
                    retry_count=len(selected) - len(logical),
                    known_input_tokens=sum(
                        item.attempt.usage.input_tokens
                        for item in selected
                        if item.attempt.usage.input_tokens is not None
                    ),
                    known_output_tokens=sum(
                        item.attempt.usage.output_tokens
                        for item in selected
                        if item.attempt.usage.output_tokens is not None
                    ),
                    known_total_tokens=sum(
                        item.attempt.usage.total_tokens
                        for item in selected
                        if item.attempt.usage.total_tokens is not None
                    ),
                    missing_usage_attempt_count=sum(
                        item.attempt.usage.total_tokens is None for item in selected
                    ),
                    success_count=sum(item.attempt.status == "success" for item in selected),
                    failed_count=sum(item.attempt.status == "failed" for item in selected),
                    invalid_output_count=sum(
                        item.attempt.status == "invalid_output" for item in selected
                    ),
                )
            )
        logical_count = sum(item.logical_call_count for item in roles)
        return cls(
            max_total_tokens=max_total_tokens,
            logical_call_count=logical_count,
            physical_attempt_count=len(attempts),
            retry_count=len(attempts)
            - len(
                {
                    (item.run_id, item.attempt.service_role, item.attempt.logical_call_id)
                    for item in attempts
                }
            ),
            known_input_tokens=sum(item.known_input_tokens for item in roles),
            known_output_tokens=sum(item.known_output_tokens for item in roles),
            known_total_tokens=sum(item.known_total_tokens for item in roles),
            missing_usage_attempt_count=sum(
                item.missing_usage_attempt_count for item in roles
            ),
            budget_state=budget_state,
            role_usage=tuple(roles),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "max_total_tokens": self.max_total_tokens,
            "logical_call_count": self.logical_call_count,
            "physical_attempt_count": self.physical_attempt_count,
            "retry_count": self.retry_count,
            "known_input_tokens": self.known_input_tokens,
            "known_output_tokens": self.known_output_tokens,
            "known_total_tokens": self.known_total_tokens,
            "missing_usage_attempt_count": self.missing_usage_attempt_count,
            "usage_complete": self.usage_complete,
            "actual_total_tokens": self.actual_total_tokens,
            "budget_state": self.budget_state,
            "role_usage": [item.to_dict() for item in self.role_usage],
            "evaluator_logical_calls": 0,
            "evaluator_physical_attempts": 0,
        }


@dataclass(frozen=True)
class GitSourceIdentity:
    commit: str
    branch: str
    tags: tuple[str, ...]
    clean: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.commit, str)
            or len(self.commit) != 40
            or any(character not in "0123456789abcdef" for character in self.commit)
        ):
            raise ExperimentContractError("source commit must be full lowercase Git SHA")
        _safe_identifier(self.branch, "branch")
        if not isinstance(self.tags, tuple) or any(
            not isinstance(tag, str) or not tag for tag in self.tags
        ):
            raise ExperimentContractError("source tags must be an immutable string tuple")
        if not isinstance(self.clean, bool):
            raise ExperimentContractError("source clean state must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "tags": list(self.tags),
            "clean": self.clean,
        }


def default_plan(
    *,
    mode: str = "dry-run",
    experiment_id: str = "phase13_full_experiment",
) -> ExperimentPlan:
    return build_full_experiment_plan(
        experiment_id=experiment_id,
        mode=mode,
        config_directory=DEFAULT_CONFIG_DIRECTORY,
    )


__all__ = [
    "CONFIGURATION_MAJOR_ORDER",
    "EXPECTED_RUN_COUNT",
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentAttempt",
    "ExperimentAuthorizationError",
    "ExperimentBudgetError",
    "ExperimentContractError",
    "ExperimentExecutionError",
    "ExperimentPlan",
    "ExperimentPreflightError",
    "ExperimentRunSpec",
    "ExperimentUsageError",
    "ExperimentUsageSummary",
    "FULL_EXPERIMENT_CONFIRMATION",
    "FULL_EXPERIMENT_MODES",
    "FullExperimentError",
    "FullRunScope",
    "GitSourceIdentity",
    "ProviderCallLimits",
    "RoleUsageSummary",
    "build_full_experiment_plan",
    "canonical_json_bytes",
    "content_hash",
    "default_plan",
    "derive_full_run_contract",
]
