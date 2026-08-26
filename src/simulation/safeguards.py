"""Phase 11 real-pilot authorization, scope, path, and usage safeguards."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

from src.config_loader import ExperimentConfig
from src.domain import NaturalInteractionBlockStep, TargetEventStep
from src.llm import LLMCallType, LLMClient, LLMRequest, TextEmbeddingProvider
from src.llm.prompt_loader import PROMPT_VERSION, PromptLoader
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    ANTHROPIC_PROVIDER,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_IDENTITY,
    EMBEDDING_PROFILE,
    EVALUATOR_MODEL_IDENTITY,
    EVALUATOR_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    OPENAI_PROVIDER,
    RETRY_DELAY_SECONDS,
    SIMULATOR_MODEL_IDENTITY,
    SIMULATOR_PROFILE,
)
from src.llm.usage import ProviderAttemptRecord, UsageRecorder
from src.logging_system import (
    APIUsageLogRecord,
    RunLogMetadata,
    StepLogMetadata,
    TurnLogMetadata,
    append_jsonl,
)
from src.simulation.turn_executor import (
    build_main_turn_id,
    build_reflection_operation_id,
)
from src.simulation.turn_policy import TurnPolicyManager
from src.storage import RunConflictError, validate_identifier


PILOT_SCHEMA_VERSION = "2.0"
PILOT_MODES = frozenset({"dry-run", "dummy", "real"})
PILOT_ACCEPTANCE_STATES = frozenset({"pending", "accepted", "rejected"})
PILOT_RETRIEVAL_CONFIG_ID = "balanced"
PILOT_REPETITION_INDEX = 1
PILOT_DAYS = (1, 2)
PILOT_STEP_IDS = (
    "day_01_natural_01",
    "day_01_natural_02",
    "day_02_natural_03",
    "day_02_target_01",
)
PILOT_NATURAL_BLOCKS = 3
PILOT_TARGET_EVENTS = 1
PILOT_MAIN_TURNS = 40
PILOT_POST_STEP_REFLECTIONS = 16
PILOT_TARGET_EVALUATION_UNITS = 4

APPROVED_PHASE10_PROMPT_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "plan": "3280e43625f91500cf8f2695db5dae9bcd7ce2accac671638b7c44d0d993e95f",
        "natural_interaction": "67a431a68bd12974c0c80c0c87d9f7b6a046f669ea50918a4879ca334e95ffbc",
        "target_event_response": "5946a503f61c2f10235a7031f21db7dd4d394c8cccc79e92000fe6ad2b9443a1",
        "reflect": "9d143c44c425be8daa6f996a48db2e461df3a49dc513c15812aa6f6b6b42ca57",
        "importance_scoring": "defc1811c94a9aec9b42f1a5a6fa12f2de302edbecc90854ec4a4d273252880e",
        "bcr_judge": "80d4cf1784e6be63eb7b4680bbea4a56eed5843f301ed527f972389b3e39253d",
    }
)

_EXPECTED_PROMPT_FOR_CALL = {
    "plan": "plan",
    "natural_response": "natural_interaction",
    "target_event_response": "target_event_response",
    "reflection": "reflect",
    "importance_score": "importance_scoring",
}


class PilotError(RuntimeError):
    """Base error for Phase 11 pilot controls."""


class PilotAuthorizationError(PilotError):
    """Raised before client creation when paid execution is not authorized."""


class PilotScopeError(PilotError, ValueError):
    """Raised when a request exceeds the fixed pilot scope."""


class PilotPreflightError(PilotError):
    """Raised when models, prompts, config, or paths fail preflight."""


class PilotPathError(PilotPreflightError):
    """Raised when a pilot artifact path is unsafe."""


class PilotUsageError(PilotError):
    """Raised when provider-attempt audit metadata cannot be trusted."""


class PilotBudgetError(PilotUsageError):
    """Base error for token or call-budget exhaustion."""


class PilotTokenBudgetExceeded(PilotBudgetError):
    """Raised before or immediately after a call crosses the token ceiling."""


class PilotCallBudgetExceeded(PilotBudgetError):
    """Raised before a logical call outside the fixed pilot plan."""


@dataclass(frozen=True)
class PilotCallLimits:
    simulator_logical_calls: int
    embedding_logical_calls: int
    evaluator_logical_calls: int
    simulator_physical_attempts: int
    embedding_physical_attempts: int
    evaluator_physical_attempts: int

    def __post_init__(self) -> None:
        for field_name in (
            "simulator_logical_calls",
            "embedding_logical_calls",
            "evaluator_logical_calls",
            "simulator_physical_attempts",
            "embedding_physical_attempts",
            "evaluator_physical_attempts",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PilotScopeError(f"{field_name} must be a non-negative integer")
        for role in ("simulator", "embedding", "evaluator"):
            logical = getattr(self, f"{role}_logical_calls")
            physical = getattr(self, f"{role}_physical_attempts")
            if physical < logical:
                raise PilotScopeError(
                    f"{role} physical-attempt limit must cover logical calls"
                )

    @property
    def total_logical_calls(self) -> int:
        return (
            self.simulator_logical_calls
            + self.embedding_logical_calls
            + self.evaluator_logical_calls
        )

    @property
    def total_physical_attempts(self) -> int:
        return (
            self.simulator_physical_attempts
            + self.embedding_physical_attempts
            + self.evaluator_physical_attempts
        )

    def logical_limit(self, role: str) -> int:
        if role not in {"simulator", "embedding", "evaluator"}:
            raise PilotUsageError(f"unsupported provider service role: {role!r}")
        return getattr(self, f"{role}_logical_calls")

    def physical_limit(self, role: str) -> int:
        if role not in {"simulator", "embedding", "evaluator"}:
            raise PilotUsageError(f"unsupported provider service role: {role!r}")
        return getattr(self, f"{role}_physical_attempts")

    def to_dict(self) -> dict[str, int]:
        return {
            "simulator_logical_calls": self.simulator_logical_calls,
            "embedding_logical_calls": self.embedding_logical_calls,
            "evaluator_logical_calls": self.evaluator_logical_calls,
            "total_logical_calls": self.total_logical_calls,
            "simulator_physical_attempts": self.simulator_physical_attempts,
            "embedding_physical_attempts": self.embedding_physical_attempts,
            "evaluator_physical_attempts": self.evaluator_physical_attempts,
            "total_physical_attempts": self.total_physical_attempts,
        }


PILOT_CALL_LIMITS = PilotCallLimits(
    simulator_logical_calls=152,
    embedding_logical_calls=108,
    evaluator_logical_calls=0,
    simulator_physical_attempts=304,
    embedding_physical_attempts=216,
    evaluator_physical_attempts=0,
)


@dataclass(frozen=True)
class PilotPlan:
    mode: str
    retrieval_config_id: str
    repetition_index: int
    days: tuple[int, ...]
    step_ids: tuple[str, ...]
    natural_block_count: int
    target_event_count: int
    main_turn_count: int
    post_step_reflection_count: int
    target_evaluation_unit_count: int
    call_limits: PilotCallLimits

    def __post_init__(self) -> None:
        if self.mode not in PILOT_MODES:
            raise PilotScopeError(f"unsupported pilot mode: {self.mode!r}")
        if self.days != PILOT_DAYS or self.step_ids != PILOT_STEP_IDS:
            raise PilotScopeError("pilot plan must be the exact Day 1-2 timeline prefix")
        expected = (
            PILOT_NATURAL_BLOCKS,
            PILOT_TARGET_EVENTS,
            PILOT_MAIN_TURNS,
            PILOT_POST_STEP_REFLECTIONS,
            PILOT_TARGET_EVALUATION_UNITS,
        )
        actual = (
            self.natural_block_count,
            self.target_event_count,
            self.main_turn_count,
            self.post_step_reflection_count,
            self.target_evaluation_unit_count,
        )
        if actual != expected or self.call_limits != PILOT_CALL_LIMITS:
            raise PilotScopeError("pilot plan does not match fixed Phase 11 limits")
        if self.mode == "real" and (
            self.retrieval_config_id != PILOT_RETRIEVAL_CONFIG_ID
            or self.repetition_index != PILOT_REPETITION_INDEX
        ):
            raise PilotScopeError(
                "real pilot requires balanced configuration and repetition 1"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "days": list(self.days),
            "step_ids": list(self.step_ids),
            "natural_block_count": self.natural_block_count,
            "target_event_count": self.target_event_count,
            "main_turn_count": self.main_turn_count,
            "post_step_reflection_count": self.post_step_reflection_count,
            "target_evaluation_unit_count": self.target_evaluation_unit_count,
            "call_limits": self.call_limits.to_dict(),
        }


def build_pilot_plan(
    config: ExperimentConfig,
    *,
    mode: str,
    retrieval_config_id: str,
    repetition_index: int,
) -> PilotPlan:
    """Derive and verify the one authorized Day 1-2 pilot plan."""
    if not isinstance(config, ExperimentConfig):
        raise PilotScopeError("config must be a validated ExperimentConfig")
    if mode not in PILOT_MODES:
        raise PilotScopeError(f"unsupported pilot mode: {mode!r}")
    known_configs = {item.config_id for item in config.retrieval_configs}
    if retrieval_config_id not in known_configs:
        raise PilotScopeError(f"unknown retrieval configuration: {retrieval_config_id!r}")
    if (
        isinstance(repetition_index, bool)
        or not isinstance(repetition_index, int)
        or not 1 <= repetition_index <= config.settings.repetitions
    ):
        raise PilotScopeError("repetition_index must be between 1 and 3")
    if mode == "real" and (
        retrieval_config_id != PILOT_RETRIEVAL_CONFIG_ID
        or repetition_index != PILOT_REPETITION_INDEX
    ):
        raise PilotScopeError(
            "real pilot scope is fixed to balanced configuration, repetition 1"
        )
    selected = tuple(step for step in config.steps if step.day in PILOT_DAYS)
    step_ids = tuple(step.step_id for step in selected)
    natural = tuple(
        step for step in selected if isinstance(step, NaturalInteractionBlockStep)
    )
    targets = tuple(step for step in selected if isinstance(step, TargetEventStep))
    main_turns = sum(step.total_turns for step in natural) + sum(
        len(step.participants) for step in targets
    )
    reflections = sum(len(step.participants) for step in selected)
    seed_embeddings = sum(len(agent.seed_memories) for agent in config.agents)
    call_limits = PilotCallLimits(
        simulator_logical_calls=main_turns * 3 + reflections * 2,
        embedding_logical_calls=seed_embeddings + main_turns * 2 + reflections,
        evaluator_logical_calls=0,
        simulator_physical_attempts=(main_turns * 3 + reflections * 2)
        * MAX_PROVIDER_ATTEMPTS,
        embedding_physical_attempts=(seed_embeddings + main_turns * 2 + reflections)
        * MAX_PROVIDER_ATTEMPTS,
        evaluator_physical_attempts=0,
    )
    if (
        config.settings.dummy_mode_default is not True
        or config.settings.allow_real_llm_default is not False
        or config.settings.allow_full_experiment_default is not False
    ):
        raise PilotPreflightError("configuration no longer has safe execution defaults")
    return PilotPlan(
        mode=mode,
        retrieval_config_id=retrieval_config_id,
        repetition_index=repetition_index,
        days=PILOT_DAYS,
        step_ids=step_ids,
        natural_block_count=len(natural),
        target_event_count=len(targets),
        main_turn_count=main_turns,
        post_step_reflection_count=reflections,
        target_evaluation_unit_count=sum(len(step.participants) for step in targets),
        call_limits=call_limits,
    )


@dataclass(frozen=True)
class PilotPaths:
    output_root: Path
    experiment_dir: Path
    source_run_dir: Path
    source_staging_dir: Path
    diagnostic_evaluation_dir: Path
    diagnostic_evaluation_staging_dir: Path
    evidence_dir: Path
    evidence_staging_dir: Path
    review_dir: Path
    review_staging_dir: Path


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_symlink(path: Path, field_name: str) -> None:
    if path.is_symlink():
        raise PilotPathError(f"{field_name} must not be a symbolic link")


def build_pilot_paths(
    output_root: str | Path,
    *,
    experiment_id: str,
    run_id: str,
) -> PilotPaths:
    """Resolve four separate sibling artifact boundaries without creating them."""
    experiment_id = validate_identifier(experiment_id, field_name="experiment_id")
    run_id = validate_identifier(run_id, field_name="run_id")
    try:
        requested = Path(output_root)
    except (TypeError, ValueError) as exc:
        raise PilotPathError("invalid output_root") from exc
    if not str(requested) or ".." in requested.parts:
        raise PilotPathError("output_root must not be empty or contain traversal")
    absolute = requested if requested.is_absolute() else Path.cwd() / requested
    _reject_symlink(absolute, "output_root")
    try:
        root = absolute.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PilotPathError("could not resolve output_root") from exc
    experiment = root / experiment_id
    _reject_symlink(experiment, "experiment directory")
    source = experiment / run_id
    evaluation = experiment / f"{run_id}_diagnostic_evaluation"
    evidence = experiment / f"{run_id}_pilot_evidence"
    review = experiment / f"{run_id}_pilot_review"
    paths = PilotPaths(
        output_root=root,
        experiment_dir=experiment,
        source_run_dir=source,
        source_staging_dir=experiment / f".{run_id}.staging",
        diagnostic_evaluation_dir=evaluation,
        diagnostic_evaluation_staging_dir=experiment
        / f".{evaluation.name}.staging",
        evidence_dir=evidence,
        evidence_staging_dir=experiment / f".{evidence.name}.staging",
        review_dir=review,
        review_staging_dir=experiment / f".{review.name}.staging",
    )
    all_paths = tuple(getattr(paths, field) for field in paths.__dataclass_fields__)
    if len(set(all_paths)) != len(all_paths):
        raise PilotPathError("pilot artifact paths must be distinct")
    return paths


def validate_fresh_pilot_paths(paths: PilotPaths) -> None:
    """Reject every final/staging conflict before any output is created."""
    if not isinstance(paths, PilotPaths):
        raise PilotPathError("paths must be PilotPaths")
    for field_name in (
        "source_run_dir",
        "source_staging_dir",
        "diagnostic_evaluation_dir",
        "diagnostic_evaluation_staging_dir",
        "evidence_dir",
        "evidence_staging_dir",
        "review_dir",
        "review_staging_dir",
    ):
        path = getattr(paths, field_name)
        if _path_exists(path):
            raise RunConflictError(f"pilot path already exists: {path}")


def validate_phase10_runtime(prompt_loader: PromptLoader | None = None) -> None:
    """Verify the exact approved pilot models, settings, and prompt bytes."""
    if (
        SIMULATOR_MODEL_IDENTITY != "openai:gpt-5-nano-2025-08-07"
        or EVALUATOR_MODEL_IDENTITY != "anthropic:claude-sonnet-5"
        or EMBEDDING_MODEL_IDENTITY != "openai:text-embedding-3-small"
        or EMBEDDING_DIMENSION != 1536
    ):
        raise PilotPreflightError("Phase 10 model identities no longer match approval")
    if SIMULATOR_MODEL_IDENTITY == EVALUATOR_MODEL_IDENTITY:
        raise PilotPreflightError("simulator and evaluator identities must differ")
    if (
        SIMULATOR_PROFILE.provider != OPENAI_PROVIDER
        or SIMULATOR_PROFILE.model_identity != SIMULATOR_MODEL_IDENTITY
        or SIMULATOR_PROFILE.reasoning_effort != "minimal"
        or SIMULATOR_PROFILE.request_timeout_seconds != 60
        or SIMULATOR_PROFILE.context_window_tokens != 400_000
        or SIMULATOR_PROFILE.stream is not False
        or SIMULATOR_PROFILE.strict_structured_output is not True
        or any(
            value is not None
            for value in (
                SIMULATOR_PROFILE.temperature,
                SIMULATOR_PROFILE.top_p,
                SIMULATOR_PROFILE.seed,
            )
        )
        or dict(SIMULATOR_PROFILE.output_token_limits)
        != {
            "plan": 300,
            "natural_response": 500,
            "target_event_response": 500,
            "reflection": 400,
            "importance_score": 150,
        }
    ):
        raise PilotPreflightError("simulator runtime profile has drifted")
    if (
        EVALUATOR_PROFILE.provider != ANTHROPIC_PROVIDER
        or EVALUATOR_PROFILE.model_identity != EVALUATOR_MODEL_IDENTITY
        or EVALUATOR_PROFILE.reasoning_effort != "medium"
        or EVALUATOR_PROFILE.request_timeout_seconds != 60
        or EVALUATOR_PROFILE.context_window_tokens != 400_000
        or dict(EVALUATOR_PROFILE.output_token_limits) != {"bcr_judge": 700}
        or EVALUATOR_PROFILE.stream is not False
        or EVALUATOR_PROFILE.strict_structured_output is not True
        or any(
            value is not None
            for value in (
                EVALUATOR_PROFILE.temperature,
                EVALUATOR_PROFILE.top_p,
                EVALUATOR_PROFILE.seed,
            )
        )
    ):
        raise PilotPreflightError("evaluator runtime profile has drifted")
    if (
        EMBEDDING_PROFILE.model_identity != EMBEDDING_MODEL_IDENTITY
        or EMBEDDING_PROFILE.expected_dimension != 1536
        or EMBEDDING_PROFILE.request_timeout_seconds != 30
        or MAX_PROVIDER_ATTEMPTS != 2
        or RETRY_DELAY_SECONDS != 1
        or BCR_RUBRIC_VERSION != "phase10_binary_bcr_rubric_v2"
    ):
        raise PilotPreflightError("embedding, retry, timeout, or rubric settings drifted")
    loader = prompt_loader or PromptLoader()
    templates = loader.load_all()
    actual = {item.spec.template_id: item.content_hash for item in templates}
    if actual != dict(APPROVED_PHASE10_PROMPT_HASHES):
        raise PilotPreflightError("Phase 10 prompt hash mismatch")
    if any(item.spec.version != PROMPT_VERSION for item in templates):
        raise PilotPreflightError("Phase 10 prompt version mismatch")


def validate_real_authorization(
    *,
    real_authorized: bool,
    max_total_tokens: int | None,
) -> int:
    """Require explicit authorization and a caller-owned positive token ceiling."""
    if real_authorized is not True:
        raise PilotAuthorizationError(
            "real pilot requires explicit --allow-real-llm true authorization"
        )
    if (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or max_total_tokens < 1
    ):
        raise PilotAuthorizationError(
            "real pilot requires a positive --max-total-tokens value"
        )
    return max_total_tokens


def require_openai_credential() -> None:
    """Check presence only; never return, print, or persist the secret."""
    value = os.environ.get(OPENAI_API_KEY_ENVIRONMENT_VARIABLE)
    if not isinstance(value, str) or not value.strip():
        raise PilotAuthorizationError(
            f"missing required credential {OPENAI_API_KEY_ENVIRONMENT_VARIABLE}"
        )


class PilotTurnMetadataResolver:
    """Map simulator requests to existing canonical run/turn metadata."""

    def __init__(self, config: ExperimentConfig, run: RunLogMetadata) -> None:
        if not isinstance(config, ExperimentConfig) or not isinstance(run, RunLogMetadata):
            raise PilotUsageError("turn resolver requires config and run metadata")
        self._main: dict[tuple[str, int], TurnLogMetadata] = {}
        self._reflections: dict[tuple[str, str], TurnLogMetadata] = {}
        timestamp = 0
        policy = TurnPolicyManager()
        for step in tuple(item for item in config.steps if item.day in PILOT_DAYS):
            if isinstance(step, NaturalInteractionBlockStep):
                order = policy.build_turn_order(step)
            elif isinstance(step, TargetEventStep):
                order = step.participants
            else:  # pragma: no cover - validated config owns the union
                raise PilotUsageError("unsupported pilot step type")
            for turn_index, agent_id in enumerate(order, start=1):
                timestamp += 1
                step_meta = StepLogMetadata(
                    run=run,
                    step_id=step.step_id,
                    step_type=step.step_type,
                    is_target_event=step.is_target_event,
                    evaluation_enabled=step.evaluation_enabled,
                    day=step.day,
                    period=step.period,
                    timestamp=timestamp,
                    event_id=getattr(step, "event_id", None),
                )
                self._main[(step.step_id, timestamp)] = TurnLogMetadata(
                    step=step_meta,
                    turn_id=build_main_turn_id(step.step_id, turn_index),
                    turn_index=turn_index,
                    agent_id=agent_id,
                )
            final_timestamp = timestamp
            for participant_index, agent_id in enumerate(step.participants, start=1):
                step_meta = StepLogMetadata(
                    run=run,
                    step_id=step.step_id,
                    step_type=step.step_type,
                    is_target_event=step.is_target_event,
                    evaluation_enabled=step.evaluation_enabled,
                    day=step.day,
                    period=step.period,
                    timestamp=final_timestamp,
                    event_id=getattr(step, "event_id", None),
                )
                self._reflections[(step.step_id, agent_id)] = TurnLogMetadata(
                    step=step_meta,
                    turn_id=build_reflection_operation_id(
                        step.step_id,
                        participant_index,
                    ),
                    turn_index=participant_index,
                    agent_id=agent_id,
                )
        if timestamp != PILOT_MAIN_TURNS:
            raise PilotUsageError("turn resolver did not build the exact pilot timeline")

    def resolve(self, request: LLMRequest) -> TurnLogMetadata:
        if not isinstance(request, LLMRequest):
            raise PilotUsageError("active simulator request is invalid")
        is_reflection = request.call_type is LLMCallType.REFLECTION or (
            request.call_type is LLMCallType.IMPORTANCE_SCORE
            and request.memory_kind == "reflection"
        )
        key: tuple[str, object]
        if is_reflection:
            key = (request.context.source_step_id, request.context.agent_id)
            result = self._reflections.get(key)
        else:
            key = (
                request.context.source_step_id,
                request.context.logical_timestamp,
            )
            result = self._main.get(key)
        if result is None or result.agent_id != request.context.agent_id:
            raise PilotUsageError("simulator request does not match the pilot timeline")
        if result.step.timestamp != request.context.logical_timestamp:
            raise PilotUsageError("simulator request logical timestamp mismatch")
        return result


def _attempt_duplicate_key(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        value["service_role"],
        value["logical_call_id"],
        value["attempt_index"],
    )


class PilotAttemptRecorder(UsageRecorder):
    """Persist every physical attempt and enforce the paid pilot budget."""

    def __init__(
        self,
        *,
        attempts_path: str | Path,
        api_usage_path: str | Path,
        run_metadata: RunLogMetadata,
        config: ExperimentConfig,
        max_total_tokens: int,
        call_limits: PilotCallLimits = PILOT_CALL_LIMITS,
    ) -> None:
        if (
            isinstance(max_total_tokens, bool)
            or not isinstance(max_total_tokens, int)
            or max_total_tokens < 1
        ):
            raise PilotUsageError("max_total_tokens must be a positive integer")
        self._attempts_path = Path(attempts_path)
        self._api_usage_path = Path(api_usage_path)
        self._run = run_metadata
        self._resolver = PilotTurnMetadataResolver(config, run_metadata)
        self._max_total_tokens = max_total_tokens
        self._limits = call_limits
        self._records: list[ProviderAttemptRecord] = []
        self._seen: set[tuple[str, str, int]] = set()
        self._logical: set[tuple[str, str]] = set()
        self._active_request: LLMRequest | None = None

    @property
    def records(self) -> tuple[ProviderAttemptRecord, ...]:
        return tuple(self._records)

    @property
    def max_total_tokens(self) -> int:
        return self._max_total_tokens

    @property
    def reported_total_tokens(self) -> int:
        return sum(
            item.usage.total_tokens
            for item in self._records
            if item.usage.total_tokens is not None
        )

    @property
    def logical_call_count(self) -> int:
        return len(self._logical)

    @property
    def retry_count(self) -> int:
        return len(self._records) - self.logical_call_count

    @property
    def missing_usage_attempt_count(self) -> int:
        return sum(item.usage.total_tokens is None for item in self._records)

    def authorize_logical_call(self, service_role: str) -> None:
        """Block a known-exhausted budget before another provider request."""
        current = sum(role == service_role for role, _ in self._logical)
        if current >= self._limits.logical_limit(service_role):
            raise PilotCallBudgetExceeded(
                f"{service_role} logical-call budget is exhausted"
            )
        if self.logical_call_count >= self._limits.total_logical_calls:
            raise PilotCallBudgetExceeded("total provider logical-call budget is exhausted")
        if self.reported_total_tokens >= self._max_total_tokens:
            raise PilotTokenBudgetExceeded(
                "pilot token budget is exhausted before the next provider call"
            )

    @contextmanager
    def simulator_request(self, request: LLMRequest) -> Iterator[None]:
        if self._active_request is not None:
            raise PilotUsageError("nested simulator request audit context is not allowed")
        self._active_request = request
        try:
            yield
        finally:
            self._active_request = None

    def _validate_attempt(self, attempt: ProviderAttemptRecord) -> None:
        if not isinstance(attempt, ProviderAttemptRecord):
            raise PilotUsageError("attempt must be ProviderAttemptRecord")
        expected_model = {
            "simulator": SIMULATOR_MODEL_IDENTITY,
            "embedding": EMBEDDING_MODEL_IDENTITY,
            "evaluator": EVALUATOR_MODEL_IDENTITY,
        }[attempt.service_role]
        if attempt.model != expected_model:
            raise PilotUsageError("provider attempt model identity mismatch")
        if attempt.service_role == "evaluator":
            raise PilotCallBudgetExceeded("Phase 11 pilot permits no evaluator calls")
        key = (
            attempt.service_role,
            attempt.logical_call_id,
            attempt.attempt_index,
        )
        if key in self._seen:
            raise PilotUsageError("duplicate provider attempt identity")
        logical = (attempt.service_role, attempt.logical_call_id)
        prior = tuple(
            record
            for record in self._records
            if (record.service_role, record.logical_call_id) == logical
        )
        if attempt.attempt_index != len(prior) + 1:
            raise PilotUsageError("provider attempt indexes must be consecutive")
        if attempt.attempt_index == 1 and logical in self._logical:
            raise PilotUsageError("duplicate provider logical call identity")
        if attempt.service_role == "simulator":
            expected_template = _EXPECTED_PROMPT_FOR_CALL.get(attempt.call_type)
            if (
                expected_template is None
                or attempt.prompt_template != expected_template
                or attempt.prompt_version != PROMPT_VERSION
                or attempt.prompt_hash
                != APPROVED_PHASE10_PROMPT_HASHES[expected_template]
            ):
                raise PilotUsageError("simulator prompt attempt identity mismatch")
        projected_logical = len(self._logical | {logical})
        role_logical = sum(
            role == attempt.service_role for role, _ in self._logical | {logical}
        )
        role_physical = sum(
            record.service_role == attempt.service_role for record in self._records
        ) + 1
        if role_logical > self._limits.logical_limit(attempt.service_role):
            raise PilotCallBudgetExceeded("provider logical-call limit exceeded")
        if role_physical > self._limits.physical_limit(attempt.service_role):
            raise PilotCallBudgetExceeded("provider physical-attempt limit exceeded")
        if projected_logical > self._limits.total_logical_calls:
            raise PilotCallBudgetExceeded("total provider logical-call limit exceeded")
        if len(self._records) + 1 > self._limits.total_physical_attempts:
            raise PilotCallBudgetExceeded("total provider attempt limit exceeded")

    def _project_simulator_usage(self, attempt: ProviderAttemptRecord) -> None:
        usage = attempt.usage
        counts = (usage.input_tokens, usage.output_tokens, usage.total_tokens)
        if all(value is None for value in counts) and attempt.status == "failed":
            return
        if any(value is None for value in counts):
            raise PilotUsageError(
                "successful provider attempt requires complete simulator token usage"
            )
        if self._active_request is None:
            raise PilotUsageError("simulator attempt has no active request context")
        if self._active_request.call_type.value != attempt.call_type:
            raise PilotUsageError("simulator attempt call_type mismatch")
        append_jsonl(
            self._api_usage_path,
            APIUsageLogRecord(
                turn=self._resolver.resolve(self._active_request),
                call_type=attempt.call_type,
                model=attempt.model,
                prompt_template=attempt.prompt_template,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                status=attempt.status,
                error_type=attempt.error_type,
            ),
        )

    def record(self, attempt: ProviderAttemptRecord) -> None:
        self._validate_attempt(attempt)
        append_jsonl(
            self._attempts_path,
            attempt,
            duplicate_key=_attempt_duplicate_key,
        )
        self._records.append(attempt)
        self._seen.add(
            (attempt.service_role, attempt.logical_call_id, attempt.attempt_index)
        )
        self._logical.add((attempt.service_role, attempt.logical_call_id))
        if attempt.service_role == "simulator":
            self._project_simulator_usage(attempt)
        elif attempt.status in {"success", "invalid_output"} and (
            attempt.usage.total_tokens is None
        ):
            raise PilotUsageError(
                "successful provider attempt requires total token usage"
            )
        total = self.reported_total_tokens
        if total > self._max_total_tokens:
            raise PilotTokenBudgetExceeded(
                f"pilot token budget exceeded: {total} > {self._max_total_tokens}"
            )
        if attempt.status != "success" and total >= self._max_total_tokens:
            raise PilotTokenBudgetExceeded(
                "pilot token budget is exhausted before a retry"
            )


class GuardedPilotLLMClient:
    """Apply a pre-call guard without changing the stable LLMClient contract."""

    def __init__(self, client: LLMClient, recorder: PilotAttemptRecorder) -> None:
        if not callable(getattr(client, "generate", None)):
            raise PilotUsageError("client must provide generate(request)")
        self._client = client
        self._recorder = recorder

    def generate(self, request: LLMRequest) -> str:
        self._recorder.authorize_logical_call("simulator")
        with self._recorder.simulator_request(request):
            return self._client.generate(request)


class GuardedPilotEmbeddingProvider:
    """Apply the same token/call guard to every embedding logical call."""

    def __init__(
        self,
        provider: TextEmbeddingProvider,
        recorder: PilotAttemptRecorder,
    ) -> None:
        if not callable(getattr(provider, "embed", None)):
            raise PilotUsageError("embedding provider must provide embed(text)")
        self._provider = provider
        self._recorder = recorder

    def embed(self, text: str) -> tuple[float, ...]:
        self._recorder.authorize_logical_call("embedding")
        return self._provider.embed(text)


__all__ = [
    "APPROVED_PHASE10_PROMPT_HASHES",
    "GuardedPilotEmbeddingProvider",
    "GuardedPilotLLMClient",
    "PILOT_ACCEPTANCE_STATES",
    "PILOT_CALL_LIMITS",
    "PILOT_DAYS",
    "PILOT_MAIN_TURNS",
    "PILOT_MODES",
    "PILOT_NATURAL_BLOCKS",
    "PILOT_POST_STEP_REFLECTIONS",
    "PILOT_REPETITION_INDEX",
    "PILOT_RETRIEVAL_CONFIG_ID",
    "PILOT_SCHEMA_VERSION",
    "PILOT_STEP_IDS",
    "PILOT_TARGET_EVALUATION_UNITS",
    "PILOT_TARGET_EVENTS",
    "PilotAttemptRecorder",
    "PilotAuthorizationError",
    "PilotBudgetError",
    "PilotCallBudgetExceeded",
    "PilotCallLimits",
    "PilotError",
    "PilotPathError",
    "PilotPaths",
    "PilotPlan",
    "PilotPreflightError",
    "PilotScopeError",
    "PilotTokenBudgetExceeded",
    "PilotTurnMetadataResolver",
    "PilotUsageError",
    "build_pilot_paths",
    "build_pilot_plan",
    "require_openai_credential",
    "validate_fresh_pilot_paths",
    "validate_phase10_runtime",
    "validate_real_authorization",
]
