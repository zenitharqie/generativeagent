"""Fail-fast, no-resume Phase 13 source-to-Phase-09 orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.config_loader import DEFAULT_CONFIG_DIRECTORY, ExperimentConfig, load_experiment_config
from src.evaluation.schemas import EVALUATION_SCHEMA_VERSION
from src.evaluation.service import EvaluationRunResult, evaluate_run
from src.llm import DummyEmbeddingProvider, LLMService, OpenAIEmbeddingProvider
from src.llm.base import LLMCallType, LLMClient, LLMRequest, TextEmbeddingProvider
from src.llm.prompt_loader import PROMPT_VERSION, PromptLoader
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    EMBEDDING_MODEL_IDENTITY,
    EMBEDDING_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    SIMULATOR_MODEL_IDENTITY,
    SIMULATOR_PROFILE,
)
from src.llm.usage import ProviderAttemptRecord, UsageRecorder
from src.logging_system import (
    SCHEMA_VERSION,
    APIUsageLogRecord,
    RunLogMetadata,
    StepLogMetadata,
    TurnLogMetadata,
    append_jsonl,
)
from src.simulation.experiment_validator import (
    EligibleRunChain,
    ExperimentValidationError,
    validate_complete_chains,
    validate_experiment_manifest,
    validate_phase09_evaluation,
    validate_run_chain_unchanged,
    validate_source_run,
)
from src.simulation.full_experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    FULL_EXPERIMENT_CONFIRMATION,
    missing_usage_cap,
    ExperimentAttempt,
    ExperimentAuthorizationError,
    ExperimentBudgetError,
    ExperimentContractError,
    ExperimentExecutionError,
    ExperimentPlan,
    ExperimentPreflightError,
    ExperimentRunSpec,
    ExperimentUsageError,
    ExperimentUsageSummary,
    GitSourceIdentity,
    build_full_experiment_plan,
    canonical_json_bytes,
)
from src.simulation.pilot import (
    PilotFreezeValidation,
    hash_config_directory,
    prompt_identities,
    validate_accepted_freeze,
)
from src.simulation.run_controller import (
    RunController,
    RunExecutionError,
    build_deterministic_dummy_llm_client,
)
from src.simulation.safeguards import (
    APPROVED_PHASE10_PROMPT_HASHES,
    require_openai_credential,
    validate_phase10_runtime,
)
from src.simulation.turn_executor import build_main_turn_id, build_reflection_operation_id
from src.simulation.turn_policy import TurnPolicyManager
from src.storage.experiment_directory import (
    ExperimentPaths,
    append_attempt,
    build_experiment_paths,
    create_experiment_directory,
    publish_experiment_manifest,
    publish_new_json,
    validate_fresh_experiment_paths,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OPERATIONAL_MINIMUM_FREE_BYTES = 3584 * 1024 * 1024  # 3.5 GiB
_EXPECTED_PROMPT_FOR_CALL = {
    "plan": "plan",
    "natural_response": "natural_interaction",
    "target_event_response": "target_event_response",
    "reflection": "reflect",
    "importance_score": "importance_scoring",
}


@dataclass(frozen=True)
class ExperimentExecutionResult:
    paths: ExperimentPaths
    plan: ExperimentPlan
    run_chains: tuple[EligibleRunChain, ...]
    usage: ExperimentUsageSummary
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.run_chains) != 12 or self.manifest.get("status") != "completed":
            raise ExperimentContractError("successful experiment result must contain 12 chains")


def dry_run_full_experiment(
    *,
    experiment_id: str = "phase13_full_experiment",
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> ExperimentPlan:
    """Build the exact plan without paths, credentials, providers, or execution."""
    return build_full_experiment_plan(
        experiment_id=experiment_id,
        mode="dry-run",
        config_directory=config_directory,
    )


def validate_execution_authorization(
    *,
    mode: str,
    confirmation: str | None,
    real_authorized: bool,
    max_total_tokens: int | None,
) -> int | None:
    if confirmation != FULL_EXPERIMENT_CONFIRMATION:
        raise ExperimentAuthorizationError(
            f"full experiment requires --confirm-full-experiment {FULL_EXPERIMENT_CONFIRMATION}"
        )
    if mode == "dummy":
        if real_authorized or max_total_tokens is not None:
            raise ExperimentAuthorizationError(
                "dummy experiment must not receive real-provider controls"
            )
        return None
    if mode != "real":
        raise ExperimentAuthorizationError("execution mode must be dummy or real")
    if real_authorized is not True:
        raise ExperimentAuthorizationError(
            "real experiment requires explicit --allow-real-llm true"
        )
    if (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or max_total_tokens < 1
    ):
        raise ExperimentAuthorizationError(
            "real experiment requires a positive --max-total-tokens value"
        )
    return max_total_tokens


def validate_ground_truth_methodological_finalization(
    config_directory: str | Path,
    evidence_directory: str | Path | None = None,
) -> None:
    """Require the evidence-bound Ground Truth lifecycle contract before real mode."""
    config = load_experiment_config(config_directory)
    status = config.ground_truth.get("status")
    freeze_required = config.ground_truth.get(
        "freeze_required_before_final_experiment"
    )
    if status == "initial_configurable" and freeze_required is True:
        raise ExperimentPreflightError(
            "real experiment requires methodological ground-truth finalization"
        )
    if status == "final" and evidence_directory is None:
        raise ExperimentPreflightError(
            "real experiment requires an explicit Ground Truth evidence directory"
        )
    try:
        from src.evaluation.ground_truth import (
            GroundTruthInputError,
            validate_final_ground_truth_identity,
        )

        validate_final_ground_truth_identity(
            config.ground_truth,
            evidence_directory=None if evidence_directory is None else str(evidence_directory),
        )
    except GroundTruthInputError as exc:
        raise ExperimentPreflightError(
            "real experiment requires validated methodological ground-truth evidence"
        ) from exc


def _run_git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentPreflightError("could not inspect Git source identity") from exc
    return result.stdout.strip()


def capture_git_source_identity(
    *,
    require_clean: bool,
    require_tag: bool,
) -> GitSourceIdentity:
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("branch", "--show-current")
    tags = tuple(sorted(line for line in _run_git("tag", "--points-at", "HEAD").splitlines() if line))
    status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    clean = not status
    if require_clean and not clean:
        raise ExperimentPreflightError("real experiment requires a clean Git worktree")
    if require_tag and not tags:
        raise ExperimentPreflightError("real experiment requires a tag at the exact source commit")
    if not branch:
        raise ExperimentPreflightError("real experiment requires an attached Git branch")
    return GitSourceIdentity(commit=commit, branch=branch, tags=tags, clean=clean)


def validate_git_identity_unchanged(expected: GitSourceIdentity) -> None:
    current = capture_git_source_identity(require_clean=True, require_tag=True)
    if current != expected:
        raise ExperimentPreflightError("Git/source identity drifted during experiment")


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ExperimentPreflightError("no existing parent for output_root")
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise ExperimentPreflightError("output_root parent is not a safe directory")
    return candidate


def validate_operational_disk_space(output_root: Path) -> int:
    """Enforce an operational staging margin, not a research-semantic threshold."""
    parent = _nearest_existing_directory(output_root)
    try:
        free = shutil.disk_usage(parent).free
    except OSError as exc:
        raise ExperimentPreflightError("could not inspect output filesystem capacity") from exc
    if free < OPERATIONAL_MINIMUM_FREE_BYTES:
        raise ExperimentPreflightError(
            "insufficient operational free space for the 12-run atomic artifact chain"
        )
    return free


class FullExperimentTurnMetadataResolver:
    """Resolve the complete 104-turn/48-reflection source timeline."""

    def __init__(self, config: ExperimentConfig, run: RunLogMetadata) -> None:
        if not isinstance(config, ExperimentConfig) or not isinstance(run, RunLogMetadata):
            raise ExperimentUsageError("turn resolver requires config and run metadata")
        self._main: dict[tuple[str, int], TurnLogMetadata] = {}
        self._reflections: dict[tuple[str, str], TurnLogMetadata] = {}
        timestamp = 0
        policy = TurnPolicyManager()
        for step in config.steps:
            order = (
                policy.build_turn_order(step)
                if step.step_type == "natural_interaction_block"
                else step.participants
            )
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
                    turn_id=build_reflection_operation_id(step.step_id, participant_index),
                    turn_index=participant_index,
                    agent_id=agent_id,
                )
        if timestamp != 104 or len(self._reflections) != 48:
            raise ExperimentUsageError("turn resolver differs from the full-run timeline")

    def resolve(self, request: LLMRequest) -> TurnLogMetadata:
        is_reflection = request.call_type is LLMCallType.REFLECTION or (
            request.call_type is LLMCallType.IMPORTANCE_SCORE
            and request.memory_kind == "reflection"
        )
        if is_reflection:
            result = self._reflections.get(
                (request.context.source_step_id, request.context.agent_id)
            )
        else:
            result = self._main.get(
                (request.context.source_step_id, request.context.logical_timestamp)
            )
        if (
            result is None
            or result.agent_id != request.context.agent_id
            or result.step.timestamp != request.context.logical_timestamp
        ):
            raise ExperimentUsageError("simulator request does not match full timeline")
        return result


class ExperimentAttemptLedger:
    """One global, run-qualified, durable attempt and token budget."""

    def __init__(
        self,
        *,
        attempts_path: Path,
        plan: ExperimentPlan,
        max_total_tokens: int,
    ) -> None:
        if isinstance(max_total_tokens, bool) or not isinstance(max_total_tokens, int) or max_total_tokens < 1:
            raise ExperimentUsageError("max_total_tokens must be positive")
        self.attempts_path = attempts_path
        self.plan = plan
        self.max_total_tokens = max_total_tokens
        self._records: list[ExperimentAttempt] = []
        self._seen: set[tuple[str, str, str, int]] = set()
        self._logical: set[tuple[str, str, str]] = set()

    @property
    def records(self) -> tuple[ExperimentAttempt, ...]:
        return tuple(self._records)

    @property
    def known_total_tokens(self) -> int:
        return sum(
            item.attempt.usage.total_tokens
            for item in self._records
            if item.attempt.usage.total_tokens is not None
        )

    @property
    def missing_usage_attempt_count(self) -> int:
        return sum(item.attempt.usage.total_tokens is None for item in self._records)

    def _run_records(self, run_id: str) -> tuple[ExperimentAttempt, ...]:
        return tuple(item for item in self._records if item.run_id == run_id)

    def authorize_logical_call(self, run_id: str, service_role: str) -> None:
        if service_role not in {"simulator", "embedding"}:
            raise ExperimentBudgetError("Phase 13 permits no evaluator calls")
        if self.missing_usage_attempt_count > missing_usage_cap(len(self._logical)):
            raise ExperimentUsageError(
                "provider usage is incomplete beyond the tolerated missing-usage cap; "
                "no later logical call is permitted"
            )
        run_logical = {
            (item.attempt.service_role, item.attempt.logical_call_id)
            for item in self._run_records(run_id)
        }
        role_count = sum(role == service_role for role, _ in run_logical)
        if role_count >= self.plan.per_run_call_limits.logical_limit(service_role):
            raise ExperimentBudgetError("per-run provider logical-call limit is exhausted")
        if len(self._logical) >= self.plan.total_logical_calls:
            raise ExperimentBudgetError("experiment provider logical-call limit is exhausted")
        if self.known_total_tokens >= self.max_total_tokens:
            raise ExperimentBudgetError(
                "global experiment token budget is exhausted before the next call"
            )

    def record(self, run_id: str, attempt: ProviderAttemptRecord) -> ExperimentAttempt:
        record = ExperimentAttempt(run_id, attempt)
        expected_model = {
            "simulator": SIMULATOR_MODEL_IDENTITY,
            "embedding": EMBEDDING_MODEL_IDENTITY,
        }[attempt.service_role]
        if attempt.model != expected_model:
            raise ExperimentUsageError("provider attempt model identity mismatch")
        key = (
            run_id,
            attempt.service_role,
            attempt.logical_call_id,
            attempt.attempt_index,
        )
        logical = (run_id, attempt.service_role, attempt.logical_call_id)
        prior = tuple(
            item for item in self._records
            if (
                item.run_id,
                item.attempt.service_role,
                item.attempt.logical_call_id,
            ) == logical
        )
        if key in self._seen or attempt.attempt_index != len(prior) + 1:
            raise ExperimentUsageError("provider attempt identity/order is invalid")
        if attempt.attempt_index == 1 and logical in self._logical:
            raise ExperimentUsageError("provider logical call identity was reused")
        if attempt.service_role == "simulator":
            expected_prompt = _EXPECTED_PROMPT_FOR_CALL.get(attempt.call_type)
            if (
                expected_prompt is None
                or attempt.prompt_template != expected_prompt
                or attempt.prompt_version != PROMPT_VERSION
                or attempt.prompt_hash != APPROVED_PHASE10_PROMPT_HASHES[expected_prompt]
            ):
                raise ExperimentUsageError("simulator attempt prompt identity mismatch")
        run_records = self._run_records(run_id)
        projected_run_logical = {
            (item.attempt.service_role, item.attempt.logical_call_id)
            for item in run_records
        } | {(attempt.service_role, attempt.logical_call_id)}
        role_logical = sum(role == attempt.service_role for role, _ in projected_run_logical)
        role_physical = sum(
            item.attempt.service_role == attempt.service_role for item in run_records
        ) + 1
        if (
            role_logical > self.plan.per_run_call_limits.logical_limit(attempt.service_role)
            or role_physical > self.plan.per_run_call_limits.physical_limit(attempt.service_role)
            or len(self._logical | {logical}) > self.plan.total_logical_calls
            or len(self._records) + 1 > self.plan.maximum_physical_attempts
        ):
            raise ExperimentBudgetError("provider call/attempt ceiling was exceeded")
        append_attempt(self.attempts_path, canonical_json_bytes(record.to_dict()))
        self._records.append(record)
        self._seen.add(key)
        self._logical.add(logical)
        known_total = self.known_total_tokens
        if known_total > self.max_total_tokens:
            raise ExperimentBudgetError("global experiment token budget was exceeded")
        if attempt.status != "success" and known_total >= self.max_total_tokens:
            raise ExperimentBudgetError("token budget is exhausted before a retry")
        return record

    def validate_completed_run(self, run_id: str) -> None:
        records = self._run_records(run_id)
        logical = {
            (item.attempt.service_role, item.attempt.logical_call_id) for item in records
        }
        role_counts = CounterRole(logical)
        final: dict[tuple[str, str], ProviderAttemptRecord] = {}
        for item in records:
            final[(item.attempt.service_role, item.attempt.logical_call_id)] = item.attempt
        missing_usage = sum(
            item.attempt.usage.total_tokens is None for item in records
        )
        if (
            role_counts.simulator != self.plan.per_run_call_limits.simulator_logical_calls
            or role_counts.embedding != self.plan.per_run_call_limits.embedding_logical_calls
            or missing_usage > missing_usage_cap(len({item.attempt.logical_call_id for item in records}))
            or any(attempt.status != "success" for attempt in final.values())
        ):
            raise ExperimentUsageError("completed real run has incomplete provider accounting")

    def summary(self) -> ExperimentUsageSummary:
        state = (
            "exceeded"
            if self.known_total_tokens > self.max_total_tokens
            else "exhausted"
            if self.known_total_tokens >= self.max_total_tokens
            else "admitted"
        )
        return ExperimentUsageSummary.from_attempts(
            self.records,
            max_total_tokens=self.max_total_tokens,
            budget_state=state,
        )


@dataclass(frozen=True)
class CounterRole:
    simulator: int
    embedding: int

    def __init__(self, logical: set[tuple[str, str]]) -> None:
        object.__setattr__(self, "simulator", sum(role == "simulator" for role, _ in logical))
        object.__setattr__(self, "embedding", sum(role == "embedding" for role, _ in logical))


class RunAttemptRecorder(UsageRecorder):
    """Run-local request context over the global experiment ledger."""

    def __init__(
        self,
        *,
        ledger: ExperimentAttemptLedger,
        spec: ExperimentRunSpec,
        run_metadata: RunLogMetadata,
        config: ExperimentConfig,
        api_usage_path: Path,
    ) -> None:
        self._ledger = ledger
        self._spec = spec
        self._resolver = FullExperimentTurnMetadataResolver(config, run_metadata)
        self._api_usage_path = api_usage_path
        self._active_request: LLMRequest | None = None

    @property
    def records(self) -> tuple[ProviderAttemptRecord, ...]:
        return tuple(
            item.attempt
            for item in self._ledger.records
            if item.run_id == self._spec.run_id
        )

    def authorize_logical_call(self, service_role: str) -> None:
        self._ledger.authorize_logical_call(self._spec.run_id, service_role)

    @contextmanager
    def simulator_request(self, request: LLMRequest) -> Iterator[None]:
        if self._active_request is not None:
            raise ExperimentUsageError("nested simulator request context is invalid")
        self._active_request = request
        try:
            yield
        finally:
            self._active_request = None

    def record(self, attempt: ProviderAttemptRecord) -> None:
        self._ledger.record(self._spec.run_id, attempt)
        if attempt.service_role == "simulator":
            self._project_simulator(attempt)
        elif attempt.status in {"success", "invalid_output"} and (
            attempt.usage.total_tokens is None
        ):
            raise ExperimentUsageError(
                "successful embedding attempt requires provider token usage"
            )

    def _project_simulator(self, attempt: ProviderAttemptRecord) -> None:
        usage = attempt.usage
        counts = (usage.input_tokens, usage.output_tokens, usage.total_tokens)
        if all(item is None for item in counts) and attempt.status == "failed":
            return
        if any(item is None for item in counts):
            raise ExperimentUsageError(
                "simulator attempt requires complete canonical token usage"
            )
        if self._active_request is None or self._active_request.call_type.value != attempt.call_type:
            raise ExperimentUsageError("simulator attempt request context is invalid")
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


class GuardedExperimentLLMClient:
    def __init__(self, client: LLMClient, recorder: RunAttemptRecorder) -> None:
        self._client = client
        self._recorder = recorder

    def generate(self, request: LLMRequest) -> str:
        self._recorder.authorize_logical_call("simulator")
        with self._recorder.simulator_request(request):
            return self._client.generate(request)


class GuardedExperimentEmbeddingProvider:
    def __init__(self, provider: TextEmbeddingProvider, recorder: RunAttemptRecorder) -> None:
        self._provider = provider
        self._recorder = recorder

    def embed(self, text: str) -> tuple[float, ...]:
        self._recorder.authorize_logical_call("embedding")
        return self._provider.embed(text)


def _default_provider_factory() -> object:
    # Imported only after every safe preflight and the credential-presence check.
    from src.llm.openai_client import OpenAIProviderAdapter

    return OpenAIProviderAdapter.from_environment(
        timeout_seconds=SIMULATOR_PROFILE.request_timeout_seconds
    )


def _freeze_identity(value: PilotFreezeValidation | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {
        "pilot_id": value.pilot_id,
        "manifest_hash": value.manifest_hash,
        "acceptance_hash": value.acceptance_hash,
        "freeze_hash": value.freeze_hash,
        "prompt_version": value.prompt_version,
    }


def _publish_plan(paths: ExperimentPaths, plan: ExperimentPlan) -> None:
    value = plan.to_dict()
    publish_new_json(
        paths.plan_file,
        canonical_json_bytes(value),
        validator=lambda parsed: (
            None
            if dict(parsed) == value
            else (_ for _ in ()).throw(ExperimentValidationError("persisted plan changed"))
        ),
    )


def _safe_error_type(exc: BaseException) -> str:
    if isinstance(exc, RunExecutionError):
        value = type(exc.original_error).__name__
    else:
        value = type(exc).__name__
    return value if value.replace("_", "").isalnum() else "ExperimentInternalError"


def _terminal_usage(
    *,
    mode: str,
    ledger: ExperimentAttemptLedger | None,
    plan: ExperimentPlan,
    completed_run_count: int,
) -> ExperimentUsageSummary:
    if ledger is not None:
        return ledger.summary()
    return ExperimentUsageSummary.from_attempts(
        (),
        max_total_tokens=None,
        budget_state="not_applicable",
        dummy_logical_counts=(
            completed_run_count * plan.per_run_call_limits.simulator_logical_calls,
            completed_run_count * plan.per_run_call_limits.embedding_logical_calls,
        ),
    )


def _publish_terminal_report(
    paths: ExperimentPaths,
    *,
    plan: ExperimentPlan,
    state: str,
    completed_run_ids: tuple[str, ...],
    failed_run_id: str | None,
    error_type: str,
    usage: ExperimentUsageSummary,
) -> None:
    value = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": plan.experiment_id,
        "mode": plan.mode,
        "status": state,
        "completed_run_ids": list(completed_run_ids),
        "failed_run_id": failed_run_id,
        "error_type": error_type,
        "message": (
            "Full experiment interrupted; inspect canonical partial artifacts."
            if state == "interrupted"
            else "Full experiment failed; inspect canonical safe evidence."
        ),
        "usage": usage.to_dict(),
        "successful_manifest_published": False,
        "resume_permitted": False,
    }
    publish_new_json(
        paths.terminal_report_file,
        canonical_json_bytes(value),
        validator=lambda parsed: (
            None
            if dict(parsed) == value
            else (_ for _ in ()).throw(ExperimentValidationError("terminal report changed"))
        ),
    )


def _build_manifest(
    *,
    plan: ExperimentPlan,
    chains: tuple[EligibleRunChain, ...],
    usage: ExperimentUsageSummary,
    source_identity: GitSourceIdentity,
    freeze: PilotFreezeValidation | None,
    config_directory: Path,
    per_run_usage: Mapping[str, ExperimentUsageSummary],
) -> dict[str, Any]:
    validate_complete_chains(plan, chains)
    loader = PromptLoader()
    run_values: list[dict[str, Any]] = []
    for chain in chains:
        item = chain.to_dict()
        item["usage"] = per_run_usage[chain.run.run_id].to_dict()
        run_values.append(item)
    expected_totals = plan.to_dict()["experiment_totals"]
    actual_totals = {
        "run_count": len(chains),
        "source_run_count": len(chains),
        "evaluation_count": len(chains),
        "step_count": sum(chain.source.step_count for chain in chains),
        "main_turn_count": sum(chain.source.main_turn_count for chain in chains),
        "post_step_reflection_count": sum(
            chain.source.reflection_count for chain in chains
        ),
        "mra_record_count": sum(
            chain.evaluation.mra_record_count for chain in chains
        ),
        "bcr_export_record_count": sum(
            chain.evaluation.bcr_export_record_count for chain in chains
        ),
        "bcr_judgment_count": 0,
    }
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": plan.experiment_id,
        "mode": plan.mode,
        "status": "completed",
        "completion_contract": "source_plus_phase09_evaluation_ready",
        "plan_hash": plan.plan_hash,
        "ordered_matrix": [run.to_dict() for run in plan.runs],
        "source_identity": source_identity.to_dict(),
        "accepted_freeze_identity": _freeze_identity(freeze),
        "configuration_hashes": [
            item.to_dict() for item in hash_config_directory(config_directory)
        ],
        "prompt_identities": [item.to_dict() for item in prompt_identities(loader)],
        "simulator_settings": SIMULATOR_PROFILE.reproducibility_metadata(),
        "embedding_settings": EMBEDDING_PROFILE.reproducibility_metadata(),
        "retry_policy": {
            "maximum_attempts_per_logical_call": MAX_PROVIDER_ATTEMPTS,
            "deterministic_delay_seconds": RETRY_DELAY_SECONDS,
            "jitter": False,
            "whole_run_retry": False,
            "resume": False,
        },
        "schema_identities": {
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "research_log_schema_version": SCHEMA_VERSION,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "bcr_rubric_version": BCR_RUBRIC_VERSION,
        },
        "expected_totals": expected_totals,
        "actual_totals": actual_totals,
        "runs": run_values,
        "usage": usage.to_dict(),
        "final_validation_state": "valid",
    }


def execute_full_experiment(
    *,
    mode: str,
    output_root: str | Path,
    experiment_id: str,
    confirmation: str | None,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
    real_authorized: bool = False,
    max_total_tokens: int | None = None,
    accepted_pilot_review_dir: str | Path | None = None,
    ground_truth_finalization_dir: str | Path | None = None,
    on_conflict: str = "fail",
    sleeper: Callable[[float], None] | None = None,
    provider_factory: Callable[[], object] | None = None,
) -> ExperimentExecutionResult:
    """Execute exactly 12 fresh source/evaluation chains, or fail fast."""
    if mode not in {"dummy", "real"}:
        raise ExperimentContractError("execute_full_experiment mode must be dummy or real")
    if on_conflict != "fail":
        raise ExperimentContractError("Phase 13 supports only conflict policy 'fail'")

    # Required ordering: plan and all safe non-credential preflights first.
    plan = build_full_experiment_plan(
        experiment_id=experiment_id,
        mode=mode,
        config_directory=config_directory,
    )
    config_path = Path(config_directory)
    config = load_experiment_config(config_path)
    paths = build_experiment_paths(output_root, experiment_id=experiment_id)
    validate_fresh_experiment_paths(paths)
    effective_budget = validate_execution_authorization(
        mode=mode,
        confirmation=confirmation,
        real_authorized=real_authorized,
        max_total_tokens=max_total_tokens,
    )

    freeze: PilotFreezeValidation | None = None
    active_provider: object | None = None
    if mode == "real":
        if accepted_pilot_review_dir is None:
            raise ExperimentPreflightError(
                "real experiment requires --accepted-pilot-review-dir"
            )
        try:
            freeze = validate_accepted_freeze(
                accepted_pilot_review_dir,
                config_directory=config_path,
                ground_truth_finalization_dir=ground_truth_finalization_dir,
            )
        except Exception:
            raise ExperimentPreflightError("accepted Phase 11 freeze is invalid") from None
        if ground_truth_finalization_dir is None:
            raise ExperimentPreflightError(
                "real experiment requires methodological ground-truth finalization evidence via --ground-truth-finalization-dir"
            )
        validate_ground_truth_methodological_finalization(
            config_path, ground_truth_finalization_dir
        )
        source_identity = capture_git_source_identity(require_clean=True, require_tag=True)
        validate_phase10_runtime()
        validate_operational_disk_space(paths.output_root)
        # This is the first credential-sensitive operation.
        require_openai_credential()
        active_provider = (provider_factory or _default_provider_factory)()
    else:
        if (
            accepted_pilot_review_dir is not None
            or ground_truth_finalization_dir is not None
            or provider_factory is not None
        ):
            raise ExperimentPreflightError(
                "dummy experiment must not receive real freeze/provider controls"
            )
        validate_phase10_runtime()
        source_identity = capture_git_source_identity(require_clean=False, require_tag=False)

    current_config_hashes = hash_config_directory(config_path)
    current_prompt_identities = prompt_identities(PromptLoader())

    def validate_freeze_identity_unchanged() -> None:
        if mode != "real":
            return
        try:
            current_freeze = validate_accepted_freeze(
                accepted_pilot_review_dir,
                config_directory=config_path,
                ground_truth_finalization_dir=ground_truth_finalization_dir,
            )
        except Exception:
            raise ExperimentPreflightError(
                "accepted freeze drifted during experiment"
            ) from None
        if current_freeze != freeze:
            raise ExperimentPreflightError("accepted freeze identity changed")

    create_experiment_directory(paths)
    _publish_plan(paths, plan)

    ledger = (
        ExperimentAttemptLedger(
            attempts_path=paths.attempts_file,
            plan=plan,
            max_total_tokens=effective_budget,
        )
        if mode == "real" and effective_budget is not None
        else None
    )
    completed: list[EligibleRunChain] = []
    per_run_usage: dict[str, ExperimentUsageSummary] = {}
    active_run_id: str | None = None

    try:
        for spec in plan.runs:
            active_run_id = spec.run_id
            if mode == "real":
                validate_git_identity_unchanged(source_identity)
                validate_freeze_identity_unchanged()
                validate_phase10_runtime()
                validate_operational_disk_space(paths.output_root)
                if (
                    hash_config_directory(config_path) != current_config_hashes
                    or prompt_identities(PromptLoader()) != current_prompt_identities
                ):
                    raise ExperimentPreflightError(
                        "configuration or prompt identity drifted during experiment"
                    )

            if mode == "dummy":
                llm_client = build_deterministic_dummy_llm_client()
                embedding_provider = DummyEmbeddingProvider()
                recorder = None
            else:
                if ledger is None or active_provider is None:
                    raise ExperimentPreflightError("real provider boundary was not initialized")
                run_metadata = RunLogMetadata(
                    experiment_id=plan.experiment_id,
                    run_id=spec.run_id,
                    retrieval_config_id=spec.retrieval_config_id,
                    repetition_index=spec.repetition_index,
                )
                recorder = RunAttemptRecorder(
                    ledger=ledger,
                    spec=spec,
                    run_metadata=run_metadata,
                    config=config,
                    api_usage_path=paths.source_run_dir(spec.run_id)
                    / "api_usage_logs.jsonl",
                )
                delay = sleeper if sleeper is not None else __import__("time").sleep
                llm_client = GuardedExperimentLLMClient(
                    LLMService(
                        active_provider,  # type: ignore[arg-type]
                        usage_recorder=recorder,
                        sleeper=delay,
                    ),
                    recorder,
                )
                embedding_provider = GuardedExperimentEmbeddingProvider(
                    OpenAIEmbeddingProvider(
                        active_provider,  # type: ignore[arg-type]
                        usage_recorder=recorder,
                        sleeper=delay,
                    ),
                    recorder,
                )

            run_result = RunController(
                llm_client=llm_client,
                embedding_provider=embedding_provider,
                config_directory=config_path,
                sanitize_failure_messages=mode == "real",
            ).run(
                output_root=paths.output_root,
                experiment_id=plan.experiment_id,
                run_id=spec.run_id,
                retrieval_config_id=spec.retrieval_config_id,
                repetition_index=spec.repetition_index,
            )
            if mode == "dummy":
                if (
                    len(llm_client.invocations)  # type: ignore[attr-defined]
                    != plan.per_run_call_limits.simulator_logical_calls
                    or len(embedding_provider.received_texts)  # type: ignore[attr-defined]
                    != plan.per_run_call_limits.embedding_logical_calls
                ):
                    raise ExperimentValidationError(
                        "dummy execution call inventory differs from derived graph"
                    )
            else:
                assert ledger is not None
                ledger.validate_completed_run(spec.run_id)

            if hash_config_directory(run_result.run_paths.config_snapshot) != current_config_hashes:
                raise ExperimentValidationError("source config snapshot differs from plan")
            source_validation = validate_source_run(
                run_result.run_paths.run_dir,
                spec=spec,
                scope=plan.run_scope,
                mode=mode,
            )
            source_before_evaluation = source_validation.source_fingerprint
            evaluation_result: EvaluationRunResult = evaluate_run(
                run_dir=run_result.run_paths.run_dir,
                output_dir=paths.evaluation_dir(spec.run_id),
                diagnostic=False,
            )
            if (
                len(evaluation_result.mra_records) != 20
                or len(evaluation_result.bcr_records) != 20
            ):
                raise ExperimentValidationError("Phase 09 returned incomplete primary output")
            evaluation_validation = validate_phase09_evaluation(
                evaluation_result.output_dir,
                spec=spec,
            )
            if validate_source_run(
                run_result.run_paths.run_dir,
                spec=spec,
                scope=plan.run_scope,
                mode=mode,
            ).source_fingerprint != source_before_evaluation:
                raise ExperimentValidationError("Phase 09 changed sealed source artifacts")
            chain = EligibleRunChain(spec, source_validation, evaluation_validation)
            validate_run_chain_unchanged(
                chain,
                source=run_result.run_paths.run_dir,
                evaluation=evaluation_result.output_dir,
            )
            completed.append(chain)
            if mode == "dummy":
                per_run_usage[spec.run_id] = ExperimentUsageSummary.from_attempts(
                    (),
                    max_total_tokens=None,
                    budget_state="not_applicable",
                    dummy_logical_counts=(
                        plan.per_run_call_limits.simulator_logical_calls,
                        plan.per_run_call_limits.embedding_logical_calls,
                    ),
                )
            else:
                assert ledger is not None
                selected = tuple(
                    item for item in ledger.records if item.run_id == spec.run_id
                )
                per_run_usage[spec.run_id] = ExperimentUsageSummary.from_attempts(
                    selected,
                    max_total_tokens=None,
                    budget_state="not_applicable",
                )

        chains = tuple(completed)
        validate_complete_chains(plan, chains)
        usage = (
            ledger.summary()
            if ledger is not None
            else ExperimentUsageSummary.from_attempts(
                (),
                max_total_tokens=None,
                budget_state="not_applicable",
                dummy_logical_counts=(
                    plan.total_simulator_logical_calls,
                    plan.total_embedding_logical_calls,
                ),
            )
        )
        if (
            usage.logical_call_count != plan.total_logical_calls
            or usage.missing_usage_attempt_count > missing_usage_cap(plan.total_logical_calls)
        ):
            raise ExperimentUsageError("experiment provider operation inventory is incomplete")
        if mode == "real" and ledger is not None:
            validate_git_identity_unchanged(source_identity)
            validate_freeze_identity_unchanged()
            validate_phase10_runtime()
            validate_operational_disk_space(paths.output_root)
            if (
                hash_config_directory(config_path) != current_config_hashes
                or prompt_identities(PromptLoader()) != current_prompt_identities
            ):
                raise ExperimentPreflightError("frozen identities drifted before manifest")
        manifest = _build_manifest(
            plan=plan,
            chains=chains,
            usage=usage,
            source_identity=source_identity,
            freeze=freeze,
            config_directory=config_path,
            per_run_usage=per_run_usage,
        )

        def recheck() -> None:
            for chain in chains:
                validate_run_chain_unchanged(
                    chain,
                    source=paths.source_run_dir(chain.run.run_id),
                    evaluation=paths.evaluation_dir(chain.run.run_id),
                )
            if mode == "real":
                validate_git_identity_unchanged(source_identity)
                validate_freeze_identity_unchanged()

        publish_experiment_manifest(
            paths,
            canonical_json_bytes(manifest),
            validator=lambda parsed: validate_experiment_manifest(
                parsed,
                plan=plan,
                expected=manifest,
            ),
            source_recheck=recheck,
        )
        return ExperimentExecutionResult(paths, plan, chains, usage, manifest)
    except BaseException as exc:
        state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error_type = _safe_error_type(exc)
        usage = _terminal_usage(
            mode=mode,
            ledger=ledger,
            plan=plan,
            completed_run_count=len(completed),
        )
        evidence_published = False
        try:
            _publish_terminal_report(
                paths,
                plan=plan,
                state=state,
                completed_run_ids=tuple(item.run.run_id for item in completed),
                failed_run_id=active_run_id,
                error_type=error_type,
                usage=usage,
            )
            evidence_published = True
        except Exception as evidence_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "terminal evidence publication also failed: "
                    f"{type(evidence_error).__name__}"
                )
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise ExperimentExecutionError(
            error_type,
            experiment_dir=paths.experiment_dir,
            terminal_state=state,
            failed_run_id=active_run_id,
            evidence_published=evidence_published,
        ) from None


__all__ = [
    "ExperimentAttemptLedger",
    "ExperimentExecutionResult",
    "FullExperimentTurnMetadataResolver",
    "GuardedExperimentEmbeddingProvider",
    "GuardedExperimentLLMClient",
    "OPERATIONAL_MINIMUM_FREE_BYTES",
    "RunAttemptRecorder",
    "capture_git_source_identity",
    "dry_run_full_experiment",
    "execute_full_experiment",
    "validate_execution_authorization",
    "validate_git_identity_unchanged",
    "validate_ground_truth_methodological_finalization",
    "validate_operational_disk_space",
]
