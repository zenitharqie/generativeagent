"""Condition-blind real BCR judging and complete-run orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

from src.evaluation.errors import (
    BCRJudgmentError,
    BCRJudgmentFailure,
    BCRJudgmentIndependenceError,
)
from src.evaluation.judged_schemas import BCRJudgedRecord, BCRJudgmentSummary
from src.evaluation.judgment_persistence import (
    prepare_judgment_output,
    publish_bcr_judgments,
)
from src.evaluation.judgment_reader import (
    LoadedBCRExport,
    load_bcr_export,
    source_directory_fingerprint,
)
from src.evaluation.schemas import (
    BCRExportRecord,
    BCRModelMetadata,
    FutureBCRJudgment,
)
from src.llm.errors import ProviderError, ProviderTransientError
from src.llm.prompt_loader import PromptLoader
from src.llm.prompt_renderer import RenderedPrompt, canonical_json, render_bcr_prompt
from src.llm.provider import StructuredTextProvider, StructuredTextRequest
from src.llm.service import validate_context_limit
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    EVALUATOR_MODEL_IDENTITY,
    EVALUATOR_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    SIMULATOR_MODEL_IDENTITY,
)
from src.llm.usage import ProviderAttemptRecord, ProviderUsage


_JUDGMENT_KEYS = {
    "communication_style_score",
    "communication_style_reason",
    "focus_consistency_score",
    "focus_consistency_reason",
    "temporal_consistency_score",
    "temporal_consistency_reason",
}


class _DuplicateJudgmentKeyError(ValueError):
    pass


def _strict_judgment_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJudgmentKeyError(f"duplicate evaluator key {key!r}")
        value[key] = item
    return value


def validate_evaluator_independence(source: LoadedBCRExport) -> None:
    """Apply exact provider-qualified model checks before any primary judge call."""
    simulator = source.simulator_model_identity
    evaluator = EVALUATOR_MODEL_IDENTITY
    if source.evaluation_mode == "primary":
        if simulator is None:
            raise BCRJudgmentIndependenceError(
                "primary BCR judgment requires explicit simulator model identity"
            )
        if simulator == evaluator:
            raise BCRJudgmentIndependenceError(
                "primary simulator and evaluator model identities must differ"
            )
        if simulator != SIMULATOR_MODEL_IDENTITY:
            raise BCRJudgmentIndependenceError(
                "primary BCR source does not use the approved simulator identity"
            )
    elif simulator == evaluator:
        raise BCRJudgmentIndependenceError(
            "diagnostic simulator and evaluator identities must not be exactly equal"
        )


def _parse_judgment(raw_output: str) -> FutureBCRJudgment:
    try:
        value = json.loads(raw_output, object_pairs_hook=_strict_judgment_object)
    except (TypeError, json.JSONDecodeError, _DuplicateJudgmentKeyError) as exc:
        raise BCRJudgmentError("evaluator output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != _JUDGMENT_KEYS:
        raise BCRJudgmentError("evaluator output fields do not match the BCR contract")
    try:
        return FutureBCRJudgment(**value)
    except (TypeError, ValueError) as exc:
        raise BCRJudgmentError("evaluator output violates the BCR score contract") from exc


@dataclass(frozen=True)
class BCRJudgeOutcome:
    judgment: FutureBCRJudgment
    rendered_prompt: RenderedPrompt
    provider_attempts: tuple[ProviderAttemptRecord, ...]


class BCRJudgeService:
    """Judge only the neutral Phase 09 payload and retain complete attempt audit."""

    def __init__(
        self,
        provider: StructuredTextProvider,
        *,
        prompt_loader: PromptLoader | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(provider, StructuredTextProvider):
            raise BCRJudgmentError("provider must implement StructuredTextProvider")
        self._provider = provider
        self._prompt_loader = prompt_loader or PromptLoader()
        self._sleeper = sleeper

    def _logical_call_id(self, record: BCRExportRecord, rendered: RenderedPrompt) -> str:
        value = {
            "identity": record.identity.to_dict(),
            "judge_input": json.loads(rendered.user_data),
        }
        return "evaluator_" + sha256(canonical_json(value).encode("utf-8")).hexdigest()

    def _attempt(
        self,
        *,
        logical_call_id: str,
        attempt_index: int,
        rendered: RenderedPrompt,
        status: str,
        usage: ProviderUsage = ProviderUsage(),
        error_type: str | None = None,
    ) -> ProviderAttemptRecord:
        return ProviderAttemptRecord(
            service_role="evaluator",
            logical_call_id=logical_call_id,
            attempt_index=attempt_index,
            call_type="bcr_judge",
            model=EVALUATOR_MODEL_IDENTITY,
            prompt_template=rendered.template_id,
            prompt_version=rendered.template_version,
            prompt_hash=rendered.template_hash,
            usage=usage,
            status=status,
            error_type=error_type,
        )

    def judge(self, record: BCRExportRecord) -> BCRJudgeOutcome:
        if not isinstance(record, BCRExportRecord):
            raise BCRJudgmentError("record must be BCRExportRecord")
        rendered = render_bcr_prompt(record.judge_input, self._prompt_loader)
        output_limit = EVALUATOR_PROFILE.output_limit_for("bcr_judge")
        validate_context_limit(rendered, EVALUATOR_PROFILE, output_limit)
        logical_call_id = self._logical_call_id(record, rendered)
        attempts: list[ProviderAttemptRecord] = []
        for attempt_index in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            request = StructuredTextRequest(
                model=EVALUATOR_PROFILE.model,
                instructions=rendered.system_instructions,
                user_data=rendered.user_data,
                schema_name=rendered.schema_name,
                output_schema_json=rendered.output_schema_json,
                max_output_tokens=output_limit,
                reasoning_effort=EVALUATOR_PROFILE.reasoning_effort,
                timeout_seconds=EVALUATOR_PROFILE.request_timeout_seconds,
                stream=False,
            )
            terminal_provider_failure: BCRJudgmentFailure | None = None
            try:
                response = self._provider.generate_structured(request)
            except ProviderError as exc:
                attempts.append(
                    self._attempt(
                        logical_call_id=logical_call_id,
                        attempt_index=attempt_index,
                        rendered=rendered,
                        status="failed",
                        error_type=type(exc).__name__,
                    )
                )
                if isinstance(exc, ProviderTransientError) and attempt_index < MAX_PROVIDER_ATTEMPTS:
                    self._sleeper(RETRY_DELAY_SECONDS)
                    continue
                terminal_provider_failure = BCRJudgmentFailure(
                    "provider_failure",
                    tuple(attempts),
                )
            if terminal_provider_failure is not None:
                raise terminal_provider_failure from None

            terminal_output_failure: BCRJudgmentFailure | None = None
            try:
                judgment = _parse_judgment(response.output_text)
            except BCRJudgmentError as exc:
                attempts.append(
                    self._attempt(
                        logical_call_id=logical_call_id,
                        attempt_index=attempt_index,
                        rendered=rendered,
                        status="invalid_output",
                        usage=response.usage,
                        error_type=type(exc).__name__,
                    )
                )
                if attempt_index < MAX_PROVIDER_ATTEMPTS:
                    self._sleeper(RETRY_DELAY_SECONDS)
                    continue
                terminal_output_failure = BCRJudgmentFailure(
                    "invalid_output",
                    tuple(attempts),
                )
            if terminal_output_failure is not None:
                raise terminal_output_failure from None
            attempts.append(
                self._attempt(
                    logical_call_id=logical_call_id,
                    attempt_index=attempt_index,
                    rendered=rendered,
                    status="success",
                    usage=response.usage,
                )
            )
            return BCRJudgeOutcome(judgment, rendered, tuple(attempts))
        raise BCRJudgmentError("approved evaluator attempt limit was exhausted")


@dataclass(frozen=True)
class BCRJudgmentRunResult:
    records: tuple[BCRJudgedRecord, ...]
    summary: BCRJudgmentSummary
    output_directory: Path


def judge_bcr_export(
    *,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    provider: StructuredTextProvider,
    sleeper: Callable[[float], None] = time.sleep,
) -> BCRJudgmentRunResult:
    """Read, preflight, judge all units, verify source immutability, and publish."""
    source = load_bcr_export(evaluation_dir)
    paths = prepare_judgment_output(output_dir, source_directory=source.source_directory)
    validate_evaluator_independence(source)
    service = BCRJudgeService(provider, sleeper=sleeper)
    records: list[BCRJudgedRecord] = []
    for export_record, source_hash in zip(
        source.records, source.source_record_hashes, strict=True
    ):
        outcome = service.judge(export_record)
        metadata = BCRModelMetadata(
            simulator_model_identity=source.simulator_model_identity,
            evaluator_model_identity=EVALUATOR_MODEL_IDENTITY,
            evaluator_prompt_version=outcome.rendered_prompt.template_version,
            evaluator_rubric_version=BCR_RUBRIC_VERSION,
        )
        records.append(
            BCRJudgedRecord(
                identity=export_record.identity,
                evaluation_mode=export_record.evaluation_mode,
                primary_eligible=export_record.primary_eligible,
                included_in_final_research_aggregation=(
                    export_record.included_in_final_research_aggregation
                ),
                judgment=outcome.judgment,
                model_metadata=metadata,
                prompt_template=outcome.rendered_prompt.template_id,
                prompt_hash=outcome.rendered_prompt.template_hash,
                source_record_hash=source_hash,
                provider_attempts=outcome.provider_attempts,
            )
        )
    if source_directory_fingerprint(source.source_directory) != source.source_fingerprint:
        raise BCRJudgmentError("source Phase 09 evaluation artifacts changed during judgment")
    record_tuple = tuple(records)
    first = record_tuple[0]
    summary = BCRJudgmentSummary(
        experiment_id=source.experiment_id,
        run_id=source.run_id,
        retrieval_config_id=source.retrieval_config_id,
        repetition_index=source.repetition_index,
        evaluation_mode=source.evaluation_mode,
        primary_eligible=source.primary_eligible,
        included_in_final_research_aggregation=(
            source.included_in_final_research_aggregation
        ),
        source_evaluation_schema_version=source.source_evaluation_schema_version,
        source_bcr_export_hash=source.source_bcr_export_hash,
        records=record_tuple,
        model_metadata=first.model_metadata,
        prompt_template=first.prompt_template,
        prompt_hash=first.prompt_hash,
    )
    published = publish_bcr_judgments(paths=paths, records=record_tuple, summary=summary)
    return BCRJudgmentRunResult(record_tuple, summary, published)


__all__ = [
    "BCRJudgeOutcome",
    "BCRJudgeService",
    "BCRJudgmentRunResult",
    "judge_bcr_export",
    "validate_evaluator_independence",
]
