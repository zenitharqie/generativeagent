"""Real simulator service implementing the stable provider-neutral LLMClient contract."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import json
import time

from src.llm.base import LLMCallType, LLMClientError, LLMRequest
from src.llm.errors import (
    ContextLimitError,
    ProviderError,
    ProviderTransientError,
)
from src.llm.output_parsers import (
    LLMOutputError,
    parse_action_output,
    parse_importance_output,
    parse_plan_output,
    parse_reflection_output,
)
from src.llm.prompt_loader import PromptLoader
from src.llm.prompt_renderer import RenderedPrompt, render_simulator_prompt
from src.llm.provider import StructuredTextProvider, StructuredTextRequest
from src.llm.settings import (
    CONTEXT_ACCOUNTING_RESERVE,
    MAX_PROVIDER_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    SIMULATOR_PROFILE,
    GenerationProfile,
)
from src.llm.usage import (
    InMemoryUsageRecorder,
    ProviderAttemptRecord,
    ProviderUsage,
    UsageRecorder,
)


_EXPECTED_KEYS = {
    LLMCallType.PLAN: {"plan"},
    LLMCallType.NATURAL_RESPONSE: {"response", "memory_to_store", "category"},
    LLMCallType.TARGET_EVENT_RESPONSE: {"response", "memory_to_store", "category"},
    LLMCallType.REFLECTION: {"reflection", "category"},
    LLMCallType.IMPORTANCE_SCORE: {"importance", "reason"},
}


class _DuplicateOutputKeyError(ValueError):
    pass


def _strict_output_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateOutputKeyError(f"duplicate output key {key!r}")
        value[key] = item
    return value


def validate_context_limit(rendered: RenderedPrompt, profile: GenerationProfile, limit: int) -> None:
    """Use UTF-8 bytes as a conservative token upper bound; never truncate input."""
    input_upper_bound = len(rendered.system_instructions.encode("utf-8")) + len(
        rendered.user_data.encode("utf-8")
    )
    if input_upper_bound + limit + CONTEXT_ACCOUNTING_RESERVE > profile.context_window_tokens:
        raise ContextLimitError(
            "rendered request exceeds the conservative approved model context bound"
        )


def _strict_output_keys(raw_output: str, expected: set[str]) -> None:
    try:
        value = json.loads(raw_output, object_pairs_hook=_strict_output_object)
    except (TypeError, json.JSONDecodeError, _DuplicateOutputKeyError) as exc:
        raise LLMClientError("provider output is not a strict JSON object") from exc
    if not isinstance(value, dict) or set(value) != expected:
        raise LLMClientError("provider output fields do not match the strict schema")


def _validate_output(call_type: LLMCallType, raw_output: str) -> None:
    _strict_output_keys(raw_output, _EXPECTED_KEYS[call_type])
    if call_type is LLMCallType.PLAN:
        parse_plan_output(raw_output)
    elif call_type in {LLMCallType.NATURAL_RESPONSE, LLMCallType.TARGET_EVENT_RESPONSE}:
        parse_action_output(raw_output)
    elif call_type is LLMCallType.REFLECTION:
        parse_reflection_output(raw_output)
    else:
        parse_importance_output(raw_output)


class LLMService:
    """Render, call, validate, retry, and audit real simulator requests."""

    def __init__(
        self,
        provider: StructuredTextProvider,
        *,
        profile: GenerationProfile = SIMULATOR_PROFILE,
        prompt_loader: PromptLoader | None = None,
        usage_recorder: UsageRecorder | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile != SIMULATOR_PROFILE:
            raise LLMClientError("simulator service requires the approved simulator profile")
        if not isinstance(provider, StructuredTextProvider):
            raise LLMClientError("provider must implement StructuredTextProvider")
        self._provider = provider
        self._profile = profile
        self._prompt_loader = prompt_loader or PromptLoader()
        self._usage_recorder = usage_recorder or InMemoryUsageRecorder()
        self._sleeper = sleeper
        self._logical_sequence = 0
        self._last_attempts: tuple[ProviderAttemptRecord, ...] = ()

    @property
    def usage_records(self) -> tuple[ProviderAttemptRecord, ...]:
        records = getattr(self._usage_recorder, "records", None)
        return tuple(records) if records is not None else ()

    @property
    def last_attempts(self) -> tuple[ProviderAttemptRecord, ...]:
        return self._last_attempts

    def _logical_call_id(self, request: LLMRequest, rendered: RenderedPrompt) -> str:
        self._logical_sequence += 1
        identity = "|".join(
            (
                str(self._logical_sequence),
                request.call_type.value,
                request.context.source_step_id,
                str(request.context.logical_timestamp),
                request.context.agent_id,
                request.memory_kind or "none",
                rendered.user_data,
            )
        )
        return "simulator_" + sha256(identity.encode("utf-8")).hexdigest()

    def _record(
        self,
        *,
        logical_call_id: str,
        attempt_index: int,
        request: LLMRequest,
        rendered: RenderedPrompt,
        status: str,
        usage: ProviderUsage = ProviderUsage(),
        error_type: str | None = None,
    ) -> ProviderAttemptRecord:
        record = ProviderAttemptRecord(
            service_role="simulator",
            logical_call_id=logical_call_id,
            attempt_index=attempt_index,
            call_type=request.call_type.value,
            model=self._profile.model_identity,
            prompt_template=rendered.template_id,
            prompt_version=rendered.template_version,
            prompt_hash=rendered.template_hash,
            usage=usage,
            status=status,
            error_type=error_type,
        )
        self._usage_recorder.record(record)
        return record

    def generate(self, request: LLMRequest) -> str:
        if not isinstance(request, LLMRequest):
            raise LLMClientError("request must be LLMRequest")
        rendered = render_simulator_prompt(request, self._prompt_loader)
        output_limit = self._profile.output_limit_for(request.call_type.value)
        validate_context_limit(rendered, self._profile, output_limit)
        logical_call_id = self._logical_call_id(request, rendered)
        attempts: list[ProviderAttemptRecord] = []
        for attempt_index in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            provider_request = StructuredTextRequest(
                model=self._profile.model,
                instructions=rendered.system_instructions,
                user_data=rendered.user_data,
                schema_name=rendered.schema_name,
                output_schema_json=rendered.output_schema_json,
                max_output_tokens=output_limit,
                reasoning_effort=self._profile.reasoning_effort,
                timeout_seconds=self._profile.request_timeout_seconds,
                stream=False,
            )
            try:
                response = self._provider.generate_structured(provider_request)
            except KeyboardInterrupt:
                record = self._record(
                    logical_call_id=logical_call_id,
                    attempt_index=attempt_index,
                    request=request,
                    rendered=rendered,
                    status="failed",
                    error_type="KeyboardInterrupt",
                )
                attempts.append(record)
                self._last_attempts = tuple(attempts)
                raise
            except ProviderError as exc:
                record = self._record(
                    logical_call_id=logical_call_id,
                    attempt_index=attempt_index,
                    request=request,
                    rendered=rendered,
                    status="failed",
                    error_type=type(exc).__name__,
                )
                attempts.append(record)
                self._last_attempts = tuple(attempts)
                if isinstance(exc, ProviderTransientError) and attempt_index < MAX_PROVIDER_ATTEMPTS:
                    self._sleeper(RETRY_DELAY_SECONDS)
                    continue
                raise
            try:
                _validate_output(request.call_type, response.output_text)
            except (LLMOutputError, LLMClientError) as exc:
                record = self._record(
                    logical_call_id=logical_call_id,
                    attempt_index=attempt_index,
                    request=request,
                    rendered=rendered,
                    status="invalid_output",
                    usage=response.usage,
                    error_type=type(exc).__name__,
                )
                attempts.append(record)
                self._last_attempts = tuple(attempts)
                if attempt_index < MAX_PROVIDER_ATTEMPTS:
                    self._sleeper(RETRY_DELAY_SECONDS)
                    continue
                raise LLMClientError(
                    "OpenAI simulator returned invalid structured output twice"
                ) from exc
            record = self._record(
                logical_call_id=logical_call_id,
                attempt_index=attempt_index,
                request=request,
                rendered=rendered,
                status="success",
                usage=response.usage,
            )
            attempts.append(record)
            self._last_attempts = tuple(attempts)
            return response.output_text
        raise LLMClientError("approved simulator attempt limit was exhausted")


__all__ = ["LLMService", "validate_context_limit"]
