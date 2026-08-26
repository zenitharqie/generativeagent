"""Offline simulator LLMService parsing, retry, and audit behavior."""

from __future__ import annotations

from collections import deque
import json

import pytest

from src.llm import LLMCallType, LLMClient, LLMClientError, LLMRequest
from src.llm.errors import ContextLimitError, ProviderAuthenticationError, ProviderTimeoutError
from src.llm.provider import StructuredTextRequest, StructuredTextResponse
from src.llm.service import LLMService
from src.llm.usage import ProviderUsage
from phase10_test_support import natural_context, target_context


VALID_OUTPUTS = {
    LLMCallType.PLAN: '{"plan":"Respond calmly."}',
    LLMCallType.NATURAL_RESPONSE: (
        '{"response":"Please share one idea.","memory_to_store":"The teacher invited one idea.",'
        '"category":"participation"}'
    ),
    LLMCallType.TARGET_EVENT_RESPONSE: (
        '{"response":"We will prepare together.","memory_to_store":"The teacher offered preparation.",'
        '"category":"support"}'
    ),
    LLMCallType.REFLECTION: '{"reflection":"Clear invitations support participation.","category":"participation"}',
    LLMCallType.IMPORTANCE_SCORE: '{"importance":6,"reason":"It may shape later participation."}',
}


class ScriptedTextProvider:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[StructuredTextRequest] = []

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return StructuredTextResponse(
            output_text=str(outcome),
            usage=ProviderUsage(10, 5, 15),
        )


def requests_by_call_type() -> dict[LLMCallType, LLMRequest]:
    plan = "Respond calmly."
    return {
        LLMCallType.PLAN: LLMRequest(LLMCallType.PLAN, natural_context()),
        LLMCallType.NATURAL_RESPONSE: LLMRequest(
            LLMCallType.NATURAL_RESPONSE,
            natural_context(),
            plan=plan,
            response_limit=2,
        ),
        LLMCallType.TARGET_EVENT_RESPONSE: LLMRequest(
            LLMCallType.TARGET_EVENT_RESPONSE,
            target_context(),
            plan=plan,
            response_limit=1,
        ),
        LLMCallType.REFLECTION: LLMRequest(
            LLMCallType.REFLECTION,
            natural_context(),
            plan=plan,
            response="Please share one idea.",
            response_limit=1,
        ),
        LLMCallType.IMPORTANCE_SCORE: LLMRequest(
            LLMCallType.IMPORTANCE_SCORE,
            natural_context(),
            memory_text="The teacher invited one idea.",
            memory_kind="action",
        ),
    }


@pytest.mark.parametrize(
    ("call_type", "expected_limit"),
    [
        (LLMCallType.PLAN, 300),
        (LLMCallType.NATURAL_RESPONSE, 500),
        (LLMCallType.TARGET_EVENT_RESPONSE, 500),
        (LLMCallType.REFLECTION, 400),
        (LLMCallType.IMPORTANCE_SCORE, 150),
    ],
)
def test_real_service_preserves_llm_contract_and_call_specific_settings(
    call_type: LLMCallType,
    expected_limit: int,
) -> None:
    provider = ScriptedTextProvider(VALID_OUTPUTS[call_type])
    service = LLMService(provider, sleeper=lambda _: None)

    assert isinstance(service, LLMClient)
    assert service.generate(requests_by_call_type()[call_type]) == VALID_OUTPUTS[call_type]
    provider_request = provider.requests[0]
    assert provider_request.model == "gpt-5.6-luna"
    assert provider_request.reasoning_effort == "minimal"
    assert provider_request.max_output_tokens == expected_limit
    assert provider_request.timeout_seconds == 60
    assert provider_request.stream is False


def test_invalid_structured_output_gets_exactly_one_retry() -> None:
    delays: list[float] = []
    provider = ScriptedTextProvider("not JSON", VALID_OUTPUTS[LLMCallType.PLAN])
    service = LLMService(provider, sleeper=delays.append)

    assert service.generate(requests_by_call_type()[LLMCallType.PLAN]) == VALID_OUTPUTS[LLMCallType.PLAN]
    assert delays == [1]
    assert tuple(record.status for record in service.last_attempts) == (
        "invalid_output",
        "success",
    )
    assert tuple(record.attempt_index for record in service.last_attempts) == (1, 2)
    assert len({record.logical_call_id for record in service.last_attempts}) == 1


def test_extra_structured_output_field_is_invalid_without_semantic_repair() -> None:
    provider = ScriptedTextProvider(
        '{"plan":"Respond calmly.","extra":"not allowed"}',
        '{"plan":"Still extra.","extra":"not allowed"}',
    )
    service = LLMService(provider, sleeper=lambda _: None)

    with pytest.raises(LLMClientError, match="invalid structured output twice"):
        service.generate(requests_by_call_type()[LLMCallType.PLAN])
    assert len(provider.requests) == 2


def test_duplicate_structured_output_key_is_rejected() -> None:
    duplicate = '{"plan":"first","plan":"second"}'
    provider = ScriptedTextProvider(duplicate, duplicate)
    service = LLMService(provider, sleeper=lambda _: None)

    with pytest.raises(LLMClientError, match="invalid structured output twice"):
        service.generate(requests_by_call_type()[LLMCallType.PLAN])


def test_transient_provider_error_retries_once_with_fixed_delay() -> None:
    delays: list[float] = []
    provider = ScriptedTextProvider(
        ProviderTimeoutError("sanitized"),
        VALID_OUTPUTS[LLMCallType.PLAN],
    )
    service = LLMService(provider, sleeper=delays.append)

    service.generate(requests_by_call_type()[LLMCallType.PLAN])

    assert delays == [1]
    assert tuple(item.status for item in service.last_attempts) == ("failed", "success")
    assert service.last_attempts[0].error_type == "ProviderTimeoutError"


def test_authentication_error_is_not_retried_and_is_sanitized_in_usage() -> None:
    sentinel = "phase10-usage-provider-secret"
    provider = ScriptedTextProvider(ProviderAuthenticationError(sentinel))
    service = LLMService(provider, sleeper=lambda _: pytest.fail("must not sleep"))

    with pytest.raises(ProviderAuthenticationError):
        service.generate(requests_by_call_type()[LLMCallType.PLAN])

    assert len(provider.requests) == 1
    assert service.last_attempts[0].error_type == "ProviderAuthenticationError"
    assert sentinel not in json.dumps(service.last_attempts[0].to_dict(), sort_keys=True)


def test_keyboard_interrupt_is_recorded_once_then_reraised_without_retry() -> None:
    sentinel = "raw-interrupt-detail-must-not-persist"
    request = requests_by_call_type()[LLMCallType.PLAN]
    services: list[LLMService] = []
    providers: list[ScriptedTextProvider] = []

    for _ in range(2):
        provider = ScriptedTextProvider(KeyboardInterrupt(sentinel))
        service = LLMService(
            provider,
            sleeper=lambda _: pytest.fail("interruption must not retry"),
        )

        with pytest.raises(KeyboardInterrupt, match=sentinel):
            service.generate(request)

        providers.append(provider)
        services.append(service)

    assert [len(provider.requests) for provider in providers] == [1, 1]
    assert [len(service.usage_records) for service in services] == [1, 1]
    first = services[0].usage_records[0]
    assert first == services[0].last_attempts[0]
    assert first.logical_call_id == services[1].usage_records[0].logical_call_id
    assert first.attempt_index == 1
    assert first.status == "failed"
    assert first.error_type == "KeyboardInterrupt"
    assert first.usage.to_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert sentinel not in json.dumps(first.to_dict(), sort_keys=True)


def test_context_limit_failure_occurs_before_provider_execution_without_truncation() -> None:
    huge_context = natural_context(observation_text="x" * 400_000)
    provider = ScriptedTextProvider(VALID_OUTPUTS[LLMCallType.PLAN])
    service = LLMService(provider, sleeper=lambda _: None)

    with pytest.raises(ContextLimitError, match="context bound"):
        service.generate(LLMRequest(LLMCallType.PLAN, huge_context))
    assert provider.requests == []
    assert service.usage_records == ()


def test_attempt_metadata_is_deterministic_across_fresh_services() -> None:
    request = requests_by_call_type()[LLMCallType.PLAN]
    first = LLMService(
        ScriptedTextProvider(VALID_OUTPUTS[LLMCallType.PLAN]), sleeper=lambda _: None
    )
    second = LLMService(
        ScriptedTextProvider(VALID_OUTPUTS[LLMCallType.PLAN]), sleeper=lambda _: None
    )

    first.generate(request)
    second.generate(request)

    assert first.usage_records == second.usage_records
    payload = first.usage_records[0].to_dict()
    assert payload["service_role"] == "simulator"
    assert payload["model"] == "openai:gpt-5.6-luna"
    assert payload["prompt_template"] == "plan"
    assert payload["prompt_version"] == "phase10_pilot_v1"
    assert len(payload["prompt_hash"]) == 64
