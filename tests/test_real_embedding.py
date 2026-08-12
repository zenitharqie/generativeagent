"""Offline production embedding validation, retries, and usage tests."""

from __future__ import annotations

from collections import deque
import json

import pytest

from src.llm import TextEmbeddingProvider
from src.llm.embedding import OpenAIEmbeddingProvider
from src.llm.errors import EmbeddingValidationError, ProviderRateLimitError, RealEmbeddingError
from src.llm.provider import EmbeddingRequest, EmbeddingResponse
from src.llm.usage import ProviderUsage


VALID_VECTOR = (1.0,) + (0.0,) * 1535


class ScriptedEmbeddingAPI:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[EmbeddingRequest] = []

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return EmbeddingResponse(
            vector=tuple(outcome),
            usage=ProviderUsage(4, None, 4),
        )


def test_embedding_provider_uses_exact_model_dimension_and_timeout() -> None:
    api = ScriptedEmbeddingAPI(VALID_VECTOR)
    provider = OpenAIEmbeddingProvider(api, sleeper=lambda _: None)

    assert isinstance(provider, TextEmbeddingProvider)
    assert provider.embed("One memory.") == VALID_VECTOR
    assert api.requests == [
        EmbeddingRequest(
            model="text-embedding-3-small",
            text="One memory.",
            timeout_seconds=30,
        )
    ]
    assert provider.usage_records[0].model == "openai:text-embedding-3-small"
    assert provider.usage_records[0].service_role == "embedding"


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (1.0, 0.0),
        (0.0,) * 1536,
        (float("nan"),) + (0.0,) * 1535,
        (float("inf"),) + (0.0,) * 1535,
        (True,) + (0.0,) * 1535,
    ],
)
def test_embedding_provider_rejects_invalid_vector_without_retry(vector: tuple[object, ...]) -> None:
    api = ScriptedEmbeddingAPI(vector)
    provider = OpenAIEmbeddingProvider(api, sleeper=lambda _: pytest.fail("must not retry"))

    with pytest.raises(EmbeddingValidationError):
        provider.embed("One memory.")
    assert len(api.requests) == 1
    assert provider.usage_records[0].status == "failed"


def test_embedding_transient_failure_retries_once() -> None:
    delays: list[float] = []
    api = ScriptedEmbeddingAPI(ProviderRateLimitError("sanitized"), VALID_VECTOR)
    provider = OpenAIEmbeddingProvider(api, sleeper=delays.append)

    assert provider.embed("One memory.") == VALID_VECTOR
    assert delays == [1]
    assert tuple(item.attempt_index for item in provider.usage_records) == (1, 2)
    assert len({item.logical_call_id for item in provider.usage_records}) == 1


def test_embedding_two_transient_failures_are_terminal() -> None:
    api = ScriptedEmbeddingAPI(
        ProviderRateLimitError("first"),
        ProviderRateLimitError("second"),
    )
    provider = OpenAIEmbeddingProvider(api, sleeper=lambda _: None)

    with pytest.raises(RealEmbeddingError, match="call failed"):
        provider.embed("One memory.")
    assert len(api.requests) == 2


def test_embedding_keyboard_interrupt_is_recorded_once_then_reraised() -> None:
    sentinel = "raw-embedding-interrupt-detail"
    api = ScriptedEmbeddingAPI(KeyboardInterrupt(sentinel))
    provider = OpenAIEmbeddingProvider(
        api,
        sleeper=lambda _: pytest.fail("interruption must not retry"),
    )

    with pytest.raises(KeyboardInterrupt, match=sentinel):
        provider.embed("One memory.")

    assert len(api.requests) == 1
    assert len(provider.usage_records) == 1
    attempt = provider.usage_records[0]
    assert attempt.attempt_index == 1
    assert attempt.status == "failed"
    assert attempt.error_type == "KeyboardInterrupt"
    assert attempt.usage.to_dict() == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    assert sentinel not in json.dumps(attempt.to_dict(), sort_keys=True)


def test_embedding_logical_identity_is_deterministic_for_fresh_services() -> None:
    first = OpenAIEmbeddingProvider(ScriptedEmbeddingAPI(VALID_VECTOR), sleeper=lambda _: None)
    second = OpenAIEmbeddingProvider(ScriptedEmbeddingAPI(VALID_VECTOR), sleeper=lambda _: None)

    first.embed("One memory.")
    second.embed("One memory.")

    assert first.usage_records == second.usage_records


def test_embedding_rejects_empty_text_before_provider_call() -> None:
    api = ScriptedEmbeddingAPI(VALID_VECTOR)
    provider = OpenAIEmbeddingProvider(api, sleeper=lambda _: None)

    with pytest.raises(RealEmbeddingError, match="non-empty"):
        provider.embed("   ")
    assert api.requests == []
