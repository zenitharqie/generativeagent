"""Offline tests for the sole OpenAI SDK adapter."""

from __future__ import annotations

import traceback
from types import SimpleNamespace

import pytest

from src.llm.errors import (
    CredentialError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from src.llm.openai_client import OpenAIProviderAdapter
from src.llm import openai_client
from src.llm.provider import EmbeddingRequest, StructuredTextRequest


class CapturingEndpoint:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def fake_client(*, text_outcome: object, embedding_outcome: object) -> SimpleNamespace:
    return SimpleNamespace(
        responses=CapturingEndpoint(text_outcome),
        embeddings=CapturingEndpoint(embedding_outcome),
    )


def text_request() -> StructuredTextRequest:
    return StructuredTextRequest(
        model="gpt-5.6-luna",
        instructions="Trusted instructions.",
        user_data='{"value":"untrusted data"}',
        schema_name="plan_output",
        output_schema_json=(
            '{"additionalProperties":false,"properties":{"plan":{"type":"string"}},'
            '"required":["plan"],"type":"object"}'
        ),
        max_output_tokens=300,
        reasoning_effort="low",
        timeout_seconds=60,
    )


def assert_sentinel_is_unreachable(exc: BaseException, sentinel: str) -> None:
    assert sentinel not in str(exc)
    assert sentinel not in repr(exc)
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(exc))


def test_responses_api_request_is_strict_non_streaming_and_complete() -> None:
    response = SimpleNamespace(
        output_text='{"plan":"Respond calmly."}',
        usage=SimpleNamespace(input_tokens=12, output_tokens=5, total_tokens=17),
    )
    client = fake_client(text_outcome=response, embedding_outcome=None)
    adapter = OpenAIProviderAdapter(client)

    normalized = adapter.generate_structured(text_request())

    assert normalized.output_text == response.output_text
    assert normalized.usage.to_dict() == {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
    }
    call = client.responses.calls[0]
    assert call == {
        "model": "gpt-5.6-luna",
        "instructions": "Trusted instructions.",
        "input": '{"value":"untrusted data"}',
        "max_output_tokens": 300,
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "plan_output",
                "schema": text_request().output_schema,
                "strict": True,
            }
        },
        "stream": False,
        "store": False,
        "truncation": "disabled",
        "timeout": 60,
    }
    for unsupported in ("temperature", "top_p", "seed"):
        assert unsupported not in call


def test_embedding_request_uses_default_dimension_without_override() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0])],
        usage=SimpleNamespace(prompt_tokens=3, total_tokens=3),
    )
    client = fake_client(text_outcome=None, embedding_outcome=response)
    adapter = OpenAIProviderAdapter(client)

    normalized = adapter.create_embedding(
        EmbeddingRequest(
            model="text-embedding-3-small",
            text="Memory text.",
            timeout_seconds=30,
        )
    )

    assert normalized.vector == (1.0, 0.0)
    assert normalized.usage.to_dict() == {
        "input_tokens": 3,
        "output_tokens": None,
        "total_tokens": 3,
    }
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "Memory text.",
            "encoding_format": "float",
            "timeout": 30,
        }
    ]
    assert "dimensions" not in client.embeddings.calls[0]


def test_missing_credential_fails_before_client_factory() -> None:
    called = False

    def factory(key: str, timeout: int) -> object:
        nonlocal called
        called = True
        raise AssertionError("factory must not run")

    with pytest.raises(CredentialError, match="OPENAI_API_KEY"):
        OpenAIProviderAdapter.from_environment(
            timeout_seconds=60,
            environ={},
            client_factory=factory,
        )
    assert called is False


def test_client_factory_receives_key_but_adapter_repr_does_not_expose_it() -> None:
    secret = "sk-test-highly-sensitive"
    received: list[tuple[str, int]] = []

    def factory(key: str, timeout: int) -> object:
        received.append((key, timeout))
        return fake_client(text_outcome=None, embedding_outcome=None)

    adapter = OpenAIProviderAdapter.from_environment(
        timeout_seconds=60,
        environ={"OPENAI_API_KEY": secret},
        client_factory=factory,
    )

    assert received == [(secret, 60)]
    assert secret not in repr(adapter)


def test_client_creation_sanitizes_exception_without_raw_chaining() -> None:
    sentinel = "phase10-client-creation-secret"

    def factory(key: str, timeout: int) -> object:
        raise RuntimeError(f"raw client body {sentinel}")

    with pytest.raises(ProviderError) as caught:
        OpenAIProviderAdapter.from_environment(
            timeout_seconds=60,
            environ={"OPENAI_API_KEY": "sk-test"},
            client_factory=factory,
        )

    assert_sentinel_is_unreachable(caught.value, sentinel)


def test_text_call_preserves_error_classification_without_raw_chaining() -> None:
    sentinel = "phase10-text-provider-secret"
    client = fake_client(
        text_outcome=ProviderTimeoutError(sentinel),
        embedding_outcome=None,
    )

    with pytest.raises(ProviderTimeoutError) as caught:
        OpenAIProviderAdapter(client).generate_structured(text_request())

    assert_sentinel_is_unreachable(caught.value, sentinel)


def test_embedding_call_preserves_error_classification_without_raw_chaining() -> None:
    sentinel = "phase10-embedding-provider-secret"
    client = fake_client(
        text_outcome=None,
        embedding_outcome=ProviderRateLimitError(sentinel),
    )

    with pytest.raises(ProviderRateLimitError) as caught:
        OpenAIProviderAdapter(client).create_embedding(
            EmbeddingRequest(
                model="text-embedding-3-small",
                text="Memory text.",
                timeout_seconds=30,
            )
        )

    assert_sentinel_is_unreachable(caught.value, sentinel)


def test_official_sdk_client_disables_automatic_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    assert openai_client._default_client_factory("sk-test", 60) is not None
    assert captured == {
        "api_key": "sk-test",
        "timeout": 60,
        "max_retries": 0,
    }
