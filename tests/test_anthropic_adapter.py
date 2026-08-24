import json

import pytest

from src.llm.anthropic_client import AnthropicProviderAdapter
from src.llm.errors import ProviderAuthenticationError, ProviderRequestError
from src.llm.provider import StructuredTextRequest


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    input_tokens = 11
    output_tokens = 7


class _Response:
    content = [_Block('{"ok":true}')]
    usage = _Usage()


class _Messages:
    def __init__(self, response=None, error=None):
        self.response = response or _Response()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class _Client:
    def __init__(self, response=None, error=None):
        self.messages = _Messages(response=response, error=error)


def _request() -> StructuredTextRequest:
    return StructuredTextRequest(
        model="claude-sonnet-5",
        instructions="system",
        user_data='{"input":true}',
        schema_name="test_schema",
        output_schema_json='{"type":"object","additionalProperties":false}',
        max_output_tokens=700,
        reasoning_effort="medium",
        timeout_seconds=60,
    )


def test_anthropic_adapter_translates_native_structured_request_and_usage() -> None:
    client = _Client()
    result = AnthropicProviderAdapter(client).generate_structured(_request())
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["system"] == "system"
    assert call["messages"] == [{"role": "user", "content": '{"input":true}'}]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert json.loads(result.output_text) == {"ok": True}
    assert result.usage.to_dict() == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


def test_anthropic_adapter_rejects_wrong_model() -> None:
    request = _request()
    request = StructuredTextRequest(
        model="gpt-5-2025-08-07",
        instructions=request.instructions,
        user_data=request.user_data,
        schema_name=request.schema_name,
        output_schema_json=request.output_schema_json,
        max_output_tokens=request.max_output_tokens,
        reasoning_effort=request.reasoning_effort,
        timeout_seconds=request.timeout_seconds,
    )
    with pytest.raises(Exception, match="unapproved model"):
        AnthropicProviderAdapter(_Client()).generate_structured(request)


def test_anthropic_error_is_sanitized_without_cause_or_secret() -> None:
    error = RuntimeError("ANTHROPIC_SECRET_SENTINEL raw response")
    with pytest.raises(ProviderRequestError) as caught:
        AnthropicProviderAdapter(_Client(error=error)).generate_structured(_request())
    assert "SENTINEL" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_anthropic_authentication_error_is_sanitized() -> None:
    error = type("AuthenticationError", (Exception,), {})("secret")
    with pytest.raises(ProviderAuthenticationError):
        AnthropicProviderAdapter(_Client(error=error)).generate_structured(_request())


def test_from_environment_uses_dedicated_credential_boundary() -> None:
    seen = {}

    def factory(key, timeout):
        seen["key"] = key
        seen["timeout"] = timeout
        return _Client()

    adapter = AnthropicProviderAdapter.from_environment(
        timeout_seconds=60,
        environ={"ANTHROPIC_API_KEY": "synthetic-key"},
        client_factory=factory,
    )
    assert isinstance(adapter, AnthropicProviderAdapter)
    assert seen == {"key": "synthetic-key", "timeout": 60}
    assert "synthetic-key" not in repr(adapter)
