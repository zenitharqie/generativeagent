"""Narrow Anthropic adapter for the provider-neutral BCR judge service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

from src.llm.errors import (
    CredentialError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from src.llm.provider import StructuredTextRequest, StructuredTextResponse
from src.llm.settings import (
    ANTHROPIC_API_KEY_ENVIRONMENT_VARIABLE,
    ANTHROPIC_PROVIDER,
    EVALUATOR_MODEL,
)
from src.llm.usage import ProviderUsage


ClientFactory = Callable[[str, int], object]


def _load_api_key(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(ANTHROPIC_API_KEY_ENVIRONMENT_VARIABLE)
    if not isinstance(value, str) or not value.strip():
        raise CredentialError(
            f"missing required credential {ANTHROPIC_API_KEY_ENVIRONMENT_VARIABLE}"
        )
    return value.strip()


def _default_client_factory(api_key: str, timeout_seconds: int) -> object:
    from anthropic import Anthropic

    return Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)


def _translate_error(exc: BaseException) -> ProviderError:
    name = type(exc).__name__.casefold()
    status = getattr(exc, "status_code", None)
    if "timeout" in name:
        return ProviderTimeoutError("Anthropic request timed out")
    if "ratelimit" in name or "rate_limit" in name or status == 429:
        return ProviderRateLimitError("Anthropic rate limit reached")
    if "connection" in name or "network" in name:
        return ProviderConnectionError("Anthropic connection failed")
    if "authentication" in name or "permission" in name or status in {401, 403}:
        return ProviderAuthenticationError("Anthropic authentication failed")
    if isinstance(status, int) and status >= 500:
        return ProviderServerError("Anthropic transient server failure")
    if "response" in name or "json" in name or "parse" in name:
        return ProviderResponseError("Anthropic provider response was invalid")
    return ProviderRequestError("Anthropic provider call failed")


def _count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(f"Anthropic {field_name} is invalid")
    return value


def _usage(value: object) -> ProviderUsage:
    if value is None:
        return ProviderUsage()
    input_tokens = _count(getattr(value, "input_tokens", None), "input_tokens")
    output_tokens = _count(getattr(value, "output_tokens", None), "output_tokens")
    total = _count(getattr(value, "total_tokens", None), "total_tokens")
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _response_text(response: object) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, (list, tuple)):
        raise ProviderResponseError("Anthropic response did not contain content blocks")
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text")
        if isinstance(text, str) and text:
            return text
    raise ProviderResponseError("Anthropic response did not contain text output")


class AnthropicProviderAdapter:
    """Translate Anthropic Messages responses into normalized text contracts."""

    def __init__(self, client: object) -> None:
        if client is None:
            raise ProviderRequestError("Anthropic client must not be None")
        self._client = client

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provider={ANTHROPIC_PROVIDER!r}>"

    @classmethod
    def from_environment(
        cls,
        *,
        timeout_seconds: int,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
    ) -> "AnthropicProviderAdapter":
        key = _load_api_key(environ)
        factory = client_factory or _default_client_factory
        try:
            client = factory(key, timeout_seconds)
        except Exception as exc:
            translated = _translate_error(exc)
        else:
            translated = None
        if translated is not None:
            raise translated from None
        return cls(client)

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        if not isinstance(request, StructuredTextRequest):
            raise ProviderRequestError("request must be StructuredTextRequest")
        if request.model != EVALUATOR_MODEL:
            raise ProviderRequestError("Anthropic adapter received an unapproved model")
        try:
            response = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_output_tokens,
                system=request.instructions,
                messages=[{"role": "user", "content": request.user_data}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": request.output_schema,
                    }
                },
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            translated = _translate_error(exc)
        else:
            translated = None
        if translated is not None:
            raise translated from None
        try:
            text = _response_text(response)
            usage = _usage(getattr(response, "usage", None))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderResponseError("Anthropic provider response was invalid") from None
        return StructuredTextResponse(output_text=text, usage=usage)


__all__ = ["AnthropicProviderAdapter"]
