"""OpenAI SDK adapter isolated behind provider-neutral normalized contracts."""

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
from src.llm.provider import (
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredTextRequest,
    StructuredTextResponse,
)
from src.llm.settings import OPENAI_API_KEY_ENVIRONMENT_VARIABLE, OPENAI_PROVIDER
from src.llm.usage import ProviderUsage


ClientFactory = Callable[[str, int], object]


_SANITIZED_INTERNAL_ERRORS: tuple[tuple[type[ProviderError], str], ...] = (
    (ProviderTimeoutError, "OpenAI request timed out"),
    (ProviderRateLimitError, "OpenAI rate limit reached"),
    (ProviderConnectionError, "OpenAI connection failed"),
    (ProviderServerError, "OpenAI transient server failure"),
    (ProviderAuthenticationError, "OpenAI authentication failed"),
    (ProviderResponseError, "OpenAI provider response was invalid"),
    (ProviderRequestError, "OpenAI request was rejected"),
    (ProviderError, "OpenAI provider call failed"),
)


def _load_api_key(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = source.get(OPENAI_API_KEY_ENVIRONMENT_VARIABLE)
    if not isinstance(value, str) or not value.strip():
        raise CredentialError(
            f"missing required credential {OPENAI_API_KEY_ENVIRONMENT_VARIABLE}"
        )
    return value.strip()


def _default_client_factory(api_key: str, timeout_seconds: int) -> object:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _translate_openai_error(exc: BaseException) -> ProviderError:
    for error_type, message in _SANITIZED_INTERNAL_ERRORS:
        if isinstance(exc, error_type):
            return error_type(message)
    try:
        import openai
    except ImportError:
        return ProviderRequestError("OpenAI SDK is unavailable")
    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError("OpenAI request timed out")
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError("OpenAI rate limit reached")
    if isinstance(exc, openai.APIConnectionError):
        return ProviderConnectionError("OpenAI connection failed")
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return ProviderAuthenticationError("OpenAI authentication failed")
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return ProviderServerError("OpenAI transient server failure")
    if isinstance(exc, openai.APIError):
        return ProviderRequestError("OpenAI request was rejected")
    return ProviderError("OpenAI provider call failed")


def _optional_count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(f"OpenAI {field_name} is invalid")
    return value


def _text_usage(value: object) -> ProviderUsage:
    if value is None:
        return ProviderUsage()
    return ProviderUsage(
        input_tokens=_optional_count(getattr(value, "input_tokens", None), "input_tokens"),
        output_tokens=_optional_count(getattr(value, "output_tokens", None), "output_tokens"),
        total_tokens=_optional_count(getattr(value, "total_tokens", None), "total_tokens"),
    )


def _embedding_usage(value: object) -> ProviderUsage:
    if value is None:
        return ProviderUsage()
    return ProviderUsage(
        input_tokens=_optional_count(getattr(value, "prompt_tokens", None), "prompt_tokens"),
        output_tokens=None,
        total_tokens=_optional_count(getattr(value, "total_tokens", None), "total_tokens"),
    )


class OpenAIProviderAdapter:
    """Translate Responses/Embeddings SDK calls into internal immutable values."""

    def __init__(self, client: object) -> None:
        if client is None:
            raise ProviderRequestError("OpenAI client must not be None")
        self._client = client

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provider={OPENAI_PROVIDER!r}>"

    @classmethod
    def from_environment(
        cls,
        *,
        timeout_seconds: int,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory | None = None,
    ) -> "OpenAIProviderAdapter":
        key = _load_api_key(environ)
        factory = client_factory or _default_client_factory
        client: object | None = None
        translated_error: ProviderError | None = None
        try:
            client = factory(key, timeout_seconds)
        except Exception as exc:
            translated_error = _translate_openai_error(exc)
        if translated_error is not None:
            raise translated_error from None
        return cls(client)

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        if not isinstance(request, StructuredTextRequest):
            raise ProviderRequestError("request must be StructuredTextRequest")
        response: object | None = None
        translated_error: ProviderError | None = None
        try:
            response = self._client.responses.create(
                model=request.model,
                instructions=request.instructions,
                input=request.user_data,
                max_output_tokens=request.max_output_tokens,
                reasoning={"effort": request.reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.schema_name,
                        "schema": request.output_schema,
                        "strict": True,
                    }
                },
                stream=False,
                store=False,
                truncation="disabled",
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            translated_error = _translate_openai_error(exc)
        if translated_error is not None:
            raise translated_error from None
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ProviderResponseError("OpenAI response did not contain normalized output text")
        return StructuredTextResponse(
            output_text=output_text,
            usage=_text_usage(getattr(response, "usage", None)),
        )

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not isinstance(request, EmbeddingRequest):
            raise ProviderRequestError("request must be EmbeddingRequest")
        response: object | None = None
        translated_error: ProviderError | None = None
        try:
            response = self._client.embeddings.create(
                model=request.model,
                input=request.text,
                encoding_format="float",
                timeout=request.timeout_seconds,
            )
        except Exception as exc:
            translated_error = _translate_openai_error(exc)
        if translated_error is not None:
            raise translated_error from None
        data = getattr(response, "data", None)
        if not isinstance(data, (list, tuple)) or len(data) != 1:
            raise ProviderResponseError("OpenAI embedding response must contain one vector")
        vector = getattr(data[0], "embedding", None)
        if not isinstance(vector, (list, tuple)):
            raise ProviderResponseError("OpenAI embedding response has no numeric vector")
        return EmbeddingResponse(
            vector=tuple(vector),
            usage=_embedding_usage(getattr(response, "usage", None)),
        )


__all__ = ["OpenAIProviderAdapter"]
