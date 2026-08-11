"""Provider-neutral normalized request/response contracts for Phase 10."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol, runtime_checkable

from src.llm.errors import ProviderRequestError, ProviderResponseError
from src.llm.usage import ProviderUsage


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderRequestError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class StructuredTextRequest:
    model: str
    instructions: str
    user_data: str
    schema_name: str
    output_schema_json: str
    max_output_tokens: int
    reasoning_effort: str
    timeout_seconds: int
    stream: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "model",
            "instructions",
            "user_data",
            "schema_name",
            "output_schema_json",
            "reasoning_effort",
        ):
            _text(getattr(self, field_name), field_name)
        for field_name in ("max_output_tokens", "timeout_seconds"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ProviderRequestError(f"{field_name} must be a positive integer")
        if self.stream is not False:
            raise ProviderRequestError("Phase 10 text generation must be non-streaming")
        try:
            schema = json.loads(self.output_schema_json)
            user_data = json.loads(self.user_data)
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("provider text request contains invalid JSON") from exc
        if not isinstance(schema, dict) or not isinstance(user_data, dict):
            raise ProviderRequestError("schema and canonical user data must be JSON objects")

    @property
    def output_schema(self) -> dict[str, Any]:
        value = json.loads(self.output_schema_json)
        if not isinstance(value, dict):
            raise ProviderRequestError("output schema must be a JSON object")
        return value


@dataclass(frozen=True)
class StructuredTextResponse:
    output_text: str
    usage: ProviderUsage

    def __post_init__(self) -> None:
        if not isinstance(self.output_text, str):
            raise ProviderResponseError("normalized output_text must be a string")
        if not isinstance(self.usage, ProviderUsage):
            raise ProviderResponseError("normalized text usage must be ProviderUsage")


@dataclass(frozen=True)
class EmbeddingRequest:
    model: str
    text: str
    timeout_seconds: int

    def __post_init__(self) -> None:
        _text(self.model, "model")
        _text(self.text, "text")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds < 1
        ):
            raise ProviderRequestError("timeout_seconds must be a positive integer")


@dataclass(frozen=True)
class EmbeddingResponse:
    vector: tuple[object, ...]
    usage: ProviderUsage

    def __post_init__(self) -> None:
        if not isinstance(self.vector, tuple):
            raise ProviderResponseError("normalized embedding vector must be a tuple")
        if not isinstance(self.usage, ProviderUsage):
            raise ProviderResponseError("normalized embedding usage must be ProviderUsage")


@runtime_checkable
class StructuredTextProvider(Protocol):
    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        """Return normalized structured text without provider SDK objects."""


@runtime_checkable
class EmbeddingAPIProvider(Protocol):
    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return one normalized embedding without provider SDK objects."""


__all__ = [
    "EmbeddingAPIProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "StructuredTextProvider",
    "StructuredTextRequest",
    "StructuredTextResponse",
]
