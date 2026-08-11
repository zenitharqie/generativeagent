"""Immutable OpenAI-only Phase 10 pilot runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from src.llm.errors import RuntimeSettingsError


OPENAI_PROVIDER = "openai"
OPENAI_API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"

SIMULATOR_MODEL = "gpt-5-nano-2025-08-07"
SIMULATOR_MODEL_IDENTITY = f"{OPENAI_PROVIDER}:{SIMULATOR_MODEL}"
EVALUATOR_MODEL = "gpt-5-2025-08-07"
EVALUATOR_MODEL_IDENTITY = f"{OPENAI_PROVIDER}:{EVALUATOR_MODEL}"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_IDENTITY = f"{OPENAI_PROVIDER}:{EMBEDDING_MODEL}"
EMBEDDING_DIMENSION = 1536

TEXT_CONTEXT_WINDOW_TOKENS = 400_000
CONTEXT_ACCOUNTING_RESERVE = 1_024
MAX_PROVIDER_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1
BCR_RUBRIC_VERSION = "phase10_bcr_rubric_v1"


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeSettingsError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True)
class GenerationProfile:
    """Research-auditable settings for one text-generation service role."""

    service_role: str
    provider: str
    model: str
    model_identity: str
    reasoning_effort: str
    request_timeout_seconds: int
    context_window_tokens: int
    output_token_limits: Mapping[str, int]
    strict_structured_output: bool = True
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.service_role not in {"simulator", "evaluator"}:
            raise RuntimeSettingsError("unsupported text-generation service_role")
        if self.provider != OPENAI_PROVIDER:
            raise RuntimeSettingsError("Phase 10 supports only the OpenAI provider")
        if self.model_identity != f"{self.provider}:{self.model}":
            raise RuntimeSettingsError("model_identity must be provider-qualified")
        if self.reasoning_effort not in {"minimal", "medium"}:
            raise RuntimeSettingsError("unsupported reasoning_effort")
        _positive_int(self.request_timeout_seconds, "request_timeout_seconds")
        _positive_int(self.context_window_tokens, "context_window_tokens")
        if self.strict_structured_output is not True or self.stream is not False:
            raise RuntimeSettingsError("text profiles require strict, non-streaming output")
        if any(value is not None for value in (self.temperature, self.top_p, self.seed)):
            raise RuntimeSettingsError(
                "temperature, top_p, and seed must remain unset for approved models"
            )
        if not isinstance(self.output_token_limits, Mapping) or not self.output_token_limits:
            raise RuntimeSettingsError("output_token_limits must be a non-empty mapping")
        limits = dict(self.output_token_limits)
        for call_type, limit in limits.items():
            if not isinstance(call_type, str) or not call_type:
                raise RuntimeSettingsError("output-token call types must be non-empty strings")
            _positive_int(limit, f"output_token_limits[{call_type!r}]")
        object.__setattr__(self, "output_token_limits", MappingProxyType(limits))

    def output_limit_for(self, call_type: str) -> int:
        try:
            return self.output_token_limits[call_type]
        except KeyError as exc:
            raise RuntimeSettingsError(
                f"call type {call_type!r} is not approved for {self.service_role}"
            ) from exc

    def reproducibility_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_identity": self.model_identity,
            "reasoning_effort": self.reasoning_effort,
            "request_timeout_seconds": self.request_timeout_seconds,
            "context_window_tokens": self.context_window_tokens,
            "strict_structured_output": self.strict_structured_output,
            "stream": self.stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "output_token_limits": dict(self.output_token_limits),
        }


@dataclass(frozen=True)
class EmbeddingProfile:
    """Frozen production embedding model and validation settings."""

    service_role: str
    provider: str
    model: str
    model_identity: str
    expected_dimension: int
    request_timeout_seconds: int

    def __post_init__(self) -> None:
        if self.service_role != "embedding" or self.provider != OPENAI_PROVIDER:
            raise RuntimeSettingsError("invalid embedding provider identity")
        if self.model_identity != f"{self.provider}:{self.model}":
            raise RuntimeSettingsError("embedding model_identity must be provider-qualified")
        _positive_int(self.expected_dimension, "expected_dimension")
        _positive_int(self.request_timeout_seconds, "request_timeout_seconds")

    def reproducibility_metadata(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_identity": self.model_identity,
            "expected_dimension": self.expected_dimension,
            "request_timeout_seconds": self.request_timeout_seconds,
            "dimensions_override": None,
        }


SIMULATOR_PROFILE = GenerationProfile(
    service_role="simulator",
    provider=OPENAI_PROVIDER,
    model=SIMULATOR_MODEL,
    model_identity=SIMULATOR_MODEL_IDENTITY,
    reasoning_effort="minimal",
    request_timeout_seconds=60,
    context_window_tokens=TEXT_CONTEXT_WINDOW_TOKENS,
    output_token_limits={
        "plan": 300,
        "natural_response": 500,
        "target_event_response": 500,
        "reflection": 400,
        "importance_score": 150,
    },
)

EVALUATOR_PROFILE = GenerationProfile(
    service_role="evaluator",
    provider=OPENAI_PROVIDER,
    model=EVALUATOR_MODEL,
    model_identity=EVALUATOR_MODEL_IDENTITY,
    reasoning_effort="medium",
    request_timeout_seconds=60,
    context_window_tokens=TEXT_CONTEXT_WINDOW_TOKENS,
    output_token_limits={"bcr_judge": 700},
)

EMBEDDING_PROFILE = EmbeddingProfile(
    service_role="embedding",
    provider=OPENAI_PROVIDER,
    model=EMBEDDING_MODEL,
    model_identity=EMBEDDING_MODEL_IDENTITY,
    expected_dimension=EMBEDDING_DIMENSION,
    request_timeout_seconds=30,
)


__all__ = [
    "BCR_RUBRIC_VERSION",
    "CONTEXT_ACCOUNTING_RESERVE",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_IDENTITY",
    "EMBEDDING_PROFILE",
    "EVALUATOR_MODEL",
    "EVALUATOR_MODEL_IDENTITY",
    "EVALUATOR_PROFILE",
    "EmbeddingProfile",
    "GenerationProfile",
    "MAX_PROVIDER_ATTEMPTS",
    "OPENAI_API_KEY_ENVIRONMENT_VARIABLE",
    "OPENAI_PROVIDER",
    "RETRY_DELAY_SECONDS",
    "SIMULATOR_MODEL",
    "SIMULATOR_MODEL_IDENTITY",
    "SIMULATOR_PROFILE",
    "TEXT_CONTEXT_WINDOW_TOKENS",
]
