"""Phase 10 prompt, provider, and real-service error hierarchy."""

from __future__ import annotations

from src.llm.base import EmbeddingError, LLMClientError, LLMError


class Phase10LLMError(LLMError):
    """Base error for Phase 10 real-provider capabilities."""


class RuntimeSettingsError(Phase10LLMError, ValueError):
    """Raised when immutable real-provider settings are invalid."""


class CredentialError(Phase10LLMError):
    """Raised before client creation when the required credential is absent."""


class PromptError(Phase10LLMError, ValueError):
    """Base error for production prompt loading and rendering."""


class PromptLoadError(PromptError):
    """Raised when a required prompt file cannot be loaded safely."""


class PromptIntegrityError(PromptError):
    """Raised when prompt inventory or placeholders violate the contract."""


class PromptRenderError(PromptError):
    """Raised when trusted instructions and canonical data cannot be rendered."""


class ContextLimitError(PromptRenderError):
    """Raised before provider execution when the conservative context bound fails."""


class ProviderError(LLMClientError):
    """Sanitized base error for an OpenAI provider failure."""


class ProviderTransientError(ProviderError):
    """Base class for the only provider failures eligible for retry."""


class ProviderTimeoutError(ProviderTransientError):
    """Raised for a bounded request timeout."""


class ProviderRateLimitError(ProviderTransientError):
    """Raised when the provider reports rate limiting."""


class ProviderConnectionError(ProviderTransientError):
    """Raised for a connection or transient transport failure."""


class ProviderServerError(ProviderTransientError):
    """Raised for a transient provider-side server failure."""


class ProviderAuthenticationError(ProviderError):
    """Raised for invalid authentication without exposing provider details."""


class ProviderRequestError(ProviderError):
    """Raised for permanent request, model, or configuration failures."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response cannot be normalized safely."""


class UsageMetadataError(Phase10LLMError, ValueError):
    """Raised when provider usage/attempt metadata violates its schema."""


class RealEmbeddingError(EmbeddingError):
    """Base error for the production embedding capability."""


class EmbeddingValidationError(RealEmbeddingError, ValueError):
    """Raised when a returned embedding violates the frozen model contract."""


__all__ = [
    "ContextLimitError",
    "CredentialError",
    "EmbeddingValidationError",
    "Phase10LLMError",
    "PromptError",
    "PromptIntegrityError",
    "PromptLoadError",
    "PromptRenderError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "ProviderTransientError",
    "RealEmbeddingError",
    "RuntimeSettingsError",
    "UsageMetadataError",
]
