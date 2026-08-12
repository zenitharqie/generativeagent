"""Validated, retried OpenAI production embedding provider."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from math import isfinite
import time

from src.llm.errors import (
    EmbeddingValidationError,
    ProviderError,
    ProviderTransientError,
    RealEmbeddingError,
)
from src.llm.provider import EmbeddingAPIProvider, EmbeddingRequest
from src.llm.settings import (
    EMBEDDING_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    EmbeddingProfile,
)
from src.llm.usage import (
    InMemoryUsageRecorder,
    ProviderAttemptRecord,
    ProviderUsage,
    UsageRecorder,
)


class OpenAIEmbeddingProvider:
    """Implement TextEmbeddingProvider without changing retrieval semantics."""

    def __init__(
        self,
        provider: EmbeddingAPIProvider,
        *,
        profile: EmbeddingProfile = EMBEDDING_PROFILE,
        usage_recorder: UsageRecorder | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if profile != EMBEDDING_PROFILE:
            raise RealEmbeddingError("embedding provider requires the approved profile")
        if not isinstance(provider, EmbeddingAPIProvider):
            raise RealEmbeddingError("provider must implement EmbeddingAPIProvider")
        self._provider = provider
        self._profile = profile
        self._usage_recorder = usage_recorder or InMemoryUsageRecorder()
        self._sleeper = sleeper
        self._logical_sequence = 0

    @property
    def usage_records(self) -> tuple[ProviderAttemptRecord, ...]:
        records = getattr(self._usage_recorder, "records", None)
        return tuple(records) if records is not None else ()

    def _logical_call_id(self, text: str) -> str:
        self._logical_sequence += 1
        digest = sha256(text.encode("utf-8")).hexdigest()
        return f"embedding_{self._logical_sequence:06d}_{digest}"

    def _record(
        self,
        logical_call_id: str,
        attempt_index: int,
        status: str,
        *,
        usage: ProviderUsage = ProviderUsage(),
        error_type: str | None = None,
    ) -> None:
        self._usage_recorder.record(
            ProviderAttemptRecord(
                service_role="embedding",
                logical_call_id=logical_call_id,
                attempt_index=attempt_index,
                call_type="embedding",
                model=self._profile.model_identity,
                status=status,
                usage=usage,
                error_type=error_type,
            )
        )

    def _validate_vector(self, value: tuple[object, ...]) -> tuple[float, ...]:
        if len(value) != self._profile.expected_dimension:
            raise EmbeddingValidationError(
                f"embedding dimension must equal {self._profile.expected_dimension}"
            )
        vector: list[float] = []
        for component in value:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise EmbeddingValidationError("embedding components must be numeric")
            normalized = float(component)
            if not isfinite(normalized):
                raise EmbeddingValidationError("embedding components must be finite")
            vector.append(normalized)
        if not any(component != 0.0 for component in vector):
            raise EmbeddingValidationError("embedding vector must be non-zero")
        return tuple(vector)

    def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise RealEmbeddingError("embedding text must be a non-empty string")
        logical_call_id = self._logical_call_id(text)
        request = EmbeddingRequest(
            model=self._profile.model,
            text=text,
            timeout_seconds=self._profile.request_timeout_seconds,
        )
        for attempt_index in range(1, MAX_PROVIDER_ATTEMPTS + 1):
            try:
                response = self._provider.create_embedding(request)
            except KeyboardInterrupt:
                self._record(
                    logical_call_id,
                    attempt_index,
                    "failed",
                    error_type="KeyboardInterrupt",
                )
                raise
            except ProviderError as exc:
                self._record(
                    logical_call_id,
                    attempt_index,
                    "failed",
                    error_type=type(exc).__name__,
                )
                if isinstance(exc, ProviderTransientError) and attempt_index < MAX_PROVIDER_ATTEMPTS:
                    self._sleeper(RETRY_DELAY_SECONDS)
                    continue
                raise RealEmbeddingError("OpenAI embedding call failed") from exc
            try:
                vector = self._validate_vector(response.vector)
            except EmbeddingValidationError as exc:
                self._record(
                    logical_call_id,
                    attempt_index,
                    "failed",
                    usage=response.usage,
                    error_type=type(exc).__name__,
                )
                raise
            self._record(
                logical_call_id,
                attempt_index,
                "success",
                usage=response.usage,
            )
            return vector
        raise RealEmbeddingError("approved embedding attempt limit was exhausted")


__all__ = ["OpenAIEmbeddingProvider"]
