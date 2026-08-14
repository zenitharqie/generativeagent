"""Offline provider doubles and helpers shared by Phase 11 tests."""

from __future__ import annotations

from pathlib import Path

from src.llm.provider import (
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredTextRequest,
    StructuredTextResponse,
)
from src.llm.errors import ProviderTimeoutError
from src.llm.usage import ProviderUsage


VECTOR = (1.0,) + (0.0,) * 1535


class Phase11MockProvider:
    """Return schema-valid deterministic values without SDK or network access."""

    def __init__(self) -> None:
        self.text_requests: list[StructuredTextRequest] = []
        self.embedding_requests: list[EmbeddingRequest] = []

    def generate_structured(
        self,
        request: StructuredTextRequest,
    ) -> StructuredTextResponse:
        self.text_requests.append(request)
        index = len(self.text_requests)
        if request.schema_name == "plan_output":
            output = f'{{"plan":"Pilot plan {index}."}}'
        elif request.schema_name == "action_output":
            output = (
                f'{{"response":"Pilot response {index}.",'
                f'"memory_to_store":"Pilot memory {index}.",'
                '"category":"classroom_norms"}'
            )
        elif request.schema_name == "reflection_output":
            output = (
                f'{{"reflection":"Pilot reflection {index}.",'
                '"category":"classroom_norms"}'
            )
        elif request.schema_name == "importance_output":
            output = '{"importance":6,"reason":"Useful classroom evidence."}'
        else:  # pragma: no cover - a provider-boundary tripwire
            raise AssertionError(f"unexpected schema: {request.schema_name}")
        return StructuredTextResponse(output, ProviderUsage(10, 5, 15))

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_requests.append(request)
        return EmbeddingResponse(VECTOR, ProviderUsage(3, None, 3))


class RetryFirstEmbeddingProvider(Phase11MockProvider):
    """Fail one physical attempt, then recover deterministically."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_attempt_count = 0

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_attempt_count += 1
        if self.embedding_attempt_count == 1:
            self.embedding_requests.append(request)
            raise ProviderTimeoutError("sanitized mocked timeout")
        return super().create_embedding(request)


class UsagePresentRetryProvider(Phase11MockProvider):
    """Return invalid simulator output with usage, then recover on retry."""

    def __init__(self) -> None:
        super().__init__()
        self.invalid_output_returned = False

    def generate_structured(
        self,
        request: StructuredTextRequest,
    ) -> StructuredTextResponse:
        if not self.invalid_output_returned:
            self.invalid_output_returned = True
            self.text_requests.append(request)
            return StructuredTextResponse("not-json", ProviderUsage(7, 2, 9))
        return super().generate_structured(request)


class SentinelTextFailureProvider(Phase11MockProvider):
    """Expose conspicuous provider text only through a mocked exception."""

    def __init__(self, sentinel: str) -> None:
        super().__init__()
        self.sentinel = sentinel

    def generate_structured(
        self,
        request: StructuredTextRequest,
    ) -> StructuredTextResponse:
        self.text_requests.append(request)
        raise ProviderTimeoutError(self.sentinel)


class TerminalUsagePresentInvalidOutputProvider(Phase11MockProvider):
    """Exhaust simulator retries with auditable usage on both attempts."""

    def generate_structured(
        self,
        request: StructuredTextRequest,
    ) -> StructuredTextResponse:
        self.text_requests.append(request)
        return StructuredTextResponse("not-json", ProviderUsage(7, 2, 9))


class TerminalEmbeddingFailureProvider(Phase11MockProvider):
    """Exhaust both approved attempts before any simulator text call."""

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_requests.append(request)
        raise ProviderTimeoutError("sanitized mocked timeout")


class OvershootingEmbeddingProvider(Phase11MockProvider):
    """Return valid output whose reported usage crosses a small pilot ceiling."""

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_requests.append(request)
        return EmbeddingResponse(VECTOR, ProviderUsage(6, None, 6))


def directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
