"""Offline provider doubles and helpers shared by Phase 11 tests."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.ground_truth.schemas import GroundTruthFinalizationIdentity

from src.llm.provider import (
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredTextRequest,
    StructuredTextResponse,
)
from src.llm.errors import ProviderTimeoutError
from src.llm.usage import ProviderUsage


VECTOR = (1.0,) + (0.0,) * 1535


def install_phase11_v2_test_contract(monkeypatch, tmp_path: Path) -> Path:
    """Supply deterministic authenticated identities for offline freeze tests."""
    import src.simulation.pilot as pilot_module

    evidence = tmp_path / "ground_truth_finalization"
    evidence.mkdir(exist_ok=True)
    identity = GroundTruthFinalizationIdentity(
        ground_truth_sha256="0" * 64,
        finalization_fingerprint="1" * 64,
        evidence_hashes={"reliability.json": "2" * 64},
        annotation_package_sha256="3" * 64,
        annotators=("gpt5", "claude_sonnet_5", "gemini_1_5_pro"),
        embedding_model="openai:text-embedding-3-small",
        embedding_dimension=1536,
        cosine_threshold=0.85,
        representative_policy=(
            "greatest_distinct_annotator_support_then_casefolded_lexical_order"
        ),
        majority_rule="positive_annotator_count_greater_than_or_equal_to_2_of_3",
        reliability_evidence_sha256="2" * 64,
        workflow_schema_version="1.0",
        finalization_version="manual_three_annotator_v1",
    )
    source_identity = {
        "commit": "a" * 40,
        "branch": "main",
        "tags": ["v1.4-claude-bcr-evaluator"],
        "clean": True,
    }
    monkeypatch.setattr(
        pilot_module,
        "authenticated_ground_truth_identity",
        lambda value, evidence_directory: identity,
    )
    monkeypatch.setattr(
        pilot_module,
        "_capture_pilot_source_identity",
        lambda require_clean: source_identity,
    )
    return evidence


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
