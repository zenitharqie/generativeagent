"""One-call, non-research real-provider smoke verification."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.base import LLMCallType, LLMContext, LLMRequest
from src.llm.openai_client import OpenAIProviderAdapter
from src.llm.provider import StructuredTextProvider
from src.llm.service import LLMService
from src.llm.settings import OPENAI_PROVIDER, SIMULATOR_MODEL_IDENTITY, SIMULATOR_PROFILE


@dataclass(frozen=True)
class LLMSmokeTestResult:
    provider: str
    model_identity: str
    attempt_count: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    def sanitized_lines(self) -> tuple[str, ...]:
        def display(value: int | None) -> str:
            return str(value) if value is not None else "unavailable"

        return (
            "Status: success",
            f"Provider: {self.provider}",
            f"Model: {self.model_identity}",
            f"Attempts: {self.attempt_count}",
            f"Input tokens: {display(self.input_tokens)}",
            f"Output tokens: {display(self.output_tokens)}",
            f"Total tokens: {display(self.total_tokens)}",
        )


def _smoke_request() -> LLMRequest:
    """Return fixed minimal data; this is not a simulation step or research run."""
    context = LLMContext(
        agent_id="teacher",
        display_name="Provider Check",
        role="teacher",
        archetype="calm_classroom_guide",
        persona_summary="A concise classroom teacher used for a non-research provider check.",
        traits=("calm",),
        goals=("return one minimal structured plan",),
        communication_style="Concise.",
        motivations=("verify structured generation",),
        source_step_id="phase10_smoke_check",
        source_step_type="natural_interaction_block",
        step_title="Non-Research Provider Check",
        step_description="Verify one structured response without running simulation.",
        day=1,
        period="smoke",
        logical_timestamp=0,
        observation_text="Return one minimal plan for a connectivity check.",
        conversation_context=(),
        selected_memory_texts=(),
    )
    return LLMRequest(LLMCallType.PLAN, context)


def run_real_llm_smoke_test(
    provider: StructuredTextProvider | None = None,
) -> LLMSmokeTestResult:
    """Execute one logical Responses API call and discard generated content."""
    active_provider = provider or OpenAIProviderAdapter.from_environment(
        timeout_seconds=SIMULATOR_PROFILE.request_timeout_seconds
    )
    service = LLMService(active_provider)
    service.generate(_smoke_request())
    attempts = service.last_attempts
    usage = attempts[-1].usage
    return LLMSmokeTestResult(
        provider=OPENAI_PROVIDER,
        model_identity=SIMULATOR_MODEL_IDENTITY,
        attempt_count=len(attempts),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


__all__ = ["LLMSmokeTestResult", "run_real_llm_smoke_test"]
