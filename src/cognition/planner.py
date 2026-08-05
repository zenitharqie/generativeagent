"""Plan-stage adapter for the provider-neutral LLM boundary."""

from __future__ import annotations

from src.llm.base import LLMCallType, LLMClient, LLMContext, LLMRequest
from src.llm.output_parsers import PlanOutput, parse_plan_output


class Planner:
    """Generate and parse one concise plan without mutating state."""

    def __init__(self, llm_client: LLMClient) -> None:
        generate = getattr(llm_client, "generate", None)
        if not callable(generate):
            raise TypeError("llm_client must provide generate(request)")
        self._llm_client = llm_client

    def generate(self, context: LLMContext) -> PlanOutput:
        request = LLMRequest(call_type=LLMCallType.PLAN, context=context)
        return parse_plan_output(self._llm_client.generate(request))
