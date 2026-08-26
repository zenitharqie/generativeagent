"""Natural and target-event action-stage adapters."""

from __future__ import annotations

import re

from src.llm.base import LLMCallType, LLMClient, LLMContext, LLMRequest
from src.llm.output_parsers import (
    ActionOutput,
    LLMOutputValidationError,
    PlanOutput,
    parse_action_output,
)


def _sentence_count(text: str) -> int:
    chunks = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    return len(chunks)


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


class Actor:
    """Generate a parsed response while applying only technical guardrails."""

    def __init__(self, llm_client: LLMClient) -> None:
        generate = getattr(llm_client, "generate", None)
        if not callable(generate):
            raise TypeError("llm_client must provide generate(request)")
        self._llm_client = llm_client

    def generate(
        self,
        *,
        context: LLMContext,
        plan: PlanOutput,
        response_limit: int,
        previous_same_agent_response: str | None = None,
    ) -> ActionOutput:
        if not isinstance(plan, PlanOutput):
            raise TypeError("plan must be a PlanOutput")
        call_type = (
            LLMCallType.TARGET_EVENT_RESPONSE
            if context.source_step_type == "target_event"
            else LLMCallType.NATURAL_RESPONSE
        )
        request = LLMRequest(
            call_type=call_type,
            context=context,
            plan=plan.plan,
            response_limit=response_limit,
        )
        output = parse_action_output(self._llm_client.generate(request))

        if previous_same_agent_response is not None:
            if (
                not isinstance(previous_same_agent_response, str)
                or not previous_same_agent_response.strip()
            ):
                raise LLMOutputValidationError(
                    "previous_same_agent_response must be non-empty or None"
                )
            if output.response == previous_same_agent_response.strip():
                raise LLMOutputValidationError(
                    "action response exactly duplicates the supplied previous response"
                )

        actual_length = (
            _paragraph_count(output.response)
            if call_type is LLMCallType.TARGET_EVENT_RESPONSE
            else _sentence_count(output.response)
        )
        if actual_length > response_limit:
            if call_type is LLMCallType.TARGET_EVENT_RESPONSE:
                trimmed_response = output.response.split("\n\n", 1)[0].strip()
            else:
                parts = [
                    part
                    for part in re.split(r"(?<=[.!?])\s+", output.response.strip())
                    if part
                ]
                trimmed_response = " ".join(parts[:response_limit])
            output = ActionOutput(
                response=trimmed_response,
                memory_to_store=output.memory_to_store,
                category=output.category,
            )
        return output
