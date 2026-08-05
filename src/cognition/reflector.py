"""Explicitly requested reflection-stage adapter."""

from __future__ import annotations

import re

from src.llm.base import LLMCallType, LLMClient, LLMContext, LLMRequest
from src.llm.output_parsers import (
    ActionOutput,
    LLMOutputValidationError,
    PlanOutput,
    ReflectionOutput,
    parse_reflection_output,
)


class Reflector:
    """Generate at most one parsed reflection when called by the pipeline."""

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
        action: ActionOutput,
        sentence_limit: int,
    ) -> ReflectionOutput:
        if not isinstance(plan, PlanOutput):
            raise TypeError("plan must be a PlanOutput")
        if not isinstance(action, ActionOutput):
            raise TypeError("action must be an ActionOutput")
        request = LLMRequest(
            call_type=LLMCallType.REFLECTION,
            context=context,
            plan=plan.plan,
            response=action.response,
            response_limit=sentence_limit,
        )
        output = parse_reflection_output(self._llm_client.generate(request))
        sentence_count = len(
            [
                part
                for part in re.split(r"(?<=[.!?])\s+", output.reflection.strip())
                if part
            ]
        )
        if sentence_count > sentence_limit:
            raise LLMOutputValidationError(
                "reflection exceeds configured limit of "
                f"{sentence_limit} sentences"
            )
        return output
