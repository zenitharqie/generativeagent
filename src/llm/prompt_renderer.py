"""Canonical prompt rendering with trusted instructions and untrusted data split."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from src.llm.base import LLMCallType, LLMRequest
from src.llm.errors import PromptRenderError
from src.llm.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from src.evaluation.schemas import BCRJudgeInput


_NON_EMPTY_STRING = {"type": "string", "minLength": 1}
_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "plan": {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan"],
        "properties": {"plan": _NON_EMPTY_STRING},
    },
    "action": {
        "type": "object",
        "additionalProperties": False,
        "required": ["response", "memory_to_store", "category"],
        "properties": {
            "response": _NON_EMPTY_STRING,
            "memory_to_store": _NON_EMPTY_STRING,
            "category": _NON_EMPTY_STRING,
        },
    },
    "reflection": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reflection", "category"],
        "properties": {
            "reflection": _NON_EMPTY_STRING,
            "category": _NON_EMPTY_STRING,
        },
    },
    "importance": {
        "type": "object",
        "additionalProperties": False,
        "required": ["importance", "reason"],
        "properties": {
            "importance": {"type": "number", "minimum": 1, "maximum": 10},
            "reason": _NON_EMPTY_STRING,
        },
    },
    "bcr_judgment": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "communication_style_score",
            "communication_style_reason",
            "focus_consistency_score",
            "focus_consistency_reason",
            "temporal_consistency_score",
            "temporal_consistency_reason",
        ],
        "properties": {
            "communication_style_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "communication_style_reason": _NON_EMPTY_STRING,
            "focus_consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "focus_consistency_reason": _NON_EMPTY_STRING,
            "temporal_consistency_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "temporal_consistency_reason": _NON_EMPTY_STRING,
        },
    },
}

_CALL_TEMPLATE_IDS = {
    LLMCallType.PLAN: "plan",
    LLMCallType.NATURAL_RESPONSE: "natural_interaction",
    LLMCallType.TARGET_EVENT_RESPONSE: "target_event_response",
    LLMCallType.REFLECTION: "reflect",
    LLMCallType.IMPORTANCE_SCORE: "importance_scoring",
}


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PromptRenderError("prompt data is not strict JSON") from exc


@dataclass(frozen=True)
class RenderedPrompt:
    template_id: str
    template_version: str
    template_hash: str
    system_instructions: str
    user_data: str
    schema_name: str
    output_schema_json: str

    @property
    def output_schema(self) -> dict[str, Any]:
        value = json.loads(self.output_schema_json)
        if not isinstance(value, dict):  # Defensive invariant for static schemas.
            raise PromptRenderError("output schema must decode to an object")
        return value


def _common_context(request: LLMRequest) -> dict[str, Any]:
    context = request.context
    situation: dict[str, Any] = {
        "source_step_id": context.source_step_id,
        "source_step_type": context.source_step_type,
        "title": context.step_title,
        "description": context.step_description,
        "day": context.day,
        "period": context.period,
        "logical_timestamp": context.logical_timestamp,
    }
    if context.event_id is not None:
        situation["target_event"] = {
            "event_id": context.event_id,
            "title": context.event_title,
            "description": context.event_description,
        }
    return {
        "persona": {
            "agent_id": context.agent_id,
            "display_name": context.display_name,
            "role": context.role,
            "archetype": context.archetype,
            "persona_summary": context.persona_summary,
            "traits": list(context.traits),
            "goals": list(context.goals),
            "communication_style": context.communication_style,
            "motivations": list(context.motivations),
        },
        "situation": situation,
        "observation": context.observation_text,
        "conversation_context": list(context.conversation_context),
        "selected_memory_texts": list(context.selected_memory_texts),
    }


def _render(template_id: str, data: object, loader: PromptLoader) -> RenderedPrompt:
    template = loader.load(template_id)
    schema = _OUTPUT_SCHEMAS[template.spec.output_contract]
    schema_json = canonical_json(schema)
    instructions = template.render_trusted({"OUTPUT_SCHEMA": schema_json})
    return RenderedPrompt(
        template_id=template.spec.template_id,
        template_version=template.spec.version,
        template_hash=template.content_hash,
        system_instructions=instructions,
        user_data=canonical_json(data),
        schema_name=f"{template.spec.output_contract}_output",
        output_schema_json=schema_json,
    )


def render_simulator_prompt(
    request: LLMRequest,
    loader: PromptLoader | None = None,
) -> RenderedPrompt:
    if not isinstance(request, LLMRequest):
        raise PromptRenderError("simulator prompt requires LLMRequest")
    data = _common_context(request)
    if request.call_type in {LLMCallType.NATURAL_RESPONSE, LLMCallType.TARGET_EVENT_RESPONSE}:
        data["plan"] = request.plan
        data["response_limit_sentences"] = request.response_limit
    elif request.call_type is LLMCallType.REFLECTION:
        data["plan"] = request.plan
        data["response"] = request.response
        data["response_limit_insights"] = request.response_limit
    elif request.call_type is LLMCallType.IMPORTANCE_SCORE:
        data = {
            "memory_kind": request.memory_kind,
            "memory_text": request.memory_text,
            "score_scale": {"minimum": 1, "maximum": 10},
        }
    return _render(_CALL_TEMPLATE_IDS[request.call_type], data, loader or PromptLoader())


def render_bcr_prompt(
    judge_input: "BCRJudgeInput",
    loader: PromptLoader | None = None,
) -> RenderedPrompt:
    from src.evaluation.schemas import BCRJudgeInput

    if not isinstance(judge_input, BCRJudgeInput):
        raise PromptRenderError("BCR prompt requires BCRJudgeInput")
    return _render("bcr_judge", judge_input.to_dict(), loader or PromptLoader())


__all__ = [
    "RenderedPrompt",
    "canonical_json",
    "render_bcr_prompt",
    "render_simulator_prompt",
]
