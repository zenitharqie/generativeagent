"""Minimal structured-output parsing without semantic response rewriting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import re
from typing import Any


class LLMOutputError(ValueError):
    """Base error for invalid structured LLM output."""


class EmptyLLMOutputError(LLMOutputError):
    """Raised when an LLM stage returns no usable text."""


class LLMOutputParseError(LLMOutputError):
    """Raised when output cannot be decoded as the required JSON object."""


class LLMOutputValidationError(LLMOutputError):
    """Raised when decoded output is missing a required valid field."""


class PlaceholderLLMOutputError(LLMOutputValidationError):
    """Raised for an unmistakable schema/template placeholder value."""


_DOCUMENTED_PLACEHOLDER_VALUES = frozenset(
    {
        "placeholder",
        "agent utterance here",
        "short factual memory text here",
        "category_label_here",
        "short plan summary",
        "one concise insight",
        "brief reason",
    }
)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMOutputValidationError(
            f"LLM output field {field_name!r} must be a non-empty string"
        )
    normalized = value.strip()
    if normalized.casefold() in _DOCUMENTED_PLACEHOLDER_VALUES or re.fullmatch(
        r"\{\{[^{}]+\}\}", normalized
    ):
        raise PlaceholderLLMOutputError(
            f"LLM output field {field_name!r} contains a placeholder"
        )
    return normalized


@dataclass(frozen=True)
class PlanOutput:
    plan: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", _require_text(self.plan, "plan"))


@dataclass(frozen=True)
class ActionOutput:
    response: str
    memory_to_store: str
    category: str

    def __post_init__(self) -> None:
        for field_name in ("response", "memory_to_store", "category"):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class ReflectionOutput:
    reflection: str
    category: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reflection",
            _require_text(self.reflection, "reflection"),
        )
        object.__setattr__(self, "category", _require_text(self.category, "category"))


@dataclass(frozen=True)
class ImportanceOutput:
    importance: float
    reason: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.importance, bool)
            or not isinstance(self.importance, (int, float))
            or not isfinite(float(self.importance))
            or not 1 <= self.importance <= 10
        ):
            raise LLMOutputValidationError(
                "LLM output field 'importance' must be a finite number between 1 and 10"
            )
        object.__setattr__(self, "importance", float(self.importance))
        object.__setattr__(self, "reason", _require_text(self.reason, "reason"))


def _strip_harmless_wrapper(raw_output: str) -> str:
    stripped = raw_output.strip()
    if not stripped:
        raise EmptyLLMOutputError("LLM output must not be empty")

    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        raise LLMOutputParseError("malformed outer code fence around LLM output")
    opening = lines[0].strip().lower()
    if opening not in {"```", "```json"}:
        raise LLMOutputParseError("only a plain or JSON outer code fence is supported")
    inner = "\n".join(lines[1:-1]).strip()
    if not inner:
        raise EmptyLLMOutputError("LLM output must not be empty")
    return inner


def _parse_json_object(raw_output: str) -> dict[str, Any]:
    if not isinstance(raw_output, str):
        raise LLMOutputParseError("LLM output must be a string")
    payload = _strip_harmless_wrapper(raw_output)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LLMOutputParseError(
            f"LLM output is not valid JSON: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(decoded, dict):
        raise LLMOutputParseError("LLM output must decode to a JSON object")
    return decoded


def parse_plan_output(raw_output: str) -> PlanOutput:
    """Parse the documented one-field plan output."""
    data = _parse_json_object(raw_output)
    if "plan" not in data:
        raise LLMOutputValidationError("LLM plan output is missing required field 'plan'")
    return PlanOutput(plan=data["plan"])


def parse_action_output(raw_output: str) -> ActionOutput:
    """Parse the shared natural/target action output structure."""
    data = _parse_json_object(raw_output)
    required = {"response", "memory_to_store", "category"}
    missing = sorted(required - data.keys())
    if missing:
        raise LLMOutputValidationError(
            f"LLM action output is missing required fields: {', '.join(missing)}"
        )
    return ActionOutput(
        response=data["response"],
        memory_to_store=data["memory_to_store"],
        category=data["category"],
    )


def parse_reflection_output(raw_output: str) -> ReflectionOutput:
    """Parse the documented bounded reflection output."""
    data = _parse_json_object(raw_output)
    required = {"reflection", "category"}
    missing = sorted(required - data.keys())
    if missing:
        raise LLMOutputValidationError(
            f"LLM reflection output is missing required fields: {', '.join(missing)}"
        )
    return ReflectionOutput(
        reflection=data["reflection"],
        category=data["category"],
    )


def parse_importance_output(raw_output: str) -> ImportanceOutput:
    """Parse a development-only importance score and audit reason."""
    data = _parse_json_object(raw_output)
    required = {"importance", "reason"}
    missing = sorted(required - data.keys())
    if missing:
        raise LLMOutputValidationError(
            f"LLM importance output is missing required fields: {', '.join(missing)}"
        )
    return ImportanceOutput(
        importance=data["importance"],
        reason=data["reason"],
    )
