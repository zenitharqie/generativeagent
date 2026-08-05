"""Provider-neutral contracts for deterministic cognitive-stage generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.simulation import STEP_TYPES


class LLMError(ValueError):
    """Base error for the Phase 06 LLM boundary."""


class LLMRequestError(LLMError):
    """Raised when a cognitive stage supplies an invalid LLM request."""


class UnsupportedLLMCallTypeError(LLMRequestError):
    """Raised when a request uses a call type outside the Phase 06 contract."""


class LLMClientError(LLMError):
    """Raised when an LLM client cannot complete a valid request."""


class EmbeddingError(LLMError):
    """Raised when a development embedding provider cannot embed text."""


class LLMCallType(str, Enum):
    """The only generation call types supported by the Phase 06 pipeline."""

    PLAN = "plan"
    NATURAL_RESPONSE = "natural_response"
    TARGET_EVENT_RESPONSE = "target_event_response"
    REFLECTION = "reflection"
    IMPORTANCE_SCORE = "importance_score"


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMRequestError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise LLMRequestError(f"{field_name} must be a tuple of strings")
    normalized = tuple(_require_text(item, field_name) for item in value)
    return normalized


@dataclass(frozen=True)
class LLMContext:
    """Sanitized context visible to a dummy or future real text model.

    Retrieval audit data is deliberately absent. Only selected memory text may
    cross this boundary.
    """

    agent_id: str
    display_name: str
    role: str
    archetype: str
    persona_summary: str
    traits: tuple[str, ...]
    goals: tuple[str, ...]
    communication_style: str
    motivations: tuple[str, ...]
    source_step_id: str
    source_step_type: str
    step_title: str
    step_description: str
    day: int
    period: str
    logical_timestamp: int
    observation_text: str
    conversation_context: tuple[str, ...]
    selected_memory_texts: tuple[str, ...]
    event_id: str | None = None
    event_title: str | None = None
    event_description: str | None = None

    def __post_init__(self) -> None:
        if self.agent_id not in FIXED_AGENT_IDS:
            raise LLMRequestError(f"unsupported LLM context agent_id: {self.agent_id!r}")
        for field_name, value in (
            ("display_name", self.display_name),
            ("role", self.role),
            ("archetype", self.archetype),
            ("persona_summary", self.persona_summary),
            ("communication_style", self.communication_style),
            ("source_step_id", self.source_step_id),
            ("step_title", self.step_title),
            ("step_description", self.step_description),
            ("period", self.period),
            ("observation_text", self.observation_text),
        ):
            object.__setattr__(self, field_name, _require_text(value, field_name))

        if self.source_step_type not in STEP_TYPES:
            raise LLMRequestError(
                f"unsupported LLM context source_step_type: {self.source_step_type!r}"
            )
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 1:
            raise LLMRequestError("day must be a positive integer")
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 0
        ):
            raise LLMRequestError("logical_timestamp must be a non-negative integer")

        for field_name in ("traits", "goals", "motivations"):
            normalized = _normalize_text_tuple(getattr(self, field_name), field_name)
            if not normalized:
                raise LLMRequestError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)
        object.__setattr__(
            self,
            "conversation_context",
            _normalize_text_tuple(self.conversation_context, "conversation_context"),
        )
        object.__setattr__(
            self,
            "selected_memory_texts",
            _normalize_text_tuple(self.selected_memory_texts, "selected_memory_texts"),
        )

        event_values = (self.event_id, self.event_title, self.event_description)
        if self.source_step_type == "target_event":
            if any(value is None for value in event_values):
                raise LLMRequestError(
                    "target-event LLM context requires event_id, event_title, and event_description"
                )
            for field_name, value in (
                ("event_id", self.event_id),
                ("event_title", self.event_title),
                ("event_description", self.event_description),
            ):
                object.__setattr__(self, field_name, _require_text(value, field_name))
        elif any(value is not None for value in event_values):
            raise LLMRequestError("natural-interaction LLM context must not include event fields")


@dataclass(frozen=True)
class LLMRequest:
    """One validated, stage-specific request sent through the LLM boundary."""

    call_type: LLMCallType | str
    context: LLMContext
    plan: str | None = None
    response: str | None = None
    memory_text: str | None = None
    memory_kind: str | None = None
    response_limit: int | None = None

    def __post_init__(self) -> None:
        try:
            call_type = (
                self.call_type
                if isinstance(self.call_type, LLMCallType)
                else LLMCallType(self.call_type)
            )
        except (TypeError, ValueError) as exc:
            raise UnsupportedLLMCallTypeError(
                f"unsupported LLM call type: {self.call_type!r}"
            ) from exc
        object.__setattr__(self, "call_type", call_type)

        if not isinstance(self.context, LLMContext):
            raise LLMRequestError("context must be an LLMContext")
        if self.response_limit is not None and (
            isinstance(self.response_limit, bool)
            or not isinstance(self.response_limit, int)
            or self.response_limit < 1
        ):
            raise LLMRequestError("response_limit must be a positive integer or None")

        if call_type is LLMCallType.PLAN:
            if any(
                value is not None
                for value in (self.plan, self.response, self.memory_text, self.memory_kind)
            ):
                raise LLMRequestError("plan requests must not contain prior stage fields")
            return

        if call_type in {
            LLMCallType.NATURAL_RESPONSE,
            LLMCallType.TARGET_EVENT_RESPONSE,
        }:
            object.__setattr__(self, "plan", _require_text(self.plan, "plan"))
            if any(
                value is not None
                for value in (self.response, self.memory_text, self.memory_kind)
            ):
                raise LLMRequestError(
                    "response requests may contain only a plan as prior stage data"
                )
            expected_step_type = (
                "natural_interaction_block"
                if call_type is LLMCallType.NATURAL_RESPONSE
                else "target_event"
            )
            if self.context.source_step_type != expected_step_type:
                raise LLMRequestError(
                    f"{call_type.value} is incompatible with "
                    f"{self.context.source_step_type!r} context"
                )
            return

        if call_type is LLMCallType.REFLECTION:
            object.__setattr__(self, "plan", _require_text(self.plan, "plan"))
            object.__setattr__(self, "response", _require_text(self.response, "response"))
            if self.memory_text is not None or self.memory_kind is not None:
                raise LLMRequestError(
                    "reflection requests must not contain memory-scoring fields"
                )
            return

        if call_type is LLMCallType.IMPORTANCE_SCORE:
            object.__setattr__(
                self,
                "memory_text",
                _require_text(self.memory_text, "memory_text"),
            )
            if self.memory_kind not in {"action", "reflection"}:
                raise LLMRequestError(
                    "importance requests require memory_kind action or reflection"
                )
            if self.plan is not None or self.response is not None:
                raise LLMRequestError(
                    "importance requests must not contain plan or response fields"
                )


@runtime_checkable
class LLMClient(Protocol):
    """Minimal generation interface shared by dummy and future real clients."""

    def generate(self, request: LLMRequest) -> str:
        """Return raw structured output for one validated request."""


@runtime_checkable
class TextEmbeddingProvider(Protocol):
    """Boundary for embedding newly generated memory text."""

    def embed(self, text: str) -> tuple[float, ...]:
        """Return a deterministic numeric vector for text."""
