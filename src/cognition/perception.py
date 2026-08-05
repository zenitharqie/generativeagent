"""Pure construction of the agent-observable context for one turn."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain import AgentProfile, SimulationStep, TargetEvent, TargetEventStep
from src.domain.simulation import STEP_TYPES


class PerceptionError(ValueError):
    """Raised when a turn cannot be converted into valid observable context."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerceptionError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_context(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise PerceptionError("conversation_context must be a tuple of strings")
    return tuple(
        _require_text(item, "conversation_context item") for item in value
    )


@dataclass(frozen=True)
class Perception:
    """A validated, auditable observation without long-term memory content."""

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
    conversation_context: tuple[str, ...]
    observation_text: str
    memory_generation_enabled: bool
    event_id: str | None = None
    event_title: str | None = None
    event_description: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "agent_id",
            "display_name",
            "role",
            "archetype",
            "persona_summary",
            "communication_style",
            "source_step_id",
            "step_title",
            "step_description",
            "period",
            "observation_text",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )
        if self.source_step_type not in STEP_TYPES:
            raise PerceptionError(
                f"unsupported source_step_type: {self.source_step_type!r}"
            )
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 1:
            raise PerceptionError("day must be a positive integer")
        if (
            isinstance(self.logical_timestamp, bool)
            or not isinstance(self.logical_timestamp, int)
            or self.logical_timestamp < 0
        ):
            raise PerceptionError("logical_timestamp must be a non-negative integer")
        if not isinstance(self.memory_generation_enabled, bool):
            raise PerceptionError("memory_generation_enabled must be boolean")
        for field_name in ("traits", "goals", "motivations"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not value:
                raise PerceptionError(f"{field_name} must be a non-empty tuple")
            object.__setattr__(
                self,
                field_name,
                tuple(_require_text(item, field_name) for item in value),
            )
        object.__setattr__(
            self,
            "conversation_context",
            _normalize_context(self.conversation_context),
        )

        event_values = (self.event_id, self.event_title, self.event_description)
        if self.source_step_type == "target_event":
            if any(value is None for value in event_values):
                raise PerceptionError(
                    "target-event perception requires event identity and description"
                )
            for field_name in ("event_id", "event_title", "event_description"):
                object.__setattr__(
                    self,
                    field_name,
                    _require_text(getattr(self, field_name), field_name),
                )
        elif any(value is not None for value in event_values):
            raise PerceptionError(
                "natural-interaction perception must not contain event fields"
            )


def build_perception(
    *,
    agent: AgentProfile,
    step: SimulationStep,
    logical_timestamp: int,
    conversation_context: tuple[str, ...] = (),
    event: TargetEvent | None = None,
) -> Perception:
    """Build one deterministic observation from validated domain entities."""
    if not isinstance(agent, AgentProfile):
        raise PerceptionError("agent must be an AgentProfile")
    if not isinstance(step, SimulationStep):
        raise PerceptionError("step must be a SimulationStep")
    if agent.agent_id not in step.participants:
        raise PerceptionError(
            f"agent {agent.agent_id!r} is not a participant in step {step.step_id!r}"
        )
    if (
        isinstance(logical_timestamp, bool)
        or not isinstance(logical_timestamp, int)
        or logical_timestamp < 0
    ):
        raise PerceptionError("logical_timestamp must be a non-negative integer")
    context = _normalize_context(conversation_context)

    event_id: str | None = None
    event_title: str | None = None
    event_description: str | None = None
    if isinstance(step, TargetEventStep):
        if not isinstance(event, TargetEvent):
            raise PerceptionError("target-event step requires its TargetEvent")
        if event.event_id != step.event_id:
            raise PerceptionError(
                f"step event_id {step.event_id!r} does not match {event.event_id!r}"
            )
        event_id = event.event_id
        event_title = event.title
        event_description = event.description
        observation_parts = (
            f"Target event: {event.title}.",
            event.description,
        )
    else:
        if event is not None:
            raise PerceptionError(
                "natural-interaction step must not receive target-event context"
            )
        observation_parts = (
            f"Classroom block: {step.title}.",
            step.description,
        )

    if context:
        observation_parts += ("Conversation context: " + " | ".join(context),)

    return Perception(
        agent_id=agent.agent_id,
        display_name=agent.display_name,
        role=agent.role,
        archetype=agent.archetype,
        persona_summary=agent.persona_summary,
        traits=agent.traits,
        goals=agent.goals,
        communication_style=agent.communication_style,
        motivations=agent.motivations,
        source_step_id=step.step_id,
        source_step_type=step.step_type,
        step_title=step.title,
        step_description=step.description,
        day=step.day,
        period=step.period,
        logical_timestamp=logical_timestamp,
        conversation_context=context,
        observation_text=" ".join(observation_parts),
        memory_generation_enabled=step.memory_generation_enabled,
        event_id=event_id,
        event_title=event_title,
        event_description=event_description,
    )
