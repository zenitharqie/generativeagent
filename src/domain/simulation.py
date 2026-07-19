"""Typed simulation schedule and target-event models."""

from __future__ import annotations

from dataclasses import dataclass


STEP_TYPES = frozenset({"natural_interaction_block", "target_event"})
TURN_POLICIES = frozenset({"controlled_fixed", "controlled_semi_dynamic"})


@dataclass(frozen=True)
class SimulationStep:
    """Fields shared by all scheduled simulation steps."""

    step_id: str
    day: int
    period: str
    step_type: str
    title: str
    description: str
    participants: tuple[str, ...]
    memory_generation_enabled: bool
    evaluation_enabled: bool

    def __post_init__(self) -> None:
        if self.step_type not in STEP_TYPES:
            raise ValueError(f"unsupported step_type: {self.step_type!r}")
        if isinstance(self.day, bool) or not isinstance(self.day, int) or self.day < 1:
            raise ValueError("step day must be a positive integer")
        for field_name, value in (
            ("step_id", self.step_id),
            ("period", self.period),
            ("title", self.title),
            ("description", self.description),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.participants, tuple) or not self.participants or any(
            not isinstance(participant, str) or not participant.strip()
            for participant in self.participants
        ):
            raise ValueError("step participants must contain non-empty strings")
        if not isinstance(self.memory_generation_enabled, bool):
            raise ValueError("memory_generation_enabled must be boolean")
        if not isinstance(self.evaluation_enabled, bool):
            raise ValueError("evaluation_enabled must be boolean")

    @property
    def is_target_event(self) -> bool:
        return self.step_type == "target_event"


@dataclass(frozen=True)
class NaturalInteractionBlockStep(SimulationStep):
    """A controlled 12-turn memory-building block."""

    turns_per_agent: int
    total_turns: int
    teacher_opens_block: bool
    turn_policy: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.step_type != "natural_interaction_block":
            raise ValueError("natural block must use natural_interaction_block step_type")
        for field_name, value in (
            ("turns_per_agent", self.turns_per_agent),
            ("total_turns", self.total_turns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.teacher_opens_block, bool):
            raise ValueError("teacher_opens_block must be boolean")
        if self.turn_policy not in TURN_POLICIES:
            raise ValueError(f"unsupported turn_policy: {self.turn_policy!r}")


@dataclass(frozen=True)
class TargetEventStep(SimulationStep):
    """A scheduled target event used as an evaluation anchor."""

    event_id: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.step_type != "target_event":
            raise ValueError("target event step must use target_event step_type")
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")


@dataclass(frozen=True)
class TargetEvent:
    """A controlled event definition referenced by the schedule."""

    event_id: str
    title: str
    description: str
    expected_memory_themes: tuple[str, ...]
    evaluation_role: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("event_id", self.event_id),
            ("title", self.title),
            ("description", self.description),
            ("evaluation_role", self.evaluation_role),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            not isinstance(self.expected_memory_themes, tuple)
            or not self.expected_memory_themes
            or any(
                not isinstance(theme, str) or not theme.strip()
                for theme in self.expected_memory_themes
            )
        ):
            raise ValueError("expected_memory_themes must contain non-empty strings")
