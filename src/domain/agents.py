"""Typed agent configuration models."""

from __future__ import annotations

from dataclasses import dataclass


FIXED_AGENT_IDS = frozenset(
    {"teacher", "active_student", "quiet_student", "academic_student"}
)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class SeedMemoryConfig:
    """A seed memory attached to an agent profile."""

    text: str
    category: str
    importance: float

    def __post_init__(self) -> None:
        _require_text(self.text, "seed memory text")
        _require_text(self.category, "seed memory category")
        if isinstance(self.importance, bool) or not isinstance(
            self.importance, (int, float)
        ):
            raise ValueError("seed memory importance must be numeric")
        if not 1 <= self.importance <= 10:
            raise ValueError("seed memory importance must be between 1 and 10")


@dataclass(frozen=True)
class PersonaProfile:
    """Stable persona attributes used by an agent."""

    summary: str
    traits: tuple[str, ...]
    goals: tuple[str, ...]
    communication_style: str
    motivations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.summary, "persona summary")
        _require_text(self.communication_style, "communication style")
        for field_name, values in (
            ("traits", self.traits),
            ("goals", self.goals),
            ("motivations", self.motivations),
        ):
            if not isinstance(values, tuple) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"persona {field_name} must contain non-empty strings")


@dataclass(frozen=True)
class AgentProfile:
    """Static identity, persona, and initial memories for one agent."""

    agent_id: str
    display_name: str
    role: str
    archetype: str
    persona: PersonaProfile
    seed_memories: tuple[SeedMemoryConfig, ...]

    def __post_init__(self) -> None:
        if self.agent_id not in FIXED_AGENT_IDS:
            raise ValueError(f"unsupported agent_id: {self.agent_id!r}")
        for field_name, value in (
            ("display_name", self.display_name),
            ("role", self.role),
            ("archetype", self.archetype),
        ):
            _require_text(value, field_name)
        if not isinstance(self.persona, PersonaProfile):
            raise ValueError("agent persona must be a PersonaProfile")
        if not isinstance(self.seed_memories, tuple) or not self.seed_memories:
            raise ValueError("agent seed_memories must not be empty")
        if any(not isinstance(memory, SeedMemoryConfig) for memory in self.seed_memories):
            raise ValueError("agent seed_memories must contain SeedMemoryConfig values")

    @property
    def persona_summary(self) -> str:
        return self.persona.summary

    @property
    def traits(self) -> tuple[str, ...]:
        return self.persona.traits

    @property
    def goals(self) -> tuple[str, ...]:
        return self.persona.goals

    @property
    def communication_style(self) -> str:
        return self.persona.communication_style

    @property
    def motivations(self) -> tuple[str, ...]:
        return self.persona.motivations
