"""Memory entity definitions without memory-stream behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.domain.agents import FIXED_AGENT_IDS


MEMORY_TYPES = frozenset({"seed_persona", "observation", "plan", "action", "reflection"})
SOURCE_STEP_TYPES = frozenset({"natural_interaction_block", "target_event"})


@dataclass(frozen=True)
class MemoryEntry:
    """One immutable memory record owned by one fixed agent."""

    memory_id: str
    agent_id: str
    memory_type: str
    text: str
    category: str
    importance: float
    created_at: int
    last_accessed_at: int
    source_step_id: str
    source_step_type: str
    embedding: tuple[float, ...] | None = None
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("memory_id", self.memory_id),
            ("text", self.text),
            ("category", self.category),
            ("source_step_id", self.source_step_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.agent_id not in FIXED_AGENT_IDS:
            raise ValueError(f"unsupported memory owner: {self.agent_id!r}")
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"unsupported memory_type: {self.memory_type!r}")
        if self.source_step_type not in SOURCE_STEP_TYPES:
            raise ValueError(f"unsupported source_step_type: {self.source_step_type!r}")
        if isinstance(self.importance, bool) or not isinstance(
            self.importance, (int, float)
        ):
            raise ValueError("importance must be numeric")
        if not 1 <= self.importance <= 10:
            raise ValueError("importance must be between 1 and 10")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.created_at, self.last_accessed_at)
        ):
            raise ValueError("memory timestamps must be non-negative")
        if self.embedding is not None and (
            not isinstance(self.embedding, tuple)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in self.embedding
            )
        ):
            raise ValueError("embedding must be a tuple of numeric values or None")
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id.strip()
        ):
            raise ValueError("run_id must be a non-empty string or None")
