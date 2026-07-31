"""Deterministic construction of validated memory entries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.memory import MemoryEntry


class MemoryFactory:
    """Create stable per-agent memory IDs in deterministic sequence order."""

    def __init__(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        start_sequence: int = 1,
    ) -> None:
        if agent_id not in FIXED_AGENT_IDS:
            raise ValueError(f"unsupported memory owner: {agent_id!r}")
        if run_id is not None and (
            not isinstance(run_id, str) or not run_id.strip()
        ):
            raise ValueError("run_id must be a non-empty string or None")
        if (
            isinstance(start_sequence, bool)
            or not isinstance(start_sequence, int)
            or start_sequence < 1
        ):
            raise ValueError("start_sequence must be a positive integer")

        self._agent_id = agent_id
        self._run_id = run_id
        self._next_sequence = start_sequence

    @property
    def agent_id(self) -> str:
        """Return the owner assigned to entries created by this factory."""
        return self._agent_id

    @property
    def run_id(self) -> str | None:
        """Return the run identifier assigned to created entries."""
        return self._run_id

    def create_memory(
        self,
        *,
        memory_type: str,
        text: str,
        category: str,
        importance: float,
        created_at: int,
        source_step_id: str,
        source_step_type: str,
        last_accessed_at: int | None = None,
        embedding: tuple[float, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create one validated entry and advance the ID only on success."""
        sequence = self._next_sequence
        memory = MemoryEntry(
            memory_id=f"mem_{self._agent_id}_{sequence:06d}",
            agent_id=self._agent_id,
            memory_type=memory_type,
            text=text,
            category=category,
            importance=importance,
            created_at=created_at,
            last_accessed_at=(
                created_at if last_accessed_at is None else last_accessed_at
            ),
            source_step_id=source_step_id,
            source_step_type=source_step_type,
            embedding=embedding,
            run_id=self._run_id,
            metadata=deepcopy(dict(metadata)) if metadata is not None else {},
        )
        self._next_sequence += 1
        return memory

    create = create_memory
