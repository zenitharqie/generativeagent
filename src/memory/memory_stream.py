"""Ordered, per-agent in-memory storage for immutable memory entries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.memory import MEMORY_TYPES, SOURCE_STEP_TYPES, MemoryEntry


class MemoryStreamError(ValueError):
    """Raised when a memory-stream operation would violate stream invariants."""


class MemoryStream:
    """An append-ordered memory stream owned by exactly one agent."""

    def __init__(self, agent_id: str, *, run_id: str | None = None) -> None:
        if agent_id not in FIXED_AGENT_IDS:
            raise MemoryStreamError(f"unsupported memory owner: {agent_id!r}")
        if run_id is not None and (
            not isinstance(run_id, str) or not run_id.strip()
        ):
            raise MemoryStreamError("run_id must be a non-empty string or None")

        self._agent_id = agent_id
        self._run_id = run_id
        self._memories: list[MemoryEntry] = []
        self._index_by_id: dict[str, int] = {}

    @property
    def agent_id(self) -> str:
        """Return the only agent permitted to own entries in this stream."""
        return self._agent_id

    @property
    def run_id(self) -> str | None:
        """Return the run to which this stream is isolated, if specified."""
        return self._run_id

    def __len__(self) -> int:
        return len(self._memories)

    def add_memory(self, memory: MemoryEntry) -> None:
        """Append a validated memory while enforcing owner, run, and ID rules."""
        if not isinstance(memory, MemoryEntry):
            raise MemoryStreamError("memory must be a MemoryEntry")
        if memory.agent_id != self._agent_id:
            raise MemoryStreamError(
                f"memory {memory.memory_id!r} belongs to {memory.agent_id!r}, "
                f"not {self._agent_id!r}"
            )
        if self._run_id is not None and memory.run_id != self._run_id:
            raise MemoryStreamError(
                f"memory {memory.memory_id!r} does not belong to run {self._run_id!r}"
            )
        if memory.memory_id in self._index_by_id:
            raise MemoryStreamError(f"duplicate memory_id: {memory.memory_id!r}")

        self._index_by_id[memory.memory_id] = len(self._memories)
        self._memories.append(memory)

    add = add_memory

    def list_memories(self) -> tuple[MemoryEntry, ...]:
        """Return an immutable snapshot in append order."""
        return tuple(self._memories)

    list_all = list_memories

    def filter_memories(
        self,
        *,
        memory_type: str | None = None,
        source_step_type: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Return matching memories without changing their access metadata."""
        if memory_type is not None and memory_type not in MEMORY_TYPES:
            raise MemoryStreamError(f"unsupported memory_type: {memory_type!r}")
        if source_step_type is not None and source_step_type not in SOURCE_STEP_TYPES:
            raise MemoryStreamError(
                f"unsupported source_step_type: {source_step_type!r}"
            )
        return tuple(
            memory
            for memory in self._memories
            if (memory_type is None or memory.memory_type == memory_type)
            and (
                source_step_type is None
                or memory.source_step_type == source_step_type
            )
        )

    def filter_by_memory_type(self, memory_type: str) -> tuple[MemoryEntry, ...]:
        """Return memories of one supported memory type."""
        return self.filter_memories(memory_type=memory_type)

    def filter_by_source_step_type(
        self, source_step_type: str
    ) -> tuple[MemoryEntry, ...]:
        """Return memories created by one supported source-step type."""
        return self.filter_memories(source_step_type=source_step_type)

    def update_last_accessed_at(
        self,
        memory_ids: Iterable[str],
        accessed_at: int,
    ) -> tuple[MemoryEntry, ...]:
        """Atomically update only selected IDs while preserving stream order."""
        if isinstance(memory_ids, (str, bytes)):
            raise MemoryStreamError("memory_ids must be an iterable of IDs")
        if (
            isinstance(accessed_at, bool)
            or not isinstance(accessed_at, int)
            or accessed_at < 0
        ):
            raise MemoryStreamError("accessed_at must be a non-negative integer")

        requested_ids = tuple(memory_ids)
        if any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in requested_ids
        ):
            raise MemoryStreamError("memory_ids must contain non-empty strings")
        if len(requested_ids) != len(set(requested_ids)):
            raise MemoryStreamError("memory_ids must not contain duplicates")

        unknown_ids = [
            memory_id
            for memory_id in requested_ids
            if memory_id not in self._index_by_id
        ]
        if unknown_ids:
            raise MemoryStreamError(
                f"unknown memory_id values: {', '.join(unknown_ids)}"
            )

        for memory_id in requested_ids:
            memory = self._memories[self._index_by_id[memory_id]]
            if accessed_at < memory.last_accessed_at:
                raise MemoryStreamError(
                    f"accessed_at must not move backward for {memory_id!r}"
                )

        selected_ids = set(requested_ids)
        updated: list[MemoryEntry] = []
        for index, memory in enumerate(self._memories):
            if memory.memory_id in selected_ids:
                memory = replace(memory, last_accessed_at=accessed_at)
                self._memories[index] = memory
                updated.append(memory)
        return tuple(updated)

    mark_accessed = update_last_accessed_at
