"""Immutable, auditable output structures for memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.memory import MemoryEntry
from src.domain.retrieval import RetrievalWeights


@dataclass(frozen=True)
class RetrievalCandidate:
    """One ranked memory with raw, normalized, and aggregate scores."""

    memory: MemoryEntry
    raw_recency: float
    raw_importance: float
    raw_relevance: float
    normalized_recency: float
    normalized_importance: float
    normalized_relevance: float
    final_score: float
    selected: bool

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryEntry):
            raise ValueError("candidate memory must be a MemoryEntry")

        raw_scores = (
            self.raw_recency,
            self.raw_importance,
            self.raw_relevance,
        )
        normalized_scores = (
            self.normalized_recency,
            self.normalized_importance,
            self.normalized_relevance,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in raw_scores
        ):
            raise ValueError("candidate raw scores must be finite numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not 0.0 <= value <= 1.0
            for value in normalized_scores
        ):
            raise ValueError("candidate normalized scores must be between 0 and 1")
        if (
            isinstance(self.final_score, bool)
            or not isinstance(self.final_score, (int, float))
            or not isfinite(float(self.final_score))
        ):
            raise ValueError("candidate final_score must be finite and numeric")
        if not isinstance(self.selected, bool):
            raise ValueError("candidate selected flag must be boolean")

    @property
    def memory_id(self) -> str:
        """Return the stable ID of the scored memory."""
        return self.memory.memory_id


@dataclass(frozen=True)
class RetrievalResult:
    """A complete ranked retrieval result suitable for later logging."""

    query: str
    agent_id: str
    retrieval_config_id: str
    applied_weights: RetrievalWeights
    top_k: int
    candidate_count: int
    candidates: tuple[RetrievalCandidate, ...]
    selected_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("retrieval query must be a non-empty string")
        if self.agent_id not in FIXED_AGENT_IDS:
            raise ValueError(f"unsupported retrieval agent_id: {self.agent_id!r}")
        if (
            not isinstance(self.retrieval_config_id, str)
            or not self.retrieval_config_id.strip()
        ):
            raise ValueError("retrieval_config_id must be a non-empty string")
        if not isinstance(self.applied_weights, RetrievalWeights):
            raise ValueError("applied_weights must be RetrievalWeights")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise ValueError("candidate_count must be a non-negative integer")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, RetrievalCandidate)
            for candidate in self.candidates
        ):
            raise ValueError("candidates must be a tuple of RetrievalCandidate values")
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must equal the number of candidates")
        if any(candidate.memory.agent_id != self.agent_id for candidate in self.candidates):
            raise ValueError("all candidates must belong to the retrieval agent")

        candidate_ids = tuple(candidate.memory_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("retrieval candidates must have unique memory IDs")
        if not isinstance(self.selected_memory_ids, tuple) or any(
            not isinstance(memory_id, str) or not memory_id.strip()
            for memory_id in self.selected_memory_ids
        ):
            raise ValueError("selected_memory_ids must contain non-empty strings")

        expected_selected_count = min(self.top_k, self.candidate_count)
        if len(self.selected_memory_ids) != expected_selected_count:
            raise ValueError(
                "selected_memory_ids must contain exactly the expected Top-K count"
            )
        if len(set(self.selected_memory_ids)) != len(self.selected_memory_ids):
            raise ValueError("selected_memory_ids must be unique")
        rank_position = {
            candidate.memory_id: index
            for index, candidate in enumerate(self.candidates)
        }
        if any(memory_id not in rank_position for memory_id in self.selected_memory_ids):
            raise ValueError("selected_memory_ids must all be ranked candidates")
        selected_positions = [
            rank_position[memory_id] for memory_id in self.selected_memory_ids
        ]
        if selected_positions != sorted(selected_positions):
            raise ValueError("selected_memory_ids must preserve ranked order")
        flagged_selected_ids = tuple(
            candidate.memory_id for candidate in self.candidates if candidate.selected
        )
        if self.selected_memory_ids != flagged_selected_ids:
            raise ValueError(
                "candidate selected flags must match selected_memory_ids in order"
            )

    @property
    def weights(self) -> RetrievalWeights:
        """Backward-friendly alias for the applied retrieval weights."""
        return self.applied_weights

    @property
    def selected_top_k_memory_ids(self) -> tuple[str, ...]:
        """Return the selected IDs using the Phase 05 terminology."""
        return self.selected_memory_ids

    @property
    def selected_memories(self) -> tuple[MemoryEntry, ...]:
        """Return selected memory entries in final ranking order."""
        return tuple(
            candidate.memory for candidate in self.candidates if candidate.selected
        )
