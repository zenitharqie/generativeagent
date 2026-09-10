"""Deterministic weighted retrieval over one agent-owned memory stream."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src.domain.retrieval import RetrievalConfig, RetrievalWeights
from src.domain.settings import ExperimentSettings
from src.memory import MemoryStream
from src.retrieval.normalization import min_max_normalize
from src.retrieval.result import RetrievalCandidate, RetrievalResult
from src.retrieval.scoring import (
    calculate_importance,
    calculate_recency,
    calculate_relevance,
    calculate_weighted_score,
    validate_vector,
)


class RetrievalEngineError(ValueError):
    """Raised when a retrieval request violates engine invariants."""


class RetrievalEngine:
    """Score and rank an entire per-agent memory stream without mutating it."""

    def __init__(self, settings: ExperimentSettings) -> None:
        if not isinstance(settings, ExperimentSettings):
            raise RetrievalEngineError("settings must be ExperimentSettings")
        self._configured_top_k = settings.top_k

    @property
    def configured_top_k(self) -> int:
        """Return Top-K from the validated experiment settings."""
        return self._configured_top_k

    def retrieve(
        self,
        *,
        memory_stream: MemoryStream,
        query: str,
        query_embedding: Sequence[float] | None,
        retrieval_config: RetrievalConfig,
        current_time: int,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Return a complete ranked result while leaving the stream untouched."""
        if not isinstance(memory_stream, MemoryStream):
            raise RetrievalEngineError("memory_stream must be a MemoryStream")
        if not isinstance(query, str) or not query.strip():
            raise RetrievalEngineError("query must be a non-empty string")
        if not isinstance(retrieval_config, RetrievalConfig):
            raise RetrievalEngineError("retrieval_config must be RetrievalConfig")
        if not isinstance(retrieval_config.weights, RetrievalWeights):
            raise RetrievalEngineError(
                "retrieval_config.weights must be RetrievalWeights"
            )
        if (
            isinstance(current_time, bool)
            or not isinstance(current_time, int)
            or current_time < 0
        ):
            raise RetrievalEngineError("current_time must be a non-negative integer")

        requested_top_k = self._configured_top_k if top_k is None else top_k
        if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
            raise RetrievalEngineError("top_k must be an integer")
        if requested_top_k != self._configured_top_k:
            raise RetrievalEngineError(
                "requested top_k must equal validated settings.top_k "
                f"({self._configured_top_k})"
            )

        validated_query_embedding = validate_vector(
            query_embedding,
            field_name="query_embedding",
        )
        memories = memory_stream.list_memories()
        memory_ids = tuple(memory.memory_id for memory in memories)
        if len(memory_ids) != len(set(memory_ids)):
            raise RetrievalEngineError("candidate memory IDs must be unique")
        if any(memory.agent_id != memory_stream.agent_id for memory in memories):
            raise RetrievalEngineError(
                "all candidate memories must belong to the memory-stream owner"
            )

        raw_recency = tuple(
            calculate_recency(memory, current_time=current_time)
            for memory in memories
        )
        raw_importance = tuple(calculate_importance(memory) for memory in memories)
        raw_relevance = tuple(
            calculate_relevance(
                memory,
                query_embedding=validated_query_embedding,
            )
            for memory in memories
        )

        normalized_recency = min_max_normalize(raw_recency)
        normalized_importance = min_max_normalize(raw_importance)
        normalized_relevance = min_max_normalize(raw_relevance)

        candidates = tuple(
            RetrievalCandidate(
                memory=memory,
                raw_recency=raw_recency[index],
                raw_importance=raw_importance[index],
                raw_relevance=raw_relevance[index],
                normalized_recency=normalized_recency[index],
                normalized_importance=normalized_importance[index],
                normalized_relevance=normalized_relevance[index],
                final_score=calculate_weighted_score(
                    normalized_recency=normalized_recency[index],
                    normalized_importance=normalized_importance[index],
                    normalized_relevance=normalized_relevance[index],
                    weights=retrieval_config.weights,
                ),
                selected=False,
            )
            for index, memory in enumerate(memories)
        )

        ranked = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.final_score,
                    -candidate.raw_relevance,
                    -candidate.raw_importance,
                    -candidate.memory.created_at,
                    candidate.memory_id,
                ),
            )
        )
        selected_count = min(requested_top_k, len(ranked))
        selected_ids: list[str] = []
        seen_categories: set[str] = set()
        for candidate in ranked:
            if len(selected_ids) >= selected_count:
                break
            category = candidate.memory.category
            if category in seen_categories:
                continue
            seen_categories.add(category)
            selected_ids.append(candidate.memory_id)
        if len(selected_ids) < selected_count:
            remaining_ids = set(selected_ids)
            for candidate in ranked:
                if len(selected_ids) >= selected_count:
                    break
                if candidate.memory_id not in remaining_ids:
                    remaining_ids.add(candidate.memory_id)
                    selected_ids.append(candidate.memory_id)
        selected_id_set = set(selected_ids)
        selected_memory_ids = tuple(
            candidate.memory_id
            for candidate in ranked
            if candidate.memory_id in selected_id_set
        )
        ranked_with_selection = tuple(
            replace(candidate, selected=candidate.memory_id in selected_id_set)
            for candidate in ranked
        )

        return RetrievalResult(
            query=query,
            agent_id=memory_stream.agent_id,
            retrieval_config_id=retrieval_config.config_id,
            applied_weights=retrieval_config.weights,
            top_k=requested_top_k,
            candidate_count=len(ranked_with_selection),
            candidates=ranked_with_selection,
            selected_memory_ids=selected_memory_ids,
        )
