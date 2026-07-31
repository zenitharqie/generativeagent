"""Pure scoring functions for the three approved retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt

from src.domain.memory import MemoryEntry
from src.domain.retrieval import RetrievalWeights


RECENCY_DECAY_FACTOR = 0.99


class RetrievalScoringError(ValueError):
    """Raised when a retrieval component cannot be scored validly."""


class MissingEmbeddingError(RetrievalScoringError):
    """Raised when semantic relevance is requested without an embedding."""


def validate_vector(
    vector: Sequence[float] | None,
    *,
    field_name: str,
) -> tuple[float, ...]:
    """Return a finite, non-empty numeric vector or raise a clear error."""
    if vector is None:
        raise MissingEmbeddingError(f"{field_name} is required")
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise RetrievalScoringError(f"{field_name} must be a numeric vector")
    if not vector:
        raise RetrievalScoringError(f"{field_name} must not be empty")

    validated: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RetrievalScoringError(
                f"{field_name} must contain only numeric values"
            )
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise RetrievalScoringError(
                f"{field_name} must contain only finite values"
            )
        validated.append(numeric_value)
    return tuple(validated)


def calculate_recency(memory: MemoryEntry, *, current_time: int) -> float:
    """Calculate fixed exponential decay from the last access timestamp."""
    if not isinstance(memory, MemoryEntry):
        raise RetrievalScoringError("memory must be a MemoryEntry")
    if (
        isinstance(current_time, bool)
        or not isinstance(current_time, int)
        or current_time < 0
    ):
        raise RetrievalScoringError("current_time must be a non-negative integer")
    if current_time < memory.last_accessed_at:
        raise RetrievalScoringError(
            "current_time must not be earlier than memory.last_accessed_at"
        )

    delta_t = current_time - memory.last_accessed_at
    return RECENCY_DECAY_FACTOR**delta_t


def calculate_importance(memory: MemoryEntry) -> float:
    """Return the validated importance stored on a memory unchanged."""
    if not isinstance(memory, MemoryEntry):
        raise RetrievalScoringError("memory must be a MemoryEntry")
    return float(memory.importance)


def cosine_similarity(
    query_embedding: Sequence[float] | None,
    memory_embedding: Sequence[float] | None,
) -> float:
    """Calculate cosine similarity for two finite, non-zero vectors."""
    query_vector = validate_vector(query_embedding, field_name="query_embedding")
    memory_vector = validate_vector(memory_embedding, field_name="memory_embedding")
    if len(query_vector) != len(memory_vector):
        raise RetrievalScoringError(
            "query_embedding and memory_embedding must have equal dimensions"
        )

    query_norm = sqrt(sum(value * value for value in query_vector))
    memory_norm = sqrt(sum(value * value for value in memory_vector))
    if query_norm == 0.0 or memory_norm == 0.0:
        raise RetrievalScoringError("cosine similarity is undefined for a zero vector")

    dot_product = sum(
        query_value * memory_value
        for query_value, memory_value in zip(query_vector, memory_vector)
    )
    return dot_product / (query_norm * memory_norm)


def calculate_relevance(
    memory: MemoryEntry,
    *,
    query_embedding: Sequence[float] | None,
) -> float:
    """Calculate semantic relevance from the memory's stored embedding."""
    if not isinstance(memory, MemoryEntry):
        raise RetrievalScoringError("memory must be a MemoryEntry")
    return cosine_similarity(query_embedding, memory.embedding)


def calculate_weighted_score(
    *,
    normalized_recency: float,
    normalized_importance: float,
    normalized_relevance: float,
    weights: RetrievalWeights,
) -> float:
    """Apply exactly the architecture-contract weighted-sum formula."""
    if not isinstance(weights, RetrievalWeights):
        raise RetrievalScoringError("weights must be RetrievalWeights")

    components = {
        "normalized_recency": normalized_recency,
        "normalized_importance": normalized_importance,
        "normalized_relevance": normalized_relevance,
    }
    validated: dict[str, float] = {}
    for field_name, value in components.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RetrievalScoringError(f"{field_name} must be numeric")
        numeric_value = float(value)
        if not isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise RetrievalScoringError(
                f"{field_name} must be a finite value between 0 and 1"
            )
        validated[field_name] = numeric_value

    return (
        weights.recency * validated["normalized_recency"]
        + weights.importance * validated["normalized_importance"]
        + weights.relevance * validated["normalized_relevance"]
    )
