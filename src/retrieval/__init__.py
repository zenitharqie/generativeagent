"""Transparent weighted retrieval over per-agent memory streams."""

from src.retrieval.engine import RetrievalEngine, RetrievalEngineError
from src.retrieval.normalization import NormalizationError, min_max_normalize
from src.retrieval.result import RetrievalCandidate, RetrievalResult
from src.retrieval.scoring import (
    RECENCY_DECAY_FACTOR,
    MissingEmbeddingError,
    RetrievalScoringError,
    calculate_importance,
    calculate_recency,
    calculate_relevance,
    calculate_weighted_score,
    cosine_similarity,
    validate_vector,
)

__all__ = [
    "MissingEmbeddingError",
    "NormalizationError",
    "RECENCY_DECAY_FACTOR",
    "RetrievalCandidate",
    "RetrievalEngine",
    "RetrievalEngineError",
    "RetrievalResult",
    "RetrievalScoringError",
    "calculate_importance",
    "calculate_recency",
    "calculate_relevance",
    "calculate_weighted_score",
    "cosine_similarity",
    "min_max_normalize",
    "validate_vector",
]
