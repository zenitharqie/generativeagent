"""Retrieval configuration models without scoring behavior."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


RETRIEVAL_COMPONENTS = frozenset({"recency", "importance", "relevance"})
WEIGHT_SUM_TOLERANCE = 0.011


@dataclass(frozen=True)
class RetrievalWeights:
    """The three transparent weights used by future retrieval scoring."""

    recency: float
    importance: float
    relevance: float

    def __post_init__(self) -> None:
        values = (self.recency, self.importance, self.relevance)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("retrieval weights must be numeric")
        if any(value < 0 for value in values):
            raise ValueError("retrieval weights must be non-negative")
        if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError("retrieval weights must approximately sum to 1.0")


@dataclass(frozen=True)
class RetrievalConfig:
    """One named retrieval-weight condition."""

    config_id: str
    weights: RetrievalWeights

    def __post_init__(self) -> None:
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise ValueError("config_id must be a non-empty string")
