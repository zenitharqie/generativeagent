"""Phase 09 evaluation error hierarchy."""

from __future__ import annotations


class EvaluationError(RuntimeError):
    """Base error for deterministic post-run evaluation."""


class EvaluationInputError(EvaluationError):
    """Raised when persisted source artifacts cannot be trusted."""


class EvaluationEligibilityError(EvaluationInputError):
    """Raised when a run lifecycle status is not eligible for evaluation."""


class EvaluationGroundTruthError(EvaluationInputError):
    """Raised when snapshotted evaluation ground truth is invalid or incomplete."""


class EvaluationJoinError(EvaluationInputError):
    """Raised when exact persisted-record joins cannot be completed."""


class EvaluationSchemaError(EvaluationError, ValueError):
    """Raised when a Phase 09 immutable model violates its schema."""


class EvaluationCalculationError(EvaluationError, ValueError):
    """Raised when an evaluation metric cannot be calculated exactly."""


class EvaluationAggregationError(EvaluationError, ValueError):
    """Raised when metric records cannot be aggregated without bias."""


class EvaluationPersistenceError(EvaluationError):
    """Raised when evaluation output cannot be safely published."""


class EvaluationConflictError(EvaluationPersistenceError):
    """Raised when a requested final or staging output already exists."""


__all__ = [
    "EvaluationAggregationError",
    "EvaluationCalculationError",
    "EvaluationConflictError",
    "EvaluationEligibilityError",
    "EvaluationError",
    "EvaluationGroundTruthError",
    "EvaluationInputError",
    "EvaluationJoinError",
    "EvaluationPersistenceError",
    "EvaluationSchemaError",
]
