"""Safe failures for the controlled Ground Truth finalization workflow."""

from __future__ import annotations


class GroundTruthWorkflowError(RuntimeError):
    """Base error whose message never includes untrusted annotation content."""


class GroundTruthInputError(GroundTruthWorkflowError):
    """Raised when an untrusted package, annotation, or embedding input is invalid."""


class GroundTruthCompletenessError(GroundTruthWorkflowError):
    """Raised when the required 20-by-3 evidence matrix is incomplete."""


class GroundTruthConsolidationError(GroundTruthWorkflowError):
    """Raised when semantic consolidation is invalid or ambiguous."""


class GroundTruthFinalizationError(GroundTruthWorkflowError):
    """Raised when methodological evidence cannot produce final Ground Truth."""


class GroundTruthPersistenceError(GroundTruthWorkflowError):
    """Raised when deterministic evidence cannot be published safely."""


__all__ = [
    "GroundTruthCompletenessError",
    "GroundTruthConsolidationError",
    "GroundTruthFinalizationError",
    "GroundTruthInputError",
    "GroundTruthPersistenceError",
    "GroundTruthWorkflowError",
]
