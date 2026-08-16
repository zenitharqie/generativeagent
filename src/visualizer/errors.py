"""Focused, secret-safe failures for Phase 12 replay export."""

from __future__ import annotations


class VisualizerError(RuntimeError):
    """Base error for read-only replay loading and export."""


class ReplayError(VisualizerError):
    """Base error for replay-specific loading, projection, and rendering."""


class ReplayInputError(ReplayError):
    """Raised when a requested replay input cannot be inspected safely."""


class ReplayEligibilityError(ReplayInputError):
    """Raised when a source run is not in an inspectable terminal state."""


class ReplaySecurityError(ReplayInputError):
    """Raised when path, size, or untrusted-content safeguards fail."""


class ReplayArtifactError(ReplayError):
    """Base error for malformed canonical persisted artifacts."""


class ReplaySchemaError(ReplayArtifactError):
    """Raised when persisted JSON does not match a supported schema."""


class ReplayIdentityError(ReplayArtifactError):
    """Raised when records from different runs or units are mixed."""


class ReplayOrderingError(ReplayArtifactError):
    """Raised when persisted logical ordering is inconsistent."""


class ReplayJoinError(ReplayArtifactError):
    """Raised when a canonical reference cannot be joined exactly."""


class ReplayOverlayError(ReplayArtifactError):
    """Raised when an explicitly supplied optional overlay is invalid."""


class ReplayExportError(ReplayError):
    """Raised when deterministic replay publication fails."""


class ReplayConflictError(ReplayExportError):
    """Raised when a final or staging output path already exists."""


class ReplayRenderError(ReplayError):
    """Raised when a replay cannot be rendered using the approved projection."""


__all__ = [
    "ReplayArtifactError",
    "ReplayConflictError",
    "ReplayEligibilityError",
    "ReplayError",
    "ReplayExportError",
    "ReplayIdentityError",
    "ReplayInputError",
    "ReplayJoinError",
    "ReplayOrderingError",
    "ReplayOverlayError",
    "ReplayRenderError",
    "ReplaySchemaError",
    "ReplaySecurityError",
    "VisualizerError",
]
