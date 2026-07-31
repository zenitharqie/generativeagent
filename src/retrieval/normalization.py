"""Candidate-pool normalization for retrieval component scores."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite


class NormalizationError(ValueError):
    """Raised when component values cannot be normalized safely."""


def min_max_normalize(values: Iterable[float]) -> tuple[float, ...]:
    """Normalize finite numeric values to 0-1, using 0.5 when all equal."""
    if isinstance(values, (str, bytes)):
        raise NormalizationError("normalization values must be an iterable of numbers")

    normalized_input: list[float] = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise NormalizationError(
            "normalization values must be an iterable of numbers"
        ) from exc

    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NormalizationError("normalization values must be numeric")
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise NormalizationError("normalization values must be finite")
        normalized_input.append(numeric_value)

    if not normalized_input:
        return ()

    minimum = min(normalized_input)
    maximum = max(normalized_input)
    if minimum == maximum:
        return tuple(0.5 for _ in normalized_input)

    value_range = maximum - minimum
    return tuple((value - minimum) / value_range for value in normalized_input)
