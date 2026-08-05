"""Deterministic, offline development doubles for Phase 06 cognition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from src.llm.base import (
    EmbeddingError,
    LLMCallType,
    LLMClientError,
    LLMRequest,
    LLMRequestError,
)


class DummyLLMError(LLMClientError):
    """Raised when a configured deterministic dummy outcome cannot be produced."""


DummyOutcome = str | Exception


DEFAULT_DUMMY_RESPONSES: Mapping[LLMCallType, str] = MappingProxyType({
    LLMCallType.PLAN: (
        '{"plan":"Use the provided classroom context to give a concise dummy response."}'
    ),
    LLMCallType.NATURAL_RESPONSE: (
        '{"response":"This is a deterministic dummy classroom response.",'
        '"memory_to_store":"The agent gave a deterministic dummy classroom response.",'
        '"category":"classroom_norms"}'
    ),
    LLMCallType.TARGET_EVENT_RESPONSE: (
        '{"response":"This is a deterministic dummy target-event response.",'
        '"memory_to_store":"The agent gave a deterministic dummy target-event response.",'
        '"category":"classroom_norms"}'
    ),
    LLMCallType.REFLECTION: (
        '{"reflection":"The current classroom context may help with a later response.",'
        '"category":"classroom_norms"}'
    ),
    LLMCallType.IMPORTANCE_SCORE: (
        '{"importance":5,"reason":"Deterministic development-only importance score."}'
    ),
})


@dataclass(frozen=True)
class LLMInvocation:
    """An immutable record of one attempted dummy generation call."""

    index: int
    request: LLMRequest


def _normalize_call_type(value: LLMCallType | str) -> LLMCallType:
    try:
        return value if isinstance(value, LLMCallType) else LLMCallType(value)
    except (TypeError, ValueError) as exc:
        raise LLMRequestError(f"unsupported scripted call type: {value!r}") from exc


def _normalize_outcomes(value: object) -> tuple[DummyOutcome, ...]:
    if isinstance(value, (str, Exception)):
        outcomes: tuple[DummyOutcome, ...] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        outcomes = tuple(value)
    else:
        raise LLMRequestError(
            "scripted dummy outcomes must be a string, exception, or sequence"
        )
    if not outcomes:
        raise LLMRequestError("scripted dummy outcomes must not be empty")
    if any(not isinstance(outcome, (str, Exception)) for outcome in outcomes):
        raise LLMRequestError("every scripted dummy outcome must be a string or exception")
    return outcomes


class DummyLLMClient:
    """Return configured stage outputs without network access or randomness.

    This client exists only for development and tests. Its outputs are not a
    substitute for the final experimental model and must not become research
    data.
    """

    def __init__(
        self,
        scripted_responses: Mapping[
            LLMCallType | str, DummyOutcome | Sequence[DummyOutcome]
        ]
        | None = None,
    ) -> None:
        scripts: dict[LLMCallType, tuple[DummyOutcome, ...]] = {}
        if scripted_responses is not None:
            if not isinstance(scripted_responses, Mapping):
                raise LLMRequestError("scripted_responses must be a mapping or None")
            for raw_call_type, raw_outcomes in scripted_responses.items():
                call_type = _normalize_call_type(raw_call_type)
                if call_type in scripts:
                    raise LLMRequestError(
                        f"duplicate scripted call type: {call_type.value!r}"
                    )
                scripts[call_type] = _normalize_outcomes(raw_outcomes)

        self._scripts = scripts
        self._calls_by_type: dict[LLMCallType, int] = {
            call_type: 0 for call_type in LLMCallType
        }
        self._invocations: list[LLMInvocation] = []

    @property
    def invocations(self) -> tuple[LLMInvocation, ...]:
        """Return an immutable snapshot of all attempted calls in order."""
        return tuple(self._invocations)

    @property
    def call_types(self) -> tuple[LLMCallType, ...]:
        """Return attempted call types in invocation order."""
        return tuple(invocation.request.call_type for invocation in self._invocations)

    def generate(self, request: LLMRequest) -> str:
        """Return the next deterministic outcome for a validated request."""
        if not isinstance(request, LLMRequest):
            raise LLMRequestError("request must be an LLMRequest")

        self._invocations.append(
            LLMInvocation(index=len(self._invocations) + 1, request=request)
        )
        call_type = request.call_type
        per_type_index = self._calls_by_type[call_type]
        self._calls_by_type[call_type] += 1

        if call_type in self._scripts:
            scripted = self._scripts[call_type]
            if per_type_index >= len(scripted):
                raise DummyLLMError(
                    f"no scripted {call_type.value} outcome at index {per_type_index + 1}"
                )
            outcome = scripted[per_type_index]
        else:
            outcome = DEFAULT_DUMMY_RESPONSES[call_type]

        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DummyEmbeddingProvider:
    """Return one configured non-zero vector for all development-only text."""

    def __init__(self, embedding: Sequence[float] = (1.0, 1.0)) -> None:
        if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
            raise EmbeddingError("dummy embedding must be a numeric vector")
        values: list[float] = []
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError("dummy embedding must contain numeric values")
            numeric = float(value)
            if not isfinite(numeric):
                raise EmbeddingError("dummy embedding must contain finite values")
            values.append(numeric)
        if not values:
            raise EmbeddingError("dummy embedding must not be empty")
        if not any(value != 0.0 for value in values):
            raise EmbeddingError("dummy embedding must be non-zero")

        self._embedding = tuple(values)
        self._received_texts: list[str] = []

    @property
    def received_texts(self) -> tuple[str, ...]:
        """Return embedded texts in deterministic call order."""
        return tuple(self._received_texts)

    def embed(self, text: str) -> tuple[float, ...]:
        """Return the configured development vector for non-empty text."""
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("text to embed must be a non-empty string")
        self._received_texts.append(text)
        return self._embedding
