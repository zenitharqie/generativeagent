"""Neutral and deterministic retrieval-query construction."""

from __future__ import annotations

from dataclasses import dataclass

from src.cognition.perception import Perception


class RetrievalQueryError(ValueError):
    """Raised when a retrieval query cannot be built from perception."""


@dataclass(frozen=True)
class RetrievalQuery:
    """The text passed unchanged to the Phase 05 retrieval engine."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise RetrievalQueryError("retrieval query text must be non-empty")
        object.__setattr__(self, "text", self.text.strip())


def build_retrieval_query(perception: Perception) -> RetrievalQuery:
    """Use current role and observable context without retrieval-condition hints."""
    if not isinstance(perception, Perception):
        raise RetrievalQueryError("perception must be a Perception")

    parts = [
        f"Agent role: {perception.role}.",
        f"Current step: {perception.step_title}.",
        perception.step_description,
    ]
    if perception.event_title is not None:
        parts.extend(
            (
                f"Current event: {perception.event_title}.",
                perception.event_description or "",
            )
        )
    if perception.conversation_context:
        parts.append(
            "Immediate conversation context: "
            + " | ".join(perception.conversation_context)
        )
    return RetrievalQuery(text=" ".join(part for part in parts if part))
