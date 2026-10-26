"""Deterministic speaker policy for natural interaction blocks."""

from __future__ import annotations

from collections import Counter
from typing import Final

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.simulation import NaturalInteractionBlockStep


FIXED_NATURAL_TURN_ORDER: Final[tuple[str, ...]] = (
    "teacher",
    "active_student",
    "academic_student",
    "quiet_student",
    "active_student",
    "teacher",
    "academic_student",
    "quiet_student",
    "academic_student",
    "active_student",
    "quiet_student",
    "teacher",
)

NATURAL_BLOCK_TOTAL_TURNS: Final = 12
NATURAL_BLOCK_TURNS_PER_AGENT: Final = 3


class TurnPolicyError(ValueError):
    """Raised when a natural-block step or speaker order violates the contract."""


class TurnPolicyManager:
    """Build and validate the single approved natural-block speaker schedule."""

    def build_turn_order(
        self,
        step: NaturalInteractionBlockStep,
    ) -> tuple[str, ...]:
        """Return the documented immutable speaker order for ``step``."""

        self.validate_turn_order(step, FIXED_NATURAL_TURN_ORDER)
        return FIXED_NATURAL_TURN_ORDER

    def validate_turn_order(
        self,
        step: NaturalInteractionBlockStep,
        turn_order: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate and return an approved immutable natural-block speaker order."""

        self._validate_step(step)
        if not isinstance(turn_order, tuple):
            raise TurnPolicyError("turn_order must be an immutable tuple")
        if len(turn_order) != NATURAL_BLOCK_TOTAL_TURNS:
            raise TurnPolicyError(
                f"turn_order must contain exactly {NATURAL_BLOCK_TOTAL_TURNS} speakers"
            )
        if any(not isinstance(agent_id, str) or not agent_id.strip() for agent_id in turn_order):
            raise TurnPolicyError("turn_order must contain non-empty agent IDs")
        if frozenset(turn_order) != FIXED_AGENT_IDS:
            raise TurnPolicyError("turn_order speakers must be exactly the fixed agent set")

        speaker_counts = Counter(turn_order)
        invalid_counts = {
            agent_id: speaker_counts[agent_id]
            for agent_id in sorted(FIXED_AGENT_IDS)
            if speaker_counts[agent_id] != NATURAL_BLOCK_TURNS_PER_AGENT
        }
        if invalid_counts:
            raise TurnPolicyError(
                "each fixed agent must appear exactly "
                f"{NATURAL_BLOCK_TURNS_PER_AGENT} times; got {invalid_counts}"
            )
        if turn_order[0] != "teacher":
            raise TurnPolicyError("teacher must open every natural interaction block")
        if any(
            current_speaker == next_speaker
            for current_speaker, next_speaker in zip(turn_order, turn_order[1:])
        ):
            raise TurnPolicyError("consecutive turns by the same speaker are not permitted")
        if turn_order != FIXED_NATURAL_TURN_ORDER:
            raise TurnPolicyError(
                "turn_order must match the documented fixed natural-block schedule"
            )
        return turn_order

    @staticmethod
    def _validate_step(step: NaturalInteractionBlockStep) -> None:
        if not isinstance(step, NaturalInteractionBlockStep):
            raise TurnPolicyError("step must be a NaturalInteractionBlockStep")
        if step.turn_policy != "controlled_fixed":
            raise TurnPolicyError("natural block must use controlled_fixed turn_policy")
        if (
            len(step.participants) != len(FIXED_AGENT_IDS)
            or len(set(step.participants)) != len(FIXED_AGENT_IDS)
            or frozenset(step.participants) != FIXED_AGENT_IDS
        ):
            raise TurnPolicyError(
                "natural block participants must be exactly the four fixed agents"
            )
        if step.total_turns != NATURAL_BLOCK_TOTAL_TURNS:
            raise TurnPolicyError(
                f"natural block total_turns must equal {NATURAL_BLOCK_TOTAL_TURNS}"
            )
        if step.turns_per_agent != NATURAL_BLOCK_TURNS_PER_AGENT:
            raise TurnPolicyError(
                "natural block turns_per_agent must equal "
                f"{NATURAL_BLOCK_TURNS_PER_AGENT}"
            )
        if not step.teacher_opens_block:
            raise TurnPolicyError("natural block must require the teacher to open")
