from __future__ import annotations

from collections import Counter

import pytest

from src.domain import NaturalInteractionBlockStep, TargetEventStep
from src.simulation.turn_policy import (
    FIXED_NATURAL_TURN_ORDER,
    NATURAL_BLOCK_TOTAL_TURNS,
    NATURAL_BLOCK_TURNS_PER_AGENT,
    TurnPolicyError,
    TurnPolicyManager,
)


FIXED_PARTICIPANTS = (
    "teacher",
    "active_student",
    "quiet_student",
    "academic_student",
)


def _natural_step(**overrides: object) -> NaturalInteractionBlockStep:
    values: dict[str, object] = {
        "step_id": "day_01_natural_01",
        "day": 1,
        "period": "morning",
        "step_type": "natural_interaction_block",
        "title": "Opening classroom interaction",
        "description": "The class begins its first controlled interaction.",
        "participants": FIXED_PARTICIPANTS,
        "memory_generation_enabled": True,
        "evaluation_enabled": False,
        "turns_per_agent": 3,
        "total_turns": 12,
        "teacher_opens_block": True,
        "turn_policy": "controlled_fixed",
    }
    values.update(overrides)
    return NaturalInteractionBlockStep(**values)  # type: ignore[arg-type]


def test_build_turn_order_returns_exact_documented_immutable_schedule() -> None:
    order = TurnPolicyManager().build_turn_order(_natural_step())

    assert isinstance(order, tuple)
    assert order == (
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
    assert order is FIXED_NATURAL_TURN_ORDER


def test_fixed_order_has_twelve_turns_three_per_agent_and_teacher_opener() -> None:
    order = TurnPolicyManager().build_turn_order(_natural_step())

    assert len(order) == NATURAL_BLOCK_TOTAL_TURNS == 12
    assert Counter(order) == Counter(
        {agent_id: NATURAL_BLOCK_TURNS_PER_AGENT for agent_id in FIXED_PARTICIPANTS}
    )
    assert order[0] == "teacher"
    assert all(left != right for left, right in zip(order, order[1:]))


def test_build_turn_order_is_deterministic_across_calls() -> None:
    manager = TurnPolicyManager()

    assert manager.build_turn_order(_natural_step()) == manager.build_turn_order(
        _natural_step(step_id="day_01_natural_02", period="afternoon")
    )


def test_participant_tuple_order_does_not_redefine_fixed_speaker_policy() -> None:
    step = _natural_step(
        participants=(
            "academic_student",
            "quiet_student",
            "active_student",
            "teacher",
        )
    )

    assert TurnPolicyManager().build_turn_order(step) == FIXED_NATURAL_TURN_ORDER


def test_validate_turn_order_returns_the_validated_tuple() -> None:
    order = tuple(FIXED_NATURAL_TURN_ORDER)

    assert TurnPolicyManager().validate_turn_order(_natural_step(), order) is order


def test_rejects_non_natural_step() -> None:
    target_step = TargetEventStep(
        step_id="day_02_target_01",
        day=2,
        period="afternoon",
        step_type="target_event",
        title="Target event",
        description="A controlled evaluation event.",
        participants=FIXED_PARTICIPANTS,
        memory_generation_enabled=True,
        evaluation_enabled=True,
        event_id="target_event_01",
    )

    with pytest.raises(TurnPolicyError, match="NaturalInteractionBlockStep"):
        TurnPolicyManager().build_turn_order(target_step)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"turn_policy": "controlled_semi_dynamic"}, "controlled_fixed"),
        ({"total_turns": 11}, "total_turns must equal 12"),
        ({"turns_per_agent": 2}, "turns_per_agent must equal 3"),
        ({"teacher_opens_block": False}, "teacher to open"),
        (
            {
                "participants": (
                    "teacher",
                    "active_student",
                    "quiet_student",
                    "teacher",
                )
            },
            "exactly the four fixed agents",
        ),
        (
            {
                "participants": (
                    "teacher",
                    "active_student",
                    "quiet_student",
                    "unknown_agent",
                )
            },
            "exactly the four fixed agents",
        ),
    ],
)
def test_rejects_natural_step_contract_mismatch(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TurnPolicyError, match=message):
        TurnPolicyManager().build_turn_order(_natural_step(**overrides))


def test_rejects_mutable_turn_order() -> None:
    with pytest.raises(TurnPolicyError, match="immutable tuple"):
        TurnPolicyManager().validate_turn_order(
            _natural_step(), list(FIXED_NATURAL_TURN_ORDER)  # type: ignore[arg-type]
        )


def test_rejects_wrong_turn_count() -> None:
    with pytest.raises(TurnPolicyError, match="exactly 12 speakers"):
        TurnPolicyManager().validate_turn_order(
            _natural_step(), FIXED_NATURAL_TURN_ORDER[:-1]
        )


def test_rejects_unknown_speaker() -> None:
    order = (*FIXED_NATURAL_TURN_ORDER[:-1], "unknown_agent")

    with pytest.raises(TurnPolicyError, match="exactly the fixed agent set"):
        TurnPolicyManager().validate_turn_order(_natural_step(), order)


def test_rejects_wrong_per_agent_quota() -> None:
    order = list(FIXED_NATURAL_TURN_ORDER)
    order[5] = "active_student"

    with pytest.raises(TurnPolicyError, match="must appear exactly 3 times"):
        TurnPolicyManager().validate_turn_order(_natural_step(), tuple(order))


def test_rejects_order_that_does_not_start_with_teacher() -> None:
    order = (*FIXED_NATURAL_TURN_ORDER[1:], FIXED_NATURAL_TURN_ORDER[0])

    with pytest.raises(TurnPolicyError, match="teacher must open"):
        TurnPolicyManager().validate_turn_order(_natural_step(), order)


def test_rejects_consecutive_same_speaker() -> None:
    order = (
        "teacher",
        "active_student",
        "active_student",
        "quiet_student",
        "academic_student",
        "teacher",
        "academic_student",
        "quiet_student",
        "academic_student",
        "active_student",
        "quiet_student",
        "teacher",
    )

    with pytest.raises(TurnPolicyError, match="Consecutive|consecutive"):
        TurnPolicyManager().validate_turn_order(_natural_step(), order)


def test_rejects_alternate_schedule_even_when_all_general_invariants_hold() -> None:
    alternate_order = (
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
    )

    with pytest.raises(TurnPolicyError, match="documented fixed"):
        TurnPolicyManager().validate_turn_order(_natural_step(), alternate_order)
