"""Exact fixed-denominator MRA calculation tests."""

from __future__ import annotations

import pytest

from src.evaluation import (
    EvaluationCalculationError,
    EvaluationSchemaError,
    EvaluationUnitIdentity,
    SelectedMemoryEvidence,
    calculate_mra,
)


def _identity() -> EvaluationUnitIdentity:
    return EvaluationUnitIdentity(
        experiment_id="experiment",
        run_id="balanced_rep01",
        retrieval_config_id="balanced",
        repetition_index=1,
        event_id="event_05",
        step_id="day_06_target_05",
        turn_id="day_06_target_05_turn_04",
        turn_index=4,
        timestamp=104,
        agent_id="academic_student",
    )


def _memories(*categories: str) -> tuple[SelectedMemoryEvidence, ...]:
    return tuple(
        SelectedMemoryEvidence(
            retrieval_rank=index,
            memory_id=f"mem_academic_student_{index:06d}",
            category=category,
            text=f"memory {index}",
        )
        for index, category in enumerate(categories, start=1)
    )


def _calculate(
    categories: tuple[str, ...],
    expected: tuple[str, ...] = (
        "exam_concern",
        "study_strategy",
        "academic_focus",
    ),
):
    memories = _memories(*categories)
    return calculate_mra(
        identity=_identity(),
        evaluation_mode="primary",
        primary_eligible=True,
        included_in_final_research_aggregation=True,
        selected_memories=memories,
        expected_categories=expected,
        candidate_count=len(memories),
        top_k=5,
    )


def test_known_three_of_five_example_is_exactly_point_six() -> None:
    result = _calculate(
        (
            "study_strategy",
            "academic_focus",
            "classroom_norms",
            "exam_concern",
            "social_learning",
        )
    )

    assert result.matched_count == 3
    assert result.denominator == 5
    assert result.mra == 0.6
    assert result.matched_memory_ids == (
        "mem_academic_student_000001",
        "mem_academic_student_000002",
        "mem_academic_student_000004",
    )
    assert result.unmatched_memory_ids == (
        "mem_academic_student_000003",
        "mem_academic_student_000005",
    )


@pytest.mark.parametrize(
    ("categories", "expected_score"),
    [
        (("exam_concern",) * 5, 1.0),
        (("classroom_norms",) * 5, 0.0),
        (("exam_concern", "other", "study_strategy", "other", "other"), 0.4),
        ((), 0.0),
        (("exam_concern", "other"), 0.2),
    ],
)
def test_full_partial_zero_and_fewer_than_k_use_fixed_denominator(
    categories: tuple[str, ...],
    expected_score: float,
) -> None:
    result = _calculate(categories)

    assert result.denominator == 5
    assert result.retrieved_count == len(categories)
    assert result.mra == expected_score


def test_duplicate_matching_retrieved_categories_count_individually() -> None:
    result = _calculate(
        ("study_strategy", "study_strategy", "other", "other", "other")
    )

    assert result.matched_count == 2
    assert result.mra == 0.4


def test_duplicate_ground_truth_categories_are_rejected() -> None:
    with pytest.raises(EvaluationCalculationError, match="duplicate"):
        _calculate(("study_strategy",), ("study_strategy", "study_strategy"))


def test_empty_ground_truth_is_rejected() -> None:
    with pytest.raises(EvaluationCalculationError, match="must not be empty"):
        _calculate(("study_strategy",), ())


@pytest.mark.parametrize("top_k", [0, 4, 6])
def test_any_top_k_other_than_five_is_rejected(top_k: int) -> None:
    with pytest.raises((EvaluationCalculationError, EvaluationSchemaError)):
        calculate_mra(
            identity=_identity(),
            evaluation_mode="primary",
            primary_eligible=True,
            included_in_final_research_aggregation=True,
            selected_memories=_memories("study_strategy"),
            expected_categories=("study_strategy",),
            candidate_count=1,
            top_k=top_k,
        )


def test_more_than_five_selected_memories_are_rejected() -> None:
    with pytest.raises(
        (EvaluationCalculationError, EvaluationSchemaError),
        match="Top-K",
    ):
        _calculate(("study_strategy",) * 6)


def test_candidate_count_cannot_be_smaller_than_retrieved_count() -> None:
    with pytest.raises(EvaluationCalculationError, match="candidate_count"):
        calculate_mra(
            identity=_identity(),
            evaluation_mode="primary",
            primary_eligible=True,
            included_in_final_research_aggregation=True,
            selected_memories=_memories("study_strategy", "academic_focus"),
            expected_categories=("study_strategy",),
            candidate_count=1,
            top_k=5,
        )


def test_repeated_calculation_is_deterministic() -> None:
    categories = ("study_strategy", "other", "academic_focus")

    assert _calculate(categories) == _calculate(categories)
    assert _calculate(categories).to_dict() == _calculate(categories).to_dict()
