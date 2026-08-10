"""Immutable Phase 09 evaluation schema contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.evaluation import (
    BCRJudgeInput,
    BCRModelMetadata,
    FutureBCRJudgment,
    MRAEvaluationResult,
    PersonaJudgeInput,
    PriorBehaviorEvidence,
    SelectedMemoryEvidence,
    TargetEventJudgeInput,
    EvaluationSchemaError,
    EvaluationUnitIdentity,
)


def identity() -> EvaluationUnitIdentity:
    return EvaluationUnitIdentity(
        experiment_id="experiment",
        run_id="balanced_rep01",
        retrieval_config_id="balanced",
        repetition_index=1,
        event_id="event_01",
        step_id="day_02_target_01",
        turn_id="day_02_target_01_turn_01",
        turn_index=1,
        timestamp=37,
        agent_id="teacher",
    )


def selected(rank: int, category: str) -> SelectedMemoryEvidence:
    return SelectedMemoryEvidence(
        retrieval_rank=rank,
        memory_id=f"mem_teacher_{rank:06d}",
        category=category,
        text=f"memory text {rank}",
    )


def test_unit_identity_is_frozen() -> None:
    value = identity()

    with pytest.raises(FrozenInstanceError):
        value.event_id = "event_02"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", ""),
        ("turn_index", 0),
        ("timestamp", -1),
        ("repetition_index", 0),
        ("agent_id", "unknown"),
    ],
)
def test_unit_identity_rejects_invalid_fields(field: str, value: object) -> None:
    arguments = identity().__dict__.copy()
    arguments[field] = value

    with pytest.raises(EvaluationSchemaError):
        EvaluationUnitIdentity(**arguments)


def test_selected_memory_evidence_is_frozen_and_keeps_text_out_of_audit_dict() -> None:
    value = selected(1, "teacher_guidance")

    assert value.to_audit_dict() == {
        "retrieval_rank": 1,
        "memory_id": "mem_teacher_000001",
        "category": "teacher_guidance",
    }
    with pytest.raises(FrozenInstanceError):
        value.category = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("rank", [0, -1, True])
def test_selected_memory_evidence_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(EvaluationSchemaError):
        SelectedMemoryEvidence(
            retrieval_rank=rank,  # type: ignore[arg-type]
            memory_id="mem_teacher_000001",
            category="teacher_guidance",
            text="memory text",
        )


def test_prior_behavior_evidence_serializes_source_and_neutral_views_separately() -> None:
    value = PriorBehaviorEvidence(
        event_id="event_01",
        step_id="day_02_target_01",
        turn_id="day_02_target_01_turn_01",
        turn_index=1,
        timestamp=37,
        event_title="Announcement",
        event_description="The teacher announces a test.",
        response="I will prepare.",
    )

    assert set(value.to_audit_dict()) == {
        "event_id",
        "step_id",
        "turn_id",
        "turn_index",
        "timestamp",
    }
    assert value.to_judge_dict() == {
        "event_title": "Announcement",
        "event_description": "The teacher announces a test.",
        "response": "I will prepare.",
    }


def test_judge_input_has_only_the_approved_behavior_information() -> None:
    judge_input = BCRJudgeInput(
        persona=PersonaJudgeInput(
            display_name="Ms. Sunday",
            role="teacher",
            archetype="calm_classroom_guide",
            persona_summary="A calm teacher.",
            traits=("calm",),
            goals=("help students",),
            communication_style="Clear and patient.",
            motivations=("student understanding",),
        ),
        target_event=TargetEventJudgeInput(
            title="Announcement",
            description="The teacher announces a test.",
        ),
        response="Please prepare carefully.",
        prior_target_behaviors=(),
        selected_memory_texts=("A relevant prior memory.",),
    )

    assert set(judge_input.to_dict()) == {
        "persona",
        "target_event",
        "response",
        "prior_target_behaviors",
        "selected_memory_texts",
    }
    serialized = repr(judge_input.to_dict())
    for forbidden in (
        "retrieval_config_id",
        "memory_id",
        "category",
        "weight",
        "score",
        "mra",
        "ground_truth",
        "expected_categories",
    ):
        assert forbidden not in serialized.lower()


def test_model_metadata_is_explicitly_unavailable_without_inference() -> None:
    metadata = BCRModelMetadata()

    assert metadata.to_dict() == {
        "simulator_model_identity": None,
        "evaluator_model_identity": None,
        "evaluator_prompt_version": None,
        "evaluator_rubric_version": None,
    }


@pytest.mark.parametrize(
    "field",
    [
        "communication_style_score",
        "focus_consistency_score",
        "temporal_consistency_score",
    ],
)
@pytest.mark.parametrize("score", [0, 6, 1.5, True])
def test_future_bcr_contract_rejects_non_integer_or_out_of_range_scores(
    field: str,
    score: object,
) -> None:
    arguments = {
        "communication_style_score": 5,
        "focus_consistency_score": 4,
        "temporal_consistency_score": 3,
        "communication_style_reason": "Matches the persona style.",
        "focus_consistency_reason": "Reflects the stated goal.",
        "temporal_consistency_reason": "Does not contradict prior behavior.",
    }
    arguments[field] = score

    with pytest.raises(EvaluationSchemaError):
        FutureBCRJudgment(**arguments)  # type: ignore[arg-type]


def test_future_bcr_contract_uses_exact_three_dimension_mean_without_decision() -> None:
    judgment = FutureBCRJudgment(
        communication_style_score=5,
        focus_consistency_score=4,
        temporal_consistency_score=3,
        communication_style_reason="Matches the persona style.",
        focus_consistency_reason="Reflects the stated goal.",
        temporal_consistency_reason="Does not contradict prior behavior.",
    )

    assert judgment.bcr == 4.0
    assert set(judgment.to_dict()) == {
        "communication_style_score",
        "focus_consistency_score",
        "temporal_consistency_score",
        "communication_style_reason",
        "focus_consistency_reason",
        "temporal_consistency_reason",
        "bcr",
    }
    assert "decision" not in judgment.to_dict()


def test_mra_result_rejects_inconsistent_partition() -> None:
    memories = (selected(1, "teacher_guidance"),)

    with pytest.raises(EvaluationSchemaError):
        MRAEvaluationResult(
            identity=identity(),
            evaluation_mode="primary",
            primary_eligible=True,
            included_in_final_research_aggregation=True,
            top_k=5,
            candidate_count=1,
            selected_memories=memories,
            expected_categories=("teacher_guidance",),
            matched_memory_ids=(),
            unmatched_memory_ids=(),
            matched_count=0,
            denominator=5,
            mra=0.0,
        )


def test_mra_result_evaluation_eligibility_fields_are_frozen() -> None:
    value = MRAEvaluationResult(
        identity=identity(),
        evaluation_mode="primary",
        primary_eligible=True,
        included_in_final_research_aggregation=True,
        top_k=5,
        candidate_count=1,
        selected_memories=(selected(1, "teacher_guidance"),),
        expected_categories=("teacher_guidance",),
        matched_memory_ids=("mem_teacher_000001",),
        unmatched_memory_ids=(),
        matched_count=1,
        denominator=5,
        mra=0.2,
    )

    with pytest.raises(FrozenInstanceError):
        value.primary_eligible = False  # type: ignore[misc]


@pytest.mark.parametrize(
    (
        "evaluation_mode",
        "primary_eligible",
        "included_in_final_research_aggregation",
    ),
    [
        ("unknown", True, True),
        ("primary", False, True),
        ("primary", True, False),
        ("diagnostic", True, False),
        ("diagnostic", False, True),
    ],
)
def test_mra_result_rejects_inconsistent_evaluation_eligibility(
    evaluation_mode: str,
    primary_eligible: bool,
    included_in_final_research_aggregation: bool,
) -> None:
    memories = (selected(1, "teacher_guidance"),)

    with pytest.raises(EvaluationSchemaError):
        MRAEvaluationResult(
            identity=identity(),
            evaluation_mode=evaluation_mode,
            primary_eligible=primary_eligible,
            included_in_final_research_aggregation=(
                included_in_final_research_aggregation
            ),
            top_k=5,
            candidate_count=1,
            selected_memories=memories,
            expected_categories=("teacher_guidance",),
            matched_memory_ids=("mem_teacher_000001",),
            unmatched_memory_ids=(),
            matched_count=1,
            denominator=5,
            mra=0.2,
        )
