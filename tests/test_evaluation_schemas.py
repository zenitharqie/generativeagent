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
    future_bcr_contract_dict,
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


def binary_judgment(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "communication_style_assessment": "consistent",
        "focus_consistency_assessment": "consistent",
        "temporal_consistency_assessment": "inconsistent",
        "consistent": 0,
        "reason": "A substantive temporal contradiction outweighs the other evidence.",
    }
    value.update(changes)
    return value


@pytest.mark.parametrize("consistent", [-1, 2, 5, 0.0, 1.0, True, "1"])
def test_future_bcr_contract_rejects_non_binary_final_values(
    consistent: object,
) -> None:
    with pytest.raises(EvaluationSchemaError):
        FutureBCRJudgment(**binary_judgment(consistent=consistent))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "communication_style_assessment",
        "focus_consistency_assessment",
        "temporal_consistency_assessment",
    ],
)
@pytest.mark.parametrize("assessment", ["", "mixed", "CONSISTENT", 1, None])
def test_future_bcr_contract_rejects_invalid_rubric_assessments(
    field: str,
    assessment: object,
) -> None:
    with pytest.raises(EvaluationSchemaError):
        FutureBCRJudgment(**binary_judgment(**{field: assessment}))  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["", "   ", None, 1])
def test_future_bcr_contract_requires_non_empty_reason(reason: object) -> None:
    with pytest.raises(EvaluationSchemaError):
        FutureBCRJudgment(**binary_judgment(reason=reason))  # type: ignore[arg-type]


def test_future_bcr_contract_uses_holistic_binary_decision_without_majority_rule() -> None:
    judgment = FutureBCRJudgment(
        **binary_judgment(),  # type: ignore[arg-type]
    )

    assert judgment.consistent == 0
    assert judgment.to_dict() == {
        "communication_style_assessment": "consistent",
        "focus_consistency_assessment": "consistent",
        "temporal_consistency_assessment": "inconsistent",
        "consistent": 0,
        "reason": "A substantive temporal contradiction outweighs the other evidence.",
    }


def test_phase09_future_bcr_metadata_describes_only_binary_methodology() -> None:
    contract = future_bcr_contract_dict()

    assert contract == {
        "status": "not_judged_phase_09",
        "decision_type": "binary",
        "allowed_final_values": [0, 1],
        "rubric_dimensions": [
            "communication_style",
            "focus_consistency",
            "temporal_consistency",
        ],
        "rubric_assessment_values": ["consistent", "inconsistent"],
        "final_judgment_field": "consistent",
        "final_metric": "proportion_of_consistent_responses",
    }
    serialized = repr(contract)
    for obsolete in ("score_minimum", "score_maximum", "arithmetic_mean", "_score"):
        assert obsolete not in serialized


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
