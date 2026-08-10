"""Fixed Top-K Memory Retrieval Accuracy calculation."""

from __future__ import annotations

from src.logging_system import FIXED_TOP_K

from src.evaluation.errors import EvaluationCalculationError
from src.evaluation.readers import LoadedEvaluationRun
from src.evaluation.schemas import (
    EvaluationUnitIdentity,
    MRAEvaluationResult,
    SelectedMemoryEvidence,
)


def calculate_mra(
    *,
    identity: EvaluationUnitIdentity,
    evaluation_mode: str,
    primary_eligible: bool,
    included_in_final_research_aggregation: bool,
    selected_memories: tuple[SelectedMemoryEvidence, ...],
    expected_categories: tuple[str, ...],
    candidate_count: int,
    top_k: int = FIXED_TOP_K,
) -> MRAEvaluationResult:
    """Calculate exact per-memory category membership divided by fixed K=5."""
    if not isinstance(identity, EvaluationUnitIdentity):
        raise EvaluationCalculationError("identity must be EvaluationUnitIdentity")
    if top_k != FIXED_TOP_K:
        raise EvaluationCalculationError(
            f"MRA top_k must equal the fixed research value {FIXED_TOP_K}"
        )
    if not isinstance(selected_memories, tuple) or any(
        not isinstance(item, SelectedMemoryEvidence) for item in selected_memories
    ):
        raise EvaluationCalculationError(
            "selected_memories must be a tuple of SelectedMemoryEvidence values"
        )
    if len(selected_memories) > FIXED_TOP_K:
        raise EvaluationCalculationError("selected memories exceed fixed Top-K")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < len(selected_memories)
    ):
        raise EvaluationCalculationError(
            "candidate_count must be an integer at least retrieved_count"
        )
    if not isinstance(expected_categories, tuple) or not expected_categories:
        raise EvaluationCalculationError("expected_categories must not be empty")
    if any(
        not isinstance(category, str) or not category.strip()
        for category in expected_categories
    ):
        raise EvaluationCalculationError(
            "expected_categories must contain non-empty strings"
        )
    if len(expected_categories) != len(set(expected_categories)):
        raise EvaluationCalculationError(
            "expected_categories must not contain duplicate categories"
        )
    selected_ids = tuple(memory.memory_id for memory in selected_memories)
    if len(selected_ids) != len(set(selected_ids)):
        raise EvaluationCalculationError("selected memory IDs must be unique")
    ranks = tuple(memory.retrieval_rank for memory in selected_memories)
    if ranks != tuple(range(1, len(selected_memories) + 1)):
        raise EvaluationCalculationError(
            "selected memory ranks must preserve one-based retrieval order"
        )

    expected = frozenset(expected_categories)
    matched = tuple(
        memory.memory_id
        for memory in selected_memories
        if memory.category in expected
    )
    unmatched = tuple(
        memory.memory_id
        for memory in selected_memories
        if memory.category not in expected
    )
    matched_count = len(matched)
    return MRAEvaluationResult(
        identity=identity,
        evaluation_mode=evaluation_mode,
        primary_eligible=primary_eligible,
        included_in_final_research_aggregation=(
            included_in_final_research_aggregation
        ),
        top_k=FIXED_TOP_K,
        candidate_count=candidate_count,
        selected_memories=selected_memories,
        expected_categories=expected_categories,
        matched_memory_ids=matched,
        unmatched_memory_ids=unmatched,
        matched_count=matched_count,
        denominator=FIXED_TOP_K,
        mra=matched_count / FIXED_TOP_K,
    )


def calculate_run_mra(
    loaded_run: LoadedEvaluationRun,
) -> tuple[MRAEvaluationResult, ...]:
    """Calculate target-event-only MRA in validated timeline/participant order."""
    if not isinstance(loaded_run, LoadedEvaluationRun):
        raise EvaluationCalculationError("loaded_run must be LoadedEvaluationRun")
    return tuple(
        calculate_mra(
            identity=unit.identity,
            evaluation_mode=loaded_run.evaluation_mode,
            primary_eligible=loaded_run.primary_eligible,
            included_in_final_research_aggregation=(
                loaded_run.included_in_final_research_aggregation
            ),
            selected_memories=unit.selected_memories,
            expected_categories=unit.expected_categories,
            candidate_count=unit.candidate_count,
            top_k=unit.top_k,
        )
        for unit in loaded_run.target_units
    )


__all__ = ["calculate_mra", "calculate_run_mra"]
