from math import isclose

import pytest

from src.domain import RetrievalWeights
from src.memory import MemoryFactory
from src.retrieval import (
    RECENCY_DECAY_FACTOR,
    MissingEmbeddingError,
    RetrievalScoringError,
    calculate_importance,
    calculate_recency,
    calculate_relevance,
    calculate_weighted_score,
    cosine_similarity,
)


def make_memory(
    *,
    importance: float = 6,
    created_at: int = 0,
    last_accessed_at: int | None = None,
    embedding: tuple[float, ...] | None = (1.0, 0.0),
):
    return MemoryFactory("teacher").create_memory(
        memory_type="observation",
        text="Ms. Sunday observed the class.",
        category="teacher_guidance",
        importance=importance,
        created_at=created_at,
        last_accessed_at=last_accessed_at,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        embedding=embedding,
    )


def test_recency_uses_the_approved_public_decay_constant() -> None:
    assert RECENCY_DECAY_FACTOR == 0.99
    assert calculate_recency(make_memory(last_accessed_at=5), current_time=5) == 1.0
    assert calculate_recency(make_memory(last_accessed_at=5), current_time=6) == 0.99


def test_larger_delta_t_produces_lower_recency() -> None:
    memory = make_memory(last_accessed_at=2)

    delta_one = calculate_recency(memory, current_time=3)
    delta_five = calculate_recency(memory, current_time=7)

    assert delta_five == pytest.approx(0.99**5)
    assert delta_five < delta_one


def test_recency_uses_last_accessed_at_instead_of_created_at() -> None:
    recently_accessed = make_memory(created_at=0, last_accessed_at=9)
    never_reaccessed = make_memory(created_at=0, last_accessed_at=0)

    assert calculate_recency(recently_accessed, current_time=10) == pytest.approx(0.99)
    assert calculate_recency(recently_accessed, current_time=10) > calculate_recency(
        never_reaccessed,
        current_time=10,
    )


def test_recency_rejects_current_time_before_last_access() -> None:
    with pytest.raises(RetrievalScoringError, match="must not be earlier"):
        calculate_recency(make_memory(last_accessed_at=5), current_time=4)


def test_importance_uses_the_stored_memory_value() -> None:
    assert calculate_importance(make_memory(importance=8.5)) == 8.5


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((1.0, 0.0), (1.0, 0.0), 1.0),
        ((1.0, 0.0), (0.0, 1.0), 0.0),
        ((1.0, 0.0), (-1.0, 0.0), -1.0),
    ],
)
def test_cosine_similarity_known_vectors(
    left: tuple[float, ...],
    right: tuple[float, ...],
    expected: float,
) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


def test_relevance_uses_the_memory_embedding() -> None:
    memory = make_memory(embedding=(0.0, 1.0))

    assert calculate_relevance(memory, query_embedding=(0.0, 2.0)) == pytest.approx(
        1.0
    )


def test_cosine_similarity_exposes_clear_vector_errors() -> None:
    with pytest.raises(MissingEmbeddingError, match="memory_embedding is required"):
        cosine_similarity((1.0, 0.0), None)
    with pytest.raises(RetrievalScoringError, match="equal dimensions"):
        cosine_similarity((1.0, 0.0), (1.0,))
    with pytest.raises(RetrievalScoringError, match="zero vector"):
        cosine_similarity((1.0, 0.0), (0.0, 0.0))
    with pytest.raises(RetrievalScoringError, match="finite"):
        cosine_similarity((1.0, float("nan")), (1.0, 0.0))


def test_weighted_score_uses_only_the_three_approved_terms() -> None:
    weights = RetrievalWeights(recency=0.33, importance=0.33, relevance=0.33)
    result = calculate_weighted_score(
        normalized_recency=1.0,
        normalized_importance=0.5,
        normalized_relevance=0.0,
        weights=weights,
    )

    expected = 0.33 * 1.0 + 0.33 * 0.5 + 0.33 * 0.0
    assert isclose(result, expected, rel_tol=0.0, abs_tol=1e-12)
    assert calculate_weighted_score(
        normalized_recency=1.0,
        normalized_importance=1.0,
        normalized_relevance=1.0,
        weights=weights,
    ) == pytest.approx(0.99)
