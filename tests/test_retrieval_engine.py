from dataclasses import replace

import pytest

from src.config_loader import load_experiment_config
from src.domain import RetrievalConfig, RetrievalWeights
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import (
    MissingEmbeddingError,
    RetrievalEngine,
    RetrievalEngineError,
)


def loaded_engine_and_config(
    config_id: str = "balanced",
) -> tuple[RetrievalEngine, RetrievalConfig]:
    experiment = load_experiment_config()
    retrieval_config = next(
        item for item in experiment.retrieval_configs if item.config_id == config_id
    )
    return RetrievalEngine(experiment.settings), retrieval_config


def build_stream(count: int, *, run_id: str = "run_01") -> MemoryStream:
    factory = MemoryFactory("academic_student", run_id=run_id)
    stream = MemoryStream("academic_student", run_id=run_id)
    for index in range(count):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"Academic memory {index + 1}.",
                category="academic_focus",
                importance=min(index + 1, 10),
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=(1.0, float(index + 1)),
            )
        )
    return stream


def retrieve(
    engine: RetrievalEngine,
    retrieval_config: RetrievalConfig,
    stream: MemoryStream,
    *,
    current_time: int = 10,
    top_k: int | None = None,
):
    return engine.retrieve(
        memory_stream=stream,
        query="How should the student prepare?",
        query_embedding=(1.0, 1.0),
        retrieval_config=retrieval_config,
        current_time=current_time,
        top_k=top_k,
    )


def test_result_contains_complete_ranked_audit_metadata() -> None:
    engine, config = loaded_engine_and_config()
    result = retrieve(engine, config, build_stream(6))

    assert result.query == "How should the student prepare?"
    assert result.agent_id == "academic_student"
    assert result.retrieval_config_id == "balanced"
    assert result.applied_weights == config.weights
    assert result.weights == config.weights
    assert result.top_k == engine.configured_top_k
    assert result.candidate_count == 6
    assert len(result.candidates) == 6
    assert len(result.selected_memory_ids) == 5
    assert result.selected_top_k_memory_ids == result.selected_memory_ids
    assert tuple(memory.memory_id for memory in result.selected_memories) == (
        result.selected_memory_ids
    )
    assert all(candidate.selected for candidate in result.candidates[:5])
    assert result.candidates[5].selected is False
    assert all(
        0.0 <= component <= 1.0
        for candidate in result.candidates
        for component in (
            candidate.normalized_recency,
            candidate.normalized_importance,
            candidate.normalized_relevance,
        )
    )


def test_balanced_configuration_produces_a_deterministic_ranking() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_stream(6)

    first = retrieve(engine, config, stream)
    second = retrieve(engine, config, stream)

    assert first == second
    assert [candidate.final_score for candidate in first.candidates] == sorted(
        (candidate.final_score for candidate in first.candidates),
        reverse=True,
    )


def test_top_k_comes_from_validated_settings_and_selects_exactly_that_many() -> None:
    engine, config = loaded_engine_and_config()
    result = retrieve(engine, config, build_stream(7))

    assert engine.configured_top_k == load_experiment_config().settings.top_k
    assert result.top_k == load_experiment_config().settings.top_k
    assert len(result.selected_memory_ids) == result.top_k


def test_requested_top_k_must_match_validated_settings() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_stream(6)

    with pytest.raises(RetrievalEngineError, match=r"settings\.top_k"):
        retrieve(engine, config, stream, top_k=4)
    with pytest.raises(RetrievalEngineError, match="top_k must be an integer"):
        retrieve(engine, config, stream, top_k=True)


def test_fewer_than_k_returns_every_available_memory() -> None:
    engine, config = loaded_engine_and_config()
    result = retrieve(engine, config, build_stream(3))

    assert result.top_k == engine.configured_top_k
    assert result.candidate_count == 3
    assert len(result.selected_memory_ids) == 3
    assert all(candidate.selected for candidate in result.candidates)


def test_empty_stream_returns_an_empty_auditable_result() -> None:
    engine, config = loaded_engine_and_config()
    stream = MemoryStream("academic_student", run_id="run_01")
    result = retrieve(engine, config, stream)

    assert result.candidate_count == 0
    assert result.candidates == ()
    assert result.selected_memory_ids == ()
    assert result.retrieval_config_id == config.config_id


def two_memory_stream(
    *,
    first_importance: float,
    second_importance: float,
    first_created_at: int,
    second_created_at: int,
    first_embedding: tuple[float, ...],
    second_embedding: tuple[float, ...],
    last_accessed_at: int,
    reverse_add: bool = False,
) -> MemoryStream:
    factory = MemoryFactory("teacher", run_id="tie_run")
    stream = MemoryStream("teacher", run_id="tie_run")
    first = factory.create_memory(
        memory_type="observation",
        text="First tied memory.",
        category="teacher_guidance",
        importance=first_importance,
        created_at=first_created_at,
        last_accessed_at=last_accessed_at,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        embedding=first_embedding,
    )
    second = factory.create_memory(
        memory_type="observation",
        text="Second tied memory.",
        category="classroom_norms",
        importance=second_importance,
        created_at=second_created_at,
        last_accessed_at=last_accessed_at,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        embedding=second_embedding,
    )
    for memory in ((second, first) if reverse_add else (first, second)):
        stream.add_memory(memory)
    return stream


def tie_engine_and_config() -> tuple[RetrievalEngine, RetrievalConfig]:
    experiment = load_experiment_config()
    return RetrievalEngine(experiment.settings), RetrievalConfig(
        config_id="tie_test",
        weights=RetrievalWeights(recency=1.0, importance=0.0, relevance=0.0),
    )


def test_tie_breaks_first_by_higher_raw_relevance() -> None:
    engine, config = tie_engine_and_config()
    stream = two_memory_stream(
        first_importance=5,
        second_importance=5,
        first_created_at=1,
        second_created_at=1,
        first_embedding=(1.0, 0.0),
        second_embedding=(0.0, 1.0),
        last_accessed_at=1,
    )

    result = engine.retrieve(
        memory_stream=stream,
        query="query",
        query_embedding=(0.0, 1.0),
        retrieval_config=config,
        current_time=2,
    )

    assert result.candidates[0].memory.text == "Second tied memory."


def test_tie_breaks_next_by_higher_raw_importance() -> None:
    engine, config = tie_engine_and_config()
    stream = two_memory_stream(
        first_importance=4,
        second_importance=8,
        first_created_at=1,
        second_created_at=1,
        first_embedding=(1.0, 0.0),
        second_embedding=(1.0, 0.0),
        last_accessed_at=1,
    )

    result = engine.retrieve(
        memory_stream=stream,
        query="query",
        query_embedding=(1.0, 0.0),
        retrieval_config=config,
        current_time=2,
    )

    assert result.candidates[0].raw_importance == 8


def test_tie_breaks_next_by_newer_creation_timestamp() -> None:
    engine, config = tie_engine_and_config()
    stream = two_memory_stream(
        first_importance=5,
        second_importance=5,
        first_created_at=1,
        second_created_at=2,
        first_embedding=(1.0, 0.0),
        second_embedding=(1.0, 0.0),
        last_accessed_at=2,
    )

    result = engine.retrieve(
        memory_stream=stream,
        query="query",
        query_embedding=(1.0, 0.0),
        retrieval_config=config,
        current_time=3,
    )

    assert result.candidates[0].memory.created_at == 2


def test_completely_unresolved_tie_uses_memory_id_not_append_order() -> None:
    engine, config = tie_engine_and_config()
    stream = two_memory_stream(
        first_importance=5,
        second_importance=5,
        first_created_at=1,
        second_created_at=1,
        first_embedding=(1.0, 0.0),
        second_embedding=(1.0, 0.0),
        last_accessed_at=1,
        reverse_add=True,
    )

    result = engine.retrieve(
        memory_stream=stream,
        query="query",
        query_embedding=(1.0, 0.0),
        retrieval_config=config,
        current_time=2,
    )

    assert [candidate.memory_id for candidate in result.candidates] == [
        "mem_teacher_000001",
        "mem_teacher_000002",
    ]


def test_retrieval_does_not_update_access_before_result_is_available() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_stream(6)
    before = stream.list_memories()

    result = retrieve(engine, config, stream)

    assert stream.list_memories() == before
    stream.update_last_accessed_at(result.selected_memory_ids, accessed_at=10)
    after = stream.list_memories()
    assert all(
        memory.last_accessed_at == 10
        for memory in after
        if memory.memory_id in result.selected_memory_ids
    )
    assert all(
        memory.last_accessed_at == before[index].last_accessed_at
        for index, memory in enumerate(after)
        if memory.memory_id not in result.selected_memory_ids
    )


def test_missing_memory_embedding_fails_instead_of_using_a_fallback() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_stream(1)
    memory = replace(stream.list_memories()[0], embedding=None)
    missing_stream = MemoryStream("academic_student", run_id="run_01")
    missing_stream.add_memory(memory)

    with pytest.raises(MissingEmbeddingError, match="memory_embedding is required"):
        retrieve(engine, config, missing_stream)


def build_mixed_category_stream(
    categories: tuple[str, ...], *, run_id: str = "run_01"
) -> MemoryStream:
    factory = MemoryFactory("academic_student", run_id=run_id)
    stream = MemoryStream("academic_student", run_id=run_id)
    for index, category in enumerate(categories):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"Academic memory {index + 1}.",
                category=category,
                importance=10,
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=(1.0, 1.0),
            )
        )
    return stream


def test_top_k_selection_deduplicates_categories_first() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_mixed_category_stream(
        ("study_strategy", "study_strategy", "study_strategy", "exam_concern", "social_learning")
    )

    result = retrieve(engine, config, stream)

    selected = tuple(
        candidate.memory.category for candidate in result.candidates if candidate.selected
    )
    assert len(selected) == engine.configured_top_k
    unique_in_rank_order = list(dict.fromkeys(selected))
    assert selected[: len(unique_in_rank_order)] == tuple(unique_in_rank_order)
    assert len(unique_in_rank_order) == 3
    assert selected_memory_ids_preserve_rank_order(result)


def selected_memory_ids_preserve_rank_order(result) -> bool:
    rank_position = {
        candidate.memory_id: index for index, candidate in enumerate(result.candidates)
    }
    positions = [rank_position[memory_id] for memory_id in result.selected_memory_ids]
    return positions == sorted(positions)


def test_duplicate_backfill_fills_remaining_slots_in_rank_order() -> None:
    engine, config = loaded_engine_and_config()
    stream = build_mixed_category_stream(
        ("study_strategy", "study_strategy", "study_strategy", "study_strategy", "study_strategy", "study_strategy")
    )

    result = retrieve(engine, config, stream)

    assert len(result.selected_memory_ids) == engine.configured_top_k
    rank_position = {
        candidate.memory_id: index for index, candidate in enumerate(result.candidates)
    }
    positions = [rank_position[memory_id] for memory_id in result.selected_memory_ids]
    assert positions == sorted(positions)
    flagged = tuple(
        candidate.memory_id for candidate in result.candidates if candidate.selected
    )
    assert flagged == result.selected_memory_ids
