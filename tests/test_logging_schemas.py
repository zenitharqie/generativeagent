from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from math import inf, nan

import pytest

from src.cognition import CognitivePipeline, CognitiveTurnInput
from src.config_loader import load_experiment_config
from src.llm import DummyEmbeddingProvider, DummyLLMClient, LLMCallType
import src.logging_system as logging_system
from src.logging_system.schemas import (
    APIUsageLogRecord,
    BehaviorLogRecord,
    ErrorLogRecord,
    LogSchemaError,
    MemoryLogRecord,
    RetrievalCandidateRecord,
    RetrievalWeightsRecord,
    RunLogMetadata,
    SCHEMA_VERSION,
    SerializationError,
    StepLogMetadata,
    TurnLogMetadata,
    cognitive_records_from_result,
    index_memory_records,
    memory_record_from_entry,
    record_to_dict,
    retrieval_record_from_result,
    serialize_record,
    turn_metadata_from_result,
    validate_candidate_memory_categories,
)
from src.memory import MemoryFactory, MemoryStream


RUN_ID = "phase_07_run"
QUERY_EMBEDDING = (1.0, 1.0)


def test_logging_package_exports_the_phase_07_public_api() -> None:
    expected_names = {
        "JsonlLogger",
        "RunSummary",
        "RunLogMetadata",
        "RetrievalLogRecord",
        "BehaviorLogRecord",
        "MemoryLogRecord",
        "ReflectionLogRecord",
        "APIUsageLogRecord",
        "ErrorLogRecord",
        "cognitive_records_from_result",
    }

    assert expected_names <= set(logging_system.__all__)
    assert "copy_config_snapshot" not in logging_system.__all__


def run_metadata(
    *,
    run_id: str = RUN_ID,
    retrieval_config_id: str = "balanced",
) -> RunLogMetadata:
    return RunLogMetadata(
        experiment_id="classroom_experiment",
        run_id=run_id,
        retrieval_config_id=retrieval_config_id,
        repetition_index=0,
    )


def completed_turn(
    *,
    reflection_requested: bool = False,
    target_event: bool = False,
):
    agent_id = "academic_student" if target_event else "teacher"
    factory = MemoryFactory(agent_id, run_id=RUN_ID)
    stream = MemoryStream(agent_id, run_id=RUN_ID)
    for index in range(6):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"Memori kelas ke-{index + 1} untuk siswa.",
                category=("academic_focus" if index % 2 else "classroom_norms"),
                importance=float(index + 1),
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=(1.0, float(index + 1)),
                metadata={"source": "uji", "nested": {"index": index}},
            )
        )
    client = DummyLLMClient()
    provider = DummyEmbeddingProvider(QUERY_EMBEDDING)
    pipeline = CognitivePipeline(
        config=load_experiment_config(),
        llm_client=client,
        embedding_provider=provider,
    )
    source_step_id = "day_02_target_01" if target_event else "day_01_natural_01"
    result = pipeline.run_turn(
        CognitiveTurnInput(
            agent_id=agent_id,
            source_step_id=source_step_id,
            turn_id="turn_001",
            retrieval_config_id="balanced",
            memory_stream=stream,
            logical_timestamp=20,
            query_embedding=QUERY_EMBEDDING,
            conversation_context=("Konteks kelas tetap.",),
            event_id="event_01" if target_event else None,
            reflection_requested=reflection_requested,
        )
    )
    turn = turn_metadata_from_result(
        result,
        run=run_metadata(),
        turn_index=0,
        evaluation_enabled=target_event,
    )
    return result, turn, client, provider


def test_run_and_turn_metadata_are_immutable_and_flatten_canonical_fields() -> None:
    result, turn, _, _ = completed_turn(target_event=True)

    assert turn.to_dict() == {
        "experiment_id": "classroom_experiment",
        "run_id": RUN_ID,
        "retrieval_config_id": "balanced",
        "repetition_index": 0,
        "step_id": result.source_step_id,
        "step_type": "target_event",
        "is_target_event": True,
        "evaluation_enabled": True,
        "day": 2,
        "period": "afternoon",
        "timestamp": 20,
        "event_id": "event_01",
        "turn_id": "turn_001",
        "turn_index": 0,
        "agent_id": "academic_student",
    }
    with pytest.raises(FrozenInstanceError):
        turn.turn_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("experiment_id", "../escape"),
        ("run_id", "/absolute"),
        ("retrieval_config_id", "bad/config"),
        ("experiment_id", "contains.dot"),
    ],
)
def test_run_metadata_rejects_unsafe_identifiers(
    field_name: str,
    value: str,
) -> None:
    values = {
        "experiment_id": "experiment",
        "run_id": RUN_ID,
        "retrieval_config_id": "balanced",
        "repetition_index": 0,
    }
    values[field_name] = value

    with pytest.raises(LogSchemaError, match="safe identifier"):
        RunLogMetadata(**values)


def test_run_metadata_rejects_unknown_retrieval_configuration() -> None:
    with pytest.raises(LogSchemaError, match="unsupported retrieval_config_id"):
        RunLogMetadata("experiment", RUN_ID, "unknown_condition", 0)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_repetition_index_must_be_a_non_negative_integer(value: object) -> None:
    with pytest.raises(LogSchemaError, match="repetition_index"):
        RunLogMetadata("experiment", RUN_ID, "balanced", value)  # type: ignore[arg-type]


def test_step_metadata_validates_type_flags_event_and_logical_timestamp() -> None:
    base = {
        "run": run_metadata(),
        "step_id": "day_01_natural_01",
        "step_type": "natural_interaction_block",
        "is_target_event": False,
        "evaluation_enabled": False,
        "day": 1,
        "period": "morning",
        "timestamp": 0,
    }
    assert StepLogMetadata(**base).timestamp == 0

    with pytest.raises(LogSchemaError, match="is_target_event"):
        StepLogMetadata(**{**base, "is_target_event": True})
    with pytest.raises(LogSchemaError, match="evaluation_enabled"):
        StepLogMetadata(**{**base, "evaluation_enabled": True})
    with pytest.raises(LogSchemaError, match="timestamp"):
        StepLogMetadata(**{**base, "timestamp": -1})
    with pytest.raises(LogSchemaError, match="6-day timeline"):
        StepLogMetadata(**{**base, "day": 7})
    with pytest.raises(LogSchemaError, match="event_id"):
        StepLogMetadata(**{**base, "event_id": "event_01"})

    target = {
        **base,
        "step_id": "day_02_target_01",
        "step_type": "target_event",
        "is_target_event": True,
        "evaluation_enabled": True,
    }
    with pytest.raises(LogSchemaError, match="unsupported event_id"):
        StepLogMetadata(**target, event_id="event_99")


def test_turn_metadata_adapter_rejects_result_identity_mismatches() -> None:
    result, turn, _, _ = completed_turn()

    wrong_run = replace(
        turn,
        step=replace(
            turn.step,
            run=run_metadata(retrieval_config_id="recency_heavy"),
        ),
    )
    with pytest.raises(LogSchemaError, match="identity"):
        retrieval_record_from_result(
            result,
            wrong_run,
            query_embedding=QUERY_EMBEDDING,
        )

    wrong_period = replace(turn, step=replace(turn.step, period="afternoon"))
    with pytest.raises(LogSchemaError, match="perception"):
        retrieval_record_from_result(
            result,
            wrong_period,
            query_embedding=QUERY_EMBEDDING,
        )


def test_retrieval_projection_preserves_full_score_audit_and_ranking() -> None:
    result, turn, _, _ = completed_turn()
    record = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )
    data = record_to_dict(record)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["query"] == result.retrieval_result.query
    assert data["query_embedding"] == [1.0, 1.0]
    assert data["weights"] == {
        "recency": result.retrieval_result.applied_weights.recency,
        "importance": result.retrieval_result.applied_weights.importance,
        "relevance": result.retrieval_result.applied_weights.relevance,
    }
    assert data["top_k"] == result.retrieval_result.top_k
    assert data["candidate_count"] == len(data["candidates"]) == 6
    assert [item["memory_id"] for item in data["candidates"]] == [
        item.memory_id for item in result.retrieval_result.candidates
    ]
    assert data["selected_memory_ids"] == list(result.selected_memory_ids)
    assert data["accessed_memory_ids"] == list(result.accessed_memory_ids)

    for source, persisted in zip(
        result.retrieval_result.candidates,
        data["candidates"],
    ):
        assert persisted == {
            "memory_id": source.memory_id,
            "category": source.memory.category,
            "raw_recency": source.raw_recency,
            "raw_importance": source.raw_importance,
            "raw_relevance": source.raw_relevance,
            "normalized_recency": source.normalized_recency,
            "normalized_importance": source.normalized_importance,
            "normalized_relevance": source.normalized_relevance,
            "final_score": source.final_score,
            "selected": source.selected,
        }
        assert "rank" not in persisted
        assert "text" not in persisted
        assert "embedding" not in persisted
        assert not any(key.startswith("norm_") for key in persisted)


def test_retrieval_record_rejects_noncanonical_top_k_or_weights() -> None:
    result, turn, _, _ = completed_turn()
    record = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )

    with pytest.raises(LogSchemaError, match="fixed research value 5"):
        replace(record, top_k=4)
    with pytest.raises(LogSchemaError, match="sum to 1.0"):
        RetrievalWeightsRecord(recency=0.2, importance=0.2, relevance=0.2)
    with pytest.raises(LogSchemaError, match="retrieval_config_id"):
        replace(
            record,
            weights=RetrievalWeightsRecord(
                recency=0.60,
                importance=0.20,
                relevance=0.20,
            ),
        )


def test_query_embedding_is_explicit_null_when_unavailable() -> None:
    result, turn, _, _ = completed_turn()

    record = retrieval_record_from_result(result, turn, query_embedding=None)

    assert record.query_embedding is None
    assert record_to_dict(record)["query_embedding"] is None


@pytest.mark.parametrize(
    "embedding",
    [(nan, 1.0), (inf, 1.0), (1.0,), ()],
)
def test_query_embedding_rejects_non_finite_empty_or_wrong_dimension_values(
    embedding: tuple[float, ...],
) -> None:
    result, turn, _, _ = completed_turn()

    with pytest.raises(LogSchemaError, match="query_embedding"):
        retrieval_record_from_result(
            result,
            turn,
            query_embedding=embedding,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_recency",
        "raw_importance",
        "raw_relevance",
        "normalized_recency",
        "normalized_importance",
        "normalized_relevance",
        "final_score",
    ],
)
def test_retrieval_candidate_record_rejects_non_finite_scores(
    field_name: str,
) -> None:
    values = {
        "memory_id": "mem_teacher_000001",
        "category": "classroom_norms",
        "raw_recency": 1.0,
        "raw_importance": 5.0,
        "raw_relevance": 0.5,
        "normalized_recency": 1.0,
        "normalized_importance": 0.5,
        "normalized_relevance": 0.5,
        "final_score": 0.5,
        "selected": True,
    }
    values[field_name] = nan

    with pytest.raises(LogSchemaError, match=field_name):
        RetrievalCandidateRecord(**values)


def test_behavior_projection_contains_no_unselected_retrieval_data() -> None:
    result, turn, _, _ = completed_turn()
    records = cognitive_records_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )
    data = record_to_dict(records.behavior)

    assert data["response"] == result.action.response
    assert data["retrieved_memory_ids"] == list(result.selected_memory_ids)
    assert data["plan_summary"] == result.plan.plan
    assert data["evaluation_enabled"] is False
    assert set(data).isdisjoint(
        {
            "query",
            "query_embedding",
            "weights",
            "candidates",
            "memory_to_store",
            "raw_relevance",
            "normalized_relevance",
        }
    )
    assert all(
        candidate.memory.text not in serialize_record(records.behavior)
        for candidate in result.retrieval_result.candidates
    )


def test_memory_projection_preserves_full_text_embedding_and_source_mapping() -> None:
    result, turn, _, _ = completed_turn()
    source = result.created_memories[0]
    source.metadata["mutable"] = {"values": [1, 2]}

    record = memory_record_from_entry(source, run=turn.run, turn=turn)
    source.metadata["mutable"]["values"].append(3)
    data = record_to_dict(record)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["memory_id"] == source.memory_id
    assert data["agent_id"] == source.agent_id
    assert data["memory_type"] == source.memory_type
    assert data["category"] == source.category
    assert data["text"] == source.text
    assert data["importance"] == source.importance
    assert data["created_at"] == source.created_at
    assert data["last_accessed_at"] == source.last_accessed_at
    assert data["step_id"] == source.source_step_id
    assert data["step_type"] == source.source_step_type
    assert data["embedding"] == [1.0, 1.0]
    assert data["metadata"]["mutable"] == {"values": [1, 2]}

    data["embedding"].append(99)
    data["metadata"]["mutable"]["values"].append(99)
    assert record_to_dict(record)["embedding"] == [1.0, 1.0]
    assert record_to_dict(record)["metadata"]["mutable"] == {"values": [1, 2]}


def test_seed_memory_without_embedding_has_explicit_null_and_no_turn_context() -> None:
    factory = MemoryFactory("quiet_student", run_id=RUN_ID)
    memory = factory.create_memory(
        memory_type="seed_persona",
        text="Siswa lebih nyaman mengamati sebelum berbicara.",
        category="quiet_participation",
        importance=8,
        created_at=0,
        source_step_id="seed_memory_initialization",
        source_step_type="agent_initialization",
        embedding=None,
    )

    data = record_to_dict(memory_record_from_entry(memory, run=run_metadata()))

    assert data["embedding"] is None
    assert data["timestamp"] == 0
    assert data["step_type"] == "agent_initialization"
    assert data["turn_id"] is None
    assert data["turn_index"] is None
    assert data["day"] is None
    assert data["period"] is None


@pytest.mark.parametrize("embedding", [(nan,), (inf,)])
def test_memory_projection_rejects_non_finite_embeddings(
    embedding: tuple[float, ...],
) -> None:
    memory = MemoryFactory("teacher", run_id=RUN_ID).create_memory(
        memory_type="observation",
        text="A memory with an invalid persisted vector.",
        category="classroom_norms",
        importance=5,
        created_at=1,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        embedding=embedding,
    )

    with pytest.raises(LogSchemaError, match="embedding"):
        memory_record_from_entry(memory, run=run_metadata())


def test_memory_projection_rejects_cross_run_and_turn_identity() -> None:
    result, turn, _, _ = completed_turn()
    memory = result.created_memories[0]

    with pytest.raises(LogSchemaError, match="different or unspecified run"):
        memory_record_from_entry(memory, run=run_metadata(run_id="other_run"))
    with pytest.raises(LogSchemaError, match="turn metadata"):
        memory_record_from_entry(
            memory,
            run=turn.run,
            turn=replace(turn, agent_id="active_student"),
        )
    with pytest.raises(LogSchemaError, match="require source turn metadata"):
        memory_record_from_entry(memory, run=turn.run)


def test_reflection_and_created_memories_preserve_pipeline_order() -> None:
    result, turn, client, provider = completed_turn(reflection_requested=True)
    calls_before = client.call_types
    embedded_before = provider.received_texts

    records = cognitive_records_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )

    assert [record.memory_type for record in records.memories] == [
        "action",
        "reflection",
    ]
    assert [record.memory_id for record in records.memories] == [
        memory.memory_id for memory in result.created_memories
    ]
    assert records.reflection is not None
    assert records.reflection.reflection == result.reflection.reflection
    assert records.reflection.category == result.reflection.category
    assert records.reflection.memory_id == result.created_memories[1].memory_id
    assert client.call_types == calls_before
    assert provider.received_texts == embedded_before


def test_absent_reflection_produces_no_reflection_record() -> None:
    result, turn, _, _ = completed_turn(reflection_requested=False)

    records = cognitive_records_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )

    assert records.reflection is None
    assert len(records.memories) == 1


def candidate_memory_records(result, turn) -> tuple[MemoryLogRecord, ...]:
    records = []
    for turn_index, candidate in enumerate(result.retrieval_result.candidates):
        memory = candidate.memory
        source_turn = TurnLogMetadata(
            step=StepLogMetadata(
                run=turn.run,
                step_id=memory.source_step_id,
                step_type=memory.source_step_type,
                is_target_event=False,
                evaluation_enabled=False,
                day=1,
                period="morning",
                timestamp=memory.created_at,
            ),
            turn_id=f"source_turn_{turn_index:03d}",
            turn_index=turn_index,
            agent_id=memory.agent_id,
        )
        records.append(
            memory_record_from_entry(memory, run=turn.run, turn=source_turn)
        )
    return tuple(records)


def test_candidate_categories_cross_validate_when_memory_records_are_available() -> None:
    result, turn, _, _ = completed_turn()
    retrieval = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )
    memories = candidate_memory_records(result, turn)

    validate_candidate_memory_categories(retrieval, memories[:1])
    validate_candidate_memory_categories(retrieval, index_memory_records(memories))
    validate_candidate_memory_categories(retrieval, ())


def test_candidate_category_mismatch_is_visible_but_missing_records_are_allowed() -> None:
    result, turn, _, _ = completed_turn()
    retrieval = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )
    first = candidate_memory_records(result, turn)[0]

    with pytest.raises(LogSchemaError, match="category"):
        validate_candidate_memory_categories(
            retrieval,
            (replace(first, category="different_category"),),
        )


def test_candidate_cross_validation_rejects_mismatched_run_metadata() -> None:
    result, turn, _, _ = completed_turn()
    retrieval = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )
    first = candidate_memory_records(result, turn)[0]
    other_run = run_metadata(retrieval_config_id="recency_heavy")
    mismatched = replace(
        first,
        run=other_run,
        turn=replace(first.turn, step=replace(first.turn.step, run=other_run)),
    )

    with pytest.raises(LogSchemaError, match="different run metadata"):
        validate_candidate_memory_categories(retrieval, (mismatched,))


def test_duplicate_memory_index_entries_are_rejected() -> None:
    result, turn, _, _ = completed_turn()
    first = candidate_memory_records(result, turn)[0]

    with pytest.raises(LogSchemaError, match="duplicate memory record"):
        index_memory_records((first, first))

    with pytest.raises(LogSchemaError, match="invalid key"):
        validate_candidate_memory_categories(
            retrieval_record_from_result(
                result,
                turn,
                query_embedding=QUERY_EMBEDDING,
            ),
            {("wrong_experiment", "wrong_run", first.memory_id): first},
        )


def test_memory_index_isolates_reused_run_ids_across_experiments() -> None:
    result, turn, _, _ = completed_turn()
    first = candidate_memory_records(result, turn)[0]
    other_run = RunLogMetadata(
        "other_experiment",
        first.run.run_id,
        first.run.retrieval_config_id,
        first.run.repetition_index,
    )
    other = replace(
        first,
        run=other_run,
        turn=replace(first.turn, step=replace(first.turn.step, run=other_run)),
    )

    index = index_memory_records((first, other))

    assert len(index) == 2


def test_api_usage_schema_is_passive_and_validates_token_accounting() -> None:
    _, turn, client, provider = completed_turn()
    calls_before = client.call_types
    embedded_before = provider.received_texts
    record = APIUsageLogRecord(
        turn=turn,
        call_type=LLMCallType.PLAN,
        model="future-model-id",
        prompt_template="plan_template_v1",
        input_tokens=7,
        output_tokens=3,
        total_tokens=10,
        status="completed",
    )

    data = record_to_dict(record)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["call_type"] == "plan"
    assert data["prompt_template"] == "plan_template_v1"
    assert data["error_type"] is None
    assert client.call_types == calls_before
    assert provider.received_texts == embedded_before
    with pytest.raises(LogSchemaError, match="total_tokens"):
        replace(record, total_tokens=11)
    with pytest.raises(LogSchemaError, match="prompt_template"):
        replace(record, prompt_template="Raw prompt text must not be stored")


def test_error_schema_supports_run_only_and_contextual_failures() -> None:
    _, turn, _, _ = completed_turn()
    run_only = ErrorLogRecord(
        run=turn.run,
        error_type="StorageFailure",
        message="Sanitized write failure.",
        recoverable=False,
    )
    contextual = ErrorLogRecord(
        run=turn.run,
        error_type="TurnFailure",
        message="Sanitized turn failure.",
        recoverable=True,
        retry_count=0,
        context=turn,
    )

    run_data = record_to_dict(run_only)
    context_data = record_to_dict(contextual)

    assert run_data["schema_version"] == SCHEMA_VERSION
    assert run_data["step_id"] is None
    assert run_data["turn_id"] is None
    assert run_data["timestamp"] is None
    assert context_data["step_id"] == turn.step.step_id
    assert context_data["turn_id"] == turn.turn_id
    assert context_data["timestamp"] == 20
    assert context_data["retry_count"] == 0
    with pytest.raises(LogSchemaError, match="retry_count"):
        replace(contextual, retry_count=-1)


def test_records_serialize_deterministically_as_utf8_strict_json() -> None:
    result, turn, _, _ = completed_turn()
    behavior = BehaviorLogRecord(
        turn=turn,
        response="Saya memahami konteks kelas — dengan jelas.",
        retrieved_memory_ids=result.selected_memory_ids,
        plan_summary=None,
    )

    first = serialize_record(behavior)
    second = serialize_record(behavior)

    assert first == second
    assert "Saya memahami" in first
    assert "\\u" not in first
    assert not first.endswith("\n")
    assert json.loads(first) == record_to_dict(behavior)
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_record_to_dict_returns_a_defensive_copy() -> None:
    result, turn, _, _ = completed_turn()
    record = retrieval_record_from_result(
        result,
        turn,
        query_embedding=QUERY_EMBEDDING,
    )

    first = record_to_dict(record)
    first["query_embedding"].append(99)
    first["candidates"][0]["category"] = "changed"

    second = record_to_dict(record)
    assert second["query_embedding"] == [1.0, 1.0]
    assert second["candidates"][0]["category"] != "changed"


def test_unsupported_objects_are_never_generically_serialized() -> None:
    with pytest.raises(SerializationError, match="unsupported persisted record"):
        record_to_dict({"schema_version": SCHEMA_VERSION})  # type: ignore[arg-type]
    with pytest.raises(SerializationError, match="unsupported persisted record"):
        serialize_record(run_metadata())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [nan, inf, object()])
def test_memory_metadata_rejects_non_json_or_non_finite_values(
    bad_value: object,
) -> None:
    memory = MemoryFactory("teacher", run_id=RUN_ID).create_memory(
        memory_type="observation",
        text="A metadata validation memory.",
        category="classroom_norms",
        importance=5,
        created_at=1,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        embedding=(1.0, 1.0),
        metadata={"bad": bad_value},
    )

    with pytest.raises(LogSchemaError, match="metadata"):
        memory_record_from_entry(memory, run=run_metadata())
