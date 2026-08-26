from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from src.cognition import (
    CognitiveInputError,
    CognitivePipeline,
    CognitiveTurnInput,
)
from src.config_loader import load_experiment_config
from src.llm import (
    DummyEmbeddingProvider,
    DummyLLMClient,
    LLMCallType,
    LLMOutputValidationError,
)
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import MissingEmbeddingError, RetrievalScoringError


def build_stream(
    agent_id: str = "teacher",
    *,
    count: int = 6,
    run_id: str = "phase_06_run",
) -> MemoryStream:
    stream = MemoryStream(agent_id, run_id=run_id)
    factory = MemoryFactory(agent_id, run_id=run_id)
    for index in range(count):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"{agent_id} private memory {index + 1}.",
                category="classroom_norms",
                importance=float(index + 1),
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=(1.0, float(index + 1)),
            )
        )
    return stream


def make_pipeline(
    *,
    scripted_responses: Mapping | None = None,
    embedding_provider: object | None = None,
) -> tuple[CognitivePipeline, DummyLLMClient, object]:
    client = DummyLLMClient(scripted_responses)
    provider = embedding_provider or DummyEmbeddingProvider((1.0, 1.0))
    pipeline = CognitivePipeline(
        config=load_experiment_config(),
        llm_client=client,
        embedding_provider=provider,
    )
    return pipeline, client, provider


def turn_input(
    stream: MemoryStream,
    *,
    source_step_id: str = "day_01_natural_01",
    event_id: str | None = None,
    reflection_requested: bool = False,
    previous_same_agent_response: str | None = None,
    logical_timestamp: int = 20,
    retrieval_config_id: str = "balanced",
) -> CognitiveTurnInput:
    return CognitiveTurnInput(
        agent_id=stream.agent_id,
        source_step_id=source_step_id,
        turn_id="turn_001",
        retrieval_config_id=retrieval_config_id,
        memory_stream=stream,
        logical_timestamp=logical_timestamp,
        query_embedding=(1.0, 1.0),
        conversation_context=(
            "Monday asked how the class should participate.",
            "Tuesday listened before responding.",
        ),
        event_id=event_id,
        previous_same_agent_response=previous_same_agent_response,
        reflection_requested=reflection_requested,
    )


def test_one_dummy_turn_is_auditable_and_stores_only_action_memory() -> None:
    stream = build_stream()
    initial = stream.list_memories()
    pipeline, client, provider = make_pipeline()

    result = pipeline.run_turn(turn_input(stream))

    assert result.agent_id == "teacher"
    assert result.source_step_id == "day_01_natural_01"
    assert result.observation.startswith("Classroom block:")
    assert result.plan.plan
    assert result.response == "This is a deterministic dummy classroom response."
    assert result.memory_to_store == (
        "The agent gave a deterministic dummy classroom response."
    )
    assert result.response != result.memory_to_store
    assert result.reflection is None
    assert result.selected_memory_ids == result.retrieval_result.selected_memory_ids
    assert result.accessed_memory_ids == result.selected_memory_ids
    assert result.created_memories == (stream.list_memories()[-1],)

    action_memory = result.created_memories[0]
    assert action_memory.memory_type == "action"
    assert action_memory.text == result.action.memory_to_store
    assert action_memory.source_step_id == result.source_step_id
    assert action_memory.source_step_type == result.source_step_type
    assert action_memory.embedding == (1.0, 1.0)
    assert all(
        memory.memory_type not in {"observation", "plan"}
        for memory in stream.list_memories()[len(initial) :]
    )
    assert provider.received_texts == (result.action.memory_to_store,)
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.NATURAL_RESPONSE,
        LLMCallType.IMPORTANCE_SCORE,
    )

    initial_by_id = {memory.memory_id: memory for memory in initial}
    stored_by_id = {memory.memory_id: memory for memory in stream.list_memories()}
    assert all(
        stored_by_id[memory_id].last_accessed_at == 20
        for memory_id in result.selected_memory_ids
    )
    assert all(
        stored_by_id[memory_id].last_accessed_at
        == initial_by_id[memory_id].last_accessed_at
        for memory_id in initial_by_id.keys() - set(result.selected_memory_ids)
    )
    assert result.retrieval_result.candidates[0].memory.last_accessed_at < 20
    with pytest.raises(FrozenInstanceError):
        result.turn_id = "changed"  # type: ignore[misc]


def test_reflection_runs_only_when_explicitly_requested_and_is_stored_once() -> None:
    stream = build_stream()
    pipeline, client, provider = make_pipeline(
        scripted_responses={
            LLMCallType.PLAN: '{"plan":"A short auditable plan."}',
            LLMCallType.NATURAL_RESPONSE: (
                '{"response":"I will explain the expectation.",'
                '"memory_to_store":"Ms. Sunday explained the expectation.",'
                '"category":"teacher_guidance"}'
            ),
            LLMCallType.REFLECTION: (
                '{"reflection":"Clear expectations support participation.",'
                '"category":"classroom_norms"}'
            ),
            LLMCallType.IMPORTANCE_SCORE: (
                '{"importance":6,"reason":"The action establishes guidance."}',
                '{"importance":7,"reason":"The insight may support continuity."}',
            ),
        }
    )

    result = pipeline.run_turn(turn_input(stream, reflection_requested=True))

    assert result.reflection is not None
    assert result.reflection.reflection == (
        "Clear expectations support participation."
    )
    assert [memory.memory_type for memory in result.created_memories] == [
        "action",
        "reflection",
    ]
    assert [memory.text for memory in result.created_memories] == [
        "Ms. Sunday explained the expectation.",
        "Clear expectations support participation.",
    ]
    assert [memory.importance for memory in result.created_memories] == [6.0, 7.0]
    assert provider.received_texts == tuple(
        memory.text for memory in result.created_memories
    )
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.NATURAL_RESPONSE,
        LLMCallType.REFLECTION,
        LLMCallType.IMPORTANCE_SCORE,
        LLMCallType.IMPORTANCE_SCORE,
    )


def test_target_event_uses_target_response_stage_and_event_perception() -> None:
    stream = build_stream("academic_student")
    pipeline, client, _ = make_pipeline()

    result = pipeline.run_turn(
        turn_input(
            stream,
            source_step_id="day_02_target_01",
            event_id="event_01",
        )
    )

    assert result.source_step_type == "target_event"
    assert result.event_id == "event_01"
    assert result.perception.event_title == "Pengumuman Ujian"
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.TARGET_EVENT_RESPONSE,
        LLMCallType.IMPORTANCE_SCORE,
    )
    action_request = client.invocations[1].request
    assert action_request.response_limit == 1
    assert action_request.context.event_description is not None


@pytest.mark.parametrize(
    ("scripts", "reflection_requested"),
    [
        ({LLMCallType.PLAN: RuntimeError("plan failed")}, False),
        ({LLMCallType.NATURAL_RESPONSE: "not JSON"}, False),
        ({LLMCallType.REFLECTION: ""}, True),
        ({LLMCallType.IMPORTANCE_SCORE: '{"importance":20,"reason":"bad"}'}, False),
        (
            {
                LLMCallType.IMPORTANCE_SCORE: (
                    '{"importance":6,"reason":"Valid action importance."}',
                    '{"importance":20,"reason":"Invalid reflection importance."}',
                )
            },
            True,
        ),
    ],
)
def test_stage_or_parsing_failure_leaves_memory_stream_unchanged(
    scripts: Mapping,
    reflection_requested: bool,
) -> None:
    stream = build_stream()
    before = stream.list_memories()
    pipeline, _, _ = make_pipeline(scripted_responses=scripts)

    with pytest.raises((RuntimeError, ValueError)):
        pipeline.run_turn(
            turn_input(stream, reflection_requested=reflection_requested)
        )

    assert stream.list_memories() == before


class FailingEmbeddingProvider:
    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise RuntimeError("dummy embedding failed")
        return (1.0, 1.0)


class InvalidEmbeddingProvider:
    def __init__(self, embedding: tuple[float, ...]) -> None:
        self.embedding = embedding

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embedding


@pytest.mark.parametrize(
    ("fail_on_call", "reflection_requested"),
    [(1, False), (2, True)],
)
def test_embedding_failure_leaves_memory_stream_unchanged(
    fail_on_call: int,
    reflection_requested: bool,
) -> None:
    stream = build_stream()
    before = stream.list_memories()
    provider = FailingEmbeddingProvider(fail_on_call)
    pipeline, _, _ = make_pipeline(embedding_provider=provider)

    with pytest.raises(RuntimeError, match="dummy embedding failed"):
        pipeline.run_turn(
            turn_input(stream, reflection_requested=reflection_requested)
        )

    assert stream.list_memories() == before


@pytest.mark.parametrize("embedding", [(0.0, 0.0), (1.0, 1.0, 1.0)])
def test_invalid_generated_embedding_leaves_memory_stream_unchanged(
    embedding: tuple[float, ...],
) -> None:
    stream = build_stream()
    before = stream.list_memories()
    pipeline, _, _ = make_pipeline(
        embedding_provider=InvalidEmbeddingProvider(embedding)
    )

    with pytest.raises(RetrievalScoringError):
        pipeline.run_turn(turn_input(stream))

    assert stream.list_memories() == before


def test_exact_duplicate_response_is_tolerated_and_hinted_to_model() -> None:
    duplicate = "This response was already used."
    stream = build_stream()
    before_len = len(stream.list_memories())
    pipeline, client, _ = make_pipeline(
        scripted_responses={
            LLMCallType.NATURAL_RESPONSE: (
                '{"response":"'
                + duplicate
                + '","memory_to_store":"A valid action memory.",'
                '"category":"classroom_norms"}'
            )
        }
    )

    result = pipeline.run_turn(
        turn_input(stream, previous_same_agent_response=duplicate)
    )

    assert result.response == duplicate
    assert len(stream.list_memories()) == before_len + 1
    action_request = client.invocations[1].request
    assert action_request.previous_same_agent_response == duplicate


def test_excessive_response_is_trimmed_to_configured_limit() -> None:
    stream = build_stream()
    pipeline, _, _ = make_pipeline(
        scripted_responses={
            LLMCallType.NATURAL_RESPONSE: (
                '{"response":"First sentence. Second sentence. Third sentence.",'
                '"memory_to_store":"A valid action memory.",'
                '"category":"classroom_norms"}'
            )
        }
    )

    before_len = len(stream.list_memories())
    result = pipeline.run_turn(turn_input(stream, previous_same_agent_response=None))

    assert result.response == "First sentence. Second sentence."
    assert result.action.response == "First sentence. Second sentence."
    assert len(stream.list_memories()) == before_len + 1


def test_empty_and_fewer_than_top_k_streams_complete_without_fallbacks() -> None:
    for count in (0, 3):
        stream = build_stream(count=count)
        pipeline, _, _ = make_pipeline()

        result = pipeline.run_turn(turn_input(stream))

        assert result.retrieval_result.candidate_count == count
        assert len(result.selected_memory_ids) == count
        assert len(stream) == count + 1


def test_repeated_turns_with_fresh_equal_state_produce_equal_results() -> None:
    first_stream = build_stream()
    second_stream = build_stream()
    first_pipeline, _, _ = make_pipeline()
    second_pipeline, _, _ = make_pipeline()

    first = first_pipeline.run_turn(turn_input(first_stream))
    second = second_pipeline.run_turn(turn_input(second_stream))

    assert first == second
    assert first_stream.list_memories() == second_stream.list_memories()


def test_runtime_input_errors_fail_without_mutation() -> None:
    invalid_cases = (
        {"source_step_id": "missing_step"},
        {"retrieval_config_id": "missing_config"},
        {
            "source_step_id": "day_02_target_01",
            "event_id": "event_02",
        },
    )
    for overrides in invalid_cases:
        stream = build_stream()
        before = stream.list_memories()
        input_value = turn_input(
            stream,
            source_step_id=overrides.get(
                "source_step_id", "day_01_natural_01"
            ),
            event_id=overrides.get("event_id"),
            retrieval_config_id=overrides.get("retrieval_config_id", "balanced"),
        )
        pipeline, _, _ = make_pipeline()

        with pytest.raises(CognitiveInputError):
            pipeline.run_turn(input_value)
        assert stream.list_memories() == before


def test_cross_agent_stream_invalid_time_and_missing_embedding_are_rejected() -> None:
    pipeline, _, _ = make_pipeline()
    foreign_stream = build_stream("active_student")
    cross_agent_input = CognitiveTurnInput(
        agent_id="teacher",
        source_step_id="day_01_natural_01",
        turn_id="turn_cross_agent",
        retrieval_config_id="balanced",
        memory_stream=foreign_stream,
        logical_timestamp=20,
        query_embedding=(1.0, 1.0),
    )
    with pytest.raises(CognitiveInputError, match="ownership"):
        pipeline.run_turn(cross_agent_input)

    stream = build_stream()
    with pytest.raises(CognitiveInputError, match="must not precede"):
        pipeline.run_turn(turn_input(stream, logical_timestamp=1))

    with pytest.raises(MissingEmbeddingError, match="query_embedding is required"):
        CognitiveTurnInput(
            agent_id="teacher",
            source_step_id="day_01_natural_01",
            turn_id="turn_missing_embedding",
            retrieval_config_id="balanced",
            memory_stream=stream,
            logical_timestamp=20,
            query_embedding=None,
        )
