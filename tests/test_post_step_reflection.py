from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from src.cognition import (
    CognitiveInputError,
    CognitivePipeline,
    CognitiveTurnInput,
    PostStepReflectionInput,
)
from src.config_loader import load_experiment_config
from src.llm import DummyEmbeddingProvider, DummyLLMClient, LLMCallType
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import RetrievalEngine, RetrievalScoringError


NATURAL_SPEAKERS = (
    "Ms. Sunday",
    "Monday",
    "Wednesday",
    "Tuesday",
    "Monday",
    "Ms. Sunday",
    "Wednesday",
    "Tuesday",
    "Wednesday",
    "Monday",
    "Tuesday",
    "Ms. Sunday",
)


class CountingRetrievalEngine:
    def __init__(self, settings: object) -> None:
        self._delegate = RetrievalEngine(settings)  # type: ignore[arg-type]
        self.calls = 0

    def retrieve(self, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self._delegate.retrieve(**kwargs)  # type: ignore[arg-type]


class FailingSecondEmbeddingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("post-step embedding failed")
        return (1.0, 1.0)


def build_stream(
    agent_id: str = "teacher",
    *,
    run_id: str = "phase_08_reflection_run",
) -> MemoryStream:
    stream = MemoryStream(agent_id, run_id=run_id)
    factory = MemoryFactory(agent_id, run_id=run_id)
    for index in range(6):
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


def scripted_client(
    overrides: Mapping[LLMCallType, object] | None = None,
) -> DummyLLMClient:
    scripts: dict[LLMCallType, object] = {
        LLMCallType.PLAN: '{"plan":"Use the completed classroom context."}',
        LLMCallType.NATURAL_RESPONSE: (
            '{"response":"I will keep the expectation clear.",'
            '"memory_to_store":"Ms. Sunday reinforced the expectation.",'
            '"category":"teacher_guidance"}'
        ),
        LLMCallType.TARGET_EVENT_RESPONSE: (
            '{"response":"I will prepare for the announced test.",'
            '"memory_to_store":"Wednesday planned for the announced test.",'
            '"category":"exam_concern"}'
        ),
        LLMCallType.REFLECTION: (
            '{"reflection":"Clear expectations supported participation.",'
            '"category":"classroom_norms"}'
        ),
        LLMCallType.IMPORTANCE_SCORE: (
            '{"importance":6,"reason":"The action was relevant."}',
            '{"importance":7,"reason":"The reflection supports continuity."}',
        ),
    }
    if overrides:
        scripts.update(overrides)
    return DummyLLMClient(scripts)  # type: ignore[arg-type]


def natural_context(count: int) -> tuple[str, ...]:
    return tuple(
        f"{display_name}: response {index + 1}"
        for index, display_name in enumerate(NATURAL_SPEAKERS[:count])
    )


def prepare_natural_reflection(
    *,
    client: DummyLLMClient | None = None,
    provider: object | None = None,
    agent_id: str = "teacher",
    main_timestamp: int = 12,
    prior_count: int = 11,
):
    config = load_experiment_config()
    stream = build_stream(agent_id)
    engine = CountingRetrievalEngine(config.settings)
    llm_client = client or scripted_client()
    embedding_provider = provider or DummyEmbeddingProvider((1.0, 1.0))
    pipeline = CognitivePipeline(
        config=config,
        llm_client=llm_client,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        retrieval_engine=engine,
    )
    prior_context = natural_context(prior_count)
    final_result = pipeline.run_turn(
        CognitiveTurnInput(
            agent_id=agent_id,
            source_step_id="day_01_natural_01",
            turn_id=f"day_01_natural_01_turn_{main_timestamp:02d}",
            retrieval_config_id="balanced",
            memory_stream=stream,
            logical_timestamp=main_timestamp,
            query_embedding=(1.0, 1.0),
            conversation_context=prior_context,
        )
    )
    agent = next(item for item in config.agents if item.agent_id == agent_id)
    complete_context = prior_context + (
        f"{agent.display_name}: {final_result.response}",
    )
    complete_context += natural_context(12)[len(complete_context) :]
    reflection_input = PostStepReflectionInput(
        agent_id=agent_id,
        source_step_id="day_01_natural_01",
        memory_stream=stream,
        logical_timestamp=12,
        conversation_context=complete_context,
        final_turn_result=final_result,
    )
    return (
        pipeline,
        stream,
        engine,
        llm_client,
        embedding_provider,
        final_result,
        reflection_input,
    )


def test_post_step_reflection_creates_exactly_one_reflection_memory() -> None:
    (
        pipeline,
        stream,
        engine,
        client,
        provider,
        final_result,
        reflection_input,
    ) = prepare_natural_reflection()
    before = stream.list_memories()
    access_before = {
        memory.memory_id: memory.last_accessed_at for memory in before
    }

    result = pipeline.run_post_step_reflection(reflection_input)

    assert engine.calls == 1
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.NATURAL_RESPONSE,
        LLMCallType.IMPORTANCE_SCORE,
        LLMCallType.REFLECTION,
        LLMCallType.IMPORTANCE_SCORE,
    )
    assert provider.received_texts == (  # type: ignore[attr-defined]
        final_result.memory_to_store,
        result.reflection.reflection,
    )
    assert len(stream) == len(before) + 1
    assert result.created_memories == (stream.list_memories()[-1],)
    assert result.reflection_memory.memory_type == "reflection"
    assert result.reflection_memory.created_at == 12
    assert result.reflection_memory.last_accessed_at == 12
    assert result.reflection_memory.embedding == (1.0, 1.0)
    assert result.final_turn_id == final_result.turn_id
    assert result.retrieval_config_id == final_result.retrieval_config_id
    assert result.selected_memory_ids == final_result.selected_memory_ids
    assert result.selected_memory_texts == tuple(
        memory.text for memory in final_result.retrieval_result.selected_memories
    )
    assert {
        memory.memory_id: memory.last_accessed_at
        for memory in stream.list_memories()[:-1]
    } == access_before
    with pytest.raises(FrozenInstanceError):
        result.final_turn_id = "changed"  # type: ignore[misc]


def test_reflection_receives_complete_final_context_and_final_turn_outputs() -> None:
    pipeline, _, _, client, _, final_result, reflection_input = (
        prepare_natural_reflection()
    )

    result = pipeline.run_post_step_reflection(reflection_input)

    reflection_request = client.invocations[-2].request
    importance_request = client.invocations[-1].request
    assert reflection_request.call_type is LLMCallType.REFLECTION
    assert reflection_request.context.conversation_context == (
        reflection_input.conversation_context
    )
    assert reflection_request.context.selected_memory_texts == (
        result.selected_memory_texts
    )
    assert reflection_request.plan == final_result.plan.plan
    assert reflection_request.response == final_result.action.response
    assert reflection_request.context.logical_timestamp == 12
    assert importance_request.call_type is LLMCallType.IMPORTANCE_SCORE
    assert importance_request.context == reflection_request.context
    assert importance_request.memory_kind == "reflection"


def test_reflection_uses_step_final_timestamp_without_advancing_access_time() -> None:
    (
        pipeline,
        stream,
        _,
        _,
        _,
        final_result,
        reflection_input,
    ) = prepare_natural_reflection(
        agent_id="active_student",
        main_timestamp=10,
        prior_count=9,
    )
    before = stream.list_memories()

    result = pipeline.run_post_step_reflection(reflection_input)

    assert final_result.logical_timestamp == 10
    assert result.logical_timestamp == 12
    assert result.reflection_memory.created_at == 12
    assert stream.list_memories()[:-1] == before


def test_target_reflection_includes_resolved_event_context() -> None:
    config = load_experiment_config()
    stream = build_stream("academic_student")
    engine = CountingRetrievalEngine(config.settings)
    client = scripted_client()
    pipeline = CognitivePipeline(
        config=config,
        llm_client=client,
        embedding_provider=DummyEmbeddingProvider((1.0, 1.0)),
        retrieval_engine=engine,
    )
    prior_context = (
        "Ms. Sunday: The test is on Day 6.",
        "Monday: I will review with classmates.",
        "Tuesday: I will prepare my notes.",
    )
    final_result = pipeline.run_turn(
        CognitiveTurnInput(
            agent_id="academic_student",
            source_step_id="day_02_target_01",
            turn_id="day_02_target_01_turn_04",
            retrieval_config_id="balanced",
            memory_stream=stream,
            logical_timestamp=40,
            query_embedding=(1.0, 1.0),
            conversation_context=prior_context,
            event_id="event_01",
        )
    )
    complete_context = prior_context + (
        f"Wednesday: {final_result.response}",
    )

    result = pipeline.run_post_step_reflection(
        PostStepReflectionInput(
            agent_id="academic_student",
            source_step_id="day_02_target_01",
            memory_stream=stream,
            logical_timestamp=40,
            conversation_context=complete_context,
            final_turn_result=final_result,
            event_id="event_01",
        )
    )

    context = client.invocations[-2].request.context
    assert result.event_id == "event_01"
    assert context.event_id == "event_01"
    assert context.event_title == "Exam Announcement"
    assert context.event_description is not None
    assert context.conversation_context == complete_context
    assert engine.calls == 1


@pytest.mark.parametrize(
    "client",
    [
        scripted_client(
            {LLMCallType.REFLECTION: RuntimeError("reflection failed")}
        ),
        scripted_client({LLMCallType.REFLECTION: "not JSON"}),
        scripted_client(
            {
                LLMCallType.IMPORTANCE_SCORE: (
                    '{"importance":6,"reason":"The action was relevant."}',
                    '{"importance":20,"reason":"Invalid reflection score."}',
                )
            }
        ),
    ],
)
def test_reflection_or_enrichment_failure_leaves_stream_unchanged(
    client: DummyLLMClient,
) -> None:
    pipeline, stream, engine, _, _, _, reflection_input = (
        prepare_natural_reflection(client=client)
    )
    before = stream.list_memories()

    with pytest.raises((RuntimeError, ValueError)):
        pipeline.run_post_step_reflection(reflection_input)

    assert stream.list_memories() == before
    assert engine.calls == 1


def test_reflection_embedding_failure_leaves_stream_unchanged() -> None:
    provider = FailingSecondEmbeddingProvider()
    pipeline, stream, engine, _, _, _, reflection_input = (
        prepare_natural_reflection(provider=provider)
    )
    before = stream.list_memories()

    with pytest.raises(RuntimeError, match="post-step embedding failed"):
        pipeline.run_post_step_reflection(reflection_input)

    assert stream.list_memories() == before
    assert engine.calls == 1


def test_invalid_reflection_embedding_leaves_stream_unchanged() -> None:
    class WrongDimensionProvider:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> tuple[float, ...]:
            self.calls += 1
            return (1.0, 1.0) if self.calls == 1 else (1.0, 1.0, 1.0)

    pipeline, stream, _, _, _, _, reflection_input = prepare_natural_reflection(
        provider=WrongDimensionProvider()
    )
    before = stream.list_memories()

    with pytest.raises(RetrievalScoringError, match="dimensions"):
        pipeline.run_post_step_reflection(reflection_input)

    assert stream.list_memories() == before


def test_incomplete_or_misordered_final_context_is_rejected_before_llm_call() -> None:
    pipeline, stream, _, client, _, final_result, reflection_input = (
        prepare_natural_reflection()
    )
    calls_before = client.call_types
    before = stream.list_memories()

    invalid_contexts = (
        reflection_input.conversation_context[:-1],
        (
            *reflection_input.conversation_context[:11],
            "Ms. Sunday: wrong response",
        ),
    )
    for context in invalid_contexts:
        with pytest.raises(CognitiveInputError, match="conversation_context"):
            pipeline.run_post_step_reflection(
                PostStepReflectionInput(
                    agent_id="teacher",
                    source_step_id="day_01_natural_01",
                    memory_stream=stream,
                    logical_timestamp=12,
                    conversation_context=context,
                    final_turn_result=final_result,
                )
            )

    assert client.call_types == calls_before
    assert stream.list_memories() == before


def test_cross_run_result_is_rejected() -> None:
    _, _, _, _, _, foreign_result, _ = prepare_natural_reflection()
    local_stream = build_stream(run_id="different_run")

    with pytest.raises(CognitiveInputError, match="MemoryStream run"):
        PostStepReflectionInput(
            agent_id="teacher",
            source_step_id="day_01_natural_01",
            memory_stream=local_stream,
            logical_timestamp=12,
            conversation_context=natural_context(12),
            final_turn_result=foreign_result,
        )


def test_post_step_reflection_is_at_most_once_per_agent_and_step() -> None:
    pipeline, stream, engine, client, _, _, reflection_input = (
        prepare_natural_reflection()
    )
    pipeline.run_post_step_reflection(reflection_input)
    before = stream.list_memories()
    calls_before = client.call_types

    with pytest.raises(CognitiveInputError):
        pipeline.run_post_step_reflection(reflection_input)

    assert stream.list_memories() == before
    assert client.call_types == calls_before
    assert engine.calls == 1


def test_run_turn_behavior_and_alias_remain_unchanged() -> None:
    pipeline, stream, engine, client, provider, final_result, _ = (
        prepare_natural_reflection()
    )

    assert pipeline.run.__func__ is CognitivePipeline.run_turn
    assert final_result.reflection is None
    assert final_result.reflection_requested is False
    assert [memory.memory_type for memory in final_result.created_memories] == [
        "action"
    ]
    assert engine.calls == 1
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.NATURAL_RESPONSE,
        LLMCallType.IMPORTANCE_SCORE,
    )
    assert provider.received_texts == (  # type: ignore[attr-defined]
        final_result.memory_to_store,
    )
    assert stream.list_memories()[-1] == final_result.created_memories[0]
