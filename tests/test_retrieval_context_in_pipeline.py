from dataclasses import fields

import pytest

from src.cognition import (
    CognitivePipeline,
    CognitiveResultValidationError,
    CognitiveTurnInput,
)
from src.config_loader import load_experiment_config
from src.llm import DummyEmbeddingProvider, DummyLLMClient, LLMContext, LLMRequest
from src.memory import MemoryFactory, MemoryStream
from src.retrieval import RetrievalEngine


def build_teacher_stream(count: int = 7) -> MemoryStream:
    stream = MemoryStream("teacher", run_id="retrieval_context_run")
    factory = MemoryFactory("teacher", run_id="retrieval_context_run")
    for index in range(count):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"SELECTABLE PRIVATE MEMORY TEXT {index + 1}",
                category="teacher_guidance",
                importance=float(min(index + 1, 10)),
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=(1.0, float(index + 1)),
            )
        )
    return stream


class TrackingRetrievalEngine:
    def __init__(self, settings: object, events: list[str]) -> None:
        self.delegate = RetrievalEngine(settings)
        self.events = events
        self.calls: list[dict] = []
        self.result = None
        self.before = None
        self.after = None

    def retrieve(self, **kwargs: object):
        self.events.append("retrieve")
        self.calls.append(dict(kwargs))
        stream = kwargs["memory_stream"]
        self.before = stream.list_memories()
        self.result = self.delegate.retrieve(**kwargs)
        self.after = stream.list_memories()
        return self.result


class TrackingLLMClient:
    def __init__(self, events: list[str]) -> None:
        self.delegate = DummyLLMClient()
        self.events = events

    @property
    def invocations(self):
        return self.delegate.invocations

    def generate(self, request: LLMRequest) -> str:
        self.events.append(request.call_type.value)
        return self.delegate.generate(request)


def build_input(stream: MemoryStream) -> CognitiveTurnInput:
    return CognitiveTurnInput(
        agent_id="teacher",
        source_step_id="day_01_natural_01",
        turn_id="turn_retrieval_context",
        retrieval_config_id="importance_heavy",
        memory_stream=stream,
        logical_timestamp=20,
        query_embedding=(1.0, 1.0),
        conversation_context=("The class is discussing expectations.",),
    )


def test_pipeline_invokes_existing_retrieval_contract_before_llm_stages() -> None:
    config = load_experiment_config()
    events: list[str] = []
    engine = TrackingRetrievalEngine(config.settings, events)
    llm = TrackingLLMClient(events)
    pipeline = CognitivePipeline(
        config=config,
        llm_client=llm,
        embedding_provider=DummyEmbeddingProvider((1.0, 1.0)),
        retrieval_engine=engine,
    )
    stream = build_teacher_stream()

    result = pipeline.run_turn(build_input(stream))

    assert events == ["retrieve", "plan", "natural_response", "importance_score"]
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["memory_stream"] is stream
    assert call["retrieval_config"].config_id == "importance_heavy"
    assert call["query"] == result.retrieval_query.text
    assert call["query_embedding"] == (1.0, 1.0)
    assert call["current_time"] == 20
    assert "top_k" not in call
    assert engine.before == engine.after
    assert result.retrieval_result is engine.result
    assert result.retrieval_result.top_k == config.settings.top_k == 5


def test_llm_context_contains_only_selected_memory_text_in_retrieval_order() -> None:
    config = load_experiment_config()
    events: list[str] = []
    engine = TrackingRetrievalEngine(config.settings, events)
    llm = TrackingLLMClient(events)
    pipeline = CognitivePipeline(
        config=config,
        llm_client=llm,
        embedding_provider=DummyEmbeddingProvider((1.0, 1.0)),
        retrieval_engine=engine,
    )

    result = pipeline.run_turn(build_input(build_teacher_stream()))
    expected_texts = tuple(
        memory.text for memory in result.retrieval_result.selected_memories
    )
    unselected_texts = {
        candidate.memory.text
        for candidate in result.retrieval_result.candidates
        if not candidate.selected
    }

    for invocation in llm.invocations:
        context = invocation.request.context
        assert context.selected_memory_texts == expected_texts
        assert unselected_texts.isdisjoint(context.selected_memory_texts)

    context_fields = {field.name for field in fields(LLMContext)}
    forbidden_fields = {
        "memory_stream",
        "candidates",
        "selected_memory_ids",
        "retrieval_config_id",
        "retrieval_weights",
        "applied_weights",
        "raw_recency",
        "raw_importance",
        "raw_relevance",
        "normalized_recency",
        "normalized_importance",
        "normalized_relevance",
        "final_score",
        "ground_truth",
        "evaluation",
    }
    assert context_fields.isdisjoint(forbidden_fields)
    assert result.retrieval_result.candidates
    assert result.retrieval_result.applied_weights == next(
        item.weights
        for item in config.retrieval_configs
        if item.config_id == "importance_heavy"
    )
    assert "importance_heavy" not in result.retrieval_query.text
    assert all(
        term not in result.retrieval_query.text.lower()
        for term in ("recency", "relevance", "retrieval weight")
    )


def test_pipeline_candidate_source_is_only_the_requested_agent_stream() -> None:
    config = load_experiment_config()
    events: list[str] = []
    engine = TrackingRetrievalEngine(config.settings, events)
    llm = TrackingLLMClient(events)
    pipeline = CognitivePipeline(
        config=config,
        llm_client=llm,
        embedding_provider=DummyEmbeddingProvider((1.0, 1.0)),
        retrieval_engine=engine,
    )
    teacher_stream = build_teacher_stream(3)
    private_student_stream = MemoryStream(
        "active_student",
        run_id="retrieval_context_run",
    )
    private_student_stream.add_memory(
        MemoryFactory(
            "active_student",
            run_id="retrieval_context_run",
        ).create_memory(
            memory_type="observation",
            text="OTHER AGENT PRIVATE MEMORY MUST NOT BE RETRIEVED",
            category="social_learning",
            importance=10,
            created_at=0,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
            embedding=(1.0, 1.0),
        )
    )

    result = pipeline.run_turn(build_input(teacher_stream))

    assert all(
        candidate.memory.agent_id == "teacher"
        for candidate in result.retrieval_result.candidates
    )
    assert all(
        "OTHER AGENT" not in text
        for invocation in llm.invocations
        for text in invocation.request.context.selected_memory_texts
    )
    assert len(private_student_stream) == 1


class TamperingRetrievalEngine:
    def __init__(self, settings: object) -> None:
        self.delegate = RetrievalEngine(settings)

    def retrieve(self, **kwargs: object):
        return self.delegate.retrieve(**{**kwargs, "query": "tampered query"})


def test_invalid_retrieval_result_fails_before_llm_or_memory_mutation() -> None:
    config = load_experiment_config()
    client = DummyLLMClient()
    stream = build_teacher_stream()
    before = stream.list_memories()
    pipeline = CognitivePipeline(
        config=config,
        llm_client=client,
        embedding_provider=DummyEmbeddingProvider((1.0, 1.0)),
        retrieval_engine=TamperingRetrievalEngine(config.settings),
    )

    with pytest.raises(CognitiveResultValidationError, match="query"):
        pipeline.run_turn(build_input(stream))

    assert client.invocations == ()
    assert stream.list_memories() == before
