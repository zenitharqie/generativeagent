"""One fully offline cognitive turn through both real-service adapters."""

from __future__ import annotations

from collections import deque

from src.cognition import CognitivePipeline, CognitiveTurnInput
from src.config_loader import load_experiment_config
from src.llm import LLMService, OpenAIEmbeddingProvider
from src.llm.provider import (
    EmbeddingRequest,
    EmbeddingResponse,
    StructuredTextRequest,
    StructuredTextResponse,
)
from src.llm.usage import ProviderUsage
from src.memory import MemoryFactory, MemoryStream


VECTOR = (1.0,) + (0.0,) * 1535


class MockedResponsesProvider:
    def __init__(self) -> None:
        self.outputs = deque(
            (
                '{"plan":"Invite one concise contribution."}',
                (
                    '{"response":"Please share one idea.",'
                    '"memory_to_store":"The teacher invited one idea.",'
                    '"category":"participation"}'
                ),
                '{"importance":6,"reason":"It may shape later participation."}',
            )
        )
        self.requests: list[StructuredTextRequest] = []

    def generate_structured(self, request: StructuredTextRequest) -> StructuredTextResponse:
        self.requests.append(request)
        return StructuredTextResponse(
            self.outputs.popleft(),
            ProviderUsage(20, 8, 28),
        )


class MockedEmbeddingAPI:
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        return EmbeddingResponse(VECTOR, ProviderUsage(5, None, 5))


def test_mocked_real_services_complete_one_turn_without_sdk_or_network() -> None:
    stream = MemoryStream("teacher", run_id="phase10_mocked_run")
    factory = MemoryFactory("teacher", run_id="phase10_mocked_run")
    for index in range(6):
        stream.add_memory(
            factory.create_memory(
                memory_type="observation",
                text=f"Teacher memory {index + 1}.",
                category="classroom_norms",
                importance=float(index + 1),
                created_at=index,
                source_step_id="day_01_natural_01",
                source_step_type="natural_interaction_block",
                embedding=VECTOR,
            )
        )
    text_api = MockedResponsesProvider()
    embedding_api = MockedEmbeddingAPI()
    llm_service = LLMService(text_api, sleeper=lambda _: None)
    embedding_provider = OpenAIEmbeddingProvider(embedding_api, sleeper=lambda _: None)
    pipeline = CognitivePipeline(
        config=load_experiment_config(),
        llm_client=llm_service,
        embedding_provider=embedding_provider,
    )

    result = pipeline.run_turn(
        CognitiveTurnInput(
            agent_id="teacher",
            source_step_id="day_01_natural_01",
            turn_id="turn_001",
            retrieval_config_id="balanced",
            memory_stream=stream,
            logical_timestamp=20,
            query_embedding=VECTOR,
            conversation_context=("A student asked how to participate.",),
        )
    )

    assert result.response == "Please share one idea."
    assert result.action.memory_to_store == "The teacher invited one idea."
    assert result.action.category == "participation"
    assert result.action_importance is not None
    assert result.action_importance.importance == 6.0
    assert result.created_memories[0].embedding == VECTOR
    assert len(text_api.requests) == 3
    assert len(embedding_api.requests) == 1
    assert len(llm_service.usage_records) == 3
    assert len(embedding_provider.usage_records) == 1
