from dataclasses import replace
import socket

import pytest

from src.llm import (
    DummyEmbeddingProvider,
    DummyLLMClient,
    DummyLLMError,
    EmbeddingError,
    LLMCallType,
    LLMContext,
    LLMRequest,
    LLMRequestError,
    UnsupportedLLMCallTypeError,
    parse_action_output,
    parse_importance_output,
    parse_plan_output,
    parse_reflection_output,
)


def natural_context() -> LLMContext:
    return LLMContext(
        agent_id="teacher",
        display_name="Ms. Sunday",
        role="teacher",
        archetype="calm_classroom_guide",
        persona_summary="A calm and structured classroom teacher.",
        traits=("calm", "structured"),
        goals=("support students",),
        communication_style="Clear and concise.",
        motivations=("student understanding",),
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        step_title="Classroom Introductions and Expectations",
        step_description="The class discusses participation.",
        day=1,
        period="morning",
        logical_timestamp=10,
        observation_text="The class is discussing participation.",
        conversation_context=("Monday asked how to participate.",),
        selected_memory_texts=("Ms. Sunday explains expectations clearly.",),
    )


def target_context() -> LLMContext:
    return replace(
        natural_context(),
        source_step_id="day_02_target_01",
        source_step_type="target_event",
        step_title="Pengumuman Ujian",
        step_description="A written test is announced.",
        event_id="event_01",
        event_title="Pengumuman Ujian",
        event_description="A written test will take place on Day 6.",
    )


def test_dummy_returns_repeatable_stage_specific_structured_outputs() -> None:
    client = DummyLLMClient()
    context = natural_context()

    first_plan_raw = client.generate(LLMRequest(LLMCallType.PLAN, context))
    second_plan_raw = client.generate(LLMRequest(LLMCallType.PLAN, context))
    plan = parse_plan_output(first_plan_raw)
    natural = parse_action_output(
        client.generate(
            LLMRequest(
                LLMCallType.NATURAL_RESPONSE,
                context,
                plan=plan.plan,
                response_limit=2,
            )
        )
    )
    target = parse_action_output(
        client.generate(
            LLMRequest(
                LLMCallType.TARGET_EVENT_RESPONSE,
                target_context(),
                plan=plan.plan,
                response_limit=1,
            )
        )
    )
    reflection = parse_reflection_output(
        client.generate(
            LLMRequest(
                LLMCallType.REFLECTION,
                context,
                plan=plan.plan,
                response=natural.response,
                response_limit=1,
            )
        )
    )
    importance = parse_importance_output(
        client.generate(
            LLMRequest(
                LLMCallType.IMPORTANCE_SCORE,
                context,
                memory_text=natural.memory_to_store,
                memory_kind="action",
            )
        )
    )

    assert first_plan_raw == second_plan_raw
    assert plan.plan
    assert natural.response and natural.memory_to_store and natural.category
    assert target.response and target.memory_to_store and target.category
    assert reflection.reflection and reflection.category
    assert importance.importance == 5.0
    assert client.call_types == (
        LLMCallType.PLAN,
        LLMCallType.PLAN,
        LLMCallType.NATURAL_RESPONSE,
        LLMCallType.TARGET_EVENT_RESPONSE,
        LLMCallType.REFLECTION,
        LLMCallType.IMPORTANCE_SCORE,
    )
    assert tuple(item.index for item in client.invocations) == (1, 2, 3, 4, 5, 6)


def test_scripted_responses_are_consumed_per_stage_and_inspectable() -> None:
    client = DummyLLMClient(
        {
            "plan": ('{"plan":"First plan."}', '{"plan":"Second plan."}'),
        }
    )
    request = LLMRequest(LLMCallType.PLAN, natural_context())

    assert parse_plan_output(client.generate(request)).plan == "First plan."
    assert parse_plan_output(client.generate(request)).plan == "Second plan."
    assert client.invocations[0].request is request
    with pytest.raises(DummyLLMError, match="no scripted plan outcome"):
        client.generate(request)
    assert len(client.invocations) == 3


def test_dummy_can_raise_a_scripted_failure_without_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in dummy mode")

    monkeypatch.setattr(socket, "create_connection", unexpected_network)
    failure = RuntimeError("planned dummy failure")
    client = DummyLLMClient({LLMCallType.PLAN: failure})

    with pytest.raises(RuntimeError, match="planned dummy failure"):
        client.generate(LLMRequest(LLMCallType.PLAN, natural_context()))
    assert client.call_types == (LLMCallType.PLAN,)


def test_dummy_embedding_is_deterministic_and_records_received_text() -> None:
    provider = DummyEmbeddingProvider((0.25, 0.75))

    assert provider.embed("First memory.") == (0.25, 0.75)
    assert provider.embed("Second memory.") == (0.25, 0.75)
    assert provider.received_texts == ("First memory.", "Second memory.")


@pytest.mark.parametrize("embedding", [(), (0.0, 0.0), (1.0, float("nan"))])
def test_dummy_embedding_rejects_invalid_configurations(embedding: tuple) -> None:
    with pytest.raises(EmbeddingError):
        DummyEmbeddingProvider(embedding)


def test_llm_request_rejects_unsupported_or_stage_incompatible_calls() -> None:
    with pytest.raises(UnsupportedLLMCallTypeError, match="unsupported LLM call type"):
        LLMRequest("bcr_judge", natural_context())
    with pytest.raises(LLMRequestError, match="incompatible"):
        LLMRequest(
            LLMCallType.TARGET_EVENT_RESPONSE,
            natural_context(),
            plan="Respond briefly.",
        )
