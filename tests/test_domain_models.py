import pytest

from src.domain import (
    AgentProfile,
    MemoryEntry,
    NaturalInteractionBlockStep,
    PersonaProfile,
    RetrievalConfig,
    RetrievalWeights,
    SeedMemoryConfig,
    TargetEventStep,
)


def test_agent_profile_exposes_typed_persona_and_seed_memories() -> None:
    persona = PersonaProfile(
        summary="A structured classroom teacher.",
        traits=("calm", "clear"),
        goals=("support learning",),
        communication_style="Concise and instructional.",
        motivations=("student understanding",),
    )
    seed_memory = SeedMemoryConfig(
        text="Ms. Sunday explains expectations clearly.",
        category="teacher_guidance",
        importance=7,
    )
    agent = AgentProfile(
        agent_id="teacher",
        display_name="Ms. Sunday",
        role="teacher",
        archetype="calm_classroom_guide",
        persona=persona,
        seed_memories=(seed_memory,),
    )

    assert agent.persona_summary == persona.summary
    assert agent.traits == ("calm", "clear")
    assert agent.seed_memories == (seed_memory,)


def test_agent_profile_rejects_unknown_agent_id() -> None:
    persona = PersonaProfile(
        summary="A profile.",
        traits=("calm",),
        goals=("learn",),
        communication_style="Brief.",
        motivations=("clarity",),
    )

    with pytest.raises(ValueError, match="unsupported agent_id"):
        AgentProfile(
            agent_id="extra_student",
            display_name="Friday",
            role="student",
            archetype="extra",
            persona=persona,
            seed_memories=(SeedMemoryConfig("A memory.", "quiet_style", 5),),
        )


def test_simulation_step_subclasses_distinguish_step_types() -> None:
    common = {
        "day": 2,
        "period": "morning",
        "title": "Title",
        "description": "Description",
        "participants": (
            "teacher",
            "active_student",
            "quiet_student",
            "academic_student",
        ),
        "memory_generation_enabled": True,
    }
    natural_step = NaturalInteractionBlockStep(
        step_id="natural_01",
        step_type="natural_interaction_block",
        evaluation_enabled=False,
        turns_per_agent=3,
        total_turns=12,
        teacher_opens_block=True,
        turn_policy="controlled_fixed",
        **common,
    )
    target_step = TargetEventStep(
        step_id="target_01",
        step_type="target_event",
        evaluation_enabled=True,
        event_id="event_01",
        **common,
    )

    assert natural_step.is_target_event is False
    assert target_step.is_target_event is True


def test_retrieval_models_validate_weights_without_scoring() -> None:
    weights = RetrievalWeights(recency=0.33, importance=0.33, relevance=0.33)
    config = RetrievalConfig(config_id="balanced", weights=weights)

    assert config.weights == weights
    with pytest.raises(ValueError, match="non-negative"):
        RetrievalWeights(recency=-0.1, importance=0.5, relevance=0.6)
    with pytest.raises(ValueError, match="sum"):
        RetrievalWeights(recency=0.2, importance=0.2, relevance=0.2)


def test_memory_entry_is_a_validated_model_only() -> None:
    memory = MemoryEntry(
        memory_id="memory_001",
        agent_id="quiet_student",
        memory_type="observation",
        text="Tuesday heard the classroom instruction.",
        category="classroom_norms",
        importance=6,
        created_at=1,
        last_accessed_at=1,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
    )

    assert memory.embedding is None
    with pytest.raises(ValueError, match="between 1 and 10"):
        MemoryEntry(
            memory_id="memory_002",
            agent_id="quiet_student",
            memory_type="observation",
            text="Invalid importance.",
            category="classroom_norms",
            importance=11,
            created_at=1,
            last_accessed_at=1,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
        )
