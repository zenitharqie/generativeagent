import pytest

from src.memory import (
    AGENT_INITIALIZATION_SOURCE_STEP_TYPE,
    MEMORY_TYPES,
    MemoryEntry,
    MemoryFactory,
)


def test_factory_creates_every_supported_memory_type_with_required_fields() -> None:
    factory = MemoryFactory("teacher", run_id="run_01")

    memories = tuple(
        factory.create_memory(
            memory_type=memory_type,
            text=f"Memory for {memory_type}.",
            category="teacher_guidance",
            importance=6,
            created_at=index,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
        )
        for index, memory_type in enumerate(sorted(MEMORY_TYPES), start=1)
    )

    assert all(isinstance(memory, MemoryEntry) for memory in memories)
    assert [memory.memory_id for memory in memories] == [
        f"mem_teacher_{index:06d}" for index in range(1, 6)
    ]
    assert all(memory.agent_id == "teacher" for memory in memories)
    assert all(memory.run_id == "run_01" for memory in memories)
    assert all(memory.last_accessed_at == memory.created_at for memory in memories)
    assert all(memory.embedding is None for memory in memories)


def test_agent_initialization_is_a_narrow_valid_memory_source() -> None:
    memory = MemoryEntry(
        memory_id="mem_teacher_000001",
        agent_id="teacher",
        memory_type="seed_persona",
        text="Ms. Sunday explains expectations clearly.",
        category="teacher_guidance",
        importance=7,
        created_at=0,
        last_accessed_at=0,
        source_step_id="seed_memory_initialization",
        source_step_type=AGENT_INITIALIZATION_SOURCE_STEP_TYPE,
    )

    assert memory.source_step_type == "agent_initialization"
    with pytest.raises(ValueError, match="only for seed_persona"):
        MemoryEntry(
            memory_id="mem_teacher_000002",
            agent_id="teacher",
            memory_type="observation",
            text="Invalid initialization observation.",
            category="teacher_guidance",
            importance=7,
            created_at=0,
            last_accessed_at=0,
            source_step_id="seed_memory_initialization",
            source_step_type=AGENT_INITIALIZATION_SOURCE_STEP_TYPE,
        )
    with pytest.raises(ValueError, match="unsupported source_step_type"):
        MemoryEntry(
            memory_id="mem_teacher_000003",
            agent_id="teacher",
            memory_type="seed_persona",
            text="Invalid source.",
            category="teacher_guidance",
            importance=7,
            created_at=0,
            last_accessed_at=0,
            source_step_id="unknown",
            source_step_type="unknown_source",
        )


def test_memory_entry_rejects_last_access_before_creation() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        MemoryEntry(
            memory_id="mem_quiet_student_000001",
            agent_id="quiet_student",
            memory_type="observation",
            text="Tuesday observed an instruction.",
            category="classroom_norms",
            importance=6,
            created_at=4,
            last_accessed_at=3,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
        )


def test_memory_metadata_is_defensively_copied() -> None:
    supplied_metadata = {"turn_index": 1}
    memory = MemoryFactory("active_student").create_memory(
        memory_type="action",
        text="Monday asks a question.",
        category="question_asking",
        importance=5,
        created_at=2,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        metadata=supplied_metadata,
    )

    supplied_metadata["turn_index"] = 99

    assert memory.metadata == {"turn_index": 1}


def test_failed_creation_does_not_consume_a_deterministic_id() -> None:
    factory = MemoryFactory("academic_student")

    with pytest.raises(ValueError, match="between 1 and 10"):
        factory.create_memory(
            memory_type="reflection",
            text="Invalid importance.",
            category="academic_focus",
            importance=11,
            created_at=1,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
        )

    memory = factory.create_memory(
        memory_type="reflection",
        text="Wednesday plans a structured review.",
        category="academic_focus",
        importance=7,
        created_at=1,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
    )

    assert memory.memory_id == "mem_academic_student_000001"
