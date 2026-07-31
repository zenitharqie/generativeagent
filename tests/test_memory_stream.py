from dataclasses import replace

import pytest

from src.memory import MemoryFactory, MemoryStream, MemoryStreamError


def build_stream() -> tuple[MemoryStream, tuple]:
    factory = MemoryFactory("quiet_student", run_id="run_01")
    stream = MemoryStream("quiet_student", run_id="run_01")
    memories = (
        factory.create_memory(
            memory_type="seed_persona",
            text="Tuesday listens before speaking.",
            category="quiet_style",
            importance=7,
            created_at=0,
            source_step_id="seed_memory_initialization",
            source_step_type="agent_initialization",
        ),
        factory.create_memory(
            memory_type="observation",
            text="Tuesday heard the classroom rule.",
            category="classroom_norms",
            importance=6,
            created_at=1,
            source_step_id="day_01_natural_01",
            source_step_type="natural_interaction_block",
        ),
        factory.create_memory(
            memory_type="reflection",
            text="Tuesday should prepare before the test.",
            category="exam_concern",
            importance=8,
            created_at=2,
            source_step_id="day_02_target_01",
            source_step_type="target_event",
        ),
    )
    for memory in memories:
        stream.add_memory(memory)
    return stream, memories


def test_add_list_and_filter_preserve_append_order() -> None:
    stream, memories = build_stream()

    assert len(stream) == 3
    assert stream.list_memories() == memories
    assert stream.filter_by_memory_type("observation") == (memories[1],)
    assert stream.filter_by_source_step_type("target_event") == (memories[2],)
    assert stream.filter_memories(
        memory_type="seed_persona",
        source_step_type="agent_initialization",
    ) == (memories[0],)


def test_listing_and_filtering_do_not_update_access_metadata() -> None:
    stream, memories = build_stream()

    stream.list_memories()
    stream.filter_by_memory_type("reflection")

    assert [memory.last_accessed_at for memory in stream.list_memories()] == [
        memory.last_accessed_at for memory in memories
    ]


def test_update_last_accessed_at_changes_selected_memories_only() -> None:
    stream, memories = build_stream()

    updated = stream.update_last_accessed_at(
        [memories[0].memory_id, memories[2].memory_id],
        accessed_at=5,
    )
    stored = stream.list_memories()

    assert [memory.memory_id for memory in updated] == [
        memories[0].memory_id,
        memories[2].memory_id,
    ]
    assert [memory.last_accessed_at for memory in stored] == [5, 1, 5]
    assert [memory.memory_id for memory in stored] == [
        memory.memory_id for memory in memories
    ]


def test_access_update_validates_all_ids_before_mutating() -> None:
    stream, memories = build_stream()

    with pytest.raises(MemoryStreamError, match="unknown memory_id"):
        stream.update_last_accessed_at(
            [memories[0].memory_id, "mem_quiet_student_unknown"],
            accessed_at=7,
        )

    assert stream.list_memories() == memories


def test_access_timestamp_cannot_move_backward() -> None:
    stream, memories = build_stream()

    with pytest.raises(MemoryStreamError, match="must not move backward"):
        stream.update_last_accessed_at([memories[2].memory_id], accessed_at=1)

    assert stream.list_memories() == memories


def test_stream_rejects_duplicate_cross_agent_and_cross_run_memories() -> None:
    stream, memories = build_stream()
    foreign_owner = MemoryFactory("active_student", run_id="run_01").create_memory(
        memory_type="observation",
        text="Monday observed the class.",
        category="social_learning",
        importance=5,
        created_at=3,
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
    )
    foreign_run = replace(memories[0], memory_id="another_id", run_id="run_02")

    with pytest.raises(MemoryStreamError, match="duplicate memory_id"):
        stream.add_memory(memories[0])
    with pytest.raises(MemoryStreamError, match="belongs to 'active_student'"):
        stream.add_memory(foreign_owner)
    with pytest.raises(MemoryStreamError, match="does not belong to run"):
        stream.add_memory(foreign_run)


@pytest.mark.parametrize(
    ("filter_name", "invalid_value"),
    [
        ("memory_type", "summary"),
        ("source_step_type", "free_world_event"),
    ],
)
def test_filters_reject_unsupported_values(
    filter_name: str, invalid_value: str
) -> None:
    stream, _ = build_stream()

    with pytest.raises(MemoryStreamError, match=f"unsupported {filter_name}"):
        stream.filter_memories(**{filter_name: invalid_value})
