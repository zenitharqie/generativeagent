from src.config_loader import load_experiment_config
from src.memory import (
    AGENT_INITIALIZATION_SOURCE_STEP_TYPE,
    SEED_MEMORY_SOURCE_STEP_ID,
    initialize_memory_streams,
)


def test_seed_memories_load_from_validated_config_for_all_agents() -> None:
    config = load_experiment_config()
    streams = initialize_memory_streams(config, run_id="run_01")

    assert set(streams) == {
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
    }
    assert all(len(stream) == 3 for stream in streams.values())

    for agent in config.agents:
        memories = streams[agent.agent_id].list_memories()
        assert [memory.text for memory in memories] == [
            seed.text for seed in agent.seed_memories
        ]
        assert [memory.category for memory in memories] == [
            seed.category for seed in agent.seed_memories
        ]
        assert [memory.importance for memory in memories] == [
            seed.importance for seed in agent.seed_memories
        ]
        assert all(memory.memory_type == "seed_persona" for memory in memories)
        assert all(memory.agent_id == agent.agent_id for memory in memories)
        assert all(memory.run_id == "run_01" for memory in memories)
        assert all(memory.created_at == 0 for memory in memories)
        assert all(memory.last_accessed_at == 0 for memory in memories)
        assert all(
            memory.source_step_id == SEED_MEMORY_SOURCE_STEP_ID
            for memory in memories
        )
        assert all(
            memory.source_step_type == AGENT_INITIALIZATION_SOURCE_STEP_TYPE
            for memory in memories
        )


def test_seed_ids_are_stable_and_unique_within_each_run() -> None:
    config = load_experiment_config()
    first = initialize_memory_streams(config, run_id="run_01")
    second = initialize_memory_streams(config, run_id="run_02")

    first_ids = [
        memory.memory_id
        for agent in config.agents
        for memory in first[agent.agent_id].list_memories()
    ]
    second_ids = [
        memory.memory_id
        for agent in config.agents
        for memory in second[agent.agent_id].list_memories()
    ]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_repeated_initialization_does_not_share_mutable_run_state() -> None:
    config = load_experiment_config()
    first = initialize_memory_streams(config, run_id="run_01")
    second = initialize_memory_streams(config, run_id="run_01")

    for agent in config.agents:
        first_stream = first[agent.agent_id]
        second_stream = second[agent.agent_id]
        first_memories = first_stream.list_memories()
        second_memories = second_stream.list_memories()

        assert first_stream is not second_stream
        assert first_memories == second_memories
        assert all(
            first_memory is not second_memory
            for first_memory, second_memory in zip(first_memories, second_memories)
        )
        assert all(
            first_memory.metadata is not second_memory.metadata
            for first_memory, second_memory in zip(first_memories, second_memories)
        )

    first_teacher_memory = first["teacher"].list_memories()[0]
    second_teacher_memory = second["teacher"].list_memories()[0]
    first_teacher_memory.metadata["changed_in_first_run"] = True

    assert "changed_in_first_run" not in second_teacher_memory.metadata


def test_seed_initialization_uses_a_caller_supplied_logical_timestamp() -> None:
    config = load_experiment_config()
    streams = initialize_memory_streams(
        config,
        run_id="run_01",
        created_at=12,
    )

    assert all(
        memory.created_at == 12 and memory.last_accessed_at == 12
        for stream in streams.values()
        for memory in stream.list_memories()
    )
