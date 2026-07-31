"""Initialize fresh per-agent memory streams from validated agent profiles."""

from __future__ import annotations

from src.config_loader.loader import ExperimentConfig
from src.domain.agents import FIXED_AGENT_IDS, AgentProfile
from src.domain.memory import AGENT_INITIALIZATION_SOURCE_STEP_TYPE, MemoryEntry
from src.memory.memory_factory import MemoryFactory
from src.memory.memory_stream import MemoryStream, MemoryStreamError


SEED_MEMORY_SOURCE_STEP_ID = "seed_memory_initialization"


def create_seed_memories(
    agent: AgentProfile,
    *,
    run_id: str,
    created_at: int = 0,
) -> tuple[MemoryEntry, ...]:
    """Create reproducibly ordered seed memories using configured importance."""
    if not isinstance(agent, AgentProfile):
        raise TypeError("agent must be an AgentProfile")

    factory = MemoryFactory(agent.agent_id, run_id=run_id)
    return tuple(
        factory.create_memory(
            memory_type="seed_persona",
            text=seed.text,
            category=seed.category,
            importance=seed.importance,
            created_at=created_at,
            source_step_id=SEED_MEMORY_SOURCE_STEP_ID,
            source_step_type=AGENT_INITIALIZATION_SOURCE_STEP_TYPE,
            metadata={"seed_index": seed_index},
        )
        for seed_index, seed in enumerate(agent.seed_memories, start=1)
    )


def initialize_memory_streams(
    config: ExperimentConfig,
    *,
    run_id: str,
    created_at: int = 0,
) -> dict[str, MemoryStream]:
    """Build four isolated streams from validated config-loader output."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    agent_ids = tuple(agent.agent_id for agent in config.agents)
    if len(agent_ids) != 4 or set(agent_ids) != FIXED_AGENT_IDS:
        raise MemoryStreamError(
            "memory initialization requires exactly the four fixed agents"
        )

    streams: dict[str, MemoryStream] = {}
    for agent in config.agents:
        stream = MemoryStream(agent.agent_id, run_id=run_id)
        for memory in create_seed_memories(
            agent,
            run_id=run_id,
            created_at=created_at,
        ):
            stream.add_memory(memory)
        streams[agent.agent_id] = stream
    return streams
