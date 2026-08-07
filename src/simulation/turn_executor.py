"""Shared deterministic execution and persistence for one simulation main turn."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.cognition import (
    CognitivePipeline,
    CognitiveResult,
    CognitiveTurnInput,
    PostStepReflectionInput,
    PostStepReflectionResult,
    build_perception,
    build_retrieval_query,
)
from src.config_loader import ExperimentConfig
from src.domain import SimulationStep, TargetEvent, TargetEventStep
from src.domain.agents import FIXED_AGENT_IDS
from src.llm import TextEmbeddingProvider
from src.logging_system import (
    ReflectionLogRecord,
    RunLogMetadata,
    StepLogMetadata,
    TurnLogMetadata,
    append_jsonl,
    cognitive_records_from_result,
    memory_record_from_entry,
    turn_metadata_from_result,
)
from src.memory import MemoryStream
from src.retrieval import validate_vector
from src.storage import RunPaths


class SimulationExecutionError(ValueError):
    """Raised when controller-owned execution state violates the Phase 08 contract."""


@dataclass
class SimulationProgress:
    """Mutable run-local counters advanced only after their persistence boundary."""

    next_logical_timestamp: int = 1
    completed_main_turns: int = 0
    completed_natural_blocks: int = 0
    completed_target_events: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject externally corrupted counter or logical-clock state."""
        for field_name in (
            "next_logical_timestamp",
            "completed_main_turns",
            "completed_natural_blocks",
            "completed_target_events",
        ):
            value = getattr(self, field_name)
            minimum = 1 if field_name == "next_logical_timestamp" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SimulationExecutionError(
                    f"{field_name} must be an integer greater than or equal to {minimum}"
                )
        if self.next_logical_timestamp != self.completed_main_turns + 1:
            raise SimulationExecutionError(
                "next_logical_timestamp must equal completed_main_turns + 1"
            )

    @property
    def final_logical_timestamp(self) -> int:
        """Return zero before the first turn, otherwise the last successful turn time."""
        self.validate()
        return self.next_logical_timestamp - 1

    def mark_main_turn_persisted(self, timestamp: int) -> None:
        """Advance exactly one tick after one fully persisted main turn."""
        self.validate()
        if timestamp != self.next_logical_timestamp:
            raise SimulationExecutionError(
                "persisted main-turn timestamp does not match the next logical timestamp"
            )
        self.completed_main_turns += 1
        self.next_logical_timestamp += 1
        self.validate()

    def mark_natural_block_completed(self) -> None:
        self.validate()
        self.completed_natural_blocks += 1

    def mark_target_event_completed(self) -> None:
        self.validate()
        self.completed_target_events += 1


@dataclass(frozen=True)
class MainTurnExecutionResult:
    """One successful main turn using the existing immutable cognitive result."""

    cognitive_result: CognitiveResult
    turn_metadata: TurnLogMetadata
    query_embedding: tuple[float, ...]
    conversation_entry: str


@dataclass(frozen=True)
class PostStepReflectionExecutionResult:
    """One successful post-step reflection and its persisted operation identity."""

    cognitive_result: PostStepReflectionResult
    operation_metadata: TurnLogMetadata


def build_main_turn_id(step_id: str, turn_index: int) -> str:
    """Build a deterministic one-based main-turn identifier."""
    if not isinstance(step_id, str) or not step_id.strip():
        raise SimulationExecutionError("step_id must be a non-empty string")
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
        raise SimulationExecutionError("turn_index must be a one-based positive integer")
    return f"{step_id}_turn_{turn_index:02d}"


def build_reflection_operation_id(step_id: str, participant_index: int) -> str:
    """Identify a post-step reflection without treating it as a main turn."""
    if not isinstance(step_id, str) or not step_id.strip():
        raise SimulationExecutionError("step_id must be a non-empty string")
    if (
        isinstance(participant_index, bool)
        or not isinstance(participant_index, int)
        or participant_index < 1
    ):
        raise SimulationExecutionError(
            "participant_index must be a one-based positive integer"
        )
    return f"{step_id}_reflection_{participant_index:02d}"


class TurnExecutor:
    """Invoke stable cognition once and persist its records in approved order."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        pipeline: CognitivePipeline,
        embedding_provider: TextEmbeddingProvider,
        run_metadata: RunLogMetadata,
        run_paths: RunPaths,
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise SimulationExecutionError("config must be an ExperimentConfig")
        if not isinstance(pipeline, CognitivePipeline):
            raise SimulationExecutionError("pipeline must be a CognitivePipeline")
        if not callable(getattr(embedding_provider, "embed", None)):
            raise SimulationExecutionError(
                "embedding_provider must provide embed(text)"
            )
        if not isinstance(run_metadata, RunLogMetadata):
            raise SimulationExecutionError("run_metadata must be RunLogMetadata")
        if not isinstance(run_paths, RunPaths):
            raise SimulationExecutionError("run_paths must be RunPaths")
        if run_paths.run_dir.name != run_metadata.run_id:
            raise SimulationExecutionError(
                "run_paths identity does not match run_metadata.run_id"
            )

        self._config = config
        self._pipeline = pipeline
        self._embedding_provider = embedding_provider
        self._run = run_metadata
        self._paths = run_paths
        self._agents = {agent.agent_id: agent for agent in config.agents}
        self._steps = {step.step_id: step for step in config.steps}
        self._events = {event.event_id: event for event in config.target_events}
        self._seen_main_turn_ids: set[str] = set()
        self._seen_reflection_ids: set[str] = set()

    @property
    def run_metadata(self) -> RunLogMetadata:
        return self._run

    @property
    def run_paths(self) -> RunPaths:
        return self._paths

    def step_metadata(
        self,
        step: SimulationStep,
        *,
        timestamp: int,
    ) -> StepLogMetadata:
        """Build canonical metadata for a configured step and logical timestamp."""
        self._validate_configured_step(step)
        return StepLogMetadata(
            run=self._run,
            step_id=step.step_id,
            step_type=step.step_type,
            is_target_event=step.is_target_event,
            evaluation_enabled=step.evaluation_enabled,
            day=step.day,
            period=step.period,
            timestamp=timestamp,
            event_id=step.event_id if isinstance(step, TargetEventStep) else None,
        )

    def execute_main_turn(
        self,
        *,
        step: SimulationStep,
        agent_id: str,
        memory_stream: MemoryStream,
        turn_index: int,
        conversation_context: list[str],
        previous_same_agent_responses: dict[str, str],
        progress: SimulationProgress,
    ) -> MainTurnExecutionResult:
        """Execute and persist one main response without an automatic retry."""
        self._validate_configured_step(step)
        agent = self._agents.get(agent_id)
        if agent is None:
            raise SimulationExecutionError(f"unknown agent_id: {agent_id!r}")
        if agent_id not in step.participants:
            raise SimulationExecutionError(
                f"agent {agent_id!r} is not a participant in {step.step_id!r}"
            )
        self._validate_stream(memory_stream, agent_id)
        if not isinstance(conversation_context, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in conversation_context
        ):
            raise SimulationExecutionError(
                "conversation_context must be a list of non-empty strings"
            )
        if not isinstance(previous_same_agent_responses, dict):
            raise SimulationExecutionError(
                "previous_same_agent_responses must be a dictionary"
            )
        if not isinstance(progress, SimulationProgress):
            raise SimulationExecutionError("progress must be SimulationProgress")
        progress.validate()

        turn_id = build_main_turn_id(step.step_id, turn_index)
        if turn_id in self._seen_main_turn_ids:
            raise SimulationExecutionError(f"duplicate main turn_id: {turn_id!r}")
        self._seen_main_turn_ids.add(turn_id)

        timestamp = progress.next_logical_timestamp
        event = self._resolve_event(step)
        context_tuple = tuple(conversation_context)
        perception = build_perception(
            agent=agent,
            step=step,
            logical_timestamp=timestamp,
            conversation_context=context_tuple,
            event=event,
        )
        retrieval_query = build_retrieval_query(perception)
        query_embedding = validate_vector(
            self._embedding_provider.embed(retrieval_query.text),
            field_name="query_embedding",
        )
        if not any(value != 0.0 for value in query_embedding):
            raise SimulationExecutionError("query_embedding must be non-zero")

        turn_input = CognitiveTurnInput(
            agent_id=agent_id,
            source_step_id=step.step_id,
            turn_id=turn_id,
            retrieval_config_id=self._run.retrieval_config_id,
            memory_stream=memory_stream,
            logical_timestamp=timestamp,
            query_embedding=query_embedding,
            conversation_context=context_tuple,
            event_id=event.event_id if event is not None else None,
            previous_same_agent_response=previous_same_agent_responses.get(agent_id),
            reflection_requested=False,
        )
        result = self._pipeline.run_turn(turn_input)
        self._validate_main_result(
            result,
            step=step,
            agent_id=agent_id,
            turn_id=turn_id,
            timestamp=timestamp,
            expected_perception=perception,
            retrieval_query_text=retrieval_query.text,
            event=event,
            memory_stream=memory_stream,
        )

        turn = turn_metadata_from_result(
            result,
            run=self._run,
            turn_index=turn_index,
            evaluation_enabled=step.evaluation_enabled,
        )
        records = cognitive_records_from_result(
            result,
            turn,
            query_embedding=query_embedding,
        )

        append_jsonl(self._paths.retrieval_logs, records.retrieval)
        append_jsonl(self._paths.behavior_logs, records.behavior)
        for memory_record in records.memories:
            append_jsonl(self._paths.memory_logs, memory_record)
        if records.reflection is not None:
            append_jsonl(self._paths.reflection_logs, records.reflection)

        conversation_entry = f"{agent.display_name}: {result.response}"
        conversation_context.append(conversation_entry)
        previous_same_agent_responses[agent_id] = result.response
        progress.mark_main_turn_persisted(timestamp)
        return MainTurnExecutionResult(
            cognitive_result=result,
            turn_metadata=turn,
            query_embedding=query_embedding,
            conversation_entry=conversation_entry,
        )

    def execute_post_step_reflection(
        self,
        *,
        step: SimulationStep,
        agent_id: str,
        participant_index: int,
        memory_stream: MemoryStream,
        conversation_context: tuple[str, ...],
        final_turn_result: CognitiveResult,
        final_logical_timestamp: int,
    ) -> PostStepReflectionExecutionResult:
        """Generate and persist one non-main-turn reflection after a completed step."""
        self._validate_configured_step(step)
        if agent_id not in step.participants:
            raise SimulationExecutionError(
                f"agent {agent_id!r} is not a participant in {step.step_id!r}"
            )
        self._validate_stream(memory_stream, agent_id)
        reflection_id = build_reflection_operation_id(
            step.step_id,
            participant_index,
        )
        if reflection_id in self._seen_reflection_ids:
            raise SimulationExecutionError(
                f"duplicate post-step reflection identity: {reflection_id!r}"
            )
        self._seen_reflection_ids.add(reflection_id)

        event = self._resolve_event(step)
        reflection_result = self._pipeline.run_post_step_reflection(
            PostStepReflectionInput(
                agent_id=agent_id,
                source_step_id=step.step_id,
                memory_stream=memory_stream,
                logical_timestamp=final_logical_timestamp,
                conversation_context=conversation_context,
                final_turn_result=final_turn_result,
                event_id=event.event_id if event is not None else None,
            )
        )
        self._validate_post_step_reflection_result(
            reflection_result,
            step=step,
            agent_id=agent_id,
            final_turn_result=final_turn_result,
            conversation_context=conversation_context,
            final_logical_timestamp=final_logical_timestamp,
            event=event,
            memory_stream=memory_stream,
        )
        operation = TurnLogMetadata(
            step=self.step_metadata(step, timestamp=final_logical_timestamp),
            turn_id=reflection_id,
            turn_index=participant_index,
            agent_id=agent_id,
        )
        memory_record = memory_record_from_entry(
            reflection_result.reflection_memory,
            run=self._run,
            turn=operation,
        )
        reflection_record = ReflectionLogRecord(
            turn=operation,
            reflection=reflection_result.reflection.reflection,
            category=reflection_result.reflection.category,
            memory_id=reflection_result.reflection_memory.memory_id,
        )

        append_jsonl(self._paths.memory_logs, memory_record)
        append_jsonl(self._paths.reflection_logs, reflection_record)
        return PostStepReflectionExecutionResult(
            cognitive_result=reflection_result,
            operation_metadata=operation,
        )

    def _validate_configured_step(self, step: SimulationStep) -> None:
        if not isinstance(step, SimulationStep):
            raise SimulationExecutionError("step must be a SimulationStep")
        configured = self._steps.get(step.step_id)
        if configured is None:
            raise SimulationExecutionError(f"unknown step_id: {step.step_id!r}")
        if configured != step:
            raise SimulationExecutionError(
                f"step {step.step_id!r} does not match the snapshotted configuration"
            )

    def _validate_stream(self, stream: MemoryStream, agent_id: str) -> None:
        if not isinstance(stream, MemoryStream):
            raise SimulationExecutionError("memory_stream must be a MemoryStream")
        if stream.agent_id != agent_id:
            raise SimulationExecutionError(
                "memory-stream owner does not match the requested agent"
            )
        if stream.run_id != self._run.run_id:
            raise SimulationExecutionError(
                "memory-stream run does not match the current run"
            )

    def _resolve_event(self, step: SimulationStep) -> TargetEvent | None:
        if not isinstance(step, TargetEventStep):
            return None
        event = self._events.get(step.event_id)
        if event is None:
            raise SimulationExecutionError(
                f"unknown event_id referenced by step: {step.event_id!r}"
            )
        if (event.title, event.description) != (step.title, step.description):
            raise SimulationExecutionError(
                f"event {event.event_id!r} does not match step {step.step_id!r}"
            )
        return event

    def _validate_main_result(
        self,
        result: CognitiveResult,
        *,
        step: SimulationStep,
        agent_id: str,
        turn_id: str,
        timestamp: int,
        expected_perception: Any,
        retrieval_query_text: str,
        event: TargetEvent | None,
        memory_stream: MemoryStream,
    ) -> None:
        if not isinstance(result, CognitiveResult):
            raise SimulationExecutionError("pipeline must return CognitiveResult")
        expected_event_id = event.event_id if event is not None else None
        if (
            result.turn_id != turn_id
            or result.agent_id != agent_id
            or result.source_step_id != step.step_id
            or result.source_step_type != step.step_type
            or result.event_id != expected_event_id
            or result.retrieval_config_id != self._run.retrieval_config_id
            or result.logical_timestamp != timestamp
            or result.perception != expected_perception
            or result.memory_generation_enabled != step.memory_generation_enabled
        ):
            raise SimulationExecutionError(
                "CognitiveResult identity does not match the requested main turn"
            )
        if result.retrieval_query.text != retrieval_query_text:
            raise SimulationExecutionError(
                "CognitiveResult retrieval query differs from the embedded query text"
            )
        if result.reflection_requested or result.reflection is not None:
            raise SimulationExecutionError(
                "ordinary Phase 08 main turns must not generate inline reflection"
            )
        result_memories = tuple(
            candidate.memory for candidate in result.retrieval_result.candidates
        ) + result.created_memories
        if any(
            memory.agent_id != agent_id or memory.run_id != self._run.run_id
            for memory in result_memories
        ):
            raise SimulationExecutionError(
                "CognitiveResult contains cross-agent or cross-run memory"
            )
        if step.memory_generation_enabled and (
            len(result.created_memories) != 1
            or result.created_memories[0].memory_type != "action"
            or not memory_stream.list_memories()
            or memory_stream.list_memories()[-1] != result.created_memories[0]
        ):
            raise SimulationExecutionError(
                "CognitiveResult action memory does not match committed turn state"
            )

    def _validate_post_step_reflection_result(
        self,
        result: PostStepReflectionResult,
        *,
        step: SimulationStep,
        agent_id: str,
        final_turn_result: CognitiveResult,
        conversation_context: tuple[str, ...],
        final_logical_timestamp: int,
        event: TargetEvent | None,
        memory_stream: MemoryStream,
    ) -> None:
        if not isinstance(result, PostStepReflectionResult):
            raise SimulationExecutionError(
                "post-step cognition must return PostStepReflectionResult"
            )
        expected_event_id = event.event_id if event is not None else None
        selected_memories = final_turn_result.retrieval_result.selected_memories
        if (
            result.final_turn_id != final_turn_result.turn_id
            or result.agent_id != agent_id
            or result.source_step_id != step.step_id
            or result.source_step_type != step.step_type
            or result.event_id != expected_event_id
            or result.retrieval_config_id != self._run.retrieval_config_id
            or result.run_id != self._run.run_id
            or result.logical_timestamp != final_logical_timestamp
            or result.conversation_context != conversation_context
            or result.selected_memory_ids
            != tuple(memory.memory_id for memory in selected_memories)
            or result.selected_memory_texts
            != tuple(memory.text for memory in selected_memories)
        ):
            raise SimulationExecutionError(
                "post-step reflection result identity or context does not match "
                "the requested operation"
            )
        memory = result.reflection_memory
        if (
            memory.agent_id != agent_id
            or memory.run_id != self._run.run_id
            or memory.source_step_id != step.step_id
            or memory.source_step_type != step.step_type
            or memory.created_at != final_logical_timestamp
            or memory.last_accessed_at != final_logical_timestamp
            or not memory_stream.list_memories()
            or memory_stream.list_memories()[-1] != memory
        ):
            raise SimulationExecutionError(
                "post-step reflection memory does not match the requested operation"
            )


def validate_memory_stream_mapping(
    streams: Mapping[str, MemoryStream],
    *,
    run_id: str,
) -> None:
    """Validate the complete four-agent run-isolated stream mapping."""
    if not isinstance(streams, Mapping):
        raise SimulationExecutionError("streams must be a mapping")
    if set(streams) != FIXED_AGENT_IDS:
        raise SimulationExecutionError(
            "streams must contain exactly the four fixed agent IDs"
        )
    if any(
        not isinstance(stream, MemoryStream)
        or stream.agent_id != agent_id
        or stream.run_id != run_id
        for agent_id, stream in streams.items()
    ):
        raise SimulationExecutionError(
            "every memory stream must match its agent and the current run"
        )
