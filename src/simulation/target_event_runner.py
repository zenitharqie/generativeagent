"""Execution of one fixed-participant target-event response step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.cognition import CognitiveResult, PostStepReflectionResult
from src.domain import TargetEventStep
from src.domain.agents import FIXED_AGENT_IDS
from src.logging_system import TargetEventSummaryRecord, append_jsonl
from src.memory import MemoryStream
from src.simulation.turn_executor import (
    SimulationExecutionError,
    SimulationProgress,
    TurnExecutor,
    validate_memory_stream_mapping,
)


@dataclass(frozen=True)
class TargetEventExecutionResult:
    """Immutable successful output for one fully persisted target event."""

    step_id: str
    event_id: str
    main_turn_results: tuple[CognitiveResult, ...]
    reflection_results: tuple[PostStepReflectionResult, ...]
    conversation_context: tuple[str, ...]
    final_logical_timestamp: int

    @property
    def ordered_main_turn_ids(self) -> tuple[str, ...]:
        return tuple(result.turn_id for result in self.main_turn_results)


class TargetEventRunner:
    """Run one response and one post-step reflection per configured participant."""

    def __init__(self, *, turn_executor: TurnExecutor) -> None:
        if not isinstance(turn_executor, TurnExecutor):
            raise SimulationExecutionError("turn_executor must be a TurnExecutor")
        self._turn_executor = turn_executor

    def run(
        self,
        *,
        step: TargetEventStep,
        memory_streams: Mapping[str, MemoryStream],
        previous_same_agent_responses: dict[str, str],
        progress: SimulationProgress,
    ) -> TargetEventExecutionResult:
        """Execute one event in the exact ordered ``step.participants`` tuple."""
        if not isinstance(step, TargetEventStep):
            raise SimulationExecutionError(
                "target event runner requires TargetEventStep"
            )
        if (
            len(step.participants) != 4
            or len(set(step.participants)) != 4
            or set(step.participants) != FIXED_AGENT_IDS
        ):
            raise SimulationExecutionError(
                "target event participants must be the four fixed agents exactly once"
            )
        validate_memory_stream_mapping(
            memory_streams,
            run_id=self._turn_executor.run_metadata.run_id,
        )
        if not isinstance(previous_same_agent_responses, dict):
            raise SimulationExecutionError(
                "previous_same_agent_responses must be a dictionary"
            )
        if not isinstance(progress, SimulationProgress):
            raise SimulationExecutionError("progress must be SimulationProgress")
        progress.validate()

        context: list[str] = []
        main_results: list[CognitiveResult] = []
        final_by_agent: dict[str, CognitiveResult] = {}
        starting_turn_count = progress.completed_main_turns

        for turn_index, agent_id in enumerate(step.participants, start=1):
            turn = self._turn_executor.execute_main_turn(
                step=step,
                agent_id=agent_id,
                memory_stream=memory_streams[agent_id],
                turn_index=turn_index,
                conversation_context=context,
                previous_same_agent_responses=previous_same_agent_responses,
                progress=progress,
            )
            main_results.append(turn.cognitive_result)
            final_by_agent[agent_id] = turn.cognitive_result

        if progress.completed_main_turns - starting_turn_count != 4:
            raise SimulationExecutionError(
                "target event did not persist exactly four main turns"
            )
        final_timestamp = progress.final_logical_timestamp
        complete_context = tuple(context)

        reflections: list[PostStepReflectionResult] = []
        for participant_index, agent_id in enumerate(step.participants, start=1):
            reflection = self._turn_executor.execute_post_step_reflection(
                step=step,
                agent_id=agent_id,
                participant_index=participant_index,
                memory_stream=memory_streams[agent_id],
                conversation_context=complete_context,
                final_turn_result=final_by_agent[agent_id],
                final_logical_timestamp=final_timestamp,
            )
            reflections.append(reflection.cognitive_result)

        counts = {agent_id: 1 for agent_id in step.participants}
        summary = TargetEventSummaryRecord(
            step=self._turn_executor.step_metadata(
                step,
                timestamp=final_timestamp,
            ),
            ordered_main_turn_ids=tuple(result.turn_id for result in main_results),
            participant_ids=step.participants,
            per_agent_main_turn_counts=counts,
            completed_main_turn_count=len(main_results),
            reflection_count=len(reflections),
            final_logical_timestamp=final_timestamp,
            ordered_responding_agent_ids=step.participants,
            response_count=len(main_results),
        )
        append_jsonl(self._turn_executor.run_paths.event_summaries, summary)
        progress.mark_target_event_completed()
        return TargetEventExecutionResult(
            step_id=step.step_id,
            event_id=step.event_id,
            main_turn_results=tuple(main_results),
            reflection_results=tuple(reflections),
            conversation_context=complete_context,
            final_logical_timestamp=final_timestamp,
        )
