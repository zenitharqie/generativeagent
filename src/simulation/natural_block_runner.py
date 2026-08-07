"""Execution of one controlled twelve-turn natural interaction block."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from src.cognition import CognitiveResult, PostStepReflectionResult
from src.domain import NaturalInteractionBlockStep
from src.logging_system import NaturalBlockSummaryRecord, append_jsonl
from src.memory import MemoryStream
from src.simulation.turn_executor import (
    SimulationExecutionError,
    SimulationProgress,
    TurnExecutor,
    validate_memory_stream_mapping,
)
from src.simulation.turn_policy import TurnPolicyManager


@dataclass(frozen=True)
class NaturalBlockExecutionResult:
    """Immutable successful output for one fully persisted natural block."""

    step_id: str
    main_turn_results: tuple[CognitiveResult, ...]
    reflection_results: tuple[PostStepReflectionResult, ...]
    conversation_context: tuple[str, ...]
    final_logical_timestamp: int

    @property
    def ordered_main_turn_ids(self) -> tuple[str, ...]:
        return tuple(result.turn_id for result in self.main_turn_results)


class NaturalBlockRunner:
    """Run natural main turns, post-step reflections, and structural summary."""

    def __init__(
        self,
        *,
        turn_executor: TurnExecutor,
        turn_policy: TurnPolicyManager | None = None,
    ) -> None:
        if not isinstance(turn_executor, TurnExecutor):
            raise SimulationExecutionError("turn_executor must be a TurnExecutor")
        policy = TurnPolicyManager() if turn_policy is None else turn_policy
        if not isinstance(policy, TurnPolicyManager):
            raise SimulationExecutionError("turn_policy must be TurnPolicyManager")
        self._turn_executor = turn_executor
        self._turn_policy = policy

    def run(
        self,
        *,
        step: NaturalInteractionBlockStep,
        memory_streams: Mapping[str, MemoryStream],
        previous_same_agent_responses: dict[str, str],
        progress: SimulationProgress,
    ) -> NaturalBlockExecutionResult:
        """Execute one block; a failed reflection or summary leaves it incomplete."""
        if not isinstance(step, NaturalInteractionBlockStep):
            raise SimulationExecutionError(
                "natural block runner requires NaturalInteractionBlockStep"
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

        turn_order = self._turn_policy.build_turn_order(step)
        context: list[str] = []
        main_results: list[CognitiveResult] = []
        final_by_agent: dict[str, CognitiveResult] = {}
        starting_turn_count = progress.completed_main_turns

        for turn_index, agent_id in enumerate(turn_order, start=1):
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

        if progress.completed_main_turns - starting_turn_count != 12:
            raise SimulationExecutionError(
                "natural block did not persist exactly twelve main turns"
            )
        final_timestamp = progress.final_logical_timestamp
        complete_context = tuple(context)

        reflections: list[PostStepReflectionResult] = []
        for participant_index, agent_id in enumerate(step.participants, start=1):
            final_turn = final_by_agent.get(agent_id)
            if final_turn is None:
                raise SimulationExecutionError(
                    f"natural block has no final main turn for {agent_id!r}"
                )
            reflection = self._turn_executor.execute_post_step_reflection(
                step=step,
                agent_id=agent_id,
                participant_index=participant_index,
                memory_stream=memory_streams[agent_id],
                conversation_context=complete_context,
                final_turn_result=final_turn,
                final_logical_timestamp=final_timestamp,
            )
            reflections.append(reflection.cognitive_result)

        speaker_counts = Counter(turn_order)
        summary = NaturalBlockSummaryRecord(
            step=self._turn_executor.step_metadata(
                step,
                timestamp=final_timestamp,
            ),
            ordered_main_turn_ids=tuple(result.turn_id for result in main_results),
            participant_ids=step.participants,
            per_agent_main_turn_counts={
                agent_id: speaker_counts[agent_id]
                for agent_id in step.participants
            },
            completed_main_turn_count=len(main_results),
            reflection_count=len(reflections),
            final_logical_timestamp=final_timestamp,
        )
        append_jsonl(
            self._turn_executor.run_paths.natural_block_summaries,
            summary,
        )
        progress.mark_natural_block_completed()
        return NaturalBlockExecutionResult(
            step_id=step.step_id,
            main_turn_results=tuple(main_results),
            reflection_results=tuple(reflections),
            conversation_context=complete_context,
            final_logical_timestamp=final_timestamp,
        )
