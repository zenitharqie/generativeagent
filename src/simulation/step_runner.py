"""Typed dispatch for one configured simulation step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from src.domain import (
    NaturalInteractionBlockStep,
    SimulationStep,
    TargetEventStep,
)
from src.memory import MemoryStream
from src.simulation.natural_block_runner import (
    NaturalBlockExecutionResult,
    NaturalBlockRunner,
)
from src.simulation.target_event_runner import (
    TargetEventExecutionResult,
    TargetEventRunner,
)
from src.simulation.turn_executor import SimulationExecutionError, SimulationProgress


StepExecutionResult: TypeAlias = (
    NaturalBlockExecutionResult | TargetEventExecutionResult
)


class StepRunner:
    """Dispatch each configured step exactly once to its specialized runner."""

    def __init__(
        self,
        *,
        natural_block_runner: NaturalBlockRunner,
        target_event_runner: TargetEventRunner,
    ) -> None:
        if not isinstance(natural_block_runner, NaturalBlockRunner):
            raise SimulationExecutionError(
                "natural_block_runner must be NaturalBlockRunner"
            )
        if not isinstance(target_event_runner, TargetEventRunner):
            raise SimulationExecutionError(
                "target_event_runner must be TargetEventRunner"
            )
        self._natural = natural_block_runner
        self._target = target_event_runner
        self._executed_step_ids: set[str] = set()

    def run_step(
        self,
        *,
        step: SimulationStep,
        memory_streams: Mapping[str, MemoryStream],
        previous_same_agent_responses: dict[str, str],
        progress: SimulationProgress,
    ) -> StepExecutionResult:
        """Run one typed step and reject duplicate execution in this run."""
        if not isinstance(step, SimulationStep):
            raise SimulationExecutionError("step must be a SimulationStep")
        if step.step_id in self._executed_step_ids:
            raise SimulationExecutionError(
                f"duplicate step execution: {step.step_id!r}"
            )
        self._executed_step_ids.add(step.step_id)

        if isinstance(step, NaturalInteractionBlockStep):
            return self._natural.run(
                step=step,
                memory_streams=memory_streams,
                previous_same_agent_responses=previous_same_agent_responses,
                progress=progress,
            )
        if isinstance(step, TargetEventStep):
            return self._target.run(
                step=step,
                memory_streams=memory_streams,
                previous_same_agent_responses=previous_same_agent_responses,
                progress=progress,
            )
        raise SimulationExecutionError(
            f"unsupported configured step type: {type(step).__name__}"
        )
