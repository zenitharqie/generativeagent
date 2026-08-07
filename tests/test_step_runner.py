"""Typed dispatch and duplicate-execution guards for Phase 08 steps."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from src.config_loader import load_experiment_config
from src.domain import SimulationStep
from src.simulation import NaturalBlockRunner, StepRunner, TargetEventRunner
from src.simulation.turn_executor import (
    SimulationExecutionError,
    SimulationProgress,
)


def _stubbed_step_runner(monkeypatch: pytest.MonkeyPatch):
    natural = object.__new__(NaturalBlockRunner)
    target = object.__new__(TargetEventRunner)
    calls: list[tuple[str, str]] = []

    def run_natural(self, **values):
        calls.append(("natural", values["step"].step_id))
        return "natural_result"

    def run_target(self, **values):
        calls.append(("target", values["step"].step_id))
        return "target_result"

    monkeypatch.setattr(NaturalBlockRunner, "run", run_natural)
    monkeypatch.setattr(TargetEventRunner, "run", run_target)
    return (
        StepRunner(
            natural_block_runner=natural,
            target_event_runner=target,
        ),
        calls,
    )


def _dispatch(runner: StepRunner, step):
    return runner.run_step(
        step=step,
        memory_streams={},
        previous_same_agent_responses={},
        progress=SimulationProgress(),
    )


def test_step_runner_dispatches_each_validated_subtype_to_its_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config()
    natural_step = next(
        step for step in config.steps if step.step_type == "natural_interaction_block"
    )
    target_step = next(
        step for step in config.steps if step.step_type == "target_event"
    )
    runner, calls = _stubbed_step_runner(monkeypatch)

    assert _dispatch(runner, natural_step) == "natural_result"
    assert _dispatch(runner, target_step) == "target_result"
    assert calls == [
        ("natural", natural_step.step_id),
        ("target", target_step.step_id),
    ]


def test_step_runner_rejects_duplicate_step_without_second_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = load_experiment_config().steps[0]
    runner, calls = _stubbed_step_runner(monkeypatch)
    _dispatch(runner, step)

    with pytest.raises(SimulationExecutionError, match="duplicate step execution"):
        _dispatch(runner, step)

    assert calls == [("natural", step.step_id)]


def test_step_runner_rejects_unsupported_base_step_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = load_experiment_config().steps[0]
    values = asdict(configured)
    for field_name in (
        "turns_per_agent",
        "total_turns",
        "teacher_opens_block",
        "turn_policy",
    ):
        values.pop(field_name)
    base_step = SimulationStep(**values)
    runner, calls = _stubbed_step_runner(monkeypatch)

    with pytest.raises(SimulationExecutionError, match="unsupported configured step type"):
        _dispatch(runner, base_step)

    assert calls == []
