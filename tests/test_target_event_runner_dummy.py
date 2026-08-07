"""Integrated dummy checks for controlled target-event execution."""

from __future__ import annotations

import json

from src.config_loader import load_experiment_config
from src.simulation import TargetEventExecutionResult


def _read_jsonl(path):
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _target_result(phase08_pilot_artifacts) -> TargetEventExecutionResult:
    result, _, _ = phase08_pilot_artifacts
    targets = tuple(
        step
        for step in result.step_results
        if isinstance(step, TargetEventExecutionResult)
    )
    assert len(targets) == 1
    return targets[0]


def test_pilot_target_event_resolves_configured_step_and_event(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    configured = next(
        step
        for step in config.steps
        if step.day == 2 and step.step_type == "target_event"
    )
    target = _target_result(phase08_pilot_artifacts)

    assert target.step_id == configured.step_id
    assert target.event_id == configured.event_id
    assert all(turn.event_id == configured.event_id for turn in target.main_turn_results)
    assert all(
        turn.perception.event_title is not None
        and turn.perception.event_description is not None
        for turn in target.main_turn_results
    )


def test_target_event_uses_exact_configured_participant_order_once_each(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    configured = next(
        step for step in config.steps if step.step_id == "day_02_target_01"
    )
    target = _target_result(phase08_pilot_artifacts)

    assert tuple(turn.agent_id for turn in target.main_turn_results) == (
        configured.participants
    )
    assert tuple(item.agent_id for item in target.reflection_results) == (
        configured.participants
    )
    assert len(target.main_turn_results) == len(set(configured.participants)) == 4


def test_target_turn_indexes_ids_and_final_logical_time_are_exact(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    target = _target_result(phase08_pilot_artifacts)
    records = tuple(
        record
        for record in _read_jsonl(result.run_paths.behavior_logs)
        if record["step_id"] == target.step_id
    )

    assert [record["turn_index"] for record in records] == [1, 2, 3, 4]
    assert [record["turn_id"] for record in records] == [
        f"{target.step_id}_turn_{index:02d}" for index in range(1, 5)
    ]
    assert [record["timestamp"] for record in records] == [37, 38, 39, 40]
    assert target.final_logical_timestamp == 40
    assert {item.logical_timestamp for item in target.reflection_results} == {40}


def test_target_context_is_step_local_speaker_labelled_and_ordered(
    phase08_pilot_artifacts,
) -> None:
    target = _target_result(phase08_pilot_artifacts)
    expected_context: list[str] = []
    for turn in target.main_turn_results:
        assert turn.perception.conversation_context == tuple(expected_context)
        expected_context.append(f"{turn.perception.display_name}: {turn.response}")

    assert target.main_turn_results[0].perception.conversation_context == ()
    assert target.conversation_context == tuple(expected_context)
    assert all(
        reflection.conversation_context == target.conversation_context
        for reflection in target.reflection_results
    )


def test_target_logs_and_structural_summary_are_evaluation_records(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    target = _target_result(phase08_pilot_artifacts)
    retrieval = tuple(
        record
        for record in _read_jsonl(result.run_paths.retrieval_logs)
        if record["step_id"] == target.step_id
    )
    behavior = tuple(
        record
        for record in _read_jsonl(result.run_paths.behavior_logs)
        if record["step_id"] == target.step_id
    )
    summaries = _read_jsonl(result.run_paths.event_summaries)

    assert len(retrieval) == len(behavior) == 4
    assert all(record["is_target_event"] is True for record in (*retrieval, *behavior))
    assert all(record["evaluation_enabled"] is True for record in (*retrieval, *behavior))
    assert summaries == (
        {
            **summaries[0],
            "ordered_responding_agent_ids": [
                turn.agent_id for turn in target.main_turn_results
            ],
        },
    )
    assert summaries[0]["event_id"] == target.event_id
    assert summaries[0]["response_count"] == 4
    assert summaries[0]["reflection_count"] == 4
