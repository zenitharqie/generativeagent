"""Integrated dummy checks for controlled natural interaction blocks."""

from __future__ import annotations

from collections import Counter
import json

from src.config_loader import load_experiment_config
from src.simulation import (
    FIXED_NATURAL_TURN_ORDER,
    NaturalBlockExecutionResult,
)


def _read_jsonl(path):
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _natural_results(phase08_pilot_artifacts):
    result, _, _ = phase08_pilot_artifacts
    return tuple(
        step
        for step in result.step_results
        if isinstance(step, NaturalBlockExecutionResult)
    )


def test_pilot_executes_three_natural_blocks_in_snapshot_order(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    expected_ids = tuple(
        step.step_id
        for step in config.steps
        if step.day in (1, 2) and step.step_type == "natural_interaction_block"
    )

    natural_results = _natural_results(phase08_pilot_artifacts)

    assert tuple(item.step_id for item in natural_results) == expected_ids
    assert len(natural_results) == 3


def test_each_natural_block_uses_exact_documented_twelve_turn_schedule(
    phase08_pilot_artifacts,
) -> None:
    for block in _natural_results(phase08_pilot_artifacts):
        agents = tuple(turn.agent_id for turn in block.main_turn_results)

        assert agents == FIXED_NATURAL_TURN_ORDER
        assert len(agents) == 12
        assert agents[0] == "teacher"
        assert Counter(agents) == {
            "teacher": 3,
            "active_student": 3,
            "quiet_student": 3,
            "academic_student": 3,
        }
        assert all(left != right for left, right in zip(agents, agents[1:]))


def test_natural_turn_ids_indexes_and_timestamps_are_one_based(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    behavior = _read_jsonl(result.run_paths.behavior_logs)
    natural_logs = tuple(
        record
        for record in behavior
        if record["step_type"] == "natural_interaction_block"
    )

    expected_timestamp = 1
    for block in _natural_results(phase08_pilot_artifacts):
        records = tuple(
            record for record in natural_logs if record["step_id"] == block.step_id
        )
        assert [record["turn_index"] for record in records] == list(range(1, 13))
        assert [record["turn_id"] for record in records] == [
            f"{block.step_id}_turn_{index:02d}" for index in range(1, 13)
        ]
        assert [record["timestamp"] for record in records] == list(
            range(expected_timestamp, expected_timestamp + 12)
        )
        expected_timestamp += 12


def test_natural_context_is_speaker_labelled_ordered_and_reset_per_block(
    phase08_pilot_artifacts,
) -> None:
    for block in _natural_results(phase08_pilot_artifacts):
        expected_context: list[str] = []
        for turn in block.main_turn_results:
            assert turn.perception.conversation_context == tuple(expected_context)
            expected_context.append(
                f"{turn.perception.display_name}: {turn.response}"
            )
        assert block.conversation_context == tuple(expected_context)
        assert block.main_turn_results[0].perception.conversation_context == ()


def test_natural_reflections_use_configured_participant_order_and_final_time(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    steps = {step.step_id: step for step in config.steps}

    for block in _natural_results(phase08_pilot_artifacts):
        reflections = block.reflection_results
        assert tuple(item.agent_id for item in reflections) == steps[
            block.step_id
        ].participants
        assert len(reflections) == 4
        assert {item.logical_timestamp for item in reflections} == {
            block.final_logical_timestamp
        }
        assert all(
            item.conversation_context == block.conversation_context
            for item in reflections
        )


def test_natural_logs_and_structural_summaries_are_non_evaluation_records(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    retrieval = _read_jsonl(result.run_paths.retrieval_logs)
    behavior = _read_jsonl(result.run_paths.behavior_logs)
    summaries = _read_jsonl(result.run_paths.natural_block_summaries)

    natural_records = tuple(
        record
        for record in (*retrieval, *behavior)
        if record["step_type"] == "natural_interaction_block"
    )
    assert len(natural_records) == 72
    assert all(record["is_target_event"] is False for record in natural_records)
    assert all(record["evaluation_enabled"] is False for record in natural_records)
    assert len(summaries) == 3
    assert all(summary["completed_main_turn_count"] == 12 for summary in summaries)
    assert all(summary["reflection_count"] == 4 for summary in summaries)
    assert all(summary["final_logical_timestamp"] == summary["timestamp"] for summary in summaries)
