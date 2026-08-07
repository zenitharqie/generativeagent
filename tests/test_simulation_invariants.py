"""Research-critical fixed totals, time, isolation, and determinism checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_loader import load_experiment_config
from src.config_loader.loader import CONFIG_FILES
from src.llm import DummyEmbeddingProvider
from src.simulation import RunController, build_deterministic_dummy_llm_client
from src.simulation.turn_executor import (
    SimulationExecutionError,
    SimulationProgress,
)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _run_pilot(
    output_root: Path,
    *,
    experiment_id: str,
    run_id: str,
    repetition_index: int = 1,
):
    return RunController(
        llm_client=build_deterministic_dummy_llm_client(),
        embedding_provider=DummyEmbeddingProvider(),
    ).run(
        output_root=output_root,
        experiment_id=experiment_id,
        run_id=run_id,
        retrieval_config_id="balanced",
        repetition_index=repetition_index,
        days=(1, 2),
    )


def test_full_run_executes_validated_timeline_and_exact_fixed_totals(
    phase08_full_artifacts,
) -> None:
    result, llm_client, embedding_provider = phase08_full_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    summary = result.run_summary
    natural_steps = tuple(
        step for step in config.steps if step.step_type == "natural_interaction_block"
    )
    target_steps = tuple(
        step for step in config.steps if step.step_type == "target_event"
    )
    derived_main_turns = sum(step.total_turns for step in natural_steps) + sum(
        len(step.participants) for step in target_steps
    )

    assert len(natural_steps) == config.settings.natural_blocks_total == 7
    assert len(target_steps) == config.settings.target_events_total == 5
    assert derived_main_turns == (7 * 12) + (5 * 4) == 104
    assert tuple(step.step_id for step in result.step_results) == tuple(
        step.step_id for step in config.steps
    )
    assert summary.status == "completed"
    assert (
        summary.completed_natural_blocks,
        summary.completed_target_events,
        summary.completed_main_turns,
        summary.error_count,
    ) == (7, 5, derived_main_turns, 0)
    assert len(llm_client.invocations) == (104 * 3) + (48 * 2)
    assert len(embedding_provider.received_texts) == 12 + (104 * 2) + 48


def test_full_run_main_clock_is_exactly_one_through_104(
    phase08_full_artifacts,
) -> None:
    result, _, _ = phase08_full_artifacts
    turns = tuple(
        turn
        for step in result.step_results
        for turn in step.main_turn_results
    )
    behavior = _read_jsonl(result.run_paths.behavior_logs)
    retrieval = _read_jsonl(result.run_paths.retrieval_logs)

    assert len(turns) == len(behavior) == len(retrieval) == 104
    assert [turn.logical_timestamp for turn in turns] == list(range(1, 105))
    assert [record["timestamp"] for record in behavior] == list(range(1, 105))
    assert [record["timestamp"] for record in retrieval] == list(range(1, 105))
    assert turns[0].logical_timestamp == 1
    assert turns[-1].logical_timestamp == 104


def test_reflection_reuses_step_final_timestamp_without_clock_ticks(
    phase08_full_artifacts,
) -> None:
    result, _, _ = phase08_full_artifacts
    reflection_logs = _read_jsonl(result.run_paths.reflection_logs)
    assert len(reflection_logs) == 48

    prior_final = 0
    for step in result.step_results:
        main_times = tuple(turn.logical_timestamp for turn in step.main_turn_results)
        reflection_times = tuple(
            item.logical_timestamp for item in step.reflection_results
        )
        persisted_reflections = tuple(
            record
            for record in reflection_logs
            if record["step_id"] == step.step_id
        )

        assert main_times[0] == prior_final + 1
        assert main_times == tuple(range(main_times[0], main_times[-1] + 1))
        assert reflection_times == (main_times[-1],) * 4
        assert tuple(record["timestamp"] for record in persisted_reflections) == (
            main_times[-1],
        ) * 4
        prior_final = main_times[-1]
    assert prior_final == 104


def test_full_artifact_counts_match_main_and_post_step_policies(
    phase08_full_artifacts,
) -> None:
    result, _, _ = phase08_full_artifacts
    paths = result.run_paths
    memories = _read_jsonl(paths.memory_logs)

    assert len(memories) == 12 + 104 + 48
    assert sum(record["memory_type"] == "seed_persona" for record in memories) == 12
    assert sum(record["memory_type"] == "action" for record in memories) == 104
    assert sum(record["memory_type"] == "reflection" for record in memories) == 48
    assert len(_read_jsonl(paths.reflection_logs)) == 48
    assert len(_read_jsonl(paths.natural_block_summaries)) == 7
    assert len(_read_jsonl(paths.event_summaries)) == 5
    assert paths.api_usage_logs.read_bytes() == b""
    assert paths.errors.read_bytes() == b""


def test_identical_fresh_runs_produce_byte_identical_results_and_logs(
    tmp_path: Path,
) -> None:
    first = _run_pilot(
        tmp_path / "first_root",
        experiment_id="determinism",
        run_id="same_identity",
    )
    second = _run_pilot(
        tmp_path / "second_root",
        experiment_id="determinism",
        run_id="same_identity",
    )

    assert first.step_results == second.step_results
    assert first.run_summary == second.run_summary
    for first_path, second_path in zip(
        first.run_paths.jsonl_files,
        second.run_paths.jsonl_files,
    ):
        assert first_path.name == second_path.name
        assert first_path.read_bytes() == second_path.read_bytes()
    assert first.run_paths.run_summary.read_bytes() == (
        second.run_paths.run_summary.read_bytes()
    )


def test_identical_fresh_full_runs_produce_byte_identical_complete_artifacts(
    tmp_path: Path,
) -> None:
    def run_fresh(output_root: Path):
        return RunController(
            llm_client=build_deterministic_dummy_llm_client(),
            embedding_provider=DummyEmbeddingProvider(),
        ).run(
            output_root=output_root,
            experiment_id="full_run_determinism",
            run_id="balanced_rep01",
            retrieval_config_id="balanced",
            repetition_index=1,
        )

    first = run_fresh(tmp_path / "first_root")
    second = run_fresh(tmp_path / "second_root")

    for result in (first, second):
        assert result.scope_succeeded is True
        assert result.run_summary.status == "completed"
        assert (
            result.run_summary.completed_natural_blocks,
            result.run_summary.completed_target_events,
            result.run_summary.completed_main_turns,
        ) == (7, 5, 104)

        turns = tuple(
            turn
            for step in result.step_results
            for turn in step.main_turn_results
        )
        expected_main_timestamps = tuple(range(1, 105))
        assert tuple(turn.logical_timestamp for turn in turns) == (
            expected_main_timestamps
        )
        assert tuple(
            record["timestamp"]
            for record in _read_jsonl(result.run_paths.behavior_logs)
        ) == expected_main_timestamps
        assert tuple(
            record["timestamp"]
            for record in _read_jsonl(result.run_paths.retrieval_logs)
        ) == expected_main_timestamps

        reflection_records = _read_jsonl(result.run_paths.reflection_logs)
        assert len(reflection_records) == 48
        previous_final_timestamp = 0
        for step in result.step_results:
            main_timestamps = tuple(
                turn.logical_timestamp for turn in step.main_turn_results
            )
            reflection_timestamps = tuple(
                reflection.logical_timestamp
                for reflection in step.reflection_results
            )
            persisted_reflection_timestamps = tuple(
                record["timestamp"]
                for record in reflection_records
                if record["step_id"] == step.step_id
            )
            assert main_timestamps[0] == previous_final_timestamp + 1
            assert reflection_timestamps == (main_timestamps[-1],) * 4
            assert persisted_reflection_timestamps == (main_timestamps[-1],) * 4
            previous_final_timestamp = main_timestamps[-1]
        assert previous_final_timestamp == 104

    assert first.requested_days == second.requested_days
    assert first.scope_succeeded == second.scope_succeeded
    assert first.step_results == second.step_results
    assert first.run_summary == second.run_summary

    expected_jsonl_names = (
        "retrieval_logs.jsonl",
        "behavior_logs.jsonl",
        "memory_logs.jsonl",
        "reflection_logs.jsonl",
        "errors.jsonl",
        "event_summaries.jsonl",
        "natural_block_summaries.jsonl",
        "api_usage_logs.jsonl",
    )
    assert tuple(path.name for path in first.run_paths.jsonl_files) == (
        expected_jsonl_names
    )
    assert tuple(path.name for path in second.run_paths.jsonl_files) == (
        expected_jsonl_names
    )
    for filename in expected_jsonl_names:
        assert (first.run_paths.run_dir / filename).read_bytes() == (
            second.run_paths.run_dir / filename
        ).read_bytes()

    assert first.run_paths.run_summary.read_bytes() == (
        second.run_paths.run_summary.read_bytes()
    )
    expected_config_names = tuple(CONFIG_FILES.values())
    assert tuple(
        sorted(path.name for path in first.run_paths.config_snapshot.iterdir())
    ) == tuple(sorted(expected_config_names))
    assert tuple(
        sorted(path.name for path in second.run_paths.config_snapshot.iterdir())
    ) == tuple(sorted(expected_config_names))
    for filename in expected_config_names:
        assert (first.run_paths.config_snapshot / filename).read_bytes() == (
            second.run_paths.config_snapshot / filename
        ).read_bytes()


def test_fresh_runs_with_different_ids_remain_strictly_isolated(
    tmp_path: Path,
) -> None:
    first = _run_pilot(
        tmp_path,
        experiment_id="isolation",
        run_id="run_one",
    )
    second = _run_pilot(
        tmp_path,
        experiment_id="isolation",
        run_id="run_two",
    )

    first_memories = _read_jsonl(first.run_paths.memory_logs)
    second_memories = _read_jsonl(second.run_paths.memory_logs)
    assert {record["run_id"] for record in first_memories} == {"run_one"}
    assert {record["run_id"] for record in second_memories} == {"run_two"}
    assert [record["memory_id"] for record in first_memories] == [
        record["memory_id"] for record in second_memories
    ]
    assert [record["text"] for record in first_memories] == [
        record["text"] for record in second_memories
    ]


def test_repetition_number_three_persists_without_zero_based_mapping(
    tmp_path: Path,
) -> None:
    result = _run_pilot(
        tmp_path,
        experiment_id="repetition_numbering",
        run_id="balanced_rep03",
        repetition_index=3,
    )

    assert result.run_summary.repetition_index == 3
    for path in result.run_paths.jsonl_files:
        for record in _read_jsonl(path):
            assert record["repetition_index"] == 3


def test_corrupted_progress_clock_is_rejected_before_execution() -> None:
    progress = SimulationProgress()
    progress.next_logical_timestamp = 2

    with pytest.raises(
        SimulationExecutionError,
        match="next_logical_timestamp must equal completed_main_turns",
    ):
        progress.validate()
