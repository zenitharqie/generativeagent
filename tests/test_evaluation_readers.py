"""Strict persisted-run loading and eligibility tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase09_test_support import (
    copy_run,
    read_json,
    read_jsonl,
    write_bytes,
    write_json,
    write_jsonl,
)
from src.evaluation import (
    EvaluationEligibilityError,
    EvaluationInputError,
    EvaluationJoinError,
    load_evaluation_run,
)


def test_completed_full_run_is_primary_eligible(phase08_full_artifacts) -> None:
    run_dir = phase08_full_artifacts[0].run_paths.run_dir

    loaded = load_evaluation_run(run_dir)

    assert loaded.evaluation_mode == "primary"
    assert loaded.primary_eligible is True
    assert loaded.included_in_final_research_aggregation is True
    assert loaded.summary.status == "completed"
    assert len(loaded.target_units) == 20


def test_excluded_pilot_requires_explicit_diagnostic_mode(
    phase08_pilot_artifacts,
) -> None:
    run_dir = phase08_pilot_artifacts[0].run_paths.run_dir

    with pytest.raises(EvaluationEligibilityError, match="diagnostic"):
        load_evaluation_run(run_dir)


def test_excluded_pilot_is_non_primary_in_diagnostic_mode(
    phase08_pilot_artifacts,
) -> None:
    run_dir = phase08_pilot_artifacts[0].run_paths.run_dir

    loaded = load_evaluation_run(run_dir, diagnostic=True)

    assert loaded.evaluation_mode == "diagnostic"
    assert loaded.primary_eligible is False
    assert loaded.included_in_final_research_aggregation is False
    assert loaded.summary.status == "excluded"
    assert len(loaded.target_units) == 4


@pytest.mark.parametrize("status", ["failed", "interrupted", "initialized"])
def test_non_evaluable_status_is_rejected_even_in_diagnostic_mode(
    phase08_full_artifacts,
    tmp_path: Path,
    status: str,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path)
    summary["status"] = status
    if status == "initialized":
        summary["completed_natural_blocks"] = 0
        summary["completed_target_events"] = 0
        summary["completed_main_turns"] = 0
        summary["error_count"] = 0
    write_json(summary_path, summary)

    with pytest.raises(EvaluationEligibilityError, match=status):
        load_evaluation_run(run_dir, diagnostic=True)


@pytest.mark.parametrize(
    "filename",
    [
        "retrieval_logs.jsonl",
        "behavior_logs.jsonl",
        "memory_logs.jsonl",
        "event_summaries.jsonl",
        "natural_block_summaries.jsonl",
        "reflection_logs.jsonl",
        "errors.jsonl",
        "api_usage_logs.jsonl",
    ],
)
def test_missing_mandatory_log_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
    filename: str,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    (run_dir / filename).unlink()

    with pytest.raises(EvaluationInputError, match=filename):
        load_evaluation_run(run_dir)


def test_corrupt_jsonl_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "retrieval_logs.jsonl"
    write_bytes(path, path.read_bytes() + b"{not-json}\n")

    with pytest.raises(EvaluationInputError, match="retrieval_logs"):
        load_evaluation_run(run_dir)


def test_unsupported_log_schema_version_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "behavior_logs.jsonl"
    records = read_jsonl(path)
    records[0]["schema_version"] = "999"
    write_jsonl(path, records)

    with pytest.raises(EvaluationInputError, match="schema_version"):
        load_evaluation_run(run_dir)


def test_duplicate_retrieval_identity_is_rejected(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "retrieval_logs.jsonl"
    records = read_jsonl(path)
    records.append(records[-1].copy())
    write_jsonl(path, records)

    with pytest.raises((EvaluationInputError, EvaluationJoinError), match="duplicate|count"):
        load_evaluation_run(run_dir)


def test_cross_run_contamination_is_rejected(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "behavior_logs.jsonl"
    records = read_jsonl(path)
    records[0]["run_id"] = "foreign_run"
    write_jsonl(path, records)

    with pytest.raises(EvaluationInputError, match="run|path"):
        load_evaluation_run(run_dir)


def test_missing_behavior_turn_is_rejected_without_zero_filling(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "behavior_logs.jsonl"
    records = read_jsonl(path)
    records.pop()
    write_jsonl(path, records)

    with pytest.raises((EvaluationInputError, EvaluationJoinError), match="count|missing"):
        load_evaluation_run(run_dir)


def test_missing_selected_memory_is_rejected(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    retrievals = read_jsonl(run_dir / "retrieval_logs.jsonl")
    selected_id = next(
        record["selected_memory_ids"][0]
        for record in retrievals
        if record["selected_memory_ids"]
    )
    memory_path = run_dir / "memory_logs.jsonl"
    memories = [
        memory
        for memory in read_jsonl(memory_path)
        if memory["memory_id"] != selected_id
    ]
    write_jsonl(memory_path, memories)

    with pytest.raises(EvaluationJoinError, match="selected memory"):
        load_evaluation_run(run_dir)


def test_behavior_and_retrieval_selected_ids_must_match(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "behavior_logs.jsonl"
    records = read_jsonl(path)
    records[-1]["retrieved_memory_ids"] = []
    write_jsonl(path, records)

    with pytest.raises(EvaluationInputError, match="retrieved_memory_ids|retrieval"):
        load_evaluation_run(run_dir)


def test_missing_config_snapshot_is_rejected(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    (run_dir / "config_snapshot" / "agents.json").unlink()

    with pytest.raises(EvaluationInputError, match="config_snapshot|agents.json"):
        load_evaluation_run(run_dir)


def test_empty_evaluation_scope_is_rejected_for_excluded_run(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_pilot_artifacts[0].run_paths.run_dir, tmp_path)
    write_bytes(run_dir / "event_summaries.jsonl", b"")

    with pytest.raises(EvaluationInputError, match="count|scope"):
        load_evaluation_run(run_dir, diagnostic=True)
