"""Atomic, deterministic, source-preserving evaluation publication."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import src.evaluation.persistence as persistence_module
from phase09_test_support import artifact_bytes, read_jsonl
from src.evaluation import (
    EvaluationConflictError,
    EvaluationPersistenceError,
    evaluate_run,
    publish_evaluation,
)


EXPECTED_FILES = {
    "mra_records.jsonl",
    "bcr_export.jsonl",
    "evaluation_summary.json",
}


def test_primary_evaluation_publishes_exact_three_file_layout(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    result = evaluate_run(
        run_dir=phase08_full_artifacts[0].run_paths.run_dir,
        output_dir=output,
    )

    assert result.output_dir == output.resolve()
    assert {path.name for path in output.iterdir()} == EXPECTED_FILES
    mra_records = read_jsonl(output / "mra_records.jsonl")
    assert len(mra_records) == 20
    assert len(read_jsonl(output / "bcr_export.jsonl")) == 20
    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["evaluation_mode"] == "primary"
    assert summary["primary_eligible"] is True
    assert summary["included_in_final_research_aggregation"] is True
    assert summary["bcr_judgment"] == {
        "status": "not_judged_phase_09",
        "decision_type": "binary",
        "allowed_final_values": [0, 1],
        "rubric_dimensions": [
            "communication_style",
            "focus_consistency",
            "temporal_consistency",
        ],
        "rubric_assessment_values": ["consistent", "inconsistent"],
        "final_judgment_field": "consistent",
        "final_metric": "proportion_of_consistent_responses",
    }
    summary_text = json.dumps(summary["bcr_judgment"], sort_keys=True)
    for obsolete in (
        "communication_style_score",
        "score_minimum",
        "score_maximum",
        "arithmetic_mean_of_three_dimension_scores",
    ):
        assert obsolete not in summary_text
    for record in mra_records:
        assert record["evaluation_mode"] == summary["evaluation_mode"]
        assert record["primary_eligible"] is summary["primary_eligible"]
        assert record["included_in_final_research_aggregation"] is summary[
            "included_in_final_research_aggregation"
        ]


def test_diagnostic_output_is_explicitly_non_primary(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"

    evaluate_run(
        run_dir=phase08_pilot_artifacts[0].run_paths.run_dir,
        output_dir=output,
        diagnostic=True,
    )

    summary = json.loads((output / "evaluation_summary.json").read_text())
    mra_records = read_jsonl(output / "mra_records.jsonl")
    assert summary["evaluation_mode"] == "diagnostic"
    assert summary["primary_eligible"] is False
    assert summary["included_in_final_research_aggregation"] is False
    assert summary["source_run_status"] == "excluded"
    assert summary["mra_aggregates"]["unit_count"] == 4
    assert summary["bcr_export_count"] == 4
    assert len(mra_records) == 4
    for record in mra_records:
        assert record["evaluation_mode"] == summary["evaluation_mode"]
        assert record["primary_eligible"] is summary["primary_eligible"]
        assert record["included_in_final_research_aggregation"] is summary[
            "included_in_final_research_aggregation"
        ]


def test_publication_rejects_mra_eligibility_that_differs_from_summary(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    result = evaluate_run(
        run_dir=phase08_full_artifacts[0].run_paths.run_dir,
        output_dir=tmp_path / "valid_evaluation",
    )
    mismatched_first = replace(
        result.mra_records[0],
        evaluation_mode="diagnostic",
        primary_eligible=False,
        included_in_final_research_aggregation=False,
    )
    output = tmp_path / "invalid_evaluation"

    with pytest.raises(EvaluationPersistenceError, match="eligibility"):
        publish_evaluation(
            output_dir=output,
            mra_records=(mismatched_first, *result.mra_records[1:]),
            bcr_records=result.bcr_records,
            summary=result.summary,
        )

    assert not output.exists()


def test_existing_output_is_conflict_and_is_never_overwritten(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(EvaluationConflictError, match="already exists"):
        evaluate_run(
            run_dir=phase08_full_artifacts[0].run_paths.run_dir,
            output_dir=output,
        )

    assert sentinel.read_bytes() == b"preserve"
    assert {path.name for path in output.iterdir()} == {"sentinel.txt"}


def test_preexisting_staging_output_is_conflict_and_is_not_repaired(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    staging = tmp_path / ".evaluation.staging"
    staging.mkdir()
    sentinel = staging / "sentinel.txt"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(EvaluationConflictError, match="staging"):
        evaluate_run(
            run_dir=phase08_full_artifacts[0].run_paths.run_dir,
            output_dir=output,
        )

    assert not output.exists()
    assert sentinel.read_bytes() == b"preserve"


def test_mid_write_failure_leaves_no_final_or_staging_output(
    phase08_full_artifacts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evaluation"
    real_write = persistence_module._write_new_file
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected persistence failure")
        real_write(path, payload)

    monkeypatch.setattr(persistence_module, "_write_new_file", fail_second_write)

    with pytest.raises(EvaluationPersistenceError, match="injected"):
        evaluate_run(
            run_dir=phase08_full_artifacts[0].run_paths.run_dir,
            output_dir=output,
        )

    assert not output.exists()
    assert not (tmp_path / ".evaluation.staging").exists()


def test_evaluation_never_changes_source_run_artifacts(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = phase08_full_artifacts[0].run_paths.run_dir
    before = artifact_bytes(run_dir)

    evaluate_run(run_dir=run_dir, output_dir=tmp_path / "evaluation")

    assert artifact_bytes(run_dir) == before


def test_two_fresh_outputs_are_byte_identical(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = phase08_full_artifacts[0].run_paths.run_dir
    first = tmp_path / "evaluation_one"
    second = tmp_path / "evaluation_two"

    evaluate_run(run_dir=run_dir, output_dir=first)
    evaluate_run(run_dir=run_dir, output_dir=second)

    assert {
        name: (first / name).read_bytes() for name in EXPECTED_FILES
    } == {
        name: (second / name).read_bytes() for name in EXPECTED_FILES
    }


def test_outputs_contain_no_wall_clock_uuid_or_output_path(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation_unique_path"

    evaluate_run(
        run_dir=phase08_full_artifacts[0].run_paths.run_dir,
        output_dir=output,
    )

    payload = b"".join((output / name).read_bytes() for name in sorted(EXPECTED_FILES))
    assert str(output).encode() not in payload
    assert b"created_at" not in payload
    assert b"evaluated_at" not in payload
    assert b"timestamp_uuid" not in payload


def test_output_must_be_separate_from_source_run(
    phase08_full_artifacts,
) -> None:
    run_dir = phase08_full_artifacts[0].run_paths.run_dir

    with pytest.raises(EvaluationPersistenceError, match="source run"):
        evaluate_run(run_dir=run_dir, output_dir=run_dir / "evaluation")
