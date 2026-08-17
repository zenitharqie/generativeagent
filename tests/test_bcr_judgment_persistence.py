"""Read-only source handling and atomic deterministic BCR publication tests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src.evaluation.errors import (
    BCRJudgmentConflictError,
    BCRJudgmentError,
    BCRJudgmentFailure,
    BCRJudgmentInputError,
    BCRJudgmentPersistenceError,
)
from src.evaluation.judgment_persistence import prepare_judgment_output
from src.evaluation.judgment_reader import load_bcr_export
from src.evaluation.judgment_service import judge_bcr_export
from phase09_test_support import artifact_bytes
from test_bcr_judge_phase10 import (
    RepeatingJudgeProvider,
    VALID_INCONSISTENT_JUDGMENT,
    VALID_JUDGMENT,
)
from src.llm.errors import ProviderRateLimitError
from phase10_test_support import (
    directory_bytes,
    prepare_phase09_evaluation,
)


def source_tree_snapshot(path: Path) -> tuple[tuple[str, str, str | None], ...]:
    """Capture source layout and file hashes without following external links."""
    snapshot: list[tuple[str, str, str | None]] = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            snapshot.append((relative, "symlink", None))
        elif item.is_dir():
            snapshot.append((relative, "directory", None))
        else:
            snapshot.append((relative, "file", sha256(item.read_bytes()).hexdigest()))
    return tuple(snapshot)


def test_reader_loads_exactly_twenty_chronological_primary_units(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )

    loaded = load_bcr_export(evaluation)

    assert len(loaded.records) == 20
    assert loaded.evaluation_mode == "primary"
    assert loaded.primary_eligible is True
    assert loaded.included_in_final_research_aggregation is True
    assert loaded.simulator_model_identity == "openai:gpt-5-nano-2025-08-07"
    assert tuple(record.identity.timestamp for record in loaded.records) == tuple(
        sorted(record.identity.timestamp for record in loaded.records)
    )


def test_complete_judgment_publishes_exact_two_files_and_binary_run_bcr(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    output = tmp_path / "judged"

    result = judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=output,
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )

    assert result.output_directory == output.resolve()
    assert {item.name for item in output.iterdir()} == {
        "bcr_judged_records.jsonl",
        "bcr_judgment_summary.json",
    }
    assert len(result.records) == 20
    assert all(record.consistent == 1 for record in result.records)
    assert result.summary.consistent_count == 20
    assert result.summary.inconsistent_count == 0
    assert result.summary.run_bcr == 1.0
    assert result.summary.judged_unit_count == 20
    assert result.summary.provider_attempt_count == 20
    summary = json.loads((output / "bcr_judgment_summary.json").read_text(encoding="utf-8"))
    assert summary["consistent_count"] == 20
    assert summary["inconsistent_count"] == 0
    assert summary["run_bcr"] == 1.0
    assert summary["bcr_formula"] == "consistent_count_divided_by_judged_unit_count"
    assert "configuration" not in summary
    assert "currency_cost" not in summary["usage"]
    assert not (tmp_path / ".judged.staging").exists()


def test_judged_rows_preserve_binary_decision_assessments_reason_and_metadata(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    output = tmp_path / "judged"

    judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=output,
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )
    row = json.loads(
        (output / "bcr_judged_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert row["judgment"] == json.loads(VALID_JUDGMENT)
    assert row["model_metadata"] == {
        "simulator_model_identity": "openai:gpt-5-nano-2025-08-07",
        "evaluator_model_identity": "anthropic:claude-sonnet-4-6",
        "evaluator_prompt_version": "phase10_pilot_v1",
        "evaluator_rubric_version": "phase10_binary_bcr_rubric_v2",
    }
    assert row["prompt_identity"]["prompt_template"] == "bcr_judge"
    assert len(row["prompt_identity"]["prompt_hash"]) == 64


def test_run_bcr_is_consistent_count_divided_by_twenty(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    provider = RepeatingJudgeProvider(
        [VALID_JUDGMENT] * 17 + [VALID_INCONSISTENT_JUDGMENT] * 3
    )

    result = judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=tmp_path / "judged",
        provider=provider,
        sleeper=lambda _: None,
    )

    assert result.summary.judged_unit_count == 20
    assert result.summary.consistent_count == 17
    assert result.summary.inconsistent_count == 3
    assert result.summary.run_bcr == 0.85


def test_source_phase09_artifacts_remain_byte_identical_after_judgment(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    before = directory_bytes(evaluation)

    judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=tmp_path / "judged",
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )

    assert directory_bytes(evaluation) == before


def test_source_simulation_run_remains_byte_identical_after_complete_path(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    run_dir = phase08_full_artifacts[0].run_paths.run_dir
    before = artifact_bytes(run_dir)
    evaluation = prepare_phase09_evaluation(tmp_path, run_dir)

    judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=tmp_path / "judged",
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )

    assert artifact_bytes(run_dir) == before


def test_identical_inputs_and_outputs_produce_byte_identical_judgment_artifacts(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    first = tmp_path / "judged_a"
    second = tmp_path / "judged_b"

    judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=first,
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )
    judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=second,
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )

    assert directory_bytes(first) == directory_bytes(second)


def test_existing_output_conflict_prevents_calls_and_overwrite(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    output = tmp_path / "judged"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    provider = RepeatingJudgeProvider()

    with pytest.raises(BCRJudgmentConflictError):
        judge_bcr_export(
            evaluation_dir=evaluation,
            output_dir=output,
            provider=provider,
            sleeper=lambda _: None,
        )
    assert provider.requests == []
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_nested_output_is_rejected_before_any_source_directory_mutation(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    before = source_tree_snapshot(evaluation)
    nested_parent = evaluation / "nested"

    with pytest.raises(BCRJudgmentPersistenceError, match="separate"):
        prepare_judgment_output(
            nested_parent / "judged",
            source_directory=evaluation,
        )

    assert not nested_parent.exists()
    assert source_tree_snapshot(evaluation) == before


def test_source_directory_itself_is_rejected_without_mutation(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    before = source_tree_snapshot(evaluation)

    with pytest.raises(BCRJudgmentPersistenceError, match="separate"):
        prepare_judgment_output(evaluation, source_directory=evaluation)

    assert source_tree_snapshot(evaluation) == before


def test_symlink_into_source_is_rejected_before_mutation(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    before = source_tree_snapshot(evaluation)
    source_link = tmp_path / "phase09_link"
    source_link.symlink_to(evaluation, target_is_directory=True)

    with pytest.raises(BCRJudgmentPersistenceError, match="symlink"):
        prepare_judgment_output(
            source_link / "nested" / "judged",
            source_directory=evaluation,
        )

    assert not (evaluation / "nested").exists()
    assert source_tree_snapshot(evaluation) == before


def test_path_traversal_is_rejected_before_creating_any_parent(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    before = source_tree_snapshot(evaluation)
    unused_parent = tmp_path / "unused"
    traversing_output = (
        unused_parent / ".." / evaluation.name / "nested" / "judged"
    )

    with pytest.raises(BCRJudgmentPersistenceError, match="traversal"):
        prepare_judgment_output(
            traversing_output,
            source_directory=evaluation,
        )

    assert not unused_parent.exists()
    assert not (evaluation / "nested").exists()
    assert source_tree_snapshot(evaluation) == before


def test_judge_failure_publishes_no_final_or_staging_directory(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    output = tmp_path / "judged"
    provider = RepeatingJudgeProvider(["not json", "still not json"])

    with pytest.raises(BCRJudgmentFailure) as caught:
        judge_bcr_export(
            evaluation_dir=evaluation,
            output_dir=output,
            provider=provider,
            sleeper=lambda _: None,
        )
    assert tuple(item.status for item in caught.value.provider_attempts) == (
        "invalid_output",
        "invalid_output",
    )
    assert tuple(item.attempt_index for item in caught.value.provider_attempts) == (1, 2)
    assert not output.exists()
    assert not (tmp_path / ".judged.staging").exists()


def test_terminal_provider_failure_publishes_nothing_and_exposes_attempts(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    output = tmp_path / "judged"
    provider = RepeatingJudgeProvider(
        [ProviderRateLimitError("sanitized"), ProviderRateLimitError("sanitized")]
    )

    with pytest.raises(BCRJudgmentFailure) as caught:
        judge_bcr_export(
            evaluation_dir=evaluation,
            output_dir=output,
            provider=provider,
            sleeper=lambda _: None,
        )

    assert tuple(item.status for item in caught.value.provider_attempts) == (
        "failed",
        "failed",
    )
    assert tuple(item.attempt_index for item in caught.value.provider_attempts) == (1, 2)
    assert not output.exists()
    assert not (tmp_path / ".judged.staging").exists()


def test_diagnostic_judgments_remain_explicitly_non_primary(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_pilot_artifacts[0].run_paths.run_dir,
        diagnostic=True,
        simulator_model_identity=None,
    )

    result = judge_bcr_export(
        evaluation_dir=evaluation,
        output_dir=tmp_path / "judged",
        provider=RepeatingJudgeProvider(),
        sleeper=lambda _: None,
    )

    assert len(result.records) == 4
    assert result.summary.evaluation_mode == "diagnostic"
    assert result.summary.primary_eligible is False
    assert result.summary.included_in_final_research_aggregation is False
    assert all(record.evaluation_mode == "diagnostic" for record in result.records)
    assert all(record.primary_eligible is False for record in result.records)
    assert all(
        record.included_in_final_research_aggregation is False
        for record in result.records
    )


def test_reader_rejects_corrupt_or_duplicate_bcr_json(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    path = evaluation / "bcr_export.jsonl"
    path.write_text('{"schema_version":"1.0","schema_version":"1.0"}\n', encoding="utf-8")

    with pytest.raises(BCRJudgmentInputError):
        load_bcr_export(evaluation)


def test_reader_rejects_obsolete_one_to_five_phase09_contract(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    evaluation = prepare_phase09_evaluation(
        tmp_path,
        phase08_full_artifacts[0].run_paths.run_dir,
    )
    summary_path = evaluation / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["bcr_judgment"] = {
        "status": "not_judged_phase_09",
        "score_fields": [
            "communication_style_score",
            "focus_consistency_score",
            "temporal_consistency_score",
        ],
        "score_type": "integer",
        "score_minimum": 1,
        "score_maximum": 5,
        "bcr_formula": "arithmetic_mean_of_three_dimension_scores",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BCRJudgmentInputError, match="unjudged Phase 09 contract"):
        load_bcr_export(evaluation)
