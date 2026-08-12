"""Strict post-pilot artifact validation gates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from src.simulation.run_validator import PilotRunValidationError, validate_pilot_run


def copied_run(source: Path, destination: Path) -> Path:
    target = destination / source.parent.name / source.name
    target.parent.mkdir()
    shutil.copytree(source, target)
    return target


def test_valid_dummy_pilot_passes_exact_excluded_scope_gates(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts

    validation = validate_pilot_run(result.run_paths.run_dir, mode="dummy")

    assert validation.source_run_status == "excluded"
    assert (
        validation.natural_block_count,
        validation.target_event_count,
        validation.main_turn_count,
        validation.reflection_count,
        validation.target_evaluation_unit_count,
    ) == (3, 1, 40, 16, 4)
    assert validation.simulator_model_identity is None


@pytest.mark.parametrize(
    "filename",
    [
        "retrieval_logs.jsonl",
        "behavior_logs.jsonl",
        "memory_logs.jsonl",
        "reflection_logs.jsonl",
        "errors.jsonl",
        "event_summaries.jsonl",
        "natural_block_summaries.jsonl",
        "api_usage_logs.jsonl",
    ],
)
def test_missing_required_research_log_is_rejected(
    phase08_pilot_artifacts,
    tmp_path: Path,
    filename: str,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copied_run(result.run_paths.run_dir, tmp_path)
    (run_dir / filename).unlink()

    with pytest.raises(PilotRunValidationError, match=filename):
        validate_pilot_run(run_dir, mode="dummy")


def test_partial_pilot_must_remain_excluded(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copied_run(result.run_paths.run_dir, tmp_path)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "failed"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PilotRunValidationError, match="excluded"):
        validate_pilot_run(run_dir, mode="dummy")


def test_corrupt_research_log_is_rejected_without_repair(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copied_run(result.run_paths.run_dir, tmp_path)
    retrieval_path = run_dir / "retrieval_logs.jsonl"
    retrieval_path.write_text("{}\n", encoding="utf-8")
    corrupt_bytes = retrieval_path.read_bytes()

    with pytest.raises(PilotRunValidationError, match="invalid retrieval_logs.jsonl"):
        validate_pilot_run(run_dir, mode="dummy")

    assert retrieval_path.read_bytes() == corrupt_bytes


def test_exact_supplied_credential_value_is_rejected_from_any_source_artifact(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copied_run(result.run_paths.run_dir, tmp_path)
    secret = "phase11-sentinel-credential"
    behavior_path = run_dir / "behavior_logs.jsonl"
    records = [
        json.loads(line)
        for line in behavior_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["response"] = secret
    behavior_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    with pytest.raises(PilotRunValidationError, match="credential content"):
        validate_pilot_run(run_dir, mode="dummy", secret_value=secret)
