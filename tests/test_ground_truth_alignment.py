"""Run-snapshot-only ground-truth alignment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase09_test_support import copy_run, read_json, write_json
from src.evaluation import EvaluationGroundTruthError, load_evaluation_run


def test_ground_truth_is_loaded_from_run_snapshot(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "config_snapshot" / "ground_truth.json"
    ground_truth = read_json(path)
    ground_truth["targets"][0]["expected_categories"] = ["snapshot_only_category"]
    write_json(path, ground_truth)

    loaded = load_evaluation_run(run_dir)
    unit = next(
        item
        for item in loaded.target_units
        if item.identity.event_id == "event_01"
        and item.identity.agent_id == "teacher"
    )

    assert unit.expected_categories == ("snapshot_only_category",)


def test_missing_snapshotted_ground_truth_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    (run_dir / "config_snapshot" / "ground_truth.json").unlink()

    with pytest.raises(EvaluationGroundTruthError, match="ground_truth.json"):
        load_evaluation_run(run_dir)


def test_missing_agent_event_ground_truth_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "config_snapshot" / "ground_truth.json"
    ground_truth = read_json(path)
    ground_truth["targets"].pop()
    write_json(path, ground_truth)

    with pytest.raises(EvaluationGroundTruthError, match="cover|missing"):
        load_evaluation_run(run_dir)


def test_duplicate_agent_event_ground_truth_is_a_hard_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "config_snapshot" / "ground_truth.json"
    ground_truth = read_json(path)
    ground_truth["targets"].append(ground_truth["targets"][0].copy())
    write_json(path, ground_truth)

    with pytest.raises(EvaluationGroundTruthError, match="unique|duplicate"):
        load_evaluation_run(run_dir)


def test_duplicate_categories_within_ground_truth_target_are_rejected(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "config_snapshot" / "ground_truth.json"
    ground_truth = read_json(path)
    categories = ground_truth["targets"][0]["expected_categories"]
    categories.append(categories[0])
    write_json(path, ground_truth)

    with pytest.raises(EvaluationGroundTruthError, match="duplicate"):
        load_evaluation_run(run_dir)


def test_empty_ground_truth_categories_are_rejected_in_diagnostic_mode(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    run_dir = copy_run(phase08_pilot_artifacts[0].run_paths.run_dir, tmp_path)
    path = run_dir / "config_snapshot" / "ground_truth.json"
    ground_truth = read_json(path)
    ground_truth["targets"][0]["expected_categories"] = []
    write_json(path, ground_truth)

    with pytest.raises(EvaluationGroundTruthError, match="expected_categories"):
        load_evaluation_run(run_dir, diagnostic=True)
