"""Additive Phase 09 evaluate command contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = REPOSITORY_ROOT / "main.py"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )


def test_evaluate_help_exposes_only_approved_phase09_options() -> None:
    result = _run_cli("evaluate", "--help")

    assert result.returncode == 0
    assert "--run-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--diagnostic" in result.stdout
    for forbidden in (
        "--ground-truth",
        "--judge",
        "--provider",
        "--model",
        "--evaluator",
    ):
        assert forbidden not in result.stdout
    assert result.stderr == ""


def test_evaluate_completed_run_succeeds_and_reports_primary_scope(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    result = _run_cli(
        "evaluate",
        "--run-dir",
        str(phase08_full_artifacts[0].run_paths.run_dir),
        "--output-dir",
        str(output),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "Evaluation completed successfully.",
        "Evaluation mode: primary",
        "Primary eligible: yes",
        "MRA records: 20",
        "BCR export records: 20",
        f"Run MRA: {__import__('json').loads((output / 'evaluation_summary.json').read_text())['mra_aggregates']['run']['mean_mra']}",
        f"Evaluation directory: {output.resolve()}",
    ]


def test_excluded_pilot_without_diagnostic_flag_fails_without_output(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    result = _run_cli(
        "evaluate",
        "--run-dir",
        str(phase08_pilot_artifacts[0].run_paths.run_dir),
        "--output-dir",
        str(output),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "diagnostic" in result.stderr
    assert not output.exists()


def test_excluded_pilot_with_diagnostic_flag_succeeds_as_non_primary(
    phase08_pilot_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"

    result = _run_cli(
        "evaluate",
        "--run-dir",
        str(phase08_pilot_artifacts[0].run_paths.run_dir),
        "--output-dir",
        str(output),
        "--diagnostic",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "Evaluation mode: diagnostic" in result.stdout
    assert "Primary eligible: no" in result.stdout
    assert "MRA records: 4" in result.stdout
    assert "BCR export records: 4" in result.stdout


def test_repeated_evaluate_path_is_conflict_without_overwrite(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"
    arguments = (
        "evaluate",
        "--run-dir",
        str(phase08_full_artifacts[0].run_paths.run_dir),
        "--output-dir",
        str(output),
    )
    first = _run_cli(*arguments)
    original = (output / "evaluation_summary.json").read_bytes()

    second = _run_cli(*arguments)

    assert first.returncode == 0
    assert second.returncode == 1
    assert second.stdout == ""
    assert "already exists" in second.stderr
    assert (output / "evaluation_summary.json").read_bytes() == original


def test_missing_run_directory_fails_without_publishing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation"

    result = _run_cli(
        "evaluate",
        "--run-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(output),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Evaluation failed" in result.stderr
    assert not output.exists()


def test_evaluate_rejects_unknown_options_as_argparse_error(
    phase08_full_artifacts,
    tmp_path: Path,
) -> None:
    result = _run_cli(
        "evaluate",
        "--run-dir",
        str(phase08_full_artifacts[0].run_paths.run_dir),
        "--output-dir",
        str(tmp_path / "evaluation"),
        "--ground-truth",
        "config/ground_truth.json",
    )

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_existing_commands_remain_visible_alongside_evaluate() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "validate-config" in result.stdout
    assert "run-pilot" in result.stdout
    assert "evaluate" in result.stdout
