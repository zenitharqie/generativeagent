"""Offline command-line contract for the approved Day 1-2 pilot."""

from __future__ import annotations

import json
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


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_cli_help_exposes_only_the_approved_phase08_pilot_surface() -> None:
    result = _run_cli("run-pilot", "--help")

    assert result.returncode == 0
    assert "--dummy" in result.stdout
    assert "--config" in result.stdout
    assert "--repetition" in result.stdout
    assert "--days" in result.stdout
    assert "--output-root" in result.stdout
    assert "--real" not in result.stdout
    assert result.stderr == ""


def test_cli_rejects_pilot_without_dummy_mode(tmp_path: Path) -> None:
    result = _run_cli("run-pilot", "--output-root", str(tmp_path))

    assert result.returncode == 1
    assert result.stdout == ""
    assert "requires --dummy" in result.stderr
    assert not (tmp_path / "phase08_dummy_pilot").exists()


def test_cli_rejects_any_day_scope_other_than_exact_day_one_to_two(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        "run-pilot",
        "--dummy",
        "--days",
        "1-3",
        "--output-root",
        str(tmp_path),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "supports only the exact day range '1-2'" in result.stderr
    assert not (tmp_path / "phase08_dummy_pilot").exists()


def test_cli_runs_offline_pilot_and_reports_excluded_research_status(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        "run-pilot",
        "--dummy",
        "--config",
        "balanced",
        "--repetition",
        "1",
        "--days",
        "1-2",
        "--output-root",
        str(tmp_path),
    )
    run_dir = (
        tmp_path
        / "phase08_dummy_pilot"
        / "balanced_rep01_pilot_day01_02"
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "Pilot scope executed successfully.",
        "Persisted run status: excluded",
        "Completed natural blocks: 3",
        "Completed target events: 1",
        "Completed main turns: 40",
        f"Run directory: {run_dir.resolve()}",
    ]
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "excluded"
    assert summary["repetition_index"] == 1
    assert summary["completed_main_turns"] == 40
    assert len(_read_jsonl(run_dir / "behavior_logs.jsonl")) == 40
    assert (run_dir / "api_usage_logs.jsonl").read_bytes() == b""


def test_cli_repeated_default_identity_fails_on_conflict_without_overwrite(
    tmp_path: Path,
) -> None:
    arguments = (
        "run-pilot",
        "--dummy",
        "--output-root",
        str(tmp_path),
    )
    first = _run_cli(*arguments)
    run_summary = (
        tmp_path
        / "phase08_dummy_pilot"
        / "balanced_rep01_pilot_day01_02"
        / "run_summary.json"
    )
    original = run_summary.read_bytes()

    second = _run_cli(*arguments)

    assert first.returncode == 0
    assert second.returncode == 1
    assert "RunConflictError" in second.stderr
    assert "already exists" in second.stderr
    assert run_summary.read_bytes() == original
