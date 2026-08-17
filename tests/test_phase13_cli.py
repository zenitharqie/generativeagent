"""Additive Phase 13 CLI execution surface and safe failure behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main as main_module
import src.simulation as simulation_module
import src.simulation.experiment_controller as controller_module


def test_full_experiment_help_contains_only_approved_controls() -> None:
    parser = main_module.build_parser()
    subparser = next(
        action.choices["run-full-experiment"]
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    help_text = subparser.format_help()
    for flag in (
        "--dry-run",
        "--dummy",
        "--real",
        "--allow-real-llm",
        "--confirm-full-experiment",
        "--max-total-tokens",
        "--accepted-pilot-review-dir",
        "--experiment-id",
        "--output-root",
        "--on-conflict",
    ):
        assert flag in help_text
    for forbidden in ("--resume", "--overwrite", "--skip-existing", "--judge-bcr"):
        assert forbidden not in help_text


def test_execution_requires_explicit_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_module.main(["run-full-experiment"]) == 1
    captured = capsys.readouterr()
    assert "requires --dry-run true, --dummy, or --real" in captured.err
    assert captured.out == ""


def test_execution_requires_output_root(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_module.main(["run-full-experiment", "--dummy"]) == 1
    captured = capsys.readouterr()
    assert "--output-root is required" in captured.err
    assert captured.out == ""


def test_real_cli_without_allow_fails_before_credential_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("credential/provider boundary was touched")

    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)
    monkeypatch.setattr(controller_module, "_default_provider_factory", forbidden)

    assert main_module.main(
        [
            "run-full-experiment",
            "--real",
            "--confirm-full-experiment",
            "RUN_12_CONFIGURED_RUNS",
            "--max-total-tokens",
            "1",
            "--accepted-pilot-review-dir",
            str(tmp_path / "review"),
            "--output-root",
            str(tmp_path),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert "allow-real-llm" in captured.err
    assert not tuple(tmp_path.iterdir())


def test_dummy_cli_wires_complete_result_without_changing_other_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_result = SimpleNamespace(
        run_chains=(None,) * 12,
        paths=SimpleNamespace(experiment_dir=tmp_path / "experiment"),
    )
    captured_arguments: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> object:
        captured_arguments.update(kwargs)
        return fake_result

    monkeypatch.setattr(simulation_module, "execute_full_experiment", fake_execute)
    assert main_module.main(
        [
            "run-full-experiment",
            "--dummy",
            "--confirm-full-experiment",
            "RUN_12_CONFIGURED_RUNS",
            "--experiment-id",
            "phase13_cli",
            "--output-root",
            str(tmp_path),
            "--on-conflict",
            "fail",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "Eligible source runs: 12" in captured.out
    assert "MRA records: 240" in captured.out
    assert "BCR export records: 240" in captured.out
    assert "BCR judgments: 0" in captured.out
    assert captured.err == ""
    assert captured_arguments["mode"] == "dummy"
    assert captured_arguments["on_conflict"] == "fail"
