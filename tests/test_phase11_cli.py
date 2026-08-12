"""Additive Phase 11 CLI authorization and compatibility tests."""

from __future__ import annotations

import pytest

import main as main_module
from src.simulation import execute_pilot


def test_pilot_help_exposes_dry_dummy_real_budget_and_conflict_controls() -> None:
    parser = main_module.build_parser()
    pilot = next(
        action.choices["run-pilot"]
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    help_text = pilot.format_help()

    for flag in (
        "--dry-run",
        "--dummy",
        "--real",
        "--allow-real-llm",
        "--max-total-tokens",
        "--on-conflict",
    ):
        assert flag in help_text


def test_dry_run_reports_exact_scope_without_requiring_output_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main_module.main(["run-pilot", "--dry-run", "true"]) == 0
    captured = capsys.readouterr()

    assert "Planned steps: 4" in captured.out
    assert "Planned main turns: 40" in captured.out
    assert "Planned provider logical calls: 260" in captured.out
    assert "Maximum physical attempts: 520" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ["run-pilot", "--real", "--output-root", "unused"],
        [
            "run-pilot",
            "--real",
            "--allow-real-llm",
            "true",
            "--output-root",
            "unused",
        ],
    ],
)
def test_real_cli_missing_required_authorization_or_budget_fails_before_provider(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main_module.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Pilot execution failed" in captured.err


def test_review_pilot_command_is_additive() -> None:
    assert "review-pilot" in main_module.build_parser().format_help()


def test_api_key_alone_never_authorizes_real_pilot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key-presence-is-insufficient")

    assert main_module.main(
        [
            "run-pilot",
            "--real",
            "--max-total-tokens",
            "1",
            "--output-root",
            str(tmp_path),
        ]
    ) == 1

    captured = capsys.readouterr()
    assert "explicit --allow-real-llm true" in captured.err
    assert not tuple(tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config", "recency_heavy", "--repetition", "1"],
        ["--config", "balanced", "--repetition", "2"],
    ],
)
def test_real_cli_rejects_scope_override_before_credentials_or_output(
    arguments: list[str],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    invocation = [
        "run-pilot",
        "--real",
        "--allow-real-llm",
        "true",
        "--max-total-tokens",
        "100",
        "--output-root",
        str(tmp_path),
        *arguments,
    ]

    assert main_module.main(invocation) == 1

    captured = capsys.readouterr()
    assert "fixed to balanced configuration, repetition 1" in captured.err
    assert "OPENAI_API_KEY" not in captured.err
    assert not tuple(tmp_path.rglob("*"))


def test_real_cli_rejects_nonpositive_budget_before_credentials_or_output(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main_module.main(
        [
            "run-pilot",
            "--real",
            "--allow-real-llm",
            "true",
            "--max-total-tokens",
            "0",
            "--output-root",
            str(tmp_path),
        ]
    ) == 1

    captured = capsys.readouterr()
    assert "positive --max-total-tokens" in captured.err
    assert not tuple(tmp_path.rglob("*"))


def test_review_pilot_cli_publishes_explicit_offline_rejection(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pilot = execute_pilot(
        mode="dummy",
        output_root=tmp_path,
        experiment_id="phase11_review_cli",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
    )

    assert main_module.main(
        [
            "review-pilot",
            "--evidence-dir",
            str(pilot.paths.evidence_dir),
            "--decision",
            "rejected",
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "Acceptance state: rejected" in captured.out
    assert "Freeze hash" not in captured.out
    assert captured.err == ""
    assert (pilot.paths.review_dir / "pilot_acceptance.json").is_file()
    assert not (pilot.paths.review_dir / "freeze_manifest.json").exists()
