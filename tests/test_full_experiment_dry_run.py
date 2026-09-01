"""Phase 13 deterministic planning and additive dry-run CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import main as main_module
import src.simulation.experiment_controller as controller_module
from src.config_loader import load_experiment_config
from src.simulation import (
    CONFIGURATION_MAJOR_ORDER,
    build_full_experiment_plan,
    derive_full_run_contract,
    dry_run_full_experiment,
)


EXPECTED_RUN_IDS = (
    "balanced_rep01",
    "balanced_rep02",
    "balanced_rep03",
    "recency_heavy_rep01",
    "recency_heavy_rep02",
    "recency_heavy_rep03",
    "relevance_heavy_rep01",
    "relevance_heavy_rep02",
    "relevance_heavy_rep03",
    "importance_heavy_rep01",
    "importance_heavy_rep02",
    "importance_heavy_rep03",
)


def test_plan_is_exact_configuration_major_four_by_three() -> None:
    plan = dry_run_full_experiment(experiment_id="phase13_plan_test")

    assert tuple(run.run_id for run in plan.runs) == EXPECTED_RUN_IDS
    assert tuple(dict.fromkeys(run.retrieval_config_id for run in plan.runs)) == (
        CONFIGURATION_MAJOR_ORDER
    )
    assert tuple(run.repetition_index for run in plan.runs) == (1, 2, 3) * 4
    assert len({run.run_id for run in plan.runs}) == 12


def test_plan_uses_exact_repository_config_weights() -> None:
    config = load_experiment_config()
    configured = {item.config_id: item.weights for item in config.retrieval_configs}
    plan = dry_run_full_experiment(experiment_id="phase13_weight_test")

    for run in plan.runs:
        expected = configured[run.retrieval_config_id]
        assert (
            run.recency_weight,
            run.importance_weight,
            run.relevance_weight,
        ) == (expected.recency, expected.importance, expected.relevance)


def test_full_run_structure_is_derived_from_frozen_graph() -> None:
    scope, calls = derive_full_run_contract(load_experiment_config())

    assert scope.to_dict() == {
        "simulation_days": 6,
        "step_count": 12,
        "natural_block_count": 7,
        "target_event_count": 5,
        "main_turn_count": 104,
        "post_step_reflection_count": 48,
        "seed_memory_count": 12,
        "memory_record_count": 164,
        "target_evaluation_unit_count": 20,
    }
    assert calls.to_dict() == {
        "simulator_logical_calls": 408,
        "embedding_logical_calls": 268,
        "evaluator_logical_calls": 0,
        "total_logical_calls": 676,
        "maximum_attempts_per_logical_call": 4,
        "maximum_physical_attempts": 2704,
    }


def test_experiment_call_totals_are_exactly_8112_and_32448() -> None:
    plan = build_full_experiment_plan(
        experiment_id="phase13_call_test",
        mode="dry-run",
    )

    assert plan.total_simulator_logical_calls == 4896
    assert plan.total_embedding_logical_calls == 3216
    assert plan.total_evaluator_logical_calls == 0
    assert plan.total_logical_calls == 8112
    assert plan.maximum_physical_attempts == 32448
    assert plan.total_main_turns == 1248
    assert plan.total_reflections == 576
    assert plan.total_target_units == 240


def test_dry_run_cli_touches_no_path_credential_provider_simulation_or_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("forbidden execution boundary was touched")

    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)
    monkeypatch.setattr(controller_module, "_default_provider_factory", forbidden)
    monkeypatch.setattr(controller_module.RunController, "run", forbidden)
    monkeypatch.setattr(controller_module, "evaluate_run", forbidden)

    assert main_module.main(["run-full-experiment", "--dry-run", "true"]) == 0
    captured = capsys.readouterr()

    assert "Planned runs: 12" in captured.out
    assert "Main turns per run: 104" in captured.out
    assert "Total main turns: 1248" in captured.out
    assert "Simulator logical calls: 4896" in captured.out
    assert "Embedding logical calls: 3216" in captured.out
    assert "Evaluator logical calls: 0" in captured.out
    assert "Maximum physical attempts: 32448" in captured.out
    assert captured.err == ""
    assert not tuple(tmp_path.iterdir())


def test_dry_run_rejects_output_and_authorization_controls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main_module.main(
        [
            "run-full-experiment",
            "--dry-run",
            "true",
            "--output-root",
            str(tmp_path),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert "accepts no output" in captured.err
    assert captured.out == ""
    assert not tuple(tmp_path.iterdir())


def test_full_experiment_cli_is_additive() -> None:
    parser = main_module.build_parser()
    help_text = parser.format_help()
    assert "run-full-experiment" in help_text
    assert "run-pilot" in help_text
    assert "evaluate" in help_text
    assert "llm-smoke-test" in help_text
    assert "review-pilot" in help_text
    assert "export-visualizer-data" in help_text
