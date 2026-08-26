"""Phase 13 explicit authorization, freeze, GT, Git, disk, and path gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.simulation.experiment_controller as controller_module
from src.simulation import (
    ExperimentAuthorizationError,
    ExperimentPreflightError,
    FULL_EXPERIMENT_CONFIRMATION,
    GitSourceIdentity,
    OPERATIONAL_MINIMUM_FREE_BYTES,
    PilotFreezeValidation,
    capture_git_source_identity,
    execute_full_experiment,
    validate_execution_authorization,
    validate_git_identity_unchanged,
    validate_ground_truth_methodological_finalization,
    validate_operational_disk_space,
)
from src.storage import (
    ExperimentPathError,
    RunConflictError,
    build_experiment_paths,
    validate_fresh_experiment_paths,
)


@pytest.mark.parametrize("confirmation", [None, "", "RUN_12_RUNS", "run_12_configured_runs"])
def test_dummy_and_real_require_exact_confirmation(confirmation: str | None) -> None:
    with pytest.raises(ExperimentAuthorizationError, match="confirm-full-experiment"):
        validate_execution_authorization(
            mode="dummy",
            confirmation=confirmation,
            real_authorized=False,
            max_total_tokens=None,
        )


def test_dummy_rejects_real_controls() -> None:
    with pytest.raises(ExperimentAuthorizationError, match="dummy"):
        validate_execution_authorization(
            mode="dummy",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=True,
            max_total_tokens=1,
        )


@pytest.mark.parametrize("budget", [None, -1, 0, True])
def test_real_requires_positive_global_budget(budget: object) -> None:
    with pytest.raises(ExperimentAuthorizationError, match="positive"):
        validate_execution_authorization(
            mode="real",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=True,
            max_total_tokens=budget,  # type: ignore[arg-type]
        )


def test_real_requires_explicit_allow_flag() -> None:
    with pytest.raises(ExperimentAuthorizationError, match="allow-real-llm"):
        validate_execution_authorization(
            mode="real",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=False,
            max_total_tokens=1,
        )


def test_wrong_confirmation_fails_before_freeze_gt_git_credential_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append("forbidden")
        raise AssertionError("later preflight was touched")

    monkeypatch.setattr(controller_module, "validate_accepted_freeze", forbidden)
    monkeypatch.setattr(
        controller_module, "validate_ground_truth_methodological_finalization", forbidden
    )
    monkeypatch.setattr(controller_module, "capture_git_source_identity", forbidden)
    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)
    monkeypatch.setattr(controller_module, "_default_provider_factory", forbidden)

    with pytest.raises(ExperimentAuthorizationError):
        execute_full_experiment(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase13_auth_order",
            confirmation="WRONG",
            real_authorized=True,
            max_total_tokens=100,
            accepted_pilot_review_dir=tmp_path / "review",
        )

    assert calls == []
    assert not tuple(tmp_path.iterdir())


def test_missing_accepted_freeze_fails_before_gt_git_credential_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("later preflight was touched")

    monkeypatch.setattr(
        controller_module, "validate_ground_truth_methodological_finalization", forbidden
    )
    monkeypatch.setattr(controller_module, "capture_git_source_identity", forbidden)
    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)
    monkeypatch.setattr(controller_module, "_default_provider_factory", forbidden)

    with pytest.raises(ExperimentPreflightError, match="freeze"):
        execute_full_experiment(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase13_missing_freeze",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=True,
            max_total_tokens=100,
            accepted_pilot_review_dir=tmp_path / "missing_review",
        )
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "freeze_error",
    [
        "pending review",
        "rejected review",
        "stale prompt hash",
        "missing usage evidence",
    ],
)
def test_any_phase11_freeze_validation_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    freeze_error: str,
) -> None:
    def reject_freeze(*args: object, **kwargs: object) -> object:
        raise RuntimeError(freeze_error)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("a later preflight boundary was touched")

    monkeypatch.setattr(controller_module, "validate_accepted_freeze", reject_freeze)
    monkeypatch.setattr(
        controller_module, "validate_ground_truth_methodological_finalization", forbidden
    )
    monkeypatch.setattr(controller_module, "capture_git_source_identity", forbidden)
    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)

    with pytest.raises(ExperimentPreflightError, match="freeze"):
        execute_full_experiment(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase13_invalid_freeze",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=True,
            max_total_tokens=100,
            accepted_pilot_review_dir=tmp_path / "review",
        )
    assert not tuple(tmp_path.iterdir())


def test_current_ground_truth_blocks_real_after_synthetic_current_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = PilotFreezeValidation(
        review_dir=tmp_path / "pilot_review",
        pilot_id="pilot",
        manifest_hash="a" * 64,
        acceptance_hash="b" * 64,
        freeze_hash="c" * 64,
        prompt_version="1.0",
    )
    monkeypatch.setattr(controller_module, "validate_accepted_freeze", lambda *a, **k: synthetic)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("credential/Git/provider boundary was touched")

    monkeypatch.setattr(controller_module, "capture_git_source_identity", forbidden)
    monkeypatch.setattr(controller_module, "require_openai_credential", forbidden)
    monkeypatch.setattr(controller_module, "_default_provider_factory", forbidden)

    with pytest.raises(ExperimentPreflightError, match="ground-truth finalization"):
        execute_full_experiment(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase13_gt_gate",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
            real_authorized=True,
            max_total_tokens=100,
            accepted_pilot_review_dir=synthetic.review_dir,
        )
    assert not tuple(tmp_path.iterdir())


def test_ground_truth_gate_requires_evidence_after_finalization() -> None:
    with pytest.raises(ExperimentPreflightError, match="Ground Truth evidence"):
        validate_ground_truth_methodological_finalization(Path("config"))


def test_git_identity_requires_clean_tagged_attached_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("branch", "--show-current"): "main",
        ("tag", "--points-at", "HEAD"): "v1.1-binary-bcr-methodology\nv1.2-test",
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        controller_module,
        "_run_git",
        lambda *arguments: outputs[arguments],
    )

    identity = capture_git_source_identity(require_clean=True, require_tag=True)
    assert identity == GitSourceIdentity(
        commit="a" * 40,
        branch="main",
        tags=("v1.1-binary-bcr-methodology", "v1.2-test"),
        clean=True,
    )

    outputs[("status", "--porcelain=v1", "--untracked-files=all")] = " M main.py"
    with pytest.raises(ExperimentPreflightError, match="clean Git"):
        capture_git_source_identity(require_clean=True, require_tag=True)


def test_git_identity_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = GitSourceIdentity("a" * 40, "main", ("v1",), True)
    drifted = GitSourceIdentity("b" * 40, "main", ("v2",), True)
    monkeypatch.setattr(
        controller_module,
        "capture_git_source_identity",
        lambda **kwargs: drifted,
    )

    with pytest.raises(ExperimentPreflightError, match="drifted"):
        validate_git_identity_unchanged(expected)


def test_operational_disk_preflight_is_separate_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=OPERATIONAL_MINIMUM_FREE_BYTES - 1),
    )

    with pytest.raises(ExperimentPreflightError, match="operational free space"):
        validate_operational_disk_space(tmp_path)


def test_output_traversal_is_rejected() -> None:
    with pytest.raises(ExperimentPathError, match="traversal"):
        build_experiment_paths("safe/../escape", experiment_id="experiment")


def test_symlink_output_root_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ExperimentPathError, match="symlink"):
        build_experiment_paths(link, experiment_id="experiment")


def test_existing_experiment_identity_is_conflict_only(tmp_path: Path) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="existing")
    paths.experiment_dir.mkdir()

    with pytest.raises(RunConflictError, match="already exists"):
        validate_fresh_experiment_paths(paths)
