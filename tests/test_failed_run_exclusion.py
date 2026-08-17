"""Phase 13 fail-fast, interruption, and safe terminal-evidence tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.simulation.experiment_controller as controller_module
from src.simulation import (
    ExperimentExecutionError,
    ExperimentValidationError,
    FULL_EXPERIMENT_CONFIRMATION,
    execute_full_experiment,
)


class _CompleteDummyClient:
    invocations = (None,) * 408


class _CompleteDummyEmbedding:
    received_texts = ("x",) * 268


def _fake_run_result() -> object:
    return SimpleNamespace(
        run_paths=SimpleNamespace(
            config_snapshot=Path("config"),
            run_dir=Path("unused_source"),
        )
    )


def test_terminal_provider_sentinel_is_never_persisted_or_exposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "RAW_PROVIDER_SENTINEL api-key-like/body/header material"
    calls = 0

    class FailingController:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError(sentinel)

    monkeypatch.setattr(controller_module, "RunController", FailingController)

    with pytest.raises(ExperimentExecutionError) as captured:
        execute_full_experiment(
            mode="dummy",
            output_root=tmp_path,
            experiment_id="phase13_provider_failure",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert calls == 1
    assert sentinel not in str(captured.value)
    assert sentinel not in repr(captured.value)
    experiment = tmp_path / "phase13_provider_failure"
    payload = b"".join(
        path.read_bytes() for path in experiment.rglob("*") if path.is_file()
    )
    assert sentinel.encode() not in payload
    report = json.loads(
        (experiment / "experiment_evidence" / "terminal_report.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert report["successful_manifest_published"] is False
    assert not (experiment / "experiment_manifest.json").exists()


def test_keyboard_interrupt_is_pending_partial_evidence_and_never_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class InterruptedController:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            raise KeyboardInterrupt

    monkeypatch.setattr(controller_module, "RunController", InterruptedController)

    with pytest.raises(KeyboardInterrupt):
        execute_full_experiment(
            mode="dummy",
            output_root=tmp_path,
            experiment_id="phase13_interrupted",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert calls == 1
    experiment = tmp_path / "phase13_interrupted"
    report = json.loads(
        (experiment / "experiment_evidence" / "terminal_report.json").read_text()
    )
    assert report["status"] == "interrupted"
    assert report["error_type"] == "KeyboardInterrupt"
    assert report["resume_permitted"] is False
    assert not (experiment / "experiment_manifest.json").exists()


def test_invalid_source_stops_before_phase09_and_before_later_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls = 0
    evaluation_calls = 0

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            nonlocal run_calls
            run_calls += 1
            return _fake_run_result()

    def reject_source(*args: object, **kwargs: object) -> object:
        raise ExperimentValidationError("invalid source count")

    def forbidden_evaluation(*args: object, **kwargs: object) -> object:
        nonlocal evaluation_calls
        evaluation_calls += 1
        raise AssertionError("Phase 09 must not execute")

    monkeypatch.setattr(controller_module, "RunController", FakeController)
    monkeypatch.setattr(
        controller_module, "build_deterministic_dummy_llm_client", _CompleteDummyClient
    )
    monkeypatch.setattr(controller_module, "DummyEmbeddingProvider", _CompleteDummyEmbedding)
    monkeypatch.setattr(controller_module, "validate_source_run", reject_source)
    monkeypatch.setattr(controller_module, "evaluate_run", forbidden_evaluation)

    with pytest.raises(ExperimentExecutionError):
        execute_full_experiment(
            mode="dummy",
            output_root=tmp_path,
            experiment_id="phase13_invalid_source",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert run_calls == 1
    assert evaluation_calls == 0
    assert not (tmp_path / "phase13_invalid_source" / "experiment_manifest.json").exists()


def test_phase09_failure_stops_before_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls = 0
    evaluation_calls = 0

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            nonlocal run_calls
            run_calls += 1
            return _fake_run_result()

    def accepted_source(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(source_fingerprint="sealed")

    def failed_evaluation(*args: object, **kwargs: object) -> object:
        nonlocal evaluation_calls
        evaluation_calls += 1
        raise RuntimeError("synthetic deterministic Phase09 failure")

    monkeypatch.setattr(controller_module, "RunController", FakeController)
    monkeypatch.setattr(
        controller_module, "build_deterministic_dummy_llm_client", _CompleteDummyClient
    )
    monkeypatch.setattr(controller_module, "DummyEmbeddingProvider", _CompleteDummyEmbedding)
    monkeypatch.setattr(controller_module, "validate_source_run", accepted_source)
    monkeypatch.setattr(controller_module, "evaluate_run", failed_evaluation)

    with pytest.raises(ExperimentExecutionError):
        execute_full_experiment(
            mode="dummy",
            output_root=tmp_path,
            experiment_id="phase13_evaluation_failure",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert run_calls == evaluation_calls == 1
    assert not (tmp_path / "phase13_evaluation_failure" / "experiment_manifest.json").exists()


def test_invalid_phase09_bundle_stops_before_second_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls = 0
    validation_calls = 0

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: object) -> object:
            nonlocal run_calls
            run_calls += 1
            return _fake_run_result()

    source_validation = SimpleNamespace(source_fingerprint="sealed")

    def fake_evaluation(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            mra_records=(None,) * 20,
            bcr_records=(None,) * 20,
            output_dir=Path("invalid_evaluation"),
        )

    def reject_evaluation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        raise ExperimentValidationError("invalid Phase 09 bundle")

    monkeypatch.setattr(controller_module, "RunController", FakeController)
    monkeypatch.setattr(
        controller_module, "build_deterministic_dummy_llm_client", _CompleteDummyClient
    )
    monkeypatch.setattr(controller_module, "DummyEmbeddingProvider", _CompleteDummyEmbedding)
    monkeypatch.setattr(
        controller_module, "validate_source_run", lambda *a, **k: source_validation
    )
    monkeypatch.setattr(controller_module, "evaluate_run", fake_evaluation)
    monkeypatch.setattr(
        controller_module, "validate_phase09_evaluation", reject_evaluation
    )

    with pytest.raises(ExperimentExecutionError):
        execute_full_experiment(
            mode="dummy",
            output_root=tmp_path,
            experiment_id="phase13_invalid_evaluation",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert run_calls == validation_calls == 1
    assert not (tmp_path / "phase13_invalid_evaluation" / "experiment_manifest.json").exists()
