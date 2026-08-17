"""Phase 11 pilot scope, authorization, and offline integration contracts."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import pytest

import main as main_module
import src.simulation as simulation_module
from phase11_test_support import (
    OvershootingEmbeddingProvider,
    Phase11MockProvider,
    RetryFirstEmbeddingProvider,
    SentinelTextFailureProvider,
    TerminalEmbeddingFailureProvider,
    TerminalUsagePresentInvalidOutputProvider,
    UsagePresentRetryProvider,
    directory_bytes,
    install_phase11_v2_test_contract,
)
from src.config_loader import load_experiment_config
from src.simulation.pilot import (
    PilotExecutionError,
    PilotReviewError,
    execute_pilot,
    review_pilot,
    validate_accepted_freeze,
)
from src.simulation.run_validator import (
    fingerprint_directory,
    validate_no_secret_artifacts,
)
from src.simulation.safeguards import (
    PilotAuthorizationError,
    PilotScopeError,
    PilotUsageError,
    build_pilot_paths,
    build_pilot_plan,
)


def test_fixed_plan_derives_exact_day_one_to_two_scope_and_call_limits() -> None:
    plan = build_pilot_plan(
        load_experiment_config(),
        mode="real",
        retrieval_config_id="balanced",
        repetition_index=1,
    )

    assert plan.days == (1, 2)
    assert plan.step_ids == (
        "day_01_natural_01",
        "day_01_natural_02",
        "day_02_natural_03",
        "day_02_target_01",
    )
    assert (
        plan.natural_block_count,
        plan.target_event_count,
        plan.main_turn_count,
        plan.post_step_reflection_count,
        plan.target_evaluation_unit_count,
    ) == (3, 1, 40, 16, 4)
    assert (
        plan.call_limits.simulator_logical_calls,
        plan.call_limits.embedding_logical_calls,
        plan.call_limits.total_logical_calls,
        plan.call_limits.total_physical_attempts,
    ) == (152, 108, 260, 520)


@pytest.mark.parametrize(
    ("retrieval_config_id", "repetition_index"),
    [("recency_heavy", 1), ("balanced", 2)],
)
def test_real_plan_rejects_any_scope_identity_override(
    retrieval_config_id: str,
    repetition_index: int,
) -> None:
    with pytest.raises(PilotScopeError):
        build_pilot_plan(
            load_experiment_config(),
            mode="real",
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
        )


def test_real_execution_requires_explicit_authorization_before_provider_use(
    tmp_path,
) -> None:
    provider = Phase11MockProvider()

    with pytest.raises(PilotAuthorizationError):
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_auth",
            run_id="unauthorized",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=False,
            max_total_tokens=3000,
            provider=provider,
            sleeper=lambda _: None,
        )

    assert provider.text_requests == []
    assert provider.embedding_requests == []
    assert not tuple(tmp_path.rglob("*"))


def test_mocked_real_pilot_is_excluded_and_publishes_separate_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "phase11-offline-test-key")
    provider = Phase11MockProvider()

    result = execute_pilot(
        mode="real",
        output_root=tmp_path,
        experiment_id="phase11_mocked",
        run_id="balanced_rep01_pilot_day01_02",
        retrieval_config_id="balanced",
        repetition_index=1,
        real_authorized=True,
        max_total_tokens=3000,
        provider=provider,
        sleeper=lambda _: None,
    )

    assert result.run_result.run_summary.status == "excluded"
    assert result.report.acceptance_state == "pending"
    assert result.report.primary_eligible is False
    assert result.report.included_in_final_research_matrix is False
    assert len(provider.text_requests) == 152
    assert len(provider.embedding_requests) == 108
    assert result.paths.source_run_dir.is_dir()
    assert result.paths.diagnostic_evaluation_dir.is_dir()
    assert result.paths.evidence_dir.is_dir()
    assert not result.paths.review_dir.exists()
    assert len(
        (result.paths.diagnostic_evaluation_dir / "bcr_export.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 4
    summary = json.loads(
        (result.paths.diagnostic_evaluation_dir / "evaluation_summary.json")
        .read_text(encoding="utf-8")
    )
    assert summary["evaluation_mode"] == "diagnostic"
    assert summary["primary_eligible"] is False
    assert summary["included_in_final_research_aggregation"] is False
    attempts = tuple(
        json.loads(line)
        for line in (result.paths.evidence_dir / "pilot_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    api_usage = tuple(
        json.loads(line)
        for line in (result.paths.source_run_dir / "api_usage_logs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(attempts) == 260
    assert sum(item["service_role"] == "simulator" for item in attempts) == 152
    assert sum(item["service_role"] == "embedding" for item in attempts) == 108
    assert all(item["service_role"] != "evaluator" for item in attempts)
    assert len(api_usage) == 152
    assert {item["model"] for item in api_usage} == {
        "openai:gpt-5-nano-2025-08-07"
    }
    assert result.report.source_fingerprint == fingerprint_directory(
        result.paths.source_run_dir
    )
    for directory in (
        result.paths.source_run_dir,
        result.paths.diagnostic_evaluation_dir,
        result.paths.evidence_dir,
    ):
        assert b"phase11-offline-test-key" not in b"".join(
            directory_bytes(directory).values()
        )


def test_recovered_retry_with_missing_usage_is_technically_failed_and_unacceptable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-retry-key")
    provider = RetryFirstEmbeddingProvider()

    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_retry",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=provider,
            sleeper=lambda _: None,
        )

    error = captured.value
    assert error.failure_type == "PilotUsageError"
    report = json.loads(
        (error.paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    usage = report["usage"]
    assert report["acceptance_state"] == "pending"
    assert report["technical_success"] is False
    assert report["eligible_for_manual_acceptance"] is False
    assert usage["logical_call_count"] == 260
    assert usage["physical_attempt_count"] == 261
    assert usage["retry_count"] == 1
    assert usage["missing_total_usage_attempt_count"] == 1
    attempts = [
        json.loads(line)
        for line in (error.paths.evidence_dir / "pilot_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert attempts[0]["logical_call_id"] == attempts[1]["logical_call_id"]
    assert [attempts[0]["attempt_index"], attempts[1]["attempt_index"]] == [1, 2]
    assert [attempts[0]["status"], attempts[1]["status"]] == ["failed", "success"]
    assert attempts[0]["total_tokens"] is None
    assert usage["reported_total_tokens"] == sum(
        item["total_tokens"]
        for item in attempts
        if item["total_tokens"] is not None
    )
    assert error.paths.diagnostic_evaluation_dir.is_dir()

    with pytest.raises(PilotReviewError, match="incomplete token usage"):
        review_pilot(
            evidence_dir=error.paths.evidence_dir,
            decision="accepted",
        )
    reviewed = review_pilot(
        evidence_dir=error.paths.evidence_dir,
        decision="rejected",
    )
    assert reviewed.acceptance_state == "rejected"
    assert reviewed.freeze_hash is None


def test_recovered_retry_with_complete_usage_remains_acceptance_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-complete-retry-key")
    gt_dir = install_phase11_v2_test_contract(monkeypatch, tmp_path)

    result = execute_pilot(
        mode="real",
        output_root=tmp_path,
        experiment_id="phase11_complete_retry",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
        real_authorized=True,
        max_total_tokens=3000,
        provider=UsagePresentRetryProvider(),
        sleeper=lambda _: None,
    )

    assert result.report.acceptance_state == "pending"
    assert result.report.technical_success is True
    assert result.usage.logical_call_count == 260
    assert result.usage.physical_attempt_count == 261
    assert result.usage.retry_count == 1
    assert result.usage.missing_total_usage_attempt_count == 0
    reviewed = review_pilot(
        evidence_dir=result.paths.evidence_dir,
        decision="accepted",
        ground_truth_finalization_dir=gt_dir,
    )
    assert reviewed.acceptance_state == "accepted"
    assert (
        validate_accepted_freeze(
            reviewed.review_dir, ground_truth_finalization_dir=gt_dir
        ).pilot_id
        == "pilot"
    )


def test_terminal_provider_failure_is_rejected_with_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-terminal-key")
    provider = TerminalEmbeddingFailureProvider()

    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_terminal",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=provider,
            sleeper=lambda _: None,
        )

    error = captured.value
    assert error.evidence_published is True
    assert error.failure_type == "RealEmbeddingError"
    assert not error.paths.diagnostic_evaluation_dir.exists()
    report = json.loads(
        (error.paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    attempts = [
        json.loads(line)
        for line in (error.paths.evidence_dir / "pilot_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["acceptance_state"] == "pending"
    assert report["technical_success"] is False
    assert report["eligible_for_manual_acceptance"] is False
    assert report["included_in_final_research_matrix"] is False
    assert len(attempts) == 2
    assert {item["logical_call_id"] for item in attempts} == {
        attempts[0]["logical_call_id"]
    }
    assert [item["attempt_index"] for item in attempts] == [1, 2]
    assert all(item["status"] == "failed" for item in attempts)
    assert all("raw" not in key for item in attempts for key in item)
    reviewed = review_pilot(
        evidence_dir=error.paths.evidence_dir,
        decision="rejected",
    )
    assert reviewed.acceptance_state == "rejected"


def test_complete_usage_technical_failure_cannot_be_manually_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-complete-failure-key")
    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_complete_usage_failure",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=TerminalUsagePresentInvalidOutputProvider(),
            sleeper=lambda _: None,
        )

    error = captured.value
    report = json.loads(
        (error.paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    assert report["acceptance_state"] == "pending"
    assert report["technical_success"] is False
    assert report["usage"]["missing_total_usage_attempt_count"] == 0
    with pytest.raises(PilotReviewError, match="technically failed"):
        review_pilot(
            evidence_dir=error.paths.evidence_dir,
            decision="accepted",
        )
    assert not error.paths.review_dir.exists()


def test_provider_error_text_is_sanitized_from_source_and_safeguard_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "RAW_PROVIDER_SENTINEL_DO_NOT_PERSIST_7fd1b48a"
    monkeypatch.setenv("OPENAI_API_KEY", "offline-source-sanitization-key")

    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_sanitized_source_failure",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=SentinelTextFailureProvider(sentinel),
            sleeper=lambda _: None,
        )

    error = captured.value
    errors_path = error.paths.source_run_dir / "errors.jsonl"
    errors = [
        json.loads(line)
        for line in errors_path.read_text(encoding="utf-8").splitlines()
    ]
    assert error.failure_type == "ProviderTimeoutError"
    assert len(errors) == 1
    assert errors[0]["error_type"] == "ProviderTimeoutError"
    assert errors[0]["message"] == (
        "Pilot run failed; inspect the canonical error type and attempt records."
    )
    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert sentinel not in "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert sentinel.encode("utf-8") not in b"".join(
        directory_bytes(tmp_path).values()
    )
    validate_no_secret_artifacts(
        error.paths.source_run_dir,
        secret_value=sentinel,
    )

    def raise_sanitized_pilot_error(**_kwargs):
        raise error

    monkeypatch.setattr(
        simulation_module,
        "execute_pilot",
        raise_sanitized_pilot_error,
    )
    assert main_module.main(
        [
            "run-pilot",
            "--dummy",
            "--output-root",
            str(tmp_path / "unused_cli_output"),
        ]
    ) == 1
    captured_cli = capsys.readouterr()
    assert sentinel not in captured_cli.out
    assert sentinel not in captured_cli.err


def test_keyboard_interrupt_persists_one_attempt_before_pilot_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingEmbeddingProvider(Phase11MockProvider):
        def create_embedding(self, request):
            self.embedding_requests.append(request)
            raise KeyboardInterrupt

    secret = "offline-interrupt-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    provider = InterruptingEmbeddingProvider()
    paths = build_pilot_paths(
        tmp_path,
        experiment_id="phase11_interrupt",
        run_id="pilot",
    )

    with pytest.raises(KeyboardInterrupt):
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_interrupt",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=provider,
            sleeper=lambda _: pytest.fail("interruption must not retry"),
        )

    attempts = [
        json.loads(line)
        for line in (paths.evidence_dir / "pilot_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    report = json.loads(
        (paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    source_summary = json.loads(
        (paths.source_run_dir / "run_summary.json").read_text(encoding="utf-8")
    )

    assert len(provider.embedding_requests) == 1
    assert provider.text_requests == []
    assert len(attempts) == 1
    assert attempts[0]["service_role"] == "embedding"
    assert attempts[0]["attempt_index"] == 1
    assert attempts[0]["status"] == "failed"
    assert attempts[0]["error_type"] == "KeyboardInterrupt"
    assert attempts[0]["input_tokens"] is None
    assert attempts[0]["output_tokens"] is None
    assert attempts[0]["total_tokens"] is None
    assert report["usage"]["physical_attempt_count"] == 1
    assert report["usage"]["logical_call_count"] == 1
    assert report["usage"]["reported_total_tokens"] == 0
    assert report["usage"]["missing_total_usage_attempt_count"] == 1
    assert report["acceptance_state"] == "pending"
    assert report["technical_success"] is False
    assert report["terminal_error_type"] == "KeyboardInterrupt"
    assert source_summary["status"] == "interrupted"
    assert not paths.diagnostic_evaluation_dir.exists()
    assert not paths.review_dir.exists()
    evidence_bytes = b"".join(directory_bytes(paths.evidence_dir).values())
    assert secret.encode("utf-8") not in evidence_bytes
    for forbidden in (
        b"raw_provider_response",
        b"rendered_prompt",
        b"credential",
        b"traceback",
    ):
        assert forbidden not in evidence_bytes


def test_simulator_keyboard_interrupt_is_persisted_once_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingTextProvider(Phase11MockProvider):
        def generate_structured(self, request):
            self.text_requests.append(request)
            raise KeyboardInterrupt

    monkeypatch.setenv("OPENAI_API_KEY", "offline-text-interrupt-key")
    provider = InterruptingTextProvider()
    paths = build_pilot_paths(
        tmp_path,
        experiment_id="phase11_text_interrupt",
        run_id="pilot",
    )

    with pytest.raises(KeyboardInterrupt):
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_text_interrupt",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=provider,
            sleeper=lambda _: pytest.fail("interruption must not retry"),
        )

    attempts = [
        json.loads(line)
        for line in (paths.evidence_dir / "pilot_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    simulator_attempts = [
        attempt for attempt in attempts if attempt["service_role"] == "simulator"
    ]
    report = json.loads(
        (paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )

    assert len(provider.text_requests) == 1
    assert len(simulator_attempts) == 1
    assert simulator_attempts[0]["logical_call_id"].startswith("simulator_")
    assert simulator_attempts[0]["attempt_index"] == 1
    assert simulator_attempts[0]["status"] == "failed"
    assert simulator_attempts[0]["error_type"] == "KeyboardInterrupt"
    assert simulator_attempts[0]["input_tokens"] is None
    assert simulator_attempts[0]["output_tokens"] is None
    assert simulator_attempts[0]["total_tokens"] is None
    assert report["usage"]["physical_attempt_count"] == len(attempts)
    assert report["usage"]["missing_total_usage_attempt_count"] == 1
    assert report["acceptance_state"] == "pending"
    assert report["terminal_error_type"] == "KeyboardInterrupt"


def test_post_call_token_overshoot_rejects_after_persisting_consumed_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-budget-key")

    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_budget",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=5,
            provider=OvershootingEmbeddingProvider(),
            sleeper=lambda _: None,
        )

    error = captured.value
    assert error.failure_type == "PilotTokenBudgetExceeded"
    attempt = json.loads(
        (error.paths.evidence_dir / "pilot_attempts.jsonl").read_text(encoding="utf-8")
    )
    report = json.loads(
        (error.paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    assert attempt["total_tokens"] == 6
    assert report["usage"]["reported_total_tokens"] == 6
    assert report["usage"]["budget_exceeded"] is True
    assert report["acceptance_state"] == "pending"


def test_required_api_usage_projection_failure_prevents_pilot_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.simulation.safeguards as safeguards_module

    monkeypatch.setenv("OPENAI_API_KEY", "offline-usage-failure-key")
    original_append = safeguards_module.append_jsonl

    def fail_api_projection(path, record, **kwargs):
        if Path(path).name == "api_usage_logs.jsonl":
            raise PilotUsageError("forced API usage persistence failure")
        return original_append(path, record, **kwargs)

    monkeypatch.setattr(safeguards_module, "append_jsonl", fail_api_projection)

    with pytest.raises(PilotExecutionError) as captured:
        execute_pilot(
            mode="real",
            output_root=tmp_path,
            experiment_id="phase11_usage_failure",
            run_id="pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            real_authorized=True,
            max_total_tokens=3000,
            provider=Phase11MockProvider(),
            sleeper=lambda _: None,
        )

    error = captured.value
    report = json.loads(
        (error.paths.evidence_dir / "pilot_report.json").read_text(encoding="utf-8")
    )
    attempts = (
        error.paths.evidence_dir / "pilot_attempts.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    source_summary = json.loads(
        (error.paths.source_run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    assert error.failure_type == "PilotUsageError"
    assert report["technical_success"] is False
    assert report["acceptance_state"] == "pending"
    assert source_summary["status"] == "failed"
    # Twelve seed embeddings and the first retrieval-query embedding precede
    # the first simulator attempt whose research-log projection is rejected.
    assert len(attempts) == 14
    assert not error.paths.diagnostic_evaluation_dir.exists()


def test_identical_mocked_real_inputs_produce_byte_identical_pilot_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-determinism-key")
    gt_dir = install_phase11_v2_test_contract(monkeypatch, tmp_path)
    results = []
    for root_name in ("first", "second"):
        results.append(
            execute_pilot(
                mode="real",
                output_root=tmp_path / root_name,
                experiment_id="phase11_deterministic",
                run_id="pilot",
                retrieval_config_id="balanced",
                repetition_index=1,
                real_authorized=True,
                max_total_tokens=3000,
                provider=Phase11MockProvider(),
                sleeper=lambda _: None,
            )
        )

    first, second = results
    assert directory_bytes(first.paths.source_run_dir) == directory_bytes(
        second.paths.source_run_dir
    )
    assert directory_bytes(first.paths.diagnostic_evaluation_dir) == directory_bytes(
        second.paths.diagnostic_evaluation_dir
    )
    assert directory_bytes(first.paths.evidence_dir) == directory_bytes(
        second.paths.evidence_dir
    )
    first_review = review_pilot(
        evidence_dir=first.paths.evidence_dir,
        decision="accepted",
        ground_truth_finalization_dir=gt_dir,
    )
    second_review = review_pilot(
        evidence_dir=second.paths.evidence_dir,
        decision="accepted",
        ground_truth_finalization_dir=gt_dir,
    )
    assert directory_bytes(first_review.review_dir) == directory_bytes(
        second_review.review_dir
    )
