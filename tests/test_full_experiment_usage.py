"""Phase 13 global provider-attempt and token accounting safeguards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.simulation.experiment_controller as controller_module
from src.llm import (
    EMBEDDING_MODEL_IDENTITY,
    EVALUATOR_MODEL_IDENTITY,
    PROMPT_VERSION,
    SIMULATOR_MODEL_IDENTITY,
    ProviderAttemptRecord,
    ProviderUsage,
)
from src.simulation import (
    APPROVED_PHASE10_PROMPT_HASHES,
    MAX_MISSING_USAGE_ATTEMPTS,
    ExperimentAttempt,
    ExperimentAttemptLedger,
    ExperimentBudgetError,
    ExperimentUsageError,
    build_full_experiment_plan,
)
from src.storage import build_experiment_paths, create_experiment_directory


def _simulator_attempt(
    logical_id: str,
    attempt_index: int,
    status: str,
    *,
    usage: ProviderUsage,
    error_type: str | None = None,
) -> ProviderAttemptRecord:
    return ProviderAttemptRecord(
        service_role="simulator",
        logical_call_id=logical_id,
        attempt_index=attempt_index,
        call_type="plan",
        model=SIMULATOR_MODEL_IDENTITY,
        prompt_template="plan",
        prompt_version=PROMPT_VERSION,
        prompt_hash=APPROVED_PHASE10_PROMPT_HASHES["plan"],
        usage=usage,
        status=status,
        error_type=error_type,
    )


def _embedding_attempt(
    logical_id: str,
    *,
    usage: ProviderUsage,
) -> ProviderAttemptRecord:
    return ProviderAttemptRecord(
        service_role="embedding",
        logical_call_id=logical_id,
        attempt_index=1,
        call_type="embedding",
        model=EMBEDDING_MODEL_IDENTITY,
        usage=usage,
        status="success",
    )


def _ledger(tmp_path: Path, *, budget: int = 100) -> ExperimentAttemptLedger:
    paths = build_experiment_paths(tmp_path, experiment_id="usage_test")
    create_experiment_directory(paths)
    return ExperimentAttemptLedger(
        attempts_path=paths.attempts_file,
        plan=build_full_experiment_plan(experiment_id="usage_test", mode="real"),
        max_total_tokens=budget,
    )


def test_recovered_retry_counts_two_physical_attempts_and_one_retry(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.record(
        "balanced_rep01",
        _simulator_attempt(
            "sim_retry",
            1,
            "failed",
            usage=ProviderUsage(2, 1, 3),
            error_type="ProviderTransientError",
        ),
    )
    ledger.record(
        "balanced_rep01",
        _simulator_attempt(
            "sim_retry",
            2,
            "success",
            usage=ProviderUsage(3, 1, 4),
        ),
    )

    usage = ledger.summary()
    assert usage.logical_call_count == 1
    assert usage.physical_attempt_count == 2
    assert usage.retry_count == 1
    assert usage.known_total_tokens == usage.actual_total_tokens == 7
    assert usage.missing_usage_attempt_count == 0
    assert usage.usage_complete is True


def test_missing_usage_within_cap_is_tolerated_and_allows_next_call(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.record(
        "balanced_rep01",
        _simulator_attempt(
            "sim_missing",
            1,
            "failed",
            usage=ProviderUsage(),
            error_type="ProviderTransientError",
        ),
    )
    ledger.record(
        "balanced_rep01",
        _simulator_attempt(
            "sim_missing",
            2,
            "success",
            usage=ProviderUsage(3, 1, 4),
        ),
    )

    usage = ledger.summary()
    assert usage.physical_attempt_count == 2
    assert usage.retry_count == 1
    assert usage.missing_usage_attempt_count == 1
    assert usage.known_total_tokens == 4
    assert usage.usage_complete is True
    assert usage.actual_total_tokens == 4
    ledger.authorize_logical_call("balanced_rep01", "simulator")


def test_missing_usage_beyond_cap_blocks_next_call(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    for index in range(1, MAX_MISSING_USAGE_ATTEMPTS + 2):
        ledger.record(
            "balanced_rep01",
            _simulator_attempt(
                f"sim_missing_{index}",
                1,
                "failed",
                usage=ProviderUsage(),
                error_type="ProviderTransientError",
            ),
        )

    usage = ledger.summary()
    assert usage.missing_usage_attempt_count == MAX_MISSING_USAGE_ATTEMPTS + 1
    assert usage.usage_complete is False
    with pytest.raises(ExperimentUsageError, match="cap"):
        ledger.authorize_logical_call("balanced_rep01", "simulator")

    persisted = [
        json.loads(line)
        for line in ledger.attempts_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted[0]["total_tokens"] is None
    assert persisted[0]["input_tokens"] is None
    assert persisted[0]["output_tokens"] is None
    assert persisted[0]["run_id"] == "balanced_rep01"


def test_budget_crossing_is_audited_then_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, budget=5)

    with pytest.raises(ExperimentBudgetError, match="exceeded"):
        ledger.record(
            "balanced_rep01",
            _simulator_attempt(
                "sim_budget",
                1,
                "success",
                usage=ProviderUsage(4, 2, 6),
            ),
        )

    assert ledger.summary().budget_state == "exceeded"
    assert ledger.summary().physical_attempt_count == 1
    assert ledger.summary().known_total_tokens == 6


def test_phase13_rejects_evaluator_attempts() -> None:
    evaluator = ProviderAttemptRecord(
        service_role="evaluator",
        logical_call_id="judge_1",
        attempt_index=1,
        call_type="bcr_judge",
        model=EVALUATOR_MODEL_IDENTITY,
        prompt_template="bcr_judge",
        prompt_version=PROMPT_VERSION,
        prompt_hash=APPROVED_PHASE10_PROMPT_HASHES["bcr_judge"],
        usage=ProviderUsage(1, 1, 2),
        status="success",
    )

    with pytest.raises(ExperimentUsageError, match="no evaluator"):
        ExperimentAttempt("balanced_rep01", evaluator)


def test_complete_real_run_inventory_accepts_all_usage_present_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies the production ledger without performing 677 fsync calls.
    monkeypatch.setattr(controller_module, "append_attempt", lambda *a, **k: None)
    ledger = _ledger(tmp_path, budget=100000)
    run_id = "balanced_rep01"
    ledger.record(
        run_id,
        _simulator_attempt(
            "simulator_000",
            1,
            "failed",
            usage=ProviderUsage(1, 0, 1),
            error_type="ProviderTransientError",
        ),
    )
    ledger.record(
        run_id,
        _simulator_attempt(
            "simulator_000",
            2,
            "success",
            usage=ProviderUsage(1, 0, 1),
        ),
    )
    for index in range(1, 408):
        ledger.record(
            run_id,
            _simulator_attempt(
                f"simulator_{index:03d}",
                1,
                "success",
                usage=ProviderUsage(1, 0, 1),
            ),
        )
    for index in range(268):
        ledger.record(
            run_id,
            _embedding_attempt(
                f"embedding_{index:03d}",
                usage=ProviderUsage(1, 0, 1),
            ),
        )

    ledger.validate_completed_run(run_id)
    usage = ledger.summary()
    assert usage.logical_call_count == 676
    assert usage.physical_attempt_count == 677
    assert usage.retry_count == 1
    assert usage.missing_usage_attempt_count == 0
