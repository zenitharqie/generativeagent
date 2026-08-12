"""Provider-attempt persistence and exact Phase 11 token-budget behavior."""

from __future__ import annotations

import json

import pytest

from src.config_loader import load_experiment_config
from src.llm.usage import ProviderAttemptRecord, ProviderUsage
from src.logging_system import RunLogMetadata
from src.simulation.safeguards import (
    PILOT_CALL_LIMITS,
    PilotAttemptRecorder,
    PilotTokenBudgetExceeded,
    PilotUsageError,
)


def embedding_attempt(
    *,
    logical_call_id: str,
    attempt_index: int = 1,
    total_tokens: int | None = 5,
    status: str = "success",
    error_type: str | None = None,
) -> ProviderAttemptRecord:
    return ProviderAttemptRecord(
        service_role="embedding",
        logical_call_id=logical_call_id,
        attempt_index=attempt_index,
        call_type="embedding",
        model="openai:text-embedding-3-small",
        usage=ProviderUsage(
            total_tokens,
            None,
            total_tokens,
        ) if total_tokens is not None else ProviderUsage(),
        status=status,
        error_type=error_type,
    )


def recorder(tmp_path, budget: int = 10) -> PilotAttemptRecorder:
    attempts = tmp_path / "pilot_attempts.jsonl"
    attempts.touch()
    return PilotAttemptRecorder(
        attempts_path=attempts,
        api_usage_path=tmp_path / "unused_api_usage_logs.jsonl",
        run_metadata=RunLogMetadata("phase11_budget", "pilot", "balanced", 1),
        config=load_experiment_config(),
        max_total_tokens=budget,
        call_limits=PILOT_CALL_LIMITS,
    )


def test_exact_budget_boundary_is_allowed_but_blocks_the_next_call(tmp_path) -> None:
    usage = recorder(tmp_path, budget=5)

    usage.record(embedding_attempt(logical_call_id="embedding_1", total_tokens=5))

    assert usage.reported_total_tokens == 5
    with pytest.raises(PilotTokenBudgetExceeded, match="exhausted"):
        usage.authorize_logical_call("embedding")


def test_post_call_overshoot_is_persisted_then_aborts(tmp_path) -> None:
    usage = recorder(tmp_path, budget=5)

    with pytest.raises(PilotTokenBudgetExceeded, match="exceeded"):
        usage.record(embedding_attempt(logical_call_id="embedding_1", total_tokens=6))

    payload = json.loads(
        (tmp_path / "pilot_attempts.jsonl").read_text(encoding="utf-8")
    )
    assert payload["logical_call_id"] == "embedding_1"
    assert payload["total_tokens"] == 6
    assert usage.reported_total_tokens == 6


def test_retry_attempts_share_one_logical_call_and_both_count(tmp_path) -> None:
    usage = recorder(tmp_path, budget=20)
    first = embedding_attempt(
        logical_call_id="embedding_retry",
        status="failed",
        error_type="ProviderTimeoutError",
        total_tokens=None,
    )
    second = embedding_attempt(
        logical_call_id="embedding_retry",
        attempt_index=2,
        total_tokens=5,
    )

    usage.record(first)
    usage.record(second)

    assert len(usage.records) == 2
    assert usage.logical_call_count == 1
    assert usage.retry_count == 1
    assert usage.missing_usage_attempt_count == 1


def test_success_without_total_usage_is_persisted_but_rejected(tmp_path) -> None:
    usage = recorder(tmp_path, budget=20)

    with pytest.raises(PilotUsageError, match="successful provider attempt"):
        usage.record(
            embedding_attempt(
                logical_call_id="embedding_missing",
                total_tokens=None,
            )
        )

    assert len(usage.records) == 1


def test_duplicate_attempt_identity_is_rejected_without_second_append(tmp_path) -> None:
    usage = recorder(tmp_path)
    attempt = embedding_attempt(logical_call_id="embedding_duplicate")
    usage.record(attempt)

    with pytest.raises(PilotUsageError, match="duplicate"):
        usage.record(attempt)

    assert len((tmp_path / "pilot_attempts.jsonl").read_text().splitlines()) == 1
