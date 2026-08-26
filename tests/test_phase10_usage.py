"""Provider attempt identity, usage validation, and serialization tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from src.llm.errors import UsageMetadataError
from src.llm.usage import ProviderAttemptRecord, ProviderUsage


HASH = "a" * 64


def simulator_attempt(**changes: object) -> ProviderAttemptRecord:
    values: dict[str, object] = {
        "service_role": "simulator",
        "logical_call_id": "simulator_" + "b" * 64,
        "attempt_index": 1,
        "call_type": "plan",
        "model": "openai:gpt-5.6-luna",
        "prompt_template": "plan",
        "prompt_version": "phase10_pilot_v1",
        "prompt_hash": HASH,
        "usage": ProviderUsage(10, 4, 14),
        "status": "success",
    }
    values.update(changes)
    return ProviderAttemptRecord(**values)


def test_attempt_serializes_every_required_audit_field_without_prompt_body() -> None:
    payload = simulator_attempt().to_dict()

    assert payload == {
        "schema_version": "1.0",
        "service_role": "simulator",
        "logical_call_id": "simulator_" + "b" * 64,
        "attempt_index": 1,
        "call_type": "plan",
        "model": "openai:gpt-5.6-luna",
        "prompt_template": "plan",
        "prompt_version": "phase10_pilot_v1",
        "prompt_hash": HASH,
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "status": "success",
        "error_type": None,
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert "system_instructions" not in serialized
    assert "user_data" not in serialized
    assert "response" not in serialized
    assert "cost" not in serialized


def test_attempt_models_are_frozen() -> None:
    record = simulator_attempt()

    with pytest.raises(FrozenInstanceError):
        record.status = "failed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"service_role": "unknown"},
        {"attempt_index": 0},
        {"attempt_index": 3},
        {"model": "moving-alias"},
        {"status": "success", "error_type": "ProviderTimeoutError"},
        {"status": "failed", "error_type": None},
        {"prompt_version": None},
        {"prompt_hash": "short"},
    ],
)
def test_attempt_rejects_invalid_audit_combinations(changes: dict[str, object]) -> None:
    with pytest.raises(UsageMetadataError):
        simulator_attempt(**changes)


def test_embedding_attempt_requires_no_prompt_identity() -> None:
    record = ProviderAttemptRecord(
        service_role="embedding",
        logical_call_id="embedding_000001_" + "c" * 64,
        attempt_index=1,
        call_type="embedding",
        model="openai:text-embedding-3-small",
        usage=ProviderUsage(7, None, 7),
        status="success",
    )

    assert record.prompt_template is None
    assert record.prompt_version is None
    assert record.prompt_hash is None


def test_usage_rejects_inconsistent_or_negative_token_counts() -> None:
    with pytest.raises(UsageMetadataError, match="total_tokens"):
        ProviderUsage(10, 2, 11)
    with pytest.raises(UsageMetadataError, match="non-negative"):
        ProviderUsage(-1, None, None)
