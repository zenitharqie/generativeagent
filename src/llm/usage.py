"""Immutable, secret-free provider-attempt usage audit records."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, runtime_checkable

from src.llm.errors import UsageMetadataError
from src.llm.settings import MAX_PROVIDER_ATTEMPTS


USAGE_SCHEMA_VERSION = "1.0"
SERVICE_ROLES = frozenset({"simulator", "embedding", "evaluator"})
ATTEMPT_STATUSES = frozenset({"success", "failed", "invalid_output"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise UsageMetadataError(f"{field_name} must be a safe identifier")
    return value


def _optional_token_count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UsageMetadataError(f"{field_name} must be a non-negative integer or None")
    return value


@dataclass(frozen=True)
class ProviderUsage:
    """Normalized provider token counts; absent counts remain explicit ``None``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            _optional_token_count(getattr(self, field_name), field_name)
        if self.total_tokens is not None and self.input_tokens is not None:
            expected = self.input_tokens + (self.output_tokens or 0)
            if self.total_tokens != expected:
                raise UsageMetadataError(
                    "total_tokens must equal available input and output token counts"
                )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ProviderAttemptRecord:
    """One provider attempt; retries share a deterministic logical call ID."""

    service_role: str
    logical_call_id: str
    attempt_index: int
    call_type: str
    model: str
    status: str
    usage: ProviderUsage = ProviderUsage()
    prompt_template: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    error_type: str | None = None
    schema_version: str = USAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != USAGE_SCHEMA_VERSION:
            raise UsageMetadataError("unsupported provider-attempt schema_version")
        if self.service_role not in SERVICE_ROLES:
            raise UsageMetadataError("unsupported service_role")
        _identifier(self.logical_call_id, "logical_call_id")
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or not 1 <= self.attempt_index <= MAX_PROVIDER_ATTEMPTS
        ):
            raise UsageMetadataError("attempt_index is outside the approved attempt range")
        _identifier(self.call_type, "call_type")
        if not isinstance(self.model, str) or not self.model.strip() or ":" not in self.model:
            raise UsageMetadataError("model must be a provider-qualified identity")
        if self.status not in ATTEMPT_STATUSES:
            raise UsageMetadataError("unsupported provider-attempt status")
        if not isinstance(self.usage, ProviderUsage):
            raise UsageMetadataError("usage must be ProviderUsage")
        prompt_values = (self.prompt_template, self.prompt_version, self.prompt_hash)
        if self.service_role == "embedding":
            if any(value is not None for value in prompt_values):
                raise UsageMetadataError("embedding attempts must not claim a prompt template")
        else:
            if any(value is None for value in prompt_values):
                raise UsageMetadataError("text attempts require complete prompt identity")
            _identifier(self.prompt_template, "prompt_template")
            _identifier(self.prompt_version, "prompt_version")
            if not isinstance(self.prompt_hash, str) or _HASH.fullmatch(self.prompt_hash) is None:
                raise UsageMetadataError("prompt_hash must be SHA-256 hex")
        if self.status == "success":
            if self.error_type is not None:
                raise UsageMetadataError("successful attempts must not have error_type")
        else:
            _identifier(self.error_type, "error_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "service_role": self.service_role,
            "logical_call_id": self.logical_call_id,
            "attempt_index": self.attempt_index,
            "call_type": self.call_type,
            "model": self.model,
            "prompt_template": self.prompt_template,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            **self.usage.to_dict(),
            "status": self.status,
            "error_type": self.error_type,
        }


@runtime_checkable
class UsageRecorder(Protocol):
    def record(self, attempt: ProviderAttemptRecord) -> None:
        """Persist or retain one already validated provider attempt."""


class InMemoryUsageRecorder:
    """Deterministic offline/default recorder; callers may persist its records."""

    def __init__(self) -> None:
        self._records: list[ProviderAttemptRecord] = []

    def record(self, attempt: ProviderAttemptRecord) -> None:
        if not isinstance(attempt, ProviderAttemptRecord):
            raise UsageMetadataError("attempt must be ProviderAttemptRecord")
        self._records.append(attempt)

    @property
    def records(self) -> tuple[ProviderAttemptRecord, ...]:
        return tuple(self._records)


__all__ = [
    "ATTEMPT_STATUSES",
    "InMemoryUsageRecorder",
    "ProviderAttemptRecord",
    "ProviderUsage",
    "SERVICE_ROLES",
    "USAGE_SCHEMA_VERSION",
    "UsageRecorder",
]
