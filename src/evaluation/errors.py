"""Phase 09 evaluation error hierarchy."""

from __future__ import annotations

from src.llm.usage import ProviderAttemptRecord


class EvaluationError(RuntimeError):
    """Base error for deterministic post-run evaluation."""


class EvaluationInputError(EvaluationError):
    """Raised when persisted source artifacts cannot be trusted."""


class EvaluationEligibilityError(EvaluationInputError):
    """Raised when a run lifecycle status is not eligible for evaluation."""


class EvaluationGroundTruthError(EvaluationInputError):
    """Raised when snapshotted evaluation ground truth is invalid or incomplete."""


class EvaluationJoinError(EvaluationInputError):
    """Raised when exact persisted-record joins cannot be completed."""


class EvaluationSchemaError(EvaluationError, ValueError):
    """Raised when a Phase 09 immutable model violates its schema."""


class EvaluationCalculationError(EvaluationError, ValueError):
    """Raised when an evaluation metric cannot be calculated exactly."""


class EvaluationAggregationError(EvaluationError, ValueError):
    """Raised when metric records cannot be aggregated without bias."""


class EvaluationPersistenceError(EvaluationError):
    """Raised when evaluation output cannot be safely published."""


class EvaluationConflictError(EvaluationPersistenceError):
    """Raised when a requested final or staging output already exists."""


class BCRJudgmentError(EvaluationError):
    """Base error for Phase 10 real BCR judgment."""


class BCRJudgmentFailure(BCRJudgmentError):
    """Terminal evaluator failure retaining only immutable sanitized attempt audit."""

    _MESSAGES = {
        "provider_failure": "OpenAI BCR evaluator call failed",
        "invalid_output": "OpenAI evaluator returned invalid structured output twice",
    }

    def __init__(
        self,
        failure_type: str,
        provider_attempts: tuple[ProviderAttemptRecord, ...],
    ) -> None:
        try:
            message = self._MESSAGES[failure_type]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported BCR judgment failure_type") from exc
        if not isinstance(provider_attempts, tuple) or not provider_attempts:
            raise ValueError("terminal BCR failure requires provider attempts")
        if any(
            not isinstance(attempt, ProviderAttemptRecord)
            or attempt.service_role != "evaluator"
            or attempt.call_type != "bcr_judge"
            for attempt in provider_attempts
        ):
            raise ValueError("terminal BCR failure attempts are invalid")
        if len({attempt.logical_call_id for attempt in provider_attempts}) != 1:
            raise ValueError("terminal BCR failure attempts must share a logical_call_id")
        if tuple(attempt.attempt_index for attempt in provider_attempts) != tuple(
            range(1, len(provider_attempts) + 1)
        ):
            raise ValueError("terminal BCR failure attempt indexes must be consecutive")
        if any(attempt.status == "success" for attempt in provider_attempts):
            raise ValueError("terminal BCR failure attempts must not contain success")
        expected_terminal_status = (
            "failed" if failure_type == "provider_failure" else "invalid_output"
        )
        if provider_attempts[-1].status != expected_terminal_status:
            raise ValueError("terminal BCR failure status does not match failure_type")
        super().__init__(message)
        self._failure_type = failure_type
        self._provider_attempts = provider_attempts

    @property
    def failure_type(self) -> str:
        return self._failure_type

    @property
    def provider_attempts(self) -> tuple[ProviderAttemptRecord, ...]:
        return self._provider_attempts


class BCRJudgmentInputError(BCRJudgmentError):
    """Raised when a Phase 09 BCR export cannot be trusted as judge input."""


class BCRJudgmentIndependenceError(BCRJudgmentError):
    """Raised before calls when primary evaluator independence is not proven."""


class BCRJudgmentPersistenceError(BCRJudgmentError):
    """Raised when judged BCR artifacts cannot be published atomically."""


class BCRJudgmentConflictError(BCRJudgmentPersistenceError):
    """Raised for an existing final or staging judgment output path."""


__all__ = [
    "BCRJudgmentConflictError",
    "BCRJudgmentError",
    "BCRJudgmentFailure",
    "BCRJudgmentIndependenceError",
    "BCRJudgmentInputError",
    "BCRJudgmentPersistenceError",
    "EvaluationAggregationError",
    "EvaluationCalculationError",
    "EvaluationConflictError",
    "EvaluationEligibilityError",
    "EvaluationError",
    "EvaluationGroundTruthError",
    "EvaluationInputError",
    "EvaluationJoinError",
    "EvaluationPersistenceError",
    "EvaluationSchemaError",
]
