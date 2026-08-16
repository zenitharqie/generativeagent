"""Immutable Phase 10 per-unit and run-level judged BCR result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
import re
from typing import Any

from src.evaluation.errors import BCRJudgmentError
from src.evaluation.schemas import (
    BCRModelMetadata,
    EvaluationUnitIdentity,
    FutureBCRJudgment,
)
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    EVALUATOR_PROFILE,
    SIMULATOR_MODEL_IDENTITY,
)
from src.llm.usage import ProviderAttemptRecord


BCR_JUDGMENT_SCHEMA_VERSION = "1.0"
_HASH = re.compile(r"[0-9a-f]{64}")


def _eligibility(mode: object, primary: object, included: object) -> None:
    if mode == "primary" and primary is True and included is True:
        return
    if mode == "diagnostic" and primary is False and included is False:
        return
    raise BCRJudgmentError("BCR judgment eligibility fields are inconsistent")


def _hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise BCRJudgmentError(f"{field_name} must be SHA-256 hex")
    return value


@dataclass(frozen=True)
class BCRJudgedRecord:
    """One completed BCR judgment joined to an exact Phase 09 export identity."""

    identity: EvaluationUnitIdentity
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    judgment: FutureBCRJudgment
    model_metadata: BCRModelMetadata
    prompt_template: str
    prompt_hash: str
    source_record_hash: str
    provider_attempts: tuple[ProviderAttemptRecord, ...]
    schema_version: str = field(default=BCR_JUDGMENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationUnitIdentity):
            raise BCRJudgmentError("identity must be EvaluationUnitIdentity")
        _eligibility(
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        if not isinstance(self.judgment, FutureBCRJudgment):
            raise BCRJudgmentError("judgment must be FutureBCRJudgment")
        if not isinstance(self.model_metadata, BCRModelMetadata):
            raise BCRJudgmentError("model_metadata must be BCRModelMetadata")
        evaluator_identity = self.model_metadata.evaluator_model_identity
        if evaluator_identity is None:
            raise BCRJudgmentError("judged record requires an evaluator model identity")
        if evaluator_identity == self.model_metadata.simulator_model_identity:
            raise BCRJudgmentError("simulator and evaluator identities must differ")
        if self.model_metadata.evaluator_prompt_version is None:
            raise BCRJudgmentError("judged record requires evaluator prompt version")
        if self.model_metadata.evaluator_rubric_version != BCR_RUBRIC_VERSION:
            raise BCRJudgmentError("judged record requires approved rubric version")
        if self.evaluation_mode == "primary" and (
            self.model_metadata.simulator_model_identity != SIMULATOR_MODEL_IDENTITY
        ):
            raise BCRJudgmentError("primary judgment requires approved simulator identity")
        if not isinstance(self.prompt_template, str) or not self.prompt_template:
            raise BCRJudgmentError("prompt_template must not be empty")
        _hash(self.prompt_hash, "prompt_hash")
        _hash(self.source_record_hash, "source_record_hash")
        if not isinstance(self.provider_attempts, tuple) or not self.provider_attempts:
            raise BCRJudgmentError("provider_attempts must be a non-empty tuple")
        logical_ids = {attempt.logical_call_id for attempt in self.provider_attempts}
        if len(logical_ids) != 1:
            raise BCRJudgmentError("provider attempts must share one logical_call_id")
        if tuple(item.attempt_index for item in self.provider_attempts) != tuple(
            range(1, len(self.provider_attempts) + 1)
        ):
            raise BCRJudgmentError("provider attempt indexes must be consecutive")
        if self.provider_attempts[-1].status != "success":
            raise BCRJudgmentError("completed judgment must end with a successful attempt")
        for attempt in self.provider_attempts:
            if (
                attempt.service_role != "evaluator"
                or attempt.call_type != "bcr_judge"
                or attempt.model != evaluator_identity
                or attempt.prompt_template != self.prompt_template
                or attempt.prompt_version != self.model_metadata.evaluator_prompt_version
                or attempt.prompt_hash != self.prompt_hash
            ):
                raise BCRJudgmentError("provider attempt metadata does not match judged record")

    @property
    def consistent(self) -> int:
        return self.judgment.consistent

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_aggregation": (
                self.included_in_final_research_aggregation
            ),
            "audit_metadata": self.identity.to_dict(),
            "source_record_hash": self.source_record_hash,
            "judgment": self.judgment.to_dict(),
            "model_metadata": self.model_metadata.to_dict(),
            "prompt_identity": {
                "prompt_template": self.prompt_template,
                "prompt_version": self.model_metadata.evaluator_prompt_version,
                "prompt_hash": self.prompt_hash,
            },
            "provider_attempts": [item.to_dict() for item in self.provider_attempts],
        }


@dataclass(frozen=True)
class BCRJudgmentSummary:
    """One run-level binary BCR proportion and reproducibility audit summary."""

    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    source_evaluation_schema_version: str
    source_bcr_export_hash: str
    records: tuple[BCRJudgedRecord, ...]
    model_metadata: BCRModelMetadata
    prompt_template: str
    prompt_hash: str
    schema_version: str = field(default=BCR_JUDGMENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _eligibility(
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        if self.source_evaluation_schema_version != "1.0":
            raise BCRJudgmentError("unsupported source evaluation schema version")
        _hash(self.source_bcr_export_hash, "source_bcr_export_hash")
        if not isinstance(self.records, tuple) or not self.records:
            raise BCRJudgmentError("judgment summary records must not be empty")
        if self.evaluation_mode == "primary" and len(self.records) != 20:
            raise BCRJudgmentError("primary BCR judgment requires exactly 20 records")
        if not isinstance(self.model_metadata, BCRModelMetadata):
            raise BCRJudgmentError("model_metadata must be BCRModelMetadata")
        _hash(self.prompt_hash, "prompt_hash")
        expected_identity = (
            self.experiment_id,
            self.run_id,
            self.retrieval_config_id,
            self.repetition_index,
            self.evaluation_mode,
            self.primary_eligible,
            self.included_in_final_research_aggregation,
        )
        seen: set[str] = set()
        for record in self.records:
            actual = (
                record.identity.experiment_id,
                record.identity.run_id,
                record.identity.retrieval_config_id,
                record.identity.repetition_index,
                record.evaluation_mode,
                record.primary_eligible,
                record.included_in_final_research_aggregation,
            )
            if actual != expected_identity:
                raise BCRJudgmentError("judged record identity does not match summary")
            if record.identity.turn_id in seen:
                raise BCRJudgmentError("duplicate judged BCR evaluation unit")
            seen.add(record.identity.turn_id)
            if (
                record.model_metadata != self.model_metadata
                or record.prompt_template != self.prompt_template
                or record.prompt_hash != self.prompt_hash
            ):
                raise BCRJudgmentError("judged record reproducibility metadata mismatch")
        order = tuple(
            (record.identity.timestamp, record.identity.turn_index)
            for record in self.records
        )
        if order != tuple(sorted(order)):
            raise BCRJudgmentError("judged BCR records must preserve source chronology")

    @property
    def judged_unit_count(self) -> int:
        return len(self.records)

    @property
    def consistent_count(self) -> int:
        return sum(record.consistent for record in self.records)

    @property
    def inconsistent_count(self) -> int:
        return self.judged_unit_count - self.consistent_count

    @property
    def run_bcr(self) -> float:
        value = self.consistent_count / self.judged_unit_count
        if not isfinite(value) or not 0 <= value <= 1:
            raise BCRJudgmentError("run-level BCR must be finite and in [0, 1]")
        return value

    @property
    def provider_attempt_count(self) -> int:
        return sum(len(record.provider_attempts) for record in self.records)

    def _usage_total(self, field_name: str) -> int | None:
        values = [
            getattr(attempt.usage, field_name)
            for record in self.records
            for attempt in record.provider_attempts
        ]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_evaluation_schema_version": self.source_evaluation_schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "retrieval_config_id": self.retrieval_config_id,
            "repetition_index": self.repetition_index,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_aggregation": (
                self.included_in_final_research_aggregation
            ),
            "source_bcr_export_hash": self.source_bcr_export_hash,
            "judged_unit_count": self.judged_unit_count,
            "consistent_count": self.consistent_count,
            "inconsistent_count": self.inconsistent_count,
            "bcr_formula": "consistent_count_divided_by_judged_unit_count",
            "run_bcr": self.run_bcr,
            "model_metadata": self.model_metadata.to_dict(),
            "prompt_identity": {
                "prompt_template": self.prompt_template,
                "prompt_version": self.model_metadata.evaluator_prompt_version,
                "prompt_hash": self.prompt_hash,
            },
            "runtime_settings": EVALUATOR_PROFILE.reproducibility_metadata(),
            "provider_attempt_count": self.provider_attempt_count,
            "usage": {
                "input_tokens": self._usage_total("input_tokens"),
                "output_tokens": self._usage_total("output_tokens"),
                "total_tokens": self._usage_total("total_tokens"),
            },
            "output_files": [
                "bcr_judged_records.jsonl",
                "bcr_judgment_summary.json",
            ],
        }


__all__ = [
    "BCR_JUDGMENT_SCHEMA_VERSION",
    "BCRJudgedRecord",
    "BCRJudgmentSummary",
]
