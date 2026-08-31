"""Guarded Day 1-2 pilot execution, evidence, review, and freeze publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from typing import Any

from src.config_loader import (
    DEFAULT_CONFIG_DIRECTORY,
    ExperimentConfig,
    load_experiment_config,
)
from src.config_loader.loader import CONFIG_FILES
from src.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationRunResult,
    evaluate_run,
    load_bcr_export,
)
from src.evaluation.ground_truth import authenticated_ground_truth_identity
from src.llm import (
    DummyEmbeddingProvider,
    LLMService,
    OpenAIEmbeddingProvider,
    OpenAIProviderAdapter,
)
from src.llm.prompt_loader import PROMPT_SPECS, PROMPT_VERSION, PromptLoader
from src.llm.provider import EmbeddingAPIProvider, StructuredTextProvider
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    EMBEDDING_MODEL_IDENTITY,
    EMBEDDING_PROFILE,
    EVALUATOR_MODEL_IDENTITY,
    EVALUATOR_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    RETRY_DELAY_SECONDS,
    SIMULATOR_MODEL_IDENTITY,
    SIMULATOR_PROFILE,
)
from src.llm.usage import ProviderAttemptRecord, ProviderUsage
from src.logging_system import SCHEMA_VERSION, RunLogMetadata, read_run_summary
from src.simulation.run_controller import (
    RunController,
    RunExecutionError,
    RunExecutionResult,
    build_deterministic_dummy_llm_client,
)
from src.simulation.run_validator import (
    PilotRunValidation,
    PilotRunValidationError,
    fingerprint_directory,
    validate_no_secret_artifacts,
    validate_pilot_run,
)
from src.simulation.full_experiment import missing_usage_cap
from src.simulation.safeguards import (
    APPROVED_PHASE10_PROMPT_HASHES,
    GuardedPilotEmbeddingProvider,
    GuardedPilotLLMClient,
    PILOT_ACCEPTANCE_STATES,
    PILOT_CALL_LIMITS,
    PILOT_SCHEMA_VERSION,
    PilotAttemptRecorder,
    PilotError,
    PilotPaths,
    PilotPlan,
    PilotPreflightError,
    PilotUsageError,
    build_pilot_paths,
    build_pilot_plan,
    require_openai_credential,
    validate_fresh_pilot_paths,
    validate_phase10_runtime,
    validate_real_authorization,
)
from src.storage import InvalidIdentifierError, validate_identifier


def _capture_pilot_source_identity(*, require_clean: bool) -> dict[str, object]:
    """Reuse Phase 13's canonical Git identity capture without duplicating it."""
    try:
        from src.simulation.experiment_controller import capture_git_source_identity

        return capture_git_source_identity(
            require_clean=require_clean, require_tag=True
        ).to_dict()
    except Exception:
        raise PilotPreflightError("could not capture pilot source identity") from None


def _validate_pilot_source_identity(value: object, *, require_clean: bool) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "commit", "branch", "tags", "clean"
    }:
        raise PilotReviewError("pilot source identity is invalid")
    commit = value["commit"]
    branch = value["branch"]
    tags = value["tags"]
    clean = value["clean"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(branch, str)
        or not branch
        or not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag for tag in tags)
        or tags != sorted(set(tags))
        or not isinstance(clean, bool)
        or require_clean and clean is not True
    ):
        raise PilotReviewError("pilot source identity is not a clean tagged source")


PILOT_EVIDENCE_FILENAMES = (
    "pilot_manifest.json",
    "pilot_attempts.jsonl",
    "pilot_usage_summary.json",
    "pilot_report.json",
)
PILOT_ACCEPTANCE_FILENAME = "pilot_acceptance.json"
PILOT_FREEZE_FILENAME = "freeze_manifest.json"
_HASH_LENGTH = 64
_SUCCESSFUL_TECHNICAL_CHECKS = (
    "authorization",
    "fixed_scope",
    "model_and_settings_identity",
    "prompt_inventory_and_hashes",
    "configuration_snapshot",
    "token_and_call_budget",
    "source_run_artifact_completeness",
    "target_only_diagnostic_evaluation",
    "secret_and_payload_isolation",
    "source_run_immutability_after_evaluation",
)


class _DuplicatePilotJsonKeyError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicatePilotJsonKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


class PilotPersistenceError(PilotError):
    """Raised when immutable pilot evidence cannot be published."""


class PilotReviewError(PilotError):
    """Raised when manual review or freeze evidence is invalid."""


class PilotExecutionError(PilotError):
    """Sanitized terminal execution error retaining safe artifact locations."""

    def __init__(
        self,
        failure_type: str,
        *,
        paths: PilotPaths,
        evidence_published: bool,
    ) -> None:
        if not isinstance(failure_type, str) or not failure_type:
            raise ValueError("failure_type must be a non-empty identifier")
        super().__init__(f"pilot failed due to {failure_type}")
        self.failure_type = failure_type
        self.paths = paths
        self.evidence_published = evidence_published


def _require_hash(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PilotPreflightError(f"{field_name} must be SHA-256 hex")
    return value


def _canonical_json_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotPersistenceError("pilot artifact is not strict JSON") from exc


def _content_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PilotPersistenceError(f"could not write fresh {path.name}") from exc


def _read_json_object(path: Path, *, error_type: type[PilotError]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise error_type(f"missing or invalid {path.name}")
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or b"\r" in payload:
            raise ValueError("artifact must be LF-terminated")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        _DuplicatePilotJsonKeyError,
    ) as exc:
        raise error_type(f"could not read strict {path.name}") from exc
    if not isinstance(value, dict):
        raise error_type(f"{path.name} must contain a JSON object")
    return value


@dataclass(frozen=True)
class ArtifactHash:
    filename: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise PilotPreflightError("artifact filename must be non-empty")
        _require_hash(self.sha256, "artifact hash")

    def to_dict(self) -> dict[str, str]:
        return {"filename": self.filename, "sha256": self.sha256}


@dataclass(frozen=True)
class PromptFreezeIdentity:
    template_id: str
    filename: str
    prompt_version: str
    prompt_hash: str
    model_role: str
    output_contract: str

    def __post_init__(self) -> None:
        spec = PROMPT_SPECS.get(self.template_id)
        if (
            spec is None
            or self.filename != spec.filename
            or self.prompt_version != spec.version
            or self.model_role != spec.model_role
            or self.output_contract != spec.output_contract
        ):
            raise PilotPreflightError("prompt identity does not match its approved spec")
        if APPROVED_PHASE10_PROMPT_HASHES.get(self.template_id) != self.prompt_hash:
            raise PilotPreflightError("prompt identity uses an unapproved hash")
        _require_hash(self.prompt_hash, "prompt_hash")

    def to_dict(self) -> dict[str, str]:
        return {
            "template_id": self.template_id,
            "filename": self.filename,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "model_role": self.model_role,
            "output_contract": self.output_contract,
        }


def hash_config_directory(directory: str | Path) -> tuple[ArtifactHash, ...]:
    """Hash the exact six canonical config bytes without normalization."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise PilotPreflightError("configuration directory is missing or a symlink")
    expected = tuple(sorted(CONFIG_FILES.values()))
    try:
        actual = tuple(sorted(item.name for item in root.iterdir()))
    except OSError as exc:
        raise PilotPreflightError("could not inspect configuration directory") from exc
    if actual != expected:
        raise PilotPreflightError("configuration directory inventory has drifted")
    result: list[ArtifactHash] = []
    for filename in expected:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise PilotPreflightError(f"invalid configuration asset: {filename}")
        try:
            digest = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PilotPreflightError(f"could not hash {filename}") from exc
        result.append(ArtifactHash(filename, digest))
    return tuple(result)


def prompt_identities(loader: PromptLoader) -> tuple[PromptFreezeIdentity, ...]:
    return tuple(
        PromptFreezeIdentity(
            template_id=template.spec.template_id,
            filename=template.spec.filename,
            prompt_version=template.spec.version,
            prompt_hash=template.content_hash,
            model_role=template.spec.model_role,
            output_contract=template.spec.output_contract,
        )
        for template in loader.load_all()
    )


@dataclass(frozen=True)
class PilotManifest:
    pilot_id: str
    experiment_id: str
    run_id: str
    plan: PilotPlan
    max_total_tokens: int | None
    configuration_hashes: tuple[ArtifactHash, ...]
    prompts: tuple[PromptFreezeIdentity, ...]
    source_identity: Mapping[str, object] | None = None
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            validate_identifier(self.experiment_id, field_name="experiment_id")
            validate_identifier(self.run_id, field_name="run_id")
        except InvalidIdentifierError:
            raise PilotPreflightError("manifest identifiers are invalid") from None
        if self.pilot_id != self.run_id:
            raise PilotPreflightError("pilot_id must equal the fresh source run_id")
        if (
            not isinstance(self.plan, PilotPlan)
            or not isinstance(self.configuration_hashes, tuple)
            or not isinstance(self.prompts, tuple)
            or any(not isinstance(item, ArtifactHash) for item in self.configuration_hashes)
            or any(not isinstance(item, PromptFreezeIdentity) for item in self.prompts)
            or len(self.configuration_hashes) != 6
            or len(self.prompts) != 6
            or {item.filename for item in self.configuration_hashes}
            != set(CONFIG_FILES.values())
            or {item.template_id for item in self.prompts} != set(PROMPT_SPECS)
            or tuple(item.filename for item in self.configuration_hashes)
            != tuple(sorted(CONFIG_FILES.values()))
            or tuple(item.template_id for item in self.prompts) != tuple(PROMPT_SPECS)
        ):
            raise PilotPreflightError("manifest must freeze six configs and six prompts")
        if self.plan.mode == "real":
            if (
                isinstance(self.max_total_tokens, bool)
                or not isinstance(self.max_total_tokens, int)
                or self.max_total_tokens < 1
            ):
                raise PilotPreflightError("real manifest requires a positive token ceiling")
        elif self.max_total_tokens is not None:
            raise PilotPreflightError("non-real manifest must not claim a token ceiling")
        if self.plan.mode == "real":
            _validate_pilot_source_identity(self.source_identity, require_clean=False)
        elif self.source_identity is not None:
            raise PilotPreflightError("dummy manifest must not claim source identity")
        object.__setattr__(self, "manifest_hash", _content_hash(self._base_dict()))

    def _base_dict(self) -> dict[str, object]:
        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "pilot_id": self.pilot_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "pilot_only": True,
            "primary_eligible": False,
            "included_in_final_research_matrix": False,
            "expected_source_run_status": "excluded",
            "plan": self.plan.to_dict(),
            "max_total_tokens": self.max_total_tokens,
            "configuration_hashes": [item.to_dict() for item in self.configuration_hashes],
            "prompts": [item.to_dict() for item in self.prompts],
            "source_identity": self.source_identity,
            "simulator_settings": SIMULATOR_PROFILE.reproducibility_metadata(),
            "evaluator_settings": EVALUATOR_PROFILE.reproducibility_metadata(),
            "embedding_settings": EMBEDDING_PROFILE.reproducibility_metadata(),
            "retry_policy": {
                "maximum_total_attempts": MAX_PROVIDER_ATTEMPTS,
                "deterministic_delay_seconds": RETRY_DELAY_SECONDS,
                "jitter": False,
            },
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "research_log_schema_version": SCHEMA_VERSION,
            "bcr_rubric_version": BCR_RUBRIC_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        value = self._base_dict()
        value["manifest_hash"] = self.manifest_hash
        return value


def build_pilot_manifest(
    *,
    plan: PilotPlan,
    experiment_id: str,
    run_id: str,
    config_directory: str | Path,
    max_total_tokens: object,
    source_identity: Mapping[str, object] | None = None,
    prompt_loader: PromptLoader | None = None,
) -> PilotManifest:
    loader = prompt_loader or PromptLoader()
    validate_phase10_runtime(loader)
    return PilotManifest(
        pilot_id=run_id,
        experiment_id=experiment_id,
        run_id=run_id,
        plan=plan,
        max_total_tokens=max_total_tokens,
        configuration_hashes=hash_config_directory(config_directory),
        prompts=prompt_identities(loader),
        source_identity=source_identity,
    )


@dataclass(frozen=True)
class RoleUsageSummary:
    service_role: str
    logical_call_count: int
    physical_attempt_count: int
    retry_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    missing_total_usage_attempt_count: int
    success_count: int
    failed_count: int
    invalid_output_count: int

    def __post_init__(self) -> None:
        if self.service_role not in {"simulator", "embedding", "evaluator"}:
            raise PilotPersistenceError("invalid pilot usage service role")
        numeric_fields = (
            "logical_call_count",
            "physical_attempt_count",
            "retry_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "missing_total_usage_attempt_count",
            "success_count",
            "failed_count",
            "invalid_output_count",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 0
            for name in numeric_fields
        ):
            raise PilotPersistenceError("pilot role usage counts must be non-negative")
        if (
            self.retry_count != self.physical_attempt_count - self.logical_call_count
            or self.success_count + self.failed_count + self.invalid_output_count
            != self.physical_attempt_count
        ):
            raise PilotPersistenceError("pilot role usage arithmetic is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "service_role": self.service_role,
            "logical_call_count": self.logical_call_count,
            "physical_attempt_count": self.physical_attempt_count,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "missing_total_usage_attempt_count": self.missing_total_usage_attempt_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "invalid_output_count": self.invalid_output_count,
        }


@dataclass(frozen=True)
class PilotUsageSummary:
    max_total_tokens: int | None
    reported_total_tokens: int
    logical_call_count: int
    physical_attempt_count: int
    retry_count: int
    missing_total_usage_attempt_count: int
    budget_exceeded: bool
    role_usage: tuple[RoleUsageSummary, ...]

    def __post_init__(self) -> None:
        if self.max_total_tokens is not None and (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < 1
        ):
            raise PilotPersistenceError("pilot token ceiling must be positive")
        numeric_fields = (
            "reported_total_tokens",
            "logical_call_count",
            "physical_attempt_count",
            "retry_count",
            "missing_total_usage_attempt_count",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 0
            for name in numeric_fields
        ):
            raise PilotPersistenceError("pilot aggregate usage counts must be non-negative")
        if (
            not isinstance(self.role_usage, tuple)
            or tuple(item.service_role for item in self.role_usage)
            != ("simulator", "embedding", "evaluator")
            or any(not isinstance(item, RoleUsageSummary) for item in self.role_usage)
        ):
            raise PilotPersistenceError("pilot role usage inventory is invalid")
        if (
            self.reported_total_tokens
            != sum(item.total_tokens for item in self.role_usage)
            or self.logical_call_count
            != sum(item.logical_call_count for item in self.role_usage)
            or self.physical_attempt_count
            != sum(item.physical_attempt_count for item in self.role_usage)
            or self.retry_count != self.physical_attempt_count - self.logical_call_count
            or self.missing_total_usage_attempt_count
            != sum(item.missing_total_usage_attempt_count for item in self.role_usage)
            or self.budget_exceeded
            != (
                self.max_total_tokens is not None
                and self.reported_total_tokens > self.max_total_tokens
            )
        ):
            raise PilotPersistenceError("pilot aggregate usage arithmetic is inconsistent")

    @classmethod
    def from_attempts(
        cls,
        attempts: tuple[ProviderAttemptRecord, ...],
        *,
        max_total_tokens: int | None,
    ) -> "PilotUsageSummary":
        roles: list[RoleUsageSummary] = []
        for role in ("simulator", "embedding", "evaluator"):
            selected = tuple(item for item in attempts if item.service_role == role)
            logical = {(item.service_role, item.logical_call_id) for item in selected}
            roles.append(
                RoleUsageSummary(
                    service_role=role,
                    logical_call_count=len(logical),
                    physical_attempt_count=len(selected),
                    retry_count=len(selected) - len(logical),
                    input_tokens=sum(
                        item.usage.input_tokens
                        for item in selected
                        if item.usage.input_tokens is not None
                    ),
                    output_tokens=sum(
                        item.usage.output_tokens
                        for item in selected
                        if item.usage.output_tokens is not None
                    ),
                    total_tokens=sum(
                        item.usage.total_tokens
                        for item in selected
                        if item.usage.total_tokens is not None
                    ),
                    missing_total_usage_attempt_count=sum(
                        item.usage.total_tokens is None for item in selected
                    ),
                    success_count=sum(item.status == "success" for item in selected),
                    failed_count=sum(item.status == "failed" for item in selected),
                    invalid_output_count=sum(
                        item.status == "invalid_output" for item in selected
                    ),
                )
            )
        reported = sum(
            item.usage.total_tokens
            for item in attempts
            if item.usage.total_tokens is not None
        )
        logical_count = len(
            {(item.service_role, item.logical_call_id) for item in attempts}
        )
        return cls(
            max_total_tokens=max_total_tokens,
            reported_total_tokens=reported,
            logical_call_count=logical_count,
            physical_attempt_count=len(attempts),
            retry_count=len(attempts) - logical_count,
            missing_total_usage_attempt_count=sum(
                item.usage.total_tokens is None for item in attempts
            ),
            budget_exceeded=(
                max_total_tokens is not None and reported > max_total_tokens
            ),
            role_usage=tuple(roles),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "max_total_tokens": self.max_total_tokens,
            "reported_total_tokens": self.reported_total_tokens,
            "logical_call_count": self.logical_call_count,
            "physical_attempt_count": self.physical_attempt_count,
            "retry_count": self.retry_count,
            "missing_total_usage_attempt_count": self.missing_total_usage_attempt_count,
            "budget_exceeded": self.budget_exceeded,
            "role_usage": [item.to_dict() for item in self.role_usage],
        }


@dataclass(frozen=True)
class PilotReport:
    pilot_id: str
    experiment_id: str
    run_id: str
    mode: str
    manifest_hash: str
    acceptance_state: str
    technical_success: bool
    eligible_for_manual_acceptance: bool
    source_run_status: str | None
    source_run_directory_name: str
    diagnostic_evaluation_directory_name: str
    primary_eligible: bool
    included_in_final_research_matrix: bool
    usage: PilotUsageSummary
    validation: PilotRunValidation | None
    source_fingerprint: str | None
    diagnostic_evaluation_fingerprint: str | None
    mra_diagnostic_record_count: int
    mra_diagnostic_mean: float | None
    bcr_export_record_count: int
    bcr_judgment_count: int
    technical_checks: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    terminal_error_type: str | None

    def __post_init__(self) -> None:
        if self.acceptance_state not in PILOT_ACCEPTANCE_STATES:
            raise PilotPersistenceError("invalid pilot acceptance state")
        if self.mode not in {"dummy", "real"}:
            raise PilotPersistenceError("invalid pilot report mode")
        if self.primary_eligible or self.included_in_final_research_matrix:
            raise PilotPersistenceError("pilot report must remain non-primary")
        if not isinstance(self.usage, PilotUsageSummary):
            raise PilotPersistenceError("pilot report usage summary is invalid")
        if self.bcr_judgment_count != 0:
            raise PilotPersistenceError("Phase 11 report cannot claim BCR judgments")
        if self.technical_success:
            if (
                self.acceptance_state != "pending"
                or self.source_run_status != "excluded"
                or not isinstance(self.validation, PilotRunValidation)
                or self.source_fingerprint is None
                or self.diagnostic_evaluation_fingerprint is None
                or self.mra_diagnostic_record_count != 4
                or self.bcr_export_record_count != 4
                or self.rejection_reasons
                or self.terminal_error_type is not None
                or self.eligible_for_manual_acceptance != (self.mode == "real")
                or self.technical_checks != _SUCCESSFUL_TECHNICAL_CHECKS
                or isinstance(self.mra_diagnostic_mean, bool)
                or not isinstance(self.mra_diagnostic_mean, (int, float))
                or not isfinite(float(self.mra_diagnostic_mean))
                or not 0.0 <= float(self.mra_diagnostic_mean) <= 1.0
            ):
                raise PilotPersistenceError("successful pilot report is inconsistent")
        elif (
            self.acceptance_state != "pending"
            or self.eligible_for_manual_acceptance
            or not self.rejection_reasons
            or self.terminal_error_type is None
            or self.technical_checks
            or self.validation is not None
            or self.diagnostic_evaluation_fingerprint is not None
            or self.mra_diagnostic_record_count != 0
            or self.mra_diagnostic_mean is not None
            or self.bcr_export_record_count != 0
        ):
            raise PilotPersistenceError(
                "failed pilot report must remain pending manual review"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "pilot_id": self.pilot_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "manifest_hash": self.manifest_hash,
            "acceptance_state": self.acceptance_state,
            "technical_success": self.technical_success,
            "eligible_for_manual_acceptance": self.eligible_for_manual_acceptance,
            "source_run_status": self.source_run_status,
            "source_run_directory_name": self.source_run_directory_name,
            "diagnostic_evaluation_directory_name": self.diagnostic_evaluation_directory_name,
            "primary_eligible": self.primary_eligible,
            "included_in_final_research_matrix": self.included_in_final_research_matrix,
            "usage": self.usage.to_dict(),
            "validation": None if self.validation is None else self.validation.to_dict(),
            "source_fingerprint": self.source_fingerprint,
            "diagnostic_evaluation_fingerprint": self.diagnostic_evaluation_fingerprint,
            "mra_diagnostic_record_count": self.mra_diagnostic_record_count,
            "mra_diagnostic_mean": self.mra_diagnostic_mean,
            "bcr_export_record_count": self.bcr_export_record_count,
            "bcr_judgment_count": self.bcr_judgment_count,
            "technical_checks": list(self.technical_checks),
            "rejection_reasons": list(self.rejection_reasons),
            "terminal_error_type": self.terminal_error_type,
        }


class PilotEvidenceWriter:
    """Incrementally retain attempts, then atomically publish complete evidence."""

    def __init__(self, paths: PilotPaths, manifest: PilotManifest) -> None:
        self.paths = paths
        self.manifest = manifest
        try:
            paths.experiment_dir.mkdir(parents=True, exist_ok=True)
            if paths.experiment_dir.resolve(strict=True) != paths.experiment_dir:
                raise PilotPersistenceError("pilot experiment path changed during preflight")
            paths.evidence_staging_dir.mkdir(mode=0o700)
        except PilotPersistenceError:
            raise
        except OSError as exc:
            raise PilotPersistenceError("could not create pilot evidence staging") from exc
        _write_new(
            paths.evidence_staging_dir / "pilot_manifest.json",
            _canonical_json_bytes(manifest.to_dict(), pretty=True),
        )
        _write_new(paths.evidence_staging_dir / "pilot_attempts.jsonl", b"")

    @property
    def attempts_path(self) -> Path:
        return self.paths.evidence_staging_dir / "pilot_attempts.jsonl"

    def publish(
        self,
        *,
        usage: PilotUsageSummary,
        report: PilotReport,
        secret_value: str | None,
    ) -> Path:
        if report.manifest_hash != self.manifest.manifest_hash:
            raise PilotPersistenceError("report does not reference the staged manifest")
        _write_new(
            self.paths.evidence_staging_dir / "pilot_usage_summary.json",
            _canonical_json_bytes(usage.to_dict(), pretty=True),
        )
        _write_new(
            self.paths.evidence_staging_dir / "pilot_report.json",
            _canonical_json_bytes(report.to_dict(), pretty=True),
        )
        try:
            actual = tuple(
                sorted(item.name for item in self.paths.evidence_staging_dir.iterdir())
            )
        except OSError as exc:
            raise PilotPersistenceError("could not inspect staged pilot evidence") from exc
        if actual != tuple(sorted(PILOT_EVIDENCE_FILENAMES)):
            raise PilotPersistenceError("staged pilot evidence layout is invalid")
        validate_no_secret_artifacts(
            self.paths.evidence_staging_dir,
            secret_value=secret_value,
        )
        if self.paths.evidence_dir.exists() or self.paths.evidence_dir.is_symlink():
            raise PilotPersistenceError("pilot evidence final directory already exists")
        try:
            self.paths.evidence_staging_dir.rename(self.paths.evidence_dir)
        except OSError as exc:
            raise PilotPersistenceError("could not atomically publish pilot evidence") from exc
        return self.paths.evidence_dir


@dataclass(frozen=True)
class PilotExecutionResult:
    paths: PilotPaths
    plan: PilotPlan
    manifest: PilotManifest
    usage: PilotUsageSummary
    report: PilotReport
    run_result: RunExecutionResult
    evaluation_result: EvaluationRunResult


def _successful_report(
    *,
    paths: PilotPaths,
    manifest: PilotManifest,
    usage: PilotUsageSummary,
    validation: PilotRunValidation,
    evaluation: EvaluationRunResult,
) -> PilotReport:
    if (
        evaluation.summary.evaluation_mode != "diagnostic"
        or evaluation.summary.primary_eligible
        or evaluation.summary.included_in_final_research_aggregation
        or len(evaluation.mra_records) != 4
        or len(evaluation.bcr_records) != 4
    ):
        raise PilotRunValidationError("pilot diagnostic evaluation is not exact")
    source_after = fingerprint_directory(paths.source_run_dir)
    if source_after != validation.source_fingerprint:
        raise PilotRunValidationError("diagnostic evaluation mutated source pilot artifacts")
    evaluation_fingerprint = fingerprint_directory(paths.diagnostic_evaluation_dir)
    return PilotReport(
        pilot_id=manifest.pilot_id,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        mode=manifest.plan.mode,
        manifest_hash=manifest.manifest_hash,
        acceptance_state="pending",
        technical_success=True,
        eligible_for_manual_acceptance=manifest.plan.mode == "real",
        source_run_status=validation.source_run_status,
        source_run_directory_name=paths.source_run_dir.name,
        diagnostic_evaluation_directory_name=paths.diagnostic_evaluation_dir.name,
        primary_eligible=False,
        included_in_final_research_matrix=False,
        usage=usage,
        validation=validation,
        source_fingerprint=source_after,
        diagnostic_evaluation_fingerprint=evaluation_fingerprint,
        mra_diagnostic_record_count=len(evaluation.mra_records),
        mra_diagnostic_mean=evaluation.summary.mra_aggregates.run.mean_mra,
        bcr_export_record_count=len(evaluation.bcr_records),
        bcr_judgment_count=0,
        technical_checks=_SUCCESSFUL_TECHNICAL_CHECKS,
        rejection_reasons=(),
        terminal_error_type=None,
    )


def _safe_source_status(path: Path) -> str | None:
    try:
        return read_run_summary(path / "run_summary.json").status
    except Exception:
        return None


def _failed_report(
    *,
    paths: PilotPaths,
    manifest: PilotManifest,
    usage: PilotUsageSummary,
    failure_type: str,
) -> PilotReport:
    source_status = _safe_source_status(paths.source_run_dir)
    source_fingerprint = (
        fingerprint_directory(paths.source_run_dir)
        if paths.source_run_dir.is_dir() and not paths.source_run_dir.is_symlink()
        else None
    )
    return PilotReport(
        pilot_id=manifest.pilot_id,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        mode=manifest.plan.mode,
        manifest_hash=manifest.manifest_hash,
        acceptance_state="pending",
        technical_success=False,
        eligible_for_manual_acceptance=False,
        source_run_status=source_status,
        source_run_directory_name=paths.source_run_dir.name,
        diagnostic_evaluation_directory_name=paths.diagnostic_evaluation_dir.name,
        primary_eligible=False,
        included_in_final_research_matrix=False,
        usage=usage,
        validation=None,
        source_fingerprint=source_fingerprint,
        diagnostic_evaluation_fingerprint=None,
        mra_diagnostic_record_count=0,
        mra_diagnostic_mean=None,
        bcr_export_record_count=0,
        bcr_judgment_count=0,
        technical_checks=(),
        rejection_reasons=(failure_type,),
        terminal_error_type=failure_type,
    )


def execute_pilot(
    *,
    mode: str,
    output_root: str | Path,
    experiment_id: str,
    run_id: str,
    retrieval_config_id: str,
    repetition_index: int,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
    real_authorized: bool = False,
    max_total_tokens: int | None = None,
    provider: StructuredTextProvider | EmbeddingAPIProvider | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> PilotExecutionResult:
    """Execute one dummy or explicitly authorized real pilot and diagnostic export."""
    if mode not in {"dummy", "real"}:
        raise PilotPreflightError("execute_pilot mode must be dummy or real")
    config = load_experiment_config(config_directory)
    plan = build_pilot_plan(
        config,
        mode=mode,
        retrieval_config_id=retrieval_config_id,
        repetition_index=repetition_index,
    )
    paths = build_pilot_paths(
        output_root,
        experiment_id=experiment_id,
        run_id=run_id,
    )
    validate_fresh_pilot_paths(paths)
    validate_phase10_runtime()
    source_identity = (
        _capture_pilot_source_identity(require_clean=False) if mode == "real" else None
    )
    effective_budget: int | None = None
    secret_value: str | None = None
    if mode == "real":
        effective_budget = validate_real_authorization(
            real_authorized=real_authorized,
            max_total_tokens=max_total_tokens,
        )
        require_openai_credential()
        secret_value = os.environ.get(OPENAI_API_KEY_ENVIRONMENT_VARIABLE)
    elif real_authorized or max_total_tokens is not None or provider is not None:
        raise PilotPreflightError("dummy pilot must not receive real-provider controls")
    manifest = build_pilot_manifest(
        plan=plan,
        experiment_id=experiment_id,
        run_id=run_id,
        config_directory=config_directory,
        max_total_tokens=effective_budget,
        source_identity=source_identity,
    )
    evidence = PilotEvidenceWriter(paths, manifest)
    recorder: PilotAttemptRecorder | None = None
    attempts: tuple[ProviderAttemptRecord, ...] = ()
    run_result: RunExecutionResult | None = None
    evaluation_result: EvaluationRunResult | None = None
    try:
        if mode == "dummy":
            llm_client = build_deterministic_dummy_llm_client()
            embedding_provider = DummyEmbeddingProvider()
        else:
            active_provider = provider or OpenAIProviderAdapter.from_environment(
                timeout_seconds=SIMULATOR_PROFILE.request_timeout_seconds
            )
            run_metadata = RunLogMetadata(
                experiment_id,
                run_id,
                retrieval_config_id,
                repetition_index,
            )
            recorder = PilotAttemptRecorder(
                attempts_path=evidence.attempts_path,
                api_usage_path=paths.source_run_dir / "api_usage_logs.jsonl",
                run_metadata=run_metadata,
                config=config,
                max_total_tokens=effective_budget,
                call_limits=PILOT_CALL_LIMITS,
            )
            delay = sleeper if sleeper is not None else __import__("time").sleep
            llm_client = GuardedPilotLLMClient(
                LLMService(
                    active_provider,
                    usage_recorder=recorder,
                    sleeper=delay,
                ),
                recorder,
            )
            embedding_provider = GuardedPilotEmbeddingProvider(
                OpenAIEmbeddingProvider(
                    active_provider,
                    usage_recorder=recorder,
                    sleeper=delay,
                ),
                recorder,
            )
        run_result = RunController(
            llm_client=llm_client,
            embedding_provider=embedding_provider,
            config_directory=config_directory,
            sanitize_failure_messages=mode == "real",
        ).run(
            output_root=paths.output_root,
            experiment_id=experiment_id,
            run_id=run_id,
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
            days=plan.days,
        )
        snapshot_hashes = hash_config_directory(run_result.run_paths.config_snapshot)
        if snapshot_hashes != manifest.configuration_hashes:
            raise PilotRunValidationError("source config snapshot differs from manifest")
        validation = validate_pilot_run(
            run_result.run_paths.run_dir,
            mode=mode,
            secret_value=secret_value,
        )
        evaluation_result = evaluate_run(
            run_dir=run_result.run_paths.run_dir,
            output_dir=paths.diagnostic_evaluation_dir,
            diagnostic=True,
        )
        validate_no_secret_artifacts(
            paths.diagnostic_evaluation_dir,
            secret_value=secret_value,
        )
        attempts = recorder.records if recorder is not None else ()
        usage = PilotUsageSummary.from_attempts(
            attempts,
            max_total_tokens=effective_budget,
        )
        if mode == "real":
            if usage.missing_total_usage_attempt_count > missing_usage_cap(usage.logical_call_count):
                raise PilotUsageError(
                    "real pilot provider usage is incomplete beyond the tolerated "
                    "missing-usage cap"
                )
            if (
                usage.logical_call_count != PILOT_CALL_LIMITS.total_logical_calls
                or usage.budget_exceeded
            ):
                raise PilotRunValidationError(
                    "real provider usage does not match pilot plan"
                )
        report = _successful_report(
            paths=paths,
            manifest=manifest,
            usage=usage,
            validation=validation,
            evaluation=evaluation_result,
        )
        evidence.publish(usage=usage, report=report, secret_value=secret_value)
        return PilotExecutionResult(
            paths=paths,
            plan=plan,
            manifest=manifest,
            usage=usage,
            report=report,
            run_result=run_result,
            evaluation_result=evaluation_result,
        )
    except BaseException as exc:
        attempts = recorder.records if recorder is not None else attempts
        usage = PilotUsageSummary.from_attempts(
            attempts,
            max_total_tokens=effective_budget,
        )
        failure_type = (
            type(exc.original_error).__name__
            if isinstance(exc, RunExecutionError)
            else type(exc).__name__
        )
        published = False
        try:
            report = _failed_report(
                paths=paths,
                manifest=manifest,
                usage=usage,
                failure_type=failure_type,
            )
            evidence.publish(usage=usage, report=report, secret_value=secret_value)
            published = True
        except Exception as evidence_error:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"pilot evidence publication also failed: {type(evidence_error).__name__}"
                )
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise PilotExecutionError(
            failure_type,
            paths=paths,
            evidence_published=published,
        ) from None


def dry_run_pilot(
    *,
    retrieval_config_id: str = "balanced",
    repetition_index: int = 1,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> PilotPlan:
    """Return the exact plan without paths, credentials, clients, or provider calls."""
    return build_pilot_plan(
        load_experiment_config(config_directory),
        mode="dry-run",
        retrieval_config_id=retrieval_config_id,
        repetition_index=repetition_index,
    )


@dataclass(frozen=True)
class PilotReviewResult:
    acceptance_state: str
    review_dir: Path
    acceptance_hash: str
    freeze_hash: str | None


@dataclass(frozen=True)
class PilotFreezeValidation:
    review_dir: Path
    pilot_id: str
    manifest_hash: str
    acceptance_hash: str
    freeze_hash: str
    prompt_version: str


def _load_attempts(path: Path) -> tuple[ProviderAttemptRecord, ...]:
    if path.is_symlink() or not path.is_file():
        raise PilotReviewError("missing pilot_attempts.jsonl")
    records: list[ProviderAttemptRecord] = []
    seen: set[tuple[str, str, int]] = set()
    seen_logical: set[tuple[str, str]] = set()
    previous_logical: tuple[str, str] | None = None
    previous_attempt_index = 0
    try:
        payload = path.read_bytes()
        if payload and (not payload.endswith(b"\n") or b"\r" in payload):
            raise ValueError("attempt stream must be LF-terminated")
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError, ValueError) as exc:
        raise PilotReviewError("could not read pilot_attempts.jsonl") from exc
    for line in lines:
        try:
            value = json.loads(line, object_pairs_hook=_strict_json_object)
            usage = ProviderUsage(
                value["input_tokens"],
                value["output_tokens"],
                value["total_tokens"],
            )
            record = ProviderAttemptRecord(
                service_role=value["service_role"],
                logical_call_id=value["logical_call_id"],
                attempt_index=value["attempt_index"],
                call_type=value["call_type"],
                model=value["model"],
                prompt_template=value["prompt_template"],
                prompt_version=value["prompt_version"],
                prompt_hash=value["prompt_hash"],
                usage=usage,
                status=value["status"],
                error_type=value["error_type"],
                schema_version=value["schema_version"],
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            _DuplicatePilotJsonKeyError,
        ) as exc:
            raise PilotReviewError("invalid pilot attempt record") from exc
        if record.to_dict() != value:
            raise PilotReviewError("pilot attempt record contains unexpected fields")
        key = (record.service_role, record.logical_call_id, record.attempt_index)
        if key in seen:
            raise PilotReviewError("duplicate persisted provider attempt")
        logical = (record.service_role, record.logical_call_id)
        if record.attempt_index == 1:
            if logical in seen_logical:
                raise PilotReviewError("duplicate persisted logical provider call")
            seen_logical.add(logical)
        elif previous_logical != logical or previous_attempt_index != 1:
            raise PilotReviewError("provider retries must immediately follow attempt one")
        expected_model = {
            "simulator": SIMULATOR_MODEL_IDENTITY,
            "embedding": EMBEDDING_MODEL_IDENTITY,
            "evaluator": EVALUATOR_MODEL_IDENTITY,
        }[record.service_role]
        if record.model != expected_model or record.service_role == "evaluator":
            raise PilotReviewError("pilot attempt uses an unapproved service or model")
        if record.service_role == "simulator":
            expected_template = {
                "plan": "plan",
                "natural_response": "natural_interaction",
                "target_event_response": "target_event_response",
                "reflection": "reflect",
                "importance_score": "importance_scoring",
            }.get(record.call_type)
            if (
                expected_template is None
                or record.prompt_template != expected_template
                or record.prompt_version != PROMPT_VERSION
                or record.prompt_hash
                != APPROVED_PHASE10_PROMPT_HASHES[expected_template]
            ):
                raise PilotReviewError("pilot simulator attempt prompt identity is invalid")
        seen.add(key)
        records.append(record)
        previous_logical = logical
        previous_attempt_index = record.attempt_index
    for role in ("simulator", "embedding", "evaluator"):
        role_logical = sum(item[0] == role for item in seen_logical)
        role_physical = sum(item.service_role == role for item in records)
        if (
            role_logical > PILOT_CALL_LIMITS.logical_limit(role)
            or role_physical > PILOT_CALL_LIMITS.physical_limit(role)
        ):
            raise PilotReviewError("persisted provider calls exceed pilot limits")
    return tuple(records)


def _reconstruct_usage_for_review(
    *,
    attempts: tuple[ProviderAttemptRecord, ...],
    max_total_tokens: int | None,
) -> dict[str, object]:
    try:
        return PilotUsageSummary.from_attempts(
            attempts,
            max_total_tokens=max_total_tokens,
        ).to_dict()
    except PilotPersistenceError:
        raise PilotReviewError("pilot usage evidence is invalid") from None


def _require_complete_provider_usage(
    *,
    usage: Mapping[str, object],
    attempts: tuple[ProviderAttemptRecord, ...],
) -> None:
    if (
        usage.get("missing_total_usage_attempt_count", 0)
        > missing_usage_cap(usage.get("logical_call_count", 0))
        or sum(
            attempt.usage.total_tokens is None for attempt in attempts
        ) > missing_usage_cap(len(attempts))
    ):
        raise PilotReviewError(
            "pilot has provider attempts with incomplete token usage beyond the "
            "tolerated cap"
        )


def _validate_manifest_mapping(value: dict[str, Any]) -> None:
    stored_hash = value.get("manifest_hash")
    base = dict(value)
    base.pop("manifest_hash", None)
    if _content_hash(base) != stored_hash:
        raise PilotReviewError("pilot manifest hash mismatch")
    if set(value) != {
        "schema_version",
        "pilot_id",
        "experiment_id",
        "run_id",
        "pilot_only",
        "primary_eligible",
        "included_in_final_research_matrix",
        "expected_source_run_status",
        "plan",
        "max_total_tokens",
        "configuration_hashes",
        "prompts",
        "source_identity",
        "simulator_settings",
        "evaluator_settings",
        "embedding_settings",
        "retry_policy",
        "evaluation_schema_version",
        "research_log_schema_version",
        "bcr_rubric_version",
        "manifest_hash",
    }:
        raise PilotReviewError("pilot manifest fields do not match the exact schema")
    if (
        value.get("schema_version") != PILOT_SCHEMA_VERSION
        or value.get("pilot_id") != value.get("run_id")
        or value.get("pilot_only") is not True
        or value.get("primary_eligible") is not False
        or value.get("included_in_final_research_matrix") is not False
        or value.get("expected_source_run_status") != "excluded"
    ):
        raise PilotReviewError("pilot manifest eligibility contract mismatch")
    if mode := value.get("plan", {}).get("mode") if isinstance(value.get("plan"), dict) else None:
        if mode == "real":
            _validate_pilot_source_identity(value.get("source_identity"), require_clean=False)
        elif value.get("source_identity") is not None:
            raise PilotReviewError("dummy pilot manifest must not claim source identity")
    prompts = value.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 6:
        raise PilotReviewError("pilot manifest prompt inventory is invalid")
    if any(
        not isinstance(item, dict)
        or set(item)
        != {
            "template_id",
            "filename",
            "prompt_version",
            "prompt_hash",
            "model_role",
            "output_contract",
        }
        for item in prompts
    ):
        raise PilotReviewError("pilot manifest prompt row schema is invalid")
    actual = {item.get("template_id"): item.get("prompt_hash") for item in prompts}
    if (
        actual != dict(APPROVED_PHASE10_PROMPT_HASHES)
        or tuple(item.get("template_id") for item in prompts) != tuple(PROMPT_SPECS)
    ):
        raise PilotReviewError("pilot manifest prompt identities have drifted")
    if {item.get("prompt_version") for item in prompts} != {PROMPT_VERSION}:
        raise PilotReviewError("pilot manifest prompt version has drifted")
    for item in prompts:
        spec = PROMPT_SPECS.get(item["template_id"])
        if (
            spec is None
            or item["filename"] != spec.filename
            or item["model_role"] != spec.model_role
            or item["output_contract"] != spec.output_contract
        ):
            raise PilotReviewError("pilot manifest prompt spec metadata is invalid")
    plan = value.get("plan")
    if not isinstance(plan, dict) or set(plan) != {
        "mode",
        "retrieval_config_id",
        "repetition_index",
        "days",
        "step_ids",
        "natural_block_count",
        "target_event_count",
        "main_turn_count",
        "post_step_reflection_count",
        "target_evaluation_unit_count",
        "call_limits",
    }:
        raise PilotReviewError("pilot manifest plan is invalid")
    mode = plan.get("mode")
    if mode not in {"dummy", "real"}:
        raise PilotReviewError("pilot manifest execution mode is invalid")
    if (
        plan.get("days") != [1, 2]
        or plan.get("step_ids")
        != [
            "day_01_natural_01",
            "day_01_natural_02",
            "day_02_natural_03",
            "day_02_target_01",
        ]
        or plan.get("natural_block_count") != 3
        or plan.get("target_event_count") != 1
        or plan.get("main_turn_count") != 40
        or plan.get("post_step_reflection_count") != 16
        or plan.get("target_evaluation_unit_count") != 4
        or plan.get("call_limits") != PILOT_CALL_LIMITS.to_dict()
    ):
        raise PilotReviewError("pilot manifest scope or call limits have drifted")
    max_total_tokens = value.get("max_total_tokens")
    if mode == "real":
        if (
            plan.get("retrieval_config_id") != "balanced"
            or plan.get("repetition_index") != 1
            or isinstance(max_total_tokens, bool)
            or not isinstance(max_total_tokens, int)
            or max_total_tokens < 1
        ):
            raise PilotReviewError("real pilot manifest identity or budget is invalid")
    elif max_total_tokens is not None:
        raise PilotReviewError("dummy pilot manifest must not contain a token ceiling")
    config_hashes = value.get("configuration_hashes")
    if not isinstance(config_hashes, list) or len(config_hashes) != 6:
        raise PilotReviewError("pilot manifest configuration inventory is invalid")
    config_names: set[str] = set()
    for item in config_hashes:
        if not isinstance(item, dict) or set(item) != {"filename", "sha256"}:
            raise PilotReviewError("pilot manifest configuration hash is invalid")
        filename = item.get("filename")
        digest = item.get("sha256")
        if not isinstance(filename, str) or filename in config_names:
            raise PilotReviewError("pilot manifest configuration names are invalid")
        config_names.add(filename)
        _require_hash(digest, "configuration hash")
    if config_names != set(CONFIG_FILES.values()):
        raise PilotReviewError("pilot manifest configuration inventory has drifted")
    if tuple(item["filename"] for item in config_hashes) != tuple(
        sorted(CONFIG_FILES.values())
    ):
        raise PilotReviewError("pilot manifest configuration order has drifted")
    if (
        value.get("simulator_settings") != SIMULATOR_PROFILE.reproducibility_metadata()
        or value.get("evaluator_settings")
        != EVALUATOR_PROFILE.reproducibility_metadata()
        or value.get("embedding_settings")
        != EMBEDDING_PROFILE.reproducibility_metadata()
        or value.get("retry_policy")
        != {
            "maximum_total_attempts": MAX_PROVIDER_ATTEMPTS,
            "deterministic_delay_seconds": RETRY_DELAY_SECONDS,
            "jitter": False,
        }
        or value.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION
        or value.get("research_log_schema_version") != SCHEMA_VERSION
        or value.get("bcr_rubric_version") != BCR_RUBRIC_VERSION
    ):
        raise PilotReviewError("pilot manifest runtime or schema settings have drifted")


def _prepare_review_paths(evidence: Path, run_id: str) -> tuple[Path, Path]:
    if evidence.name != f"{run_id}_pilot_evidence":
        raise PilotReviewError("evidence directory name does not match run identity")
    final = evidence.parent / f"{run_id}_pilot_review"
    staging = evidence.parent / f".{run_id}_pilot_review.staging"
    if os.path.lexists(final) or os.path.lexists(staging):
        raise PilotReviewError("pilot review or staging path already exists")
    return final, staging


def review_pilot(
    *,
    evidence_dir: str | Path,
    decision: str,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
    prompt_loader: PromptLoader | None = None,
    ground_truth_finalization_dir: str | Path | None = None,
) -> PilotReviewResult:
    """Publish one manual accepted/rejected decision without provider calls."""
    if decision not in {"accepted", "rejected"}:
        raise PilotReviewError("review decision must be accepted or rejected")
    requested_evidence = Path(evidence_dir)
    if requested_evidence.is_symlink():
        raise PilotReviewError("evidence_dir must be a real published directory")
    try:
        evidence = requested_evidence.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise PilotReviewError("evidence_dir must be a real published directory") from None
    if not evidence.is_dir():
        raise PilotReviewError("evidence_dir must be a real published directory")
    try:
        actual_files = tuple(sorted(item.name for item in evidence.iterdir()))
    except OSError as exc:
        raise PilotReviewError("could not inspect pilot evidence") from exc
    if actual_files != tuple(sorted(PILOT_EVIDENCE_FILENAMES)):
        raise PilotReviewError("pilot evidence layout is invalid")
    manifest = _read_json_object(
        evidence / "pilot_manifest.json",
        error_type=PilotReviewError,
    )
    usage = _read_json_object(
        evidence / "pilot_usage_summary.json",
        error_type=PilotReviewError,
    )
    report = _read_json_object(
        evidence / "pilot_report.json",
        error_type=PilotReviewError,
    )
    attempts = _load_attempts(evidence / "pilot_attempts.jsonl")
    _validate_manifest_mapping(manifest)
    reconstructed_usage = _reconstruct_usage_for_review(
        attempts=attempts,
        max_total_tokens=manifest.get("max_total_tokens"),
    )
    if (
        report.get("manifest_hash") != manifest.get("manifest_hash")
        or report.get("pilot_id") != manifest.get("pilot_id")
        or report.get("run_id") != manifest.get("run_id")
        or usage.get("physical_attempt_count") != len(attempts)
        or usage != reconstructed_usage
        or report.get("usage") != usage
    ):
        raise PilotReviewError("pilot evidence cross-file identity mismatch")
    run_id = report.get("run_id")
    if not isinstance(run_id, str):
        raise PilotReviewError("pilot report run_id is invalid")
    plan = manifest["plan"]
    if set(report) != {
        "schema_version",
        "pilot_id",
        "experiment_id",
        "run_id",
        "mode",
        "manifest_hash",
        "acceptance_state",
        "technical_success",
        "eligible_for_manual_acceptance",
        "source_run_status",
        "source_run_directory_name",
        "diagnostic_evaluation_directory_name",
        "primary_eligible",
        "included_in_final_research_matrix",
        "usage",
        "validation",
        "source_fingerprint",
        "diagnostic_evaluation_fingerprint",
        "mra_diagnostic_record_count",
        "mra_diagnostic_mean",
        "bcr_export_record_count",
        "bcr_judgment_count",
        "technical_checks",
        "rejection_reasons",
        "terminal_error_type",
    }:
        raise PilotReviewError("pilot report fields do not match the exact schema")
    if (
        report.get("schema_version") != PILOT_SCHEMA_VERSION
        or report.get("experiment_id") != manifest.get("experiment_id")
        or report.get("mode") != plan.get("mode")
        or report.get("source_run_directory_name") != run_id
        or report.get("diagnostic_evaluation_directory_name")
        != f"{run_id}_diagnostic_evaluation"
        or report.get("primary_eligible") is not False
        or report.get("included_in_final_research_matrix") is not False
        or report.get("bcr_judgment_count") != 0
    ):
        raise PilotReviewError("pilot report identity or eligibility is invalid")
    if report.get("acceptance_state") != "pending":
        raise PilotReviewError("terminally accepted or rejected pilot evidence cannot be reused")
    if decision == "accepted":
        _require_complete_provider_usage(usage=usage, attempts=attempts)
    technical_success = report.get("technical_success") is True
    if technical_success:
        if (
            report.get("source_run_status") != "excluded"
            or report.get("mra_diagnostic_record_count") != 4
            or report.get("bcr_export_record_count") != 4
            or report.get("terminal_error_type") is not None
            or report.get("technical_checks") != list(_SUCCESSFUL_TECHNICAL_CHECKS)
            or report.get("rejection_reasons") != []
            or report.get("eligible_for_manual_acceptance")
            != (plan.get("mode") == "real")
        ):
            raise PilotReviewError("successful pending pilot evidence is inconsistent")
        final_attempts: dict[tuple[str, str], ProviderAttemptRecord] = {}
        for attempt in attempts:
            final_attempts[(attempt.service_role, attempt.logical_call_id)] = attempt
        if any(attempt.status != "success" for attempt in final_attempts.values()):
            raise PilotReviewError("successful pilot contains a terminal provider failure")
        if plan.get("mode") == "real":
            roles = {
                item.get("service_role"): item
                for item in usage.get("role_usage", [])
                if isinstance(item, dict)
            }
            if (
                usage.get("logical_call_count")
                != PILOT_CALL_LIMITS.total_logical_calls
                or roles.get("simulator", {}).get("logical_call_count")
                != PILOT_CALL_LIMITS.simulator_logical_calls
                or roles.get("embedding", {}).get("logical_call_count")
                != PILOT_CALL_LIMITS.embedding_logical_calls
                or roles.get("evaluator", {}).get("logical_call_count") != 0
            ):
                raise PilotReviewError("real pilot provider call inventory is incomplete")
    else:
        rejection_reasons = report.get("rejection_reasons")
        terminal_error_type = report.get("terminal_error_type")
        if (
            report.get("eligible_for_manual_acceptance") is not False
            or not isinstance(rejection_reasons, list)
            or not rejection_reasons
            or any(not isinstance(item, str) or not item for item in rejection_reasons)
            or not isinstance(terminal_error_type, str)
            or not terminal_error_type
            or report.get("technical_checks") != []
            or report.get("validation") is not None
            or report.get("diagnostic_evaluation_fingerprint") is not None
            or report.get("mra_diagnostic_record_count") != 0
            or report.get("mra_diagnostic_mean") is not None
            or report.get("bcr_export_record_count") != 0
        ):
            raise PilotReviewError("failed pending pilot evidence is inconsistent")
        if decision == "accepted":
            raise PilotReviewError("technically failed pilot cannot be accepted")
    final, staging = _prepare_review_paths(evidence, run_id)
    source = evidence.parent / report.get("source_run_directory_name", "")
    evaluation = evidence.parent / report.get(
        "diagnostic_evaluation_directory_name",
        "",
    )
    if technical_success:
        try:
            current_validation = validate_pilot_run(source, mode=plan["mode"])
            loaded_evaluation = load_bcr_export(evaluation)
        except Exception:
            raise PilotReviewError(
                "source or diagnostic evaluation validation failed"
            ) from None
        if (
            current_validation.to_dict() != report.get("validation")
            or loaded_evaluation.evaluation_mode != "diagnostic"
            or loaded_evaluation.primary_eligible
            or loaded_evaluation.included_in_final_research_aggregation
            or len(loaded_evaluation.records) != 4
            or loaded_evaluation.experiment_id != report.get("experiment_id")
            or loaded_evaluation.run_id != run_id
            or loaded_evaluation.retrieval_config_id != plan.get("retrieval_config_id")
            or loaded_evaluation.repetition_index != plan.get("repetition_index")
            or fingerprint_directory(source) != report.get("source_fingerprint")
            or fingerprint_directory(evaluation)
            != report.get("diagnostic_evaluation_fingerprint")
        ):
            raise PilotReviewError("source or evaluation evidence changed after pilot")
    elif report.get("source_fingerprint") is not None:
        try:
            current_source_fingerprint = fingerprint_directory(source)
        except Exception:
            raise PilotReviewError("failed source evidence validation failed") from None
        if current_source_fingerprint != report.get("source_fingerprint"):
            raise PilotReviewError("failed source evidence changed after pilot")
    evidence_hash = fingerprint_directory(evidence)
    if decision == "accepted":
        if (
            report.get("acceptance_state") != "pending"
            or report.get("technical_success") is not True
            or report.get("eligible_for_manual_acceptance") is not True
            or report.get("mode") != "real"
            or report.get("source_run_status") != "excluded"
            or report.get("primary_eligible") is not False
            or report.get("included_in_final_research_matrix") is not False
            or report.get("bcr_export_record_count") != 4
            or report.get("bcr_judgment_count") != 0
            or usage.get("budget_exceeded") is not False
        ):
            raise PilotReviewError("pilot is not eligible for manual acceptance")
        _require_complete_provider_usage(usage=usage, attempts=attempts)
        _validate_pilot_source_identity(manifest.get("source_identity"), require_clean=True)
        if manifest.get("source_identity") != _capture_pilot_source_identity(require_clean=True):
            raise PilotReviewError("current Git/source identity differs from pilot")
        if ground_truth_finalization_dir is None:
            raise PilotReviewError(
                "accepted review requires explicit Ground Truth finalization evidence"
            )
        try:
            current_config = load_experiment_config(config_directory)
            ground_truth_identity = authenticated_ground_truth_identity(
                current_config.ground_truth,
                evidence_directory=str(ground_truth_finalization_dir),
            )
        except Exception:
            raise PilotReviewError(
                "Ground Truth finalization evidence is invalid"
            ) from None
        validate_phase10_runtime(prompt_loader or PromptLoader())
        current_configs = [item.to_dict() for item in hash_config_directory(config_directory)]
        if current_configs != manifest.get("configuration_hashes"):
            raise PilotReviewError("current configuration differs from pilot manifest")
        snapshot_configs = [
            item.to_dict() for item in hash_config_directory(source / "config_snapshot")
        ]
        if snapshot_configs != manifest.get("configuration_hashes"):
            raise PilotReviewError("source snapshot differs from pilot manifest")
    acceptance_base: dict[str, object] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot_id": report["pilot_id"],
        "experiment_id": report["experiment_id"],
        "run_id": run_id,
        "acceptance_state": decision,
        "manual_review": True,
        "manifest_hash": manifest["manifest_hash"],
        "evidence_hash": evidence_hash,
        "source_run_status": report["source_run_status"],
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
    }
    acceptance_hash = _content_hash(acceptance_base)
    acceptance = dict(acceptance_base)
    acceptance["acceptance_hash"] = acceptance_hash
    freeze: dict[str, object] | None = None
    freeze_hash: str | None = None
    if decision == "accepted":
        # Recheck at the freeze creation boundary instead of relying on the
        # preceding acceptance eligibility decision.
        _require_complete_provider_usage(usage=usage, attempts=attempts)
        freeze_base: dict[str, object] = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "freeze_state": "accepted",
            "pilot_id": report["pilot_id"],
            "experiment_id": report["experiment_id"],
            "run_id": run_id,
            "acceptance_hash": acceptance_hash,
            "evidence_hash": evidence_hash,
            "manifest_hash": manifest["manifest_hash"],
            "prompt_version": PROMPT_VERSION,
            "configuration_hashes": manifest["configuration_hashes"],
            "prompts": manifest["prompts"],
            "simulator_settings": manifest["simulator_settings"],
            "evaluator_settings": manifest["evaluator_settings"],
            "embedding_settings": manifest["embedding_settings"],
            "retry_policy": manifest["retry_policy"],
            "evaluation_schema_version": manifest["evaluation_schema_version"],
            "research_log_schema_version": manifest["research_log_schema_version"],
            "bcr_rubric_version": manifest["bcr_rubric_version"],
            "primary_eligible": False,
            "included_in_final_research_matrix": False,
            "phase13_source_and_git_verification_required": True,
            "source_identity": manifest["source_identity"],
            "ground_truth_finalization": ground_truth_identity.to_dict(),
        }
        freeze_hash = _content_hash(freeze_base)
        freeze = dict(freeze_base)
        freeze["freeze_hash"] = freeze_hash
    try:
        staging.mkdir(mode=0o700)
        _write_new(
            staging / PILOT_ACCEPTANCE_FILENAME,
            _canonical_json_bytes(acceptance, pretty=True),
        )
        if freeze is not None:
            _write_new(
                staging / PILOT_FREEZE_FILENAME,
                _canonical_json_bytes(freeze, pretty=True),
            )
        validate_no_secret_artifacts(staging)
        expected = (
            (PILOT_ACCEPTANCE_FILENAME, PILOT_FREEZE_FILENAME)
            if decision == "accepted"
            else (PILOT_ACCEPTANCE_FILENAME,)
        )
        if tuple(sorted(item.name for item in staging.iterdir())) != tuple(sorted(expected)):
            raise PilotReviewError("staged pilot review layout is invalid")
        if os.path.lexists(final):
            raise PilotReviewError("pilot review final path already exists")
        staging.rename(final)
    except BaseException as exc:
        if staging.exists() and not final.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, PilotError):
            raise
        raise PilotReviewError("could not publish pilot review") from None
    return PilotReviewResult(decision, final, acceptance_hash, freeze_hash)


def validate_accepted_freeze(
    review_dir: str | Path,
    *,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
    prompt_loader: PromptLoader | None = None,
    ground_truth_finalization_dir: str | Path | None = None,
) -> PilotFreezeValidation:
    """Verify an accepted freeze still matches exact current prompt/settings bytes."""
    requested_review = Path(review_dir)
    if requested_review.is_symlink():
        raise PilotReviewError("accepted review directory is missing or invalid")
    try:
        review = requested_review.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise PilotReviewError("accepted review directory is missing or invalid") from None
    if not review.is_dir():
        raise PilotReviewError("accepted review directory is missing or invalid")
    try:
        files = tuple(sorted(item.name for item in review.iterdir()))
    except OSError as exc:
        raise PilotReviewError("could not inspect accepted review directory") from exc
    if files != tuple(sorted((PILOT_ACCEPTANCE_FILENAME, PILOT_FREEZE_FILENAME))):
        raise PilotReviewError("accepted review layout is invalid")
    acceptance = _read_json_object(
        review / PILOT_ACCEPTANCE_FILENAME,
        error_type=PilotReviewError,
    )
    freeze = _read_json_object(
        review / PILOT_FREEZE_FILENAME,
        error_type=PilotReviewError,
    )
    acceptance_hash = acceptance.get("acceptance_hash")
    acceptance_base = dict(acceptance)
    acceptance_base.pop("acceptance_hash", None)
    freeze_hash = freeze.get("freeze_hash")
    freeze_base = dict(freeze)
    freeze_base.pop("freeze_hash", None)
    if (
        set(acceptance)
        != {
            "schema_version",
            "pilot_id",
            "experiment_id",
            "run_id",
            "acceptance_state",
            "manual_review",
            "manifest_hash",
            "evidence_hash",
            "source_run_status",
            "primary_eligible",
            "included_in_final_research_matrix",
            "acceptance_hash",
        }
        or set(freeze)
        != {
            "schema_version",
            "freeze_state",
            "pilot_id",
            "experiment_id",
            "run_id",
            "acceptance_hash",
            "evidence_hash",
            "manifest_hash",
            "prompt_version",
            "configuration_hashes",
            "prompts",
            "simulator_settings",
            "evaluator_settings",
            "embedding_settings",
            "retry_policy",
            "evaluation_schema_version",
            "research_log_schema_version",
            "bcr_rubric_version",
            "primary_eligible",
            "included_in_final_research_matrix",
            "phase13_source_and_git_verification_required",
            "source_identity",
            "ground_truth_finalization",
            "freeze_hash",
        }
        or _content_hash(acceptance_base) != acceptance_hash
        or _content_hash(freeze_base) != freeze_hash
    ):
        raise PilotReviewError("accepted review content hash is invalid")
    if (
        acceptance.get("schema_version") != PILOT_SCHEMA_VERSION
        or acceptance.get("acceptance_state") != "accepted"
        or acceptance.get("manual_review") is not True
        or acceptance.get("source_run_status") != "excluded"
        or acceptance.get("primary_eligible") is not False
        or acceptance.get("included_in_final_research_matrix") is not False
        or freeze.get("schema_version") != PILOT_SCHEMA_VERSION
        or freeze.get("freeze_state") != "accepted"
        or freeze.get("pilot_id") != acceptance.get("pilot_id")
        or freeze.get("experiment_id") != acceptance.get("experiment_id")
        or freeze.get("run_id") != acceptance.get("run_id")
        or freeze.get("acceptance_hash") != acceptance_hash
        or freeze.get("evidence_hash") != acceptance.get("evidence_hash")
        or freeze.get("manifest_hash") != acceptance.get("manifest_hash")
        or freeze.get("prompt_version") != PROMPT_VERSION
        or freeze.get("primary_eligible") is not False
        or freeze.get("included_in_final_research_matrix") is not False
        or freeze.get("phase13_source_and_git_verification_required") is not True
    ):
        raise PilotReviewError("accepted freeze identity or eligibility is invalid")
    if ground_truth_finalization_dir is None:
        raise PilotReviewError("schema 2.0 freeze requires Ground Truth evidence")
    _validate_pilot_source_identity(freeze.get("source_identity"), require_clean=True)
    try:
        current_config = load_experiment_config(config_directory)
        current_gt_identity = authenticated_ground_truth_identity(
            current_config.ground_truth,
            evidence_directory=str(ground_truth_finalization_dir),
        )
    except Exception:
        raise PilotReviewError("Ground Truth finalization evidence is invalid") from None
    if freeze.get("ground_truth_finalization") != current_gt_identity.to_dict():
        raise PilotReviewError("Ground Truth finalization identity drifted")
    run_id = acceptance.get("run_id")
    if not isinstance(run_id, str) or review.name != f"{run_id}_pilot_review":
        raise PilotReviewError("accepted freeze path identity is invalid")
    evidence = review.parent / f"{run_id}_pilot_evidence"
    if fingerprint_directory(evidence) != acceptance.get("evidence_hash"):
        raise PilotReviewError("accepted pilot evidence has changed after review")
    manifest = _read_json_object(
        evidence / "pilot_manifest.json",
        error_type=PilotReviewError,
    )
    _validate_manifest_mapping(manifest)
    if manifest.get("manifest_hash") != acceptance.get("manifest_hash"):
        raise PilotReviewError("accepted freeze references a different pilot manifest")
    if freeze.get("source_identity") != manifest.get("source_identity"):
        raise PilotReviewError("pilot source identity differs from freeze")
    if freeze.get("source_identity") != _capture_pilot_source_identity(require_clean=True):
        raise PilotReviewError("current Git/source identity differs from freeze")
    usage = _read_json_object(
        evidence / "pilot_usage_summary.json",
        error_type=PilotReviewError,
    )
    report = _read_json_object(
        evidence / "pilot_report.json",
        error_type=PilotReviewError,
    )
    attempts = _load_attempts(evidence / "pilot_attempts.jsonl")
    reconstructed_usage = _reconstruct_usage_for_review(
        attempts=attempts,
        max_total_tokens=manifest.get("max_total_tokens"),
    )
    if (
        usage != reconstructed_usage
        or report.get("usage") != usage
        or report.get("manifest_hash") != manifest.get("manifest_hash")
        or report.get("pilot_id") != acceptance.get("pilot_id")
        or report.get("experiment_id") != acceptance.get("experiment_id")
        or report.get("run_id") != run_id
    ):
        raise PilotReviewError("accepted freeze usage evidence is inconsistent")
    _require_complete_provider_usage(usage=usage, attempts=attempts)
    if (
        report.get("acceptance_state") != "pending"
        or report.get("technical_success") is not True
        or report.get("eligible_for_manual_acceptance") is not True
        or report.get("mode") != "real"
        or report.get("source_run_status") != "excluded"
        or report.get("primary_eligible") is not False
        or report.get("included_in_final_research_matrix") is not False
        or report.get("mra_diagnostic_record_count") != 4
        or report.get("bcr_export_record_count") != 4
        or report.get("bcr_judgment_count") != 0
        or report.get("terminal_error_type") is not None
        or report.get("technical_checks") != list(_SUCCESSFUL_TECHNICAL_CHECKS)
        or report.get("rejection_reasons") != []
        or usage.get("budget_exceeded") is not False
    ):
        raise PilotReviewError("accepted freeze references ineligible pilot evidence")
    loader = prompt_loader or PromptLoader()
    try:
        validate_phase10_runtime(loader)
    except Exception:
        raise PilotReviewError("current prompt or runtime settings invalidate the freeze") from None
    current_configs = [item.to_dict() for item in hash_config_directory(config_directory)]
    current_prompts = [item.to_dict() for item in prompt_identities(loader)]
    if (
        freeze.get("configuration_hashes") != current_configs
        or freeze.get("prompts") != current_prompts
        or freeze.get("simulator_settings")
        != SIMULATOR_PROFILE.reproducibility_metadata()
        or freeze.get("evaluator_settings")
        != EVALUATOR_PROFILE.reproducibility_metadata()
        or freeze.get("embedding_settings")
        != EMBEDDING_PROFILE.reproducibility_metadata()
        or freeze.get("retry_policy")
        != {
            "maximum_total_attempts": MAX_PROVIDER_ATTEMPTS,
            "deterministic_delay_seconds": RETRY_DELAY_SECONDS,
            "jitter": False,
        }
        or freeze.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION
        or freeze.get("research_log_schema_version") != SCHEMA_VERSION
        or freeze.get("bcr_rubric_version") != BCR_RUBRIC_VERSION
    ):
        raise PilotReviewError("accepted freeze no longer matches approved configuration")
    return PilotFreezeValidation(
        review_dir=review,
        pilot_id=run_id,
        manifest_hash=acceptance["manifest_hash"],
        acceptance_hash=acceptance_hash,
        freeze_hash=freeze_hash,
        prompt_version=PROMPT_VERSION,
    )


__all__ = [
    "ArtifactHash",
    "PILOT_ACCEPTANCE_FILENAME",
    "PILOT_EVIDENCE_FILENAMES",
    "PILOT_FREEZE_FILENAME",
    "PilotExecutionError",
    "PilotExecutionResult",
    "PilotFreezeValidation",
    "PilotManifest",
    "PilotPersistenceError",
    "PilotReport",
    "PilotReviewError",
    "PilotReviewResult",
    "PilotUsageSummary",
    "PromptFreezeIdentity",
    "RoleUsageSummary",
    "build_pilot_manifest",
    "dry_run_pilot",
    "execute_pilot",
    "hash_config_directory",
    "prompt_identities",
    "review_pilot",
    "validate_accepted_freeze",
]
