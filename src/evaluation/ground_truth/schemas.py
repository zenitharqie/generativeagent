"""Immutable contracts for manual annotation and automated finalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType
import unicodedata
from typing import Any, Mapping

from src.llm.settings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_IDENTITY

from .errors import GroundTruthInputError


GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION = "1.0"
GROUND_TRUTH_FINALIZATION_VERSION = "manual_three_annotator_v1"
CANONICAL_REPRESENTATIVE_POLICY = (
    "greatest_distinct_annotator_support_then_casefolded_lexical_order"
)
FINAL_GROUND_TRUTH_STATUS = "final"
SEMANTIC_CONSOLIDATION_MODEL_IDENTITY = "openai:text-embedding-3-small"
SEMANTIC_CONSOLIDATION_THRESHOLD = 0.85
MAJORITY_SUPPORT_REQUIRED = 2
EXPECTED_UNIT_COUNT = 20
EXPECTED_ANNOTATION_COUNT = 60
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_CATEGORY_LENGTH = 160
MAX_REASON_LENGTH = 500
MAX_CATEGORIES_PER_UNIT = 16

ANNOTATOR_MODELS: Mapping[str, str] = {
    "gpt5": "GPT-5",
    "claude_sonnet_4_6": "Claude Sonnet 4.6",
    "gemini_1_5_pro": "Gemini 1.5 Pro",
}

if SEMANTIC_CONSOLIDATION_MODEL_IDENTITY != EMBEDDING_MODEL_IDENTITY:
    raise RuntimeError("Ground Truth consolidation model drifted from the frozen embedding model")

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]{0,79}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
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
        raise GroundTruthInputError("Ground Truth evidence is not strict JSON") from exc


def sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def content_hash(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def require_exact_keys(
    value: Mapping[str, Any], required: set[str], context: str
) -> None:
    if set(value) != required:
        raise GroundTruthInputError(f"{context} has an invalid field set")


def require_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise GroundTruthInputError(f"{context} must be a safe identifier")
    return value


def require_hash(value: object, context: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise GroundTruthInputError(f"{context} must be SHA-256 hex")
    return value


def require_safe_text(value: object, context: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise GroundTruthInputError(f"{context} must be text")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > maximum:
        raise GroundTruthInputError(f"{context} has an invalid length")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise GroundTruthInputError(f"{context} contains control characters")
    return normalized


def require_category(value: object, context: str) -> str:
    category = require_safe_text(value, context, maximum=MAX_CATEGORY_LENGTH)
    if "/" in category or "\\" in category:
        raise GroundTruthInputError(f"{context} contains a path separator")
    return category


def category_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _validate_finalization_evidence_bundle(
    value: Mapping[str, Any],
    evidence_directory: str,
    evidence_hashes: Mapping[str, Any],
    fingerprint: str,
) -> None:
    """Authenticate the final Ground Truth against its immutable evidence bundle."""
    root = Path(evidence_directory)
    if not root.is_dir() or root.is_symlink() or ".." in root.parts:
        raise GroundTruthInputError("Ground Truth evidence directory is unsafe or missing")
    expected_files = set(evidence_hashes) | {
        "ground_truth.json",
        "ground_truth_fingerprint.json",
        "finalization_report.json",
    }
    try:
        from .readers import read_strict_json
    except ImportError as exc:  # pragma: no cover - package import invariant
        raise GroundTruthInputError("Ground Truth evidence reader is unavailable") from exc
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise GroundTruthInputError("Ground Truth evidence must not contain symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            actual_files.add(relative)
    if actual_files != expected_files:
        raise GroundTruthInputError("Ground Truth evidence inventory does not match final identity")
    actual_hashes: dict[str, str] = {}
    for name in sorted(evidence_hashes):
        path = root / name
        payload = path.read_bytes()
        actual_hashes[name] = sha256_hex(payload)
        if actual_hashes[name] != evidence_hashes[name]:
            raise GroundTruthInputError("Ground Truth evidence bytes do not match final identity")
        read_strict_json(path, context="Ground Truth evidence")
    stored_ground_truth = read_strict_json(root / "ground_truth.json", context="final Ground Truth")
    if canonical_json_bytes(stored_ground_truth, pretty=True) != canonical_json_bytes(value, pretty=True):
        raise GroundTruthInputError("installed Ground Truth does not match evidence bundle")
    fingerprint_file = read_strict_json(
        root / "ground_truth_fingerprint.json", context="Ground Truth fingerprint"
    )
    if not isinstance(fingerprint_file, Mapping):
        raise GroundTruthInputError("Ground Truth fingerprint evidence is invalid")
    require_exact_keys(
        fingerprint_file,
        {"schema_version", "methodology_id", "fingerprint_sha256", "binding_sha256", "ground_truth_sha256", "evidence_hashes"},
        "Ground Truth fingerprint",
    )
    if fingerprint_file["fingerprint_sha256"] != fingerprint or fingerprint_file["evidence_hashes"] != dict(evidence_hashes):
        raise GroundTruthInputError("Ground Truth fingerprint evidence does not match final identity")
    if fingerprint_file["ground_truth_sha256"] != sha256_hex(canonical_json_bytes(value, pretty=True)):
        raise GroundTruthInputError("Ground Truth content fingerprint does not match")
    report = read_strict_json(root / "finalization_report.json", context="finalization report")
    if not isinstance(report, Mapping):
        raise GroundTruthInputError("Ground Truth finalization report is invalid")
    require_exact_keys(
        report,
        {
            "schema_version", "status", "annotation_unit_count", "annotator_count",
            "annotation_result_count", "final_category_count", "fingerprint_sha256",
            "reliability_sha256", "no_experiment_outcome_data",
        },
        "Ground Truth finalization report",
    )
    categories_count = sum(len(target["expected_categories"]) for target in value["targets"])
    if (
        report["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION
        or report["status"] != FINAL_GROUND_TRUTH_STATUS
        or report["annotation_unit_count"] != EXPECTED_UNIT_COUNT
        or report["annotator_count"] != len(ANNOTATOR_MODELS)
        or report["annotation_result_count"] != EXPECTED_ANNOTATION_COUNT
        or report["final_category_count"] != categories_count
        or report["fingerprint_sha256"] != fingerprint
        or report["reliability_sha256"] != evidence_hashes["reliability.json"]
        or report["no_experiment_outcome_data"] is not True
    ):
        raise GroundTruthInputError("Ground Truth finalization report does not match identity")
    metadata = value["methodological_finalization"]
    assert isinstance(metadata, Mapping)
    without_fingerprint = dict(value)
    without_metadata_fingerprint = dict(metadata)
    without_metadata_fingerprint.pop("fingerprint_sha256", None)
    without_fingerprint["methodological_finalization"] = without_metadata_fingerprint
    binding = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "ground_truth_without_fingerprint": without_fingerprint,
        "evidence_hashes": dict(evidence_hashes),
        "reliability": read_strict_json(root / "reliability.json", context="reliability evidence"),
    }
    if content_hash(binding) != fingerprint or fingerprint_file["binding_sha256"] != fingerprint:
        raise GroundTruthInputError("Ground Truth finalization fingerprint is invalid")


def validate_final_ground_truth_identity(
    value: object,
    *,
    evidence_directory: str | None = None,
) -> str:
    """Validate the compact final identity consumed by Phase 13/freeze gates."""
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("final Ground Truth must be an object")
    if (
        value.get("evaluation_step_type") != "target_event"
        or value.get("status") != FINAL_GROUND_TRUTH_STATUS
        or value.get("freeze_required_before_final_experiment") is not True
    ):
        raise GroundTruthInputError("Ground Truth is not methodologically final")
    metadata = value.get("methodological_finalization")
    if not isinstance(metadata, Mapping):
        raise GroundTruthInputError("final Ground Truth lacks methodological evidence identity")
    require_exact_keys(
        metadata,
        {
            "schema_version",
            "methodology_id",
            "annotation_package_sha256",
            "embedding_model_identity",
            "cosine_similarity_threshold",
            "majority_rule",
            "canonical_representative_policy",
            "annotation_result_count",
            "evidence_hashes",
            "fingerprint_sha256",
        },
        "methodological finalization",
    )
    if (
        metadata["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION
        or metadata["methodology_id"] != GROUND_TRUTH_FINALIZATION_VERSION
        or metadata["embedding_model_identity"] != SEMANTIC_CONSOLIDATION_MODEL_IDENTITY
        or metadata["cosine_similarity_threshold"] != SEMANTIC_CONSOLIDATION_THRESHOLD
        or metadata["majority_rule"]
        != "positive_annotator_count_greater_than_or_equal_to_2_of_3"
        or metadata["canonical_representative_policy"]
        != CANONICAL_REPRESENTATIVE_POLICY
        or metadata["annotation_result_count"] != EXPECTED_ANNOTATION_COUNT
    ):
        raise GroundTruthInputError("methodological finalization contract drifted")
    require_hash(metadata["annotation_package_sha256"], "annotation package hash")
    fingerprint = require_hash(metadata["fingerprint_sha256"], "Ground Truth fingerprint")
    evidence_hashes = metadata["evidence_hashes"]
    expected_evidence = {
        "annotation_package.json",
        *(f"annotations/{annotator_id}.json" for annotator_id in ANNOTATOR_MODELS),
        "master_candidate_list.json",
        "consolidation_evidence.json",
        "embedding_input.json",
        "embedding_evidence.json",
        "annotation_matrix.json",
        "reliability.json",
    }
    if not isinstance(evidence_hashes, Mapping) or set(evidence_hashes) != expected_evidence:
        raise GroundTruthInputError("Ground Truth evidence inventory is incomplete")
    for name, digest in evidence_hashes.items():
        if not isinstance(name, str):
            raise GroundTruthInputError("Ground Truth evidence filename is invalid")
        require_hash(digest, "Ground Truth evidence hash")
    if evidence_directory is not None:
        _validate_finalization_evidence_bundle(value, evidence_directory, evidence_hashes, fingerprint)
    targets = value.get("targets")
    if not isinstance(targets, list) or len(targets) != EXPECTED_UNIT_COUNT:
        raise GroundTruthInputError("final Ground Truth must contain exactly 20 targets")
    pairs: set[tuple[str, str]] = set()
    for target in targets:
        if not isinstance(target, Mapping):
            raise GroundTruthInputError("final Ground Truth target must be an object")
        require_exact_keys(target, {"event_id", "agent_id", "expected_categories"}, "target")
        event_id = require_identifier(target["event_id"], "event_id")
        agent_id = require_identifier(target["agent_id"], "agent_id")
        pair = (event_id, agent_id)
        if pair in pairs:
            raise GroundTruthInputError("final Ground Truth contains duplicate targets")
        pairs.add(pair)
        categories = target["expected_categories"]
        if not isinstance(categories, list) or not categories:
            raise GroundTruthInputError("final Ground Truth categories must not be empty")
        normalized = tuple(require_category(item, "expected category") for item in categories)
        if len({category_key(item) for item in normalized}) != len(normalized):
            raise GroundTruthInputError("final Ground Truth categories must be unique")
    return fingerprint


@dataclass(frozen=True)
class GroundTruthFinalizationIdentity:
    """Authenticated, portable identity consumed by Phase 11 freeze validation."""

    ground_truth_sha256: str
    finalization_fingerprint: str
    evidence_hashes: Mapping[str, str]
    annotation_package_sha256: str
    annotators: tuple[str, ...]
    embedding_model: str
    embedding_dimension: int
    cosine_threshold: float
    representative_policy: str
    majority_rule: str
    reliability_evidence_sha256: str
    workflow_schema_version: str
    finalization_version: str

    def __post_init__(self) -> None:
        require_hash(self.ground_truth_sha256, "Ground Truth content hash")
        require_hash(self.finalization_fingerprint, "Ground Truth finalization fingerprint")
        require_hash(self.annotation_package_sha256, "annotation package hash")
        require_hash(self.reliability_evidence_sha256, "reliability evidence hash")
        if not isinstance(self.evidence_hashes, Mapping):
            raise GroundTruthInputError("Ground Truth evidence hashes must be a mapping")
        object.__setattr__(self, "evidence_hashes", MappingProxyType(dict(self.evidence_hashes)))
        if self.annotators != tuple(ANNOTATOR_MODELS):
            raise GroundTruthInputError("Ground Truth annotator identity is invalid")
        if self.embedding_model != SEMANTIC_CONSOLIDATION_MODEL_IDENTITY:
            raise GroundTruthInputError("Ground Truth embedding model identity is invalid")
        if self.embedding_dimension != EMBEDDING_DIMENSION:
            raise GroundTruthInputError("Ground Truth embedding dimension is invalid")
        if self.cosine_threshold != SEMANTIC_CONSOLIDATION_THRESHOLD:
            raise GroundTruthInputError("Ground Truth cosine threshold is invalid")
        if self.representative_policy != CANONICAL_REPRESENTATIVE_POLICY:
            raise GroundTruthInputError("Ground Truth representative policy is invalid")
        if self.majority_rule != "positive_annotator_count_greater_than_or_equal_to_2_of_3":
            raise GroundTruthInputError("Ground Truth majority rule is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "ground_truth_sha256": self.ground_truth_sha256,
            "finalization_fingerprint": self.finalization_fingerprint,
            "evidence_hashes": dict(self.evidence_hashes),
            "annotation_package_sha256": self.annotation_package_sha256,
            "annotators": list(self.annotators),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "cosine_threshold": self.cosine_threshold,
            "representative_policy": self.representative_policy,
            "majority_rule": self.majority_rule,
            "reliability_evidence_sha256": self.reliability_evidence_sha256,
            "workflow_schema_version": self.workflow_schema_version,
            "finalization_version": self.finalization_version,
        }


def authenticated_ground_truth_identity(
    value: object, *, evidence_directory: str | Path
) -> GroundTruthFinalizationIdentity:
    """Validate the complete evidence bundle and return its immutable identity."""
    if not isinstance(evidence_directory, (str, Path)) or not str(evidence_directory):
        raise GroundTruthInputError("authenticated Ground Truth identity requires evidence")
    fingerprint = validate_final_ground_truth_identity(
        value, evidence_directory=str(evidence_directory)
    )
    if not isinstance(value, Mapping):  # defensive; validator already checked this
        raise GroundTruthInputError("final Ground Truth must be an object")
    metadata = value["methodological_finalization"]
    assert isinstance(metadata, Mapping)
    evidence_hashes = metadata["evidence_hashes"]
    assert isinstance(evidence_hashes, Mapping)
    return GroundTruthFinalizationIdentity(
        ground_truth_sha256=sha256_hex(canonical_json_bytes(value, pretty=True)),
        finalization_fingerprint=fingerprint,
        evidence_hashes={str(k): str(v) for k, v in evidence_hashes.items()},
        annotation_package_sha256=str(metadata["annotation_package_sha256"]),
        annotators=tuple(ANNOTATOR_MODELS),
        embedding_model=str(metadata["embedding_model_identity"]),
        embedding_dimension=EMBEDDING_DIMENSION,
        cosine_threshold=float(metadata["cosine_similarity_threshold"]),
        representative_policy=str(metadata["canonical_representative_policy"]),
        majority_rule=str(metadata["majority_rule"]),
        reliability_evidence_sha256=str(evidence_hashes["reliability.json"]),
        workflow_schema_version=str(metadata["schema_version"]),
        finalization_version=str(metadata["methodology_id"]),
    )


@dataclass(frozen=True)
class AnnotationCategory:
    category: str
    relevant: int
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", require_category(self.category, "category"))
        if type(self.relevant) is not int or self.relevant not in (0, 1):
            raise GroundTruthInputError("category relevance must be integer 0 or 1")
        if self.reason is not None:
            object.__setattr__(
                self,
                "reason",
                require_safe_text(self.reason, "reason", maximum=MAX_REASON_LENGTH),
            )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "category": self.category,
            "relevant": self.relevant,
        }
        if self.reason is not None:
            value["reason"] = self.reason
        return value


@dataclass(frozen=True)
class AnnotationResult:
    annotation_unit_id: str
    agent_id: str
    target_event_id: str
    input_sha256: str
    categories: tuple[AnnotationCategory, ...]

    def __post_init__(self) -> None:
        for field_name in ("annotation_unit_id", "agent_id", "target_event_id"):
            require_identifier(getattr(self, field_name), field_name)
        require_hash(self.input_sha256, "input_sha256")
        if not isinstance(self.categories, tuple) or not self.categories:
            raise GroundTruthInputError("annotation categories must not be empty")
        if len(self.categories) > MAX_CATEGORIES_PER_UNIT:
            raise GroundTruthInputError("annotation contains too many categories")
        keys = tuple(category_key(item.category) for item in self.categories)
        if len(keys) != len(set(keys)):
            raise GroundTruthInputError("annotation contains duplicate categories")

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_unit_id": self.annotation_unit_id,
            "agent_id": self.agent_id,
            "target_event_id": self.target_event_id,
            "input_sha256": self.input_sha256,
            "categories": [item.to_dict() for item in self.categories],
        }


@dataclass(frozen=True)
class AnnotatorSubmission:
    annotator_id: str
    provider_model_label: str
    annotation_package_sha256: str
    results: tuple[AnnotationResult, ...]

    def __post_init__(self) -> None:
        expected_label = ANNOTATOR_MODELS.get(self.annotator_id)
        if expected_label is None or self.provider_model_label != expected_label:
            raise GroundTruthInputError("annotator identity is not methodology-approved")
        require_hash(self.annotation_package_sha256, "annotation_package_sha256")
        if len(self.results) != EXPECTED_UNIT_COUNT:
            raise GroundTruthInputError("annotator submission must contain exactly 20 units")
        unit_ids = tuple(item.annotation_unit_id for item in self.results)
        if len(unit_ids) != len(set(unit_ids)):
            raise GroundTruthInputError("annotator submission contains duplicate units")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
            "annotator_id": self.annotator_id,
            "provider_model_label": self.provider_model_label,
            "annotation_package_sha256": self.annotation_package_sha256,
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True)
class PhraseEmbedding:
    phrase: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phrase", require_category(self.phrase, "embedding phrase"))
        if len(self.vector) != EMBEDDING_DIMENSION:
            raise GroundTruthInputError(
                f"embedding vector must have exactly {EMBEDDING_DIMENSION} dimensions"
            )
        normalized: list[float] = []
        for component in self.vector:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise GroundTruthInputError("embedding components must be numeric")
            number = float(component)
            if not isfinite(number):
                raise GroundTruthInputError("embedding components must be finite")
            normalized.append(number)
        if not any(number != 0.0 for number in normalized):
            raise GroundTruthInputError("embedding vector must be non-zero")
        object.__setattr__(self, "vector", tuple(normalized))

    def to_dict(self) -> dict[str, object]:
        return {"phrase": self.phrase, "vector": list(self.vector)}


__all__ = [
    "ANNOTATOR_MODELS",
    "AnnotationCategory",
    "AnnotationResult",
    "AnnotatorSubmission",
    "GroundTruthFinalizationIdentity",
    "EXPECTED_ANNOTATION_COUNT",
    "EXPECTED_UNIT_COUNT",
    "FINAL_GROUND_TRUTH_STATUS",
    "GROUND_TRUTH_FINALIZATION_VERSION",
    "GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION",
    "CANONICAL_REPRESENTATIVE_POLICY",
    "MAJORITY_SUPPORT_REQUIRED",
    "MAX_INPUT_BYTES",
    "MAX_CATEGORIES_PER_UNIT",
    "PhraseEmbedding",
    "SEMANTIC_CONSOLIDATION_MODEL_IDENTITY",
    "SEMANTIC_CONSOLIDATION_THRESHOLD",
    "canonical_json_bytes",
    "authenticated_ground_truth_identity",
    "category_key",
    "content_hash",
    "require_category",
    "require_exact_keys",
    "require_hash",
    "require_identifier",
    "require_safe_text",
    "sha256_hex",
    "validate_final_ground_truth_identity",
]
