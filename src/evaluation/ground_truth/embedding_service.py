"""Explicit, embedding-only acquisition for Ground Truth consolidation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from src.llm import InMemoryUsageRecorder, OpenAIEmbeddingProvider, OpenAIProviderAdapter
from src.llm.settings import EMBEDDING_DIMENSION, EMBEDDING_MODEL_IDENTITY, EMBEDDING_PROFILE

from .errors import GroundTruthInputError, GroundTruthPersistenceError
from .persistence import publish_json_file
from .readers import load_embedding_evidence, read_strict_json
from .schemas import (
    GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
    canonical_json_bytes,
    require_exact_keys,
    require_hash,
    sha256_hex,
)

CONFIRMATION = "GENERATE_GT_EMBEDDINGS"


def _validate_input(path: str | Path) -> Mapping[str, Any]:
    value = read_strict_json(path, context="Ground Truth embedding input")
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("embedding input must be an object")
    require_exact_keys(
        value,
        {"schema_version", "annotation_package_sha256", "model_identity", "candidate_set_sha256", "candidate_count", "candidates"},
        "embedding input",
    )
    if value["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION or value["model_identity"] != EMBEDDING_MODEL_IDENTITY:
        raise GroundTruthInputError("embedding input model or schema is unsupported")
    require_hash(value["annotation_package_sha256"], "annotation package hash")
    require_hash(value["candidate_set_sha256"], "candidate set hash")
    candidates = value["candidates"]
    if not isinstance(value["candidate_count"], int) or isinstance(value["candidate_count"], bool) or value["candidate_count"] != len(candidates if isinstance(candidates, list) else ()):
        raise GroundTruthInputError("embedding input candidate count is invalid")
    if not isinstance(candidates, list) or not candidates:
        raise GroundTruthInputError("embedding input candidates are empty")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise GroundTruthInputError("embedding candidate must be an object")
        require_exact_keys(candidate, {"annotation_unit_id", "phrase", "phrase_sha256"}, "embedding candidate")
        require_hash(candidate["phrase_sha256"], "embedding phrase hash")
        if not isinstance(candidate["phrase"], str) or not candidate["phrase"].strip():
            raise GroundTruthInputError("embedding phrase is invalid")
        if candidate["phrase_sha256"] != sha256_hex(candidate["phrase"].encode("utf-8")):
            raise GroundTruthInputError("embedding phrase hash is invalid")
    if value["candidate_set_sha256"] != sha256_hex(
        canonical_json_bytes(
            [
                {
                    "annotation_unit_id": item["annotation_unit_id"],
                    "phrase": item["phrase"],
                    "phrase_sha256": item["phrase_sha256"],
                }
                for item in candidates
            ]
        )
    ):
        raise GroundTruthInputError("embedding input candidate-set identity is invalid")
    return value


def _validate_output_path(path: str | Path) -> Path:
    output = Path(path)
    if ".." in output.parts or output.exists() or output.is_symlink():
        raise GroundTruthPersistenceError("embedding evidence output must be fresh and safe")
    if not output.parent.is_dir():
        raise GroundTruthPersistenceError("embedding evidence output parent must exist")
    current = output.parent
    while current != current.parent:
        if current.is_symlink():
            raise GroundTruthPersistenceError("embedding evidence output path must not contain symlinks")
        current = current.parent
    return output


def _same_file(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).resolve(strict=True) == right.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def _dummy_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[index % EMBEDDING_DIMENSION] = 1.0
    return vector


def generate_embedding_evidence(
    *,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    confirmation: str | None,
    real_authorized: bool = False,
    disk_check: Callable[[Path], None] | None = None,
) -> Mapping[str, Any]:
    """Acquire only the vectors named by a validated embedding input."""
    if mode not in {"real", "dummy"}:
        raise GroundTruthInputError("embedding evidence mode must be real or dummy")
    value = _validate_input(input_path)
    output = _validate_output_path(output_path)
    if _same_file(input_path, output):
        raise GroundTruthPersistenceError("embedding evidence output must differ from input")
    if confirmation != CONFIRMATION:
        raise GroundTruthInputError("exact Ground Truth embedding confirmation is required")
    if mode == "real" and real_authorized is not True:
        raise GroundTruthInputError("real embedding requires --allow-real-embedding true")
    if mode == "dummy" and real_authorized:
        raise GroundTruthInputError("dummy embedding must not receive real authorization")
    if disk_check is not None:
        disk_check(output.parent)

    candidates = value["candidates"]
    assert isinstance(candidates, list)
    recorder = InMemoryUsageRecorder()
    provider: OpenAIEmbeddingProvider | None = None
    if mode == "real":
        adapter = OpenAIProviderAdapter.from_environment(
            timeout_seconds=EMBEDDING_PROFILE.request_timeout_seconds
        )
        provider = OpenAIEmbeddingProvider(adapter, usage_recorder=recorder)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        assert isinstance(candidate, Mapping)
        vector = _dummy_vector(index) if provider is None else list(provider.embed(candidate["phrase"]))
        records.append(
            {
                "annotation_unit_id": candidate["annotation_unit_id"],
                "phrase": candidate["phrase"],
                "phrase_sha256": candidate["phrase_sha256"],
                "vector": vector,
            }
        )
    evidence: dict[str, Any] = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "acquisition_mode": mode,
        "annotation_package_sha256": value["annotation_package_sha256"],
        "model_identity": EMBEDDING_MODEL_IDENTITY,
        "dimension": EMBEDDING_DIMENSION,
        "candidate_set_sha256": value["candidate_set_sha256"],
        "embeddings": records,
        "attempts": [record.to_dict() for record in recorder.records],
    }
    payload = canonical_json_bytes(evidence, pretty=True)
    if _validate_input(input_path)["candidate_set_sha256"] != value["candidate_set_sha256"]:
        raise GroundTruthInputError("embedding input changed during acquisition")
    publish_json_file(output, payload)
    try:
        load_embedding_evidence(output, expected_input=value, allow_dummy=True)
    except Exception:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return evidence


__all__ = ["CONFIRMATION", "generate_embedding_evidence"]
