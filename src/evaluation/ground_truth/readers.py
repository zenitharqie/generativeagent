"""Strict readers for untrusted manual Ground Truth evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.config_loader import DEFAULT_CONFIG_DIRECTORY

from .annotation_package import validate_annotation_package
from .errors import GroundTruthCompletenessError, GroundTruthInputError
from .schemas import (
    ANNOTATOR_MODELS,
    AnnotationCategory,
    AnnotationResult,
    AnnotatorSubmission,
    EXPECTED_ANNOTATION_COUNT,
    EXPECTED_UNIT_COUNT,
    GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
    MAX_INPUT_BYTES,
    PhraseEmbedding,
    SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
    category_key,
    content_hash,
    sha256_hex,
    require_exact_keys,
    require_hash,
)


class _DuplicateKey(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _reject_symlink_path(path: Path) -> None:
    if ".." in path.parts:
        raise GroundTruthInputError("input paths must not contain traversal")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if os.path.lexists(current) and current.is_symlink():
            raise GroundTruthInputError("input path components must not be symlinks")


def read_strict_json(path: str | Path, *, context: str) -> Any:
    """Read bounded UTF-8 JSON without following a final-component symlink."""
    candidate = Path(path)
    _reject_symlink_path(candidate)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        try:
            info = os.fstat(descriptor)
            if not candidate.is_file() or candidate.is_symlink():
                raise GroundTruthInputError(f"{context} is not a safe regular file")
            if info.st_size > MAX_INPUT_BYTES:
                raise GroundTruthInputError(f"{context} exceeds the size limit")
            payload = b""
            while len(payload) <= MAX_INPUT_BYTES:
                chunk = os.read(descriptor, min(65536, MAX_INPUT_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload += chunk
        finally:
            os.close(descriptor)
        if len(payload) > MAX_INPUT_BYTES:
            raise GroundTruthInputError(f"{context} exceeds the size limit")
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except GroundTruthInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, _DuplicateKey) as exc:
        raise GroundTruthInputError(f"{context} is not valid strict JSON") from exc


def load_annotation_package(
    path: str | Path,
    *,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> Mapping[str, Any]:
    value = read_strict_json(path, context="annotation package")
    return validate_annotation_package(value, config_directory=config_directory)


def _parse_category(value: object) -> AnnotationCategory:
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("annotation category must be an object")
    keys = set(value)
    if keys not in ({"category", "relevant"}, {"category", "relevant", "reason"}):
        raise GroundTruthInputError("annotation category has an invalid field set")
    return AnnotationCategory(
        category=value.get("category"),  # type: ignore[arg-type]
        relevant=value.get("relevant"),  # type: ignore[arg-type]
        reason=value.get("reason"),  # type: ignore[arg-type]
    )


def _parse_result(value: object) -> AnnotationResult:
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("annotation result must be an object")
    require_exact_keys(
        value,
        {"annotation_unit_id", "agent_id", "target_event_id", "input_sha256", "categories"},
        "annotation result",
    )
    categories = value["categories"]
    if not isinstance(categories, list):
        raise GroundTruthInputError("annotation categories must be an array")
    return AnnotationResult(
        annotation_unit_id=value["annotation_unit_id"],  # type: ignore[arg-type]
        agent_id=value["agent_id"],  # type: ignore[arg-type]
        target_event_id=value["target_event_id"],  # type: ignore[arg-type]
        input_sha256=value["input_sha256"],  # type: ignore[arg-type]
        categories=tuple(_parse_category(item) for item in categories),
    )


def parse_submission(value: object) -> AnnotatorSubmission:
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("annotator submission must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "annotator_id",
            "provider_model_label",
            "annotation_package_sha256",
            "results",
        },
        "annotator submission",
    )
    if value["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION:
        raise GroundTruthInputError("annotator submission schema is unsupported")
    results = value["results"]
    if not isinstance(results, list):
        raise GroundTruthInputError("annotator results must be an array")
    return AnnotatorSubmission(
        annotator_id=value["annotator_id"],  # type: ignore[arg-type]
        provider_model_label=value["provider_model_label"],  # type: ignore[arg-type]
        annotation_package_sha256=value["annotation_package_sha256"],  # type: ignore[arg-type]
        results=tuple(_parse_result(item) for item in results),
    )


def load_complete_annotations(
    annotations_directory: str | Path,
    *,
    package: Mapping[str, Any],
) -> tuple[AnnotatorSubmission, ...]:
    """Load exactly three complete, independent submissions in fixed order."""
    root = Path(annotations_directory)
    _reject_symlink_path(root)
    if root.is_symlink() or not root.is_dir():
        raise GroundTruthInputError("annotations directory is missing or unsafe")
    expected_names = tuple(f"{annotator_id}.json" for annotator_id in ANNOTATOR_MODELS)
    try:
        actual_names = tuple(sorted(item.name for item in root.iterdir()))
    except OSError as exc:
        raise GroundTruthInputError("could not inspect annotations directory") from exc
    if actual_names != tuple(sorted(expected_names)):
        raise GroundTruthCompletenessError("annotations directory must contain exactly three submissions")
    package_hash = require_hash(package.get("annotation_package_sha256"), "package hash")
    package_units = package.get("units")
    if not isinstance(package_units, list) or len(package_units) != EXPECTED_UNIT_COUNT:
        raise GroundTruthInputError("annotation package unit inventory is invalid")
    expected_by_id = {unit["annotation_unit_id"]: unit for unit in package_units}
    submissions: list[AnnotatorSubmission] = []
    for annotator_id in ANNOTATOR_MODELS:
        submission = parse_submission(
            read_strict_json(root / f"{annotator_id}.json", context="annotator submission")
        )
        if submission.annotator_id != annotator_id:
            raise GroundTruthInputError("submission filename and annotator identity differ")
        if submission.annotation_package_sha256 != package_hash:
            raise GroundTruthInputError("annotator submission uses a stale package hash")
        if tuple(item.annotation_unit_id for item in submission.results) != tuple(expected_by_id):
            raise GroundTruthCompletenessError("annotator units are missing or out of canonical order")
        for result in submission.results:
            unit = expected_by_id[result.annotation_unit_id]
            if (
                result.agent_id != unit["agent_id"]
                or result.target_event_id != unit["target_event_id"]
                or result.input_sha256 != unit["input_sha256"]
            ):
                raise GroundTruthInputError("annotation result identity or input hash is stale")
        submissions.append(submission)
    if sum(len(item.results) for item in submissions) != EXPECTED_ANNOTATION_COUNT:
        raise GroundTruthCompletenessError("exactly 60 annotation results are required")
    return tuple(submissions)


def positive_phrase_inventory(
    submissions: tuple[AnnotatorSubmission, ...],
) -> tuple[dict[str, object], ...]:
    """Return unit-scoped positive phrases without inventing candidates."""
    inventory: list[dict[str, object]] = []
    for unit_index in range(EXPECTED_UNIT_COUNT):
        phrases: dict[str, dict[str, object]] = {}
        unit_id = submissions[0].results[unit_index].annotation_unit_id
        for submission in submissions:
            result = submission.results[unit_index]
            for category_index, category in enumerate(result.categories):
                if category.relevant != 1:
                    continue
                key = category_key(category.category)
                entry = phrases.setdefault(
                    key,
                    {"phrase": category.category, "provenance": []},
                )
                provenance = entry["provenance"]
                assert isinstance(provenance, list)
                provenance.append(
                    {
                        "annotator_id": submission.annotator_id,
                        "annotation_artifact": (
                            f"annotations/{submission.annotator_id}.json"
                        ),
                        "annotation_unit_id": unit_id,
                        "category_index": category_index,
                        "original_phrase": category.category,
                    }
                )
        if not phrases:
            raise GroundTruthCompletenessError("every annotation unit needs a positive candidate")
        inventory.append(
            {
                "annotation_unit_id": unit_id,
                "phrases": [phrases[key] for key in sorted(phrases)],
            }
        )
    return tuple(inventory)


def build_embedding_input(
    *, package: Mapping[str, Any], submissions: tuple[AnnotatorSubmission, ...]
) -> dict[str, object]:
    inventory = positive_phrase_inventory(submissions)
    phrase_rows = [
        {
            "annotation_unit_id": unit["annotation_unit_id"],
            "phrase": phrase["phrase"],
            "phrase_sha256": sha256_hex(str(phrase["phrase"]).encode("utf-8")),
        }
        for unit in inventory
        for phrase in unit["phrases"]  # type: ignore[union-attr]
    ]
    candidate_set_sha256 = content_hash(phrase_rows)
    return {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "annotation_package_sha256": package["annotation_package_sha256"],
        "model_identity": SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
        "candidate_set_sha256": candidate_set_sha256,
        "candidate_count": len(phrase_rows),
        "candidates": phrase_rows,
    }


def load_embedding_evidence(
    path: str | Path,
    *,
    expected_input: Mapping[str, Any],
    allow_dummy: bool = False,
) -> tuple[PhraseEmbedding, ...]:
    value = read_strict_json(path, context="embedding evidence")
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("embedding evidence must be an object")
    keys = set(value)
    required = {
        "schema_version", "annotation_package_sha256", "model_identity",
        "dimension", "candidate_set_sha256", "embeddings",
    }
    optional = {"acquisition_mode", "attempts"}
    if not keys.issubset(required | optional) or not required.issubset(keys):
        raise GroundTruthInputError("embedding evidence has an invalid field set")
    acquisition_mode = value.get("acquisition_mode")
    if acquisition_mode not in (None, "real", "dummy"):
        raise GroundTruthInputError("embedding evidence acquisition mode is invalid")
    if acquisition_mode == "dummy" and not allow_dummy:
        raise GroundTruthInputError("dummy embedding evidence is not valid for finalization")
    if "attempts" in value and not isinstance(value["attempts"], list):
        raise GroundTruthInputError("embedding attempt evidence must be an array")
    if "attempts" in value:
        for attempt in value["attempts"]:
            if not isinstance(attempt, Mapping):
                raise GroundTruthInputError("embedding attempt evidence is invalid")
            require_exact_keys(
                attempt,
                {
                    "schema_version", "service_role", "logical_call_id", "attempt_index",
                    "call_type", "model", "prompt_template", "prompt_version", "prompt_hash",
                    "input_tokens", "output_tokens", "total_tokens", "status", "error_type",
                },
                "embedding attempt evidence",
            )
            if attempt["service_role"] != "embedding" or attempt["model"] != SEMANTIC_CONSOLIDATION_MODEL_IDENTITY:
                raise GroundTruthInputError("embedding attempt identity is invalid")
    if (
        value["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION
        or value["model_identity"] != SEMANTIC_CONSOLIDATION_MODEL_IDENTITY
        or value["dimension"] != 1536
        or value["annotation_package_sha256"] != expected_input["annotation_package_sha256"]
        or value["candidate_set_sha256"] != expected_input["candidate_set_sha256"]
    ):
        raise GroundTruthInputError("embedding evidence identity is stale or unsupported")
    records = value["embeddings"]
    if not isinstance(records, list) or len(records) != expected_input["candidate_count"]:
        raise GroundTruthInputError("embedding evidence inventory is incomplete")
    expected_candidates = expected_input["candidates"]
    embeddings: list[PhraseEmbedding] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise GroundTruthInputError("embedding record must be an object")
        keys = set(record)
        if keys not in (
            {"annotation_unit_id", "phrase", "phrase_sha256", "vector"},
            {"annotation_unit_id", "phrase", "vector"},
        ):
            raise GroundTruthInputError("embedding record has an invalid field set")
        expected = expected_candidates[index]  # type: ignore[index]
        if (
            record["annotation_unit_id"] != expected["annotation_unit_id"]
            or record["phrase"] != expected["phrase"]
        ):
            raise GroundTruthInputError("embedding records are missing or out of canonical order")
        if "phrase_sha256" in record and record["phrase_sha256"] != expected["phrase_sha256"]:
            raise GroundTruthInputError("embedding phrase identity is stale")
        vector = record["vector"]
        if not isinstance(vector, list):
            raise GroundTruthInputError("embedding vector must be an array")
        embeddings.append(PhraseEmbedding(phrase=record["phrase"], vector=tuple(vector)))  # type: ignore[arg-type]
    return tuple(embeddings)


__all__ = [
    "build_embedding_input",
    "load_annotation_package",
    "load_complete_annotations",
    "load_embedding_evidence",
    "parse_submission",
    "positive_phrase_inventory",
    "read_strict_json",
]
