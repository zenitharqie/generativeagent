"""Deterministic evidence builders for Ground Truth workflow tests."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Callable, Mapping

from src.evaluation.ground_truth import ANNOTATOR_MODELS, build_embedding_input
from src.evaluation.ground_truth.schemas import canonical_json_bytes
from src.evaluation.ground_truth.readers import parse_submission


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value, pretty=True))


def write_package(path: Path, package: Mapping[str, object]) -> None:
    write_json(path, package)


def submission_value(
    package: Mapping[str, object],
    annotator_id: str,
    *,
    category_factory: Callable[[int, str], str] | None = None,
) -> dict[str, object]:
    units = package["units"]
    assert isinstance(units, list)
    position = tuple(ANNOTATOR_MODELS).index(annotator_id)
    results = []
    for index, unit in enumerate(units):
        category = (
            category_factory(index, annotator_id)
            if category_factory is not None
            else (
                f"shared concept {index:02d} alpha"
                if position == 0
                else f"shared concept {index:02d} beta"
                if position == 1
                else f"single concept {index:02d}"
            )
        )
        results.append(
            {
                "annotation_unit_id": unit["annotation_unit_id"],
                "agent_id": unit["agent_id"],
                "target_event_id": unit["target_event_id"],
                "input_sha256": unit["input_sha256"],
                "categories": [{"category": category, "relevant": 1}],
            }
        )
    return {
        "schema_version": "1.0",
        "annotator_id": annotator_id,
        "provider_model_label": ANNOTATOR_MODELS[annotator_id],
        "annotation_package_sha256": package["annotation_package_sha256"],
        "results": results,
    }


def write_submissions(root: Path, package: Mapping[str, object]) -> None:
    for annotator_id in ANNOTATOR_MODELS:
        write_json(root / f"{annotator_id}.json", submission_value(package, annotator_id))


def _vector(first: float, second: float, *, axis: int = 0) -> list[float]:
    vector = [0.0] * 1536
    vector[axis] = first
    vector[axis + 1] = second
    return vector


def write_embedding_evidence(
    path: Path,
    *,
    package: Mapping[str, object],
    annotation_values: tuple[Mapping[str, object], ...],
    relation: str = "merge_first_two",
) -> dict[str, object]:
    submissions = tuple(parse_submission(value) for value in annotation_values)
    embedding_input = build_embedding_input(package=package, submissions=submissions)
    embeddings = []
    by_unit: dict[str, int] = {}
    for candidate in embedding_input["candidates"]:
        assert isinstance(candidate, dict)
        unit_id = candidate["annotation_unit_id"]
        local = by_unit.get(unit_id, 0)
        by_unit[unit_id] = local + 1
        if relation == "merge_first_two":
            vector = _vector(1.0, 0.0) if local < 2 else _vector(0.0, 1.0)
        elif relation == "no_merge":
            angle = local * 1.5707963267948966
            vector = _vector(__import__("math").cos(angle), __import__("math").sin(angle))
        elif relation == "ambiguous_chain":
            angle = local * (25.0 * 3.141592653589793 / 180.0)
            vector = _vector(__import__("math").cos(angle), __import__("math").sin(angle))
        elif relation == "exact_threshold":
            vector = (
                _vector(1.0, 0.0)
                if local == 0
                else _vector(0.85, sqrt(1.0 - 0.85**2))
                if local == 1
                else _vector(0.0, 1.0)
            )
        else:
            raise AssertionError(relation)
        embeddings.append({**candidate, "vector": vector})
    value = {
        "schema_version": "1.0",
        "annotation_package_sha256": package["annotation_package_sha256"],
        "model_identity": "openai:text-embedding-3-small",
        "dimension": 1536,
        "candidate_set_sha256": embedding_input["candidate_set_sha256"],
        "embeddings": embeddings,
    }
    write_json(path, value)
    return value


def complete_evidence(tmp_path: Path, package: Mapping[str, object]) -> tuple[Path, Path, Path]:
    package_path = tmp_path / "annotation_package.json"
    annotations = tmp_path / "annotations"
    embedding_path = tmp_path / "embeddings.json"
    write_package(package_path, package)
    write_submissions(annotations, package)
    annotation_values = tuple(
        submission_value(package, annotator_id) for annotator_id in ANNOTATOR_MODELS
    )
    write_embedding_evidence(
        embedding_path,
        package=package,
        annotation_values=annotation_values,
    )
    return package_path, annotations, embedding_path
