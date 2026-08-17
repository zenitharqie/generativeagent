"""Automated consolidation, voting, reliability, and final Ground Truth construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Mapping

from src.config_loader import DEFAULT_CONFIG_DIRECTORY

from .errors import GroundTruthConsolidationError, GroundTruthFinalizationError
from .readers import (
    build_embedding_input,
    load_annotation_package,
    load_complete_annotations,
    load_embedding_evidence,
    positive_phrase_inventory,
)
from .schemas import (
    ANNOTATOR_MODELS,
    CANONICAL_REPRESENTATIVE_POLICY,
    EXPECTED_ANNOTATION_COUNT,
    EXPECTED_UNIT_COUNT,
    FINAL_GROUND_TRUTH_STATUS,
    GROUND_TRUTH_FINALIZATION_VERSION,
    GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
    MAJORITY_SUPPORT_REQUIRED,
    SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
    SEMANTIC_CONSOLIDATION_THRESHOLD,
    AnnotatorSubmission,
    PhraseEmbedding,
    canonical_json_bytes,
    category_key,
    content_hash,
    sha256_hex,
    validate_final_ground_truth_identity,
)


@dataclass(frozen=True)
class GroundTruthFinalizationBundle:
    files: Mapping[str, bytes]
    final_ground_truth: Mapping[str, Any]
    fingerprint_sha256: str


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise GroundTruthConsolidationError("embedding norm must be non-zero")
    result = numerator / (left_norm * right_norm)
    return max(-1.0, min(1.0, result))


def _components(adjacency: tuple[frozenset[int], ...]) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(len(adjacency)))
    result: list[tuple[int, ...]] = []
    while unseen:
        first = min(unseen)
        stack = [first]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        result.append(tuple(sorted(component)))
    return tuple(result)


def _consolidate_unit(
    unit: Mapping[str, Any],
    embeddings: tuple[PhraseEmbedding, ...],
    offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    phrases = unit["phrases"]
    assert isinstance(phrases, list)
    vectors = embeddings[offset : offset + len(phrases)]
    if len(vectors) != len(phrases):
        raise GroundTruthConsolidationError("embedding inventory is incomplete")
    similarities: list[dict[str, object]] = []
    adjacent: list[set[int]] = [{index} for index in range(len(phrases))]
    for left in range(len(phrases)):
        for right in range(left + 1, len(phrases)):
            score = _cosine(vectors[left].vector, vectors[right].vector)
            merges = score >= SEMANTIC_CONSOLIDATION_THRESHOLD
            similarities.append(
                {
                    "left_phrase": phrases[left]["phrase"],
                    "right_phrase": phrases[right]["phrase"],
                    "cosine_similarity": score,
                    "meets_threshold": merges,
                }
            )
            if merges:
                adjacent[left].add(right)
                adjacent[right].add(left)
    components = _components(tuple(frozenset(item) for item in adjacent))
    for component in components:
        if any(
            right not in adjacent[left]
            for left in component
            for right in component
            if left != right
        ):
            raise GroundTruthConsolidationError(
                "semantic consolidation contains an ambiguous transitive cluster"
            )
    candidates: list[dict[str, object]] = []
    for candidate_index, component in enumerate(components, start=1):
        originals = [phrases[index] for index in component]
        support_by_phrase = []
        for item in originals:
            annotators = {
                row["annotator_id"] for row in item["provenance"]  # type: ignore[index,union-attr]
            }
            support_by_phrase.append((len(annotators), category_key(item["phrase"]), item["phrase"]))
        # Deterministic representative: strongest direct annotator support, then lexical key.
        canonical = sorted(support_by_phrase, key=lambda row: (-row[0], row[1]))[0][2]
        provenance = sorted(
            (row for item in originals for row in item["provenance"]),  # type: ignore[union-attr]
            key=lambda row: (
                row["annotator_id"],
                row["category_index"],
                category_key(row["original_phrase"]),
            ),
        )
        annotators = tuple(sorted({row["annotator_id"] for row in provenance}))
        candidates.append(
            {
                "candidate_id": f"{unit['annotation_unit_id']}_candidate_{candidate_index:03d}",
                "canonical_category": canonical,
                "original_phrases": sorted(
                    {item["phrase"] for item in originals}, key=category_key
                ),
                "positive_annotators": list(annotators),
                "positive_annotator_count": len(annotators),
                "included_by_majority": len(annotators) >= MAJORITY_SUPPORT_REQUIRED,
                "provenance": provenance,
            }
        )
    candidates.sort(key=lambda item: category_key(item["canonical_category"]))  # type: ignore[arg-type]
    return similarities, candidates, offset + len(phrases)


def _fleiss_kappa(rows: list[list[int]]) -> dict[str, object]:
    if not rows or any(len(row) != len(ANNOTATOR_MODELS) for row in rows):
        raise GroundTruthFinalizationError("annotation matrix is invalid")
    n = len(ANNOTATOR_MODELS)
    agreements: list[float] = []
    positive_total = 0
    for row in rows:
        if any(value not in {0, 1} for value in row):
            raise GroundTruthFinalizationError("annotation matrix must be binary")
        positives = sum(row)
        negatives = n - positives
        positive_total += positives
        agreements.append(
            (positives * (positives - 1) + negatives * (negatives - 1))
            / (n * (n - 1))
        )
    observed = sum(agreements) / len(agreements)
    positive_prevalence = positive_total / (len(rows) * n)
    negative_prevalence = 1.0 - positive_prevalence
    expected = positive_prevalence**2 + negative_prevalence**2
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "item_count": len(rows),
        "ratings_per_item": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "fleiss_kappa": kappa,
        "interpretation": "descriptive_only_no_pass_fail_threshold",
    }


def calculate_fleiss_kappa(rows: tuple[tuple[int, ...], ...]) -> float | None:
    """Public calculation helper used by known-matrix tests."""
    value = _fleiss_kappa([list(row) for row in rows])
    result = value["fleiss_kappa"]
    return None if result is None else float(result)


def build_finalization_bundle(
    *,
    package_path: str | Path,
    annotations_directory: str | Path,
    embedding_evidence_path: str | Path,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> GroundTruthFinalizationBundle:
    """Validate all evidence and build deterministic final artifacts without publishing."""
    package = load_annotation_package(package_path, config_directory=config_directory)
    submissions = load_complete_annotations(annotations_directory, package=package)
    embedding_input = build_embedding_input(package=package, submissions=submissions)
    embeddings = load_embedding_evidence(
        embedding_evidence_path, expected_input=embedding_input
    )
    inventory = positive_phrase_inventory(submissions)
    master = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "annotation_package_sha256": package["annotation_package_sha256"],
        "construction_rule": "union_of_categories_labeled_relevant_by_at_least_one_annotator",
        "units": list(inventory),
    }
    consolidation_units: list[dict[str, object]] = []
    matrix_items: list[dict[str, object]] = []
    final_targets: list[dict[str, object]] = []
    offset = 0
    package_units = package["units"]
    assert isinstance(package_units, list)
    for unit_index, unit in enumerate(inventory):
        similarities, candidates, offset = _consolidate_unit(unit, embeddings, offset)
        matrix_rows: list[dict[str, object]] = []
        for candidate in candidates:
            positives = set(candidate["positive_annotators"])  # type: ignore[arg-type]
            ratings = [1 if annotator_id in positives else 0 for annotator_id in ANNOTATOR_MODELS]
            matrix_row = {
                "candidate_id": candidate["candidate_id"],
                "ratings": ratings,
                "positive_annotator_count": sum(ratings),
                "included_by_majority": sum(ratings) >= MAJORITY_SUPPORT_REQUIRED,
            }
            matrix_rows.append(matrix_row)
            matrix_items.append(
                {
                    "annotation_unit_id": unit["annotation_unit_id"],
                    **matrix_row,
                }
            )
        included = [
            candidate["canonical_category"]
            for candidate in candidates
            if candidate["included_by_majority"] is True
        ]
        if not included:
            raise GroundTruthFinalizationError(
                "every final Ground Truth unit requires at least one majority category"
            )
        identity = package_units[unit_index]
        final_targets.append(
            {
                "event_id": identity["target_event_id"],
                "agent_id": identity["agent_id"],
                "expected_categories": included,
            }
        )
        consolidation_units.append(
            {
                "annotation_unit_id": unit["annotation_unit_id"],
                "similarities": similarities,
                "candidates": candidates,
                "annotation_matrix": matrix_rows,
            }
        )
    if offset != len(embeddings):
        raise GroundTruthConsolidationError("unused embedding records are not permitted")
    matrix = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "annotator_order": list(ANNOTATOR_MODELS),
        "rating_values": [0, 1],
        "items": matrix_items,
    }
    reliability_value = _fleiss_kappa([item["ratings"] for item in matrix_items])  # type: ignore[list-item]
    reliability = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "matrix_sha256": content_hash(matrix),
        **reliability_value,
    }
    canonical_embedding_evidence = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "annotation_package_sha256": package["annotation_package_sha256"],
        "model_identity": SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
        "dimension": len(embeddings[0].vector),
        "candidate_set_sha256": embedding_input["candidate_set_sha256"],
        "embeddings": [
            {
                **candidate,
                "vector": list(embedding.vector),
            }
            for candidate, embedding in zip(
                embedding_input["candidates"], embeddings, strict=True  # type: ignore[arg-type]
            )
        ],
    }
    consolidation = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "annotation_package_sha256": package["annotation_package_sha256"],
        "embedding_input_sha256": content_hash(embedding_input),
        "embedding_evidence_sha256": content_hash(
            {
                "embedding_input": embedding_input,
                "embeddings": [item.to_dict() for item in embeddings],
            }
        ),
        "model_identity": SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
        "cosine_similarity_threshold": SEMANTIC_CONSOLIDATION_THRESHOLD,
        "ambiguous_transitive_clusters": "fail_closed",
        "canonical_representative_rule": (
            CANONICAL_REPRESENTATIVE_POLICY
        ),
        "units": consolidation_units,
    }
    annotation_files = {
        f"annotations/{submission.annotator_id}.json": canonical_json_bytes(
            submission.to_dict(), pretty=True
        )
        for submission in submissions
    }
    evidence_hashes = {
        "annotation_package.json": sha256_hex(canonical_json_bytes(package, pretty=True)),
        **{name: sha256_hex(payload) for name, payload in annotation_files.items()},
        "master_candidate_list.json": sha256_hex(canonical_json_bytes(master, pretty=True)),
        "embedding_input.json": sha256_hex(
            canonical_json_bytes(embedding_input, pretty=True)
        ),
        "embedding_evidence.json": sha256_hex(
            canonical_json_bytes(canonical_embedding_evidence, pretty=True)
        ),
        "consolidation_evidence.json": sha256_hex(canonical_json_bytes(consolidation, pretty=True)),
        "annotation_matrix.json": sha256_hex(canonical_json_bytes(matrix, pretty=True)),
        "reliability.json": sha256_hex(canonical_json_bytes(reliability, pretty=True)),
    }
    ground_truth_without_fingerprint = {
        "evaluation_step_type": "target_event",
        "status": FINAL_GROUND_TRUTH_STATUS,
        "freeze_required_before_final_experiment": True,
        "notes": (
            "Finalized from three independent manual annotations before the full experiment. "
            "A new Phase 11 pilot and accepted freeze remain required."
        ),
        "methodological_finalization": {
            "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
            "methodology_id": GROUND_TRUTH_FINALIZATION_VERSION,
            "annotation_package_sha256": package["annotation_package_sha256"],
            "embedding_model_identity": SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
            "cosine_similarity_threshold": SEMANTIC_CONSOLIDATION_THRESHOLD,
            "majority_rule": "positive_annotator_count_greater_than_or_equal_to_2_of_3",
            "canonical_representative_policy": CANONICAL_REPRESENTATIVE_POLICY,
            "annotation_result_count": EXPECTED_ANNOTATION_COUNT,
            "evidence_hashes": evidence_hashes,
        },
        "targets": final_targets,
    }
    fingerprint_binding = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "ground_truth_without_fingerprint": ground_truth_without_fingerprint,
        "evidence_hashes": evidence_hashes,
        "reliability": reliability,
    }
    fingerprint_sha256 = content_hash(fingerprint_binding)
    final_ground_truth = dict(ground_truth_without_fingerprint)
    finalization_metadata = dict(final_ground_truth["methodological_finalization"])
    finalization_metadata["fingerprint_sha256"] = fingerprint_sha256
    final_ground_truth["methodological_finalization"] = finalization_metadata
    fingerprint = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "methodology_id": GROUND_TRUTH_FINALIZATION_VERSION,
        "fingerprint_sha256": fingerprint_sha256,
        "binding_sha256": content_hash(fingerprint_binding),
        "ground_truth_sha256": sha256_hex(canonical_json_bytes(final_ground_truth, pretty=True)),
        "evidence_hashes": evidence_hashes,
    }
    report = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "status": FINAL_GROUND_TRUTH_STATUS,
        "annotation_unit_count": EXPECTED_UNIT_COUNT,
        "annotator_count": len(ANNOTATOR_MODELS),
        "annotation_result_count": EXPECTED_ANNOTATION_COUNT,
        "final_category_count": sum(len(item["expected_categories"]) for item in final_targets),
        "fingerprint_sha256": fingerprint_sha256,
        "reliability_sha256": evidence_hashes["reliability.json"],
        "no_experiment_outcome_data": True,
    }
    files: dict[str, bytes] = {
        "annotation_package.json": canonical_json_bytes(package, pretty=True),
        **annotation_files,
        "embedding_input.json": canonical_json_bytes(embedding_input, pretty=True),
        "embedding_evidence.json": canonical_json_bytes(
            canonical_embedding_evidence, pretty=True
        ),
        "master_candidate_list.json": canonical_json_bytes(master, pretty=True),
        "consolidation_evidence.json": canonical_json_bytes(consolidation, pretty=True),
        "annotation_matrix.json": canonical_json_bytes(matrix, pretty=True),
        "reliability.json": canonical_json_bytes(reliability, pretty=True),
        "ground_truth.json": canonical_json_bytes(final_ground_truth, pretty=True),
        "ground_truth_fingerprint.json": canonical_json_bytes(fingerprint, pretty=True),
        "finalization_report.json": canonical_json_bytes(report, pretty=True),
    }
    if validate_final_ground_truth_identity(final_ground_truth) != fingerprint_sha256:
        raise GroundTruthFinalizationError("final Ground Truth identity validation failed")
    return GroundTruthFinalizationBundle(
        files=files,
        final_ground_truth=final_ground_truth,
        fingerprint_sha256=fingerprint_sha256,
    )


__all__ = [
    "GroundTruthFinalizationBundle",
    "build_finalization_bundle",
    "calculate_fleiss_kappa",
]
