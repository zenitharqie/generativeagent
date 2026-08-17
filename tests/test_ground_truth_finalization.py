from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.config_loader import load_experiment_config
from src.evaluation.ground_truth import (
    ANNOTATOR_MODELS,
    GroundTruthConsolidationError,
    GroundTruthFinalizationError,
    build_annotation_package,
    build_finalization_bundle,
    calculate_fleiss_kappa,
    finalize_ground_truth,
    validate_final_ground_truth_identity,
)
from src.evaluation import load_evaluation_run
from phase09_test_support import copy_run
from src.simulation import validate_ground_truth_methodological_finalization
from ground_truth_test_support import (
    complete_evidence,
    submission_value,
    write_embedding_evidence,
    write_json,
)


def test_complete_finalization_builds_twenty_units_provenance_and_majority(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
    )
    ground_truth = bundle.final_ground_truth
    assert ground_truth["status"] == "final"
    assert ground_truth["freeze_required_before_final_experiment"] is True
    assert len(ground_truth["targets"]) == 20
    assert all(len(item["expected_categories"]) == 1 for item in ground_truth["targets"])
    assert validate_final_ground_truth_identity(ground_truth) == bundle.fingerprint_sha256
    master = json.loads(bundle.files["master_candidate_list.json"])
    consolidation = json.loads(bundle.files["consolidation_evidence.json"])
    matrix = json.loads(bundle.files["annotation_matrix.json"])
    assert len(master["units"]) == 20
    assert all(len(unit["phrases"]) == 3 for unit in master["units"])
    assert all(len(unit["candidates"]) == 2 for unit in consolidation["units"])
    assert len(matrix["items"]) == 40
    assert all(item["positive_annotator_count"] in {1, 2} for item in matrix["items"])


def test_majority_rules_are_exact_and_have_no_override(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
    )
    evidence = json.loads(bundle.files["consolidation_evidence.json"])
    first = evidence["units"][0]["candidates"]
    included = [item for item in first if item["included_by_majority"]]
    excluded = [item for item in first if not item["included_by_majority"]]
    assert included[0]["positive_annotator_count"] == 2
    assert excluded[0]["positive_annotator_count"] == 1
    assert "override" not in json.dumps(evidence).casefold()


def test_fleiss_kappa_known_perfect_and_degenerate_examples() -> None:
    assert calculate_fleiss_kappa(((1, 1, 1), (0, 0, 0))) == pytest.approx(1.0)
    assert calculate_fleiss_kappa(((1, 1, 1), (1, 1, 1))) is None
    # Kappa is descriptive and does not mutate its binary input.
    rows = ((1, 1, 0), (0, 0, 1))
    before = tuple(tuple(row) for row in rows)
    assert calculate_fleiss_kappa(rows) == pytest.approx(-0.3333333333333333)
    assert rows == before


def test_threshold_equal_merges_and_below_threshold_does_not(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, _ = complete_evidence(tmp_path, package)
    values = tuple(submission_value(package, item) for item in ANNOTATOR_MODELS)
    exact = tmp_path / "exact.json"
    write_embedding_evidence(
        exact, package=package, annotation_values=values, relation="exact_threshold"
    )
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=exact,
    )
    consolidation = json.loads(bundle.files["consolidation_evidence.json"])
    assert consolidation["cosine_similarity_threshold"] == 0.85
    first_pair = consolidation["units"][0]["similarities"][0]
    assert first_pair["cosine_similarity"] == pytest.approx(0.85)
    assert first_pair["meets_threshold"] is True

    below = tmp_path / "below.json"
    write_embedding_evidence(
        below, package=package, annotation_values=values, relation="no_merge"
    )
    with pytest.raises(GroundTruthFinalizationError, match="majority"):
        build_finalization_bundle(
            package_path=package_path,
            annotations_directory=annotations,
            embedding_evidence_path=below,
        )


def test_ambiguous_transitive_consolidation_fails_closed(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, _ = complete_evidence(tmp_path, package)
    values = tuple(submission_value(package, item) for item in ANNOTATOR_MODELS)
    ambiguous = tmp_path / "ambiguous.json"
    write_embedding_evidence(
        ambiguous,
        package=package,
        annotation_values=values,
        relation="ambiguous_chain",
    )
    with pytest.raises(GroundTruthConsolidationError, match="ambiguous"):
        build_finalization_bundle(
            package_path=package_path,
            annotations_directory=annotations,
            embedding_evidence_path=ambiguous,
        )


@pytest.mark.parametrize("field,value", [("model_identity", "other:model"), ("dimension", 3)])
def test_embedding_methodology_identity_is_exact(tmp_path, field, value) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    evidence = json.loads(embeddings.read_text())
    evidence[field] = value
    write_json(embeddings, evidence)
    with pytest.raises(Exception):
        build_finalization_bundle(
            package_path=package_path,
            annotations_directory=annotations,
            embedding_evidence_path=embeddings,
        )


def test_finalization_publication_is_deterministic_and_inputs_immutable(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    inputs = tuple(path for path in tmp_path.rglob("*") if path.is_file())
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in inputs}
    first = tmp_path / "final-a"
    second = tmp_path / "final-b"
    a = finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=first,
    )
    b = finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=second,
    )
    assert a.fingerprint_sha256 == b.fingerprint_sha256
    assert {
        str(path.relative_to(first)): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        str(path.relative_to(second)): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert before == {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in inputs
    }
    assert not (first / "bcr_judged_records.jsonl").exists()


def test_final_ground_truth_remains_phase09_config_compatible_and_phase13_valid(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    output = tmp_path / "final"
    finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=output,
    )
    config_copy = tmp_path / "config"
    source_config = Path("config")
    config_copy.mkdir()
    for path in source_config.iterdir():
        (config_copy / path.name).write_bytes(path.read_bytes())
    (config_copy / "ground_truth.json").write_bytes((output / "ground_truth.json").read_bytes())
    config = load_experiment_config(config_copy)
    assert len(config.ground_truth["targets"]) == 20
    validate_ground_truth_methodological_finalization(config_copy, output)


def test_phase09_reads_finalized_ground_truth_without_methodology_change(
    phase08_full_artifacts, tmp_path
) -> None:
    package = build_annotation_package().value
    evidence_root = tmp_path / "evidence"
    package_path, annotations, embeddings = complete_evidence(evidence_root, package)
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
    )
    run_dir = copy_run(phase08_full_artifacts[0].run_paths.run_dir, tmp_path / "run-copy")
    snapshot = run_dir / "config_snapshot" / "ground_truth.json"
    snapshot.unlink()
    snapshot.write_bytes(bundle.files["ground_truth.json"])
    loaded = load_evaluation_run(run_dir)
    assert len(loaded.target_units) == 20
    expected = bundle.final_ground_truth["targets"][0]["expected_categories"]
    assert list(loaded.target_units[0].expected_categories) == expected


def test_three_of_three_is_included(tmp_path) -> None:
    package = build_annotation_package().value
    package_path = tmp_path / "package.json"
    write_json(package_path, package)
    annotations = tmp_path / "annotations"
    same = lambda index, annotator: f"unanimous concept {index:02d}"
    values = tuple(
        submission_value(package, annotator, category_factory=same)
        for annotator in ANNOTATOR_MODELS
    )
    for value in values:
        write_json(annotations / f"{value['annotator_id']}.json", value)
    embeddings = tmp_path / "embeddings.json"
    write_embedding_evidence(
        embeddings, package=package, annotation_values=values
    )
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
    )
    matrix = json.loads(bundle.files["annotation_matrix.json"])
    assert all(item["positive_annotator_count"] == 3 for item in matrix["items"])
    assert all(item["included_by_majority"] is True for item in matrix["items"])


def test_incomplete_or_directly_edited_evidence_cannot_finalize(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    (annotations / "gemini_1_5_pro.json").unlink()
    with pytest.raises(Exception):
        build_finalization_bundle(
            package_path=package_path,
            annotations_directory=annotations,
            embedding_evidence_path=embeddings,
        )
    hand_edit = {
        "evaluation_step_type": "target_event",
        "status": "final",
        "freeze_required_before_final_experiment": True,
        "targets": [],
    }
    with pytest.raises(Exception):
        validate_final_ground_truth_identity(hand_edit)


def test_phase13_identity_rejects_mutated_finalization_report(tmp_path) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    output = tmp_path / "final"
    bundle = finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=output,
    )
    report = json.loads((output / "finalization_report.json").read_text())
    report["final_category_count"] += 1
    write_json(output / "finalization_report.json", report)
    with pytest.raises(Exception):
        validate_final_ground_truth_identity(bundle.final_ground_truth, evidence_directory=str(output))
