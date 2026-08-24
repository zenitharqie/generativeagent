from __future__ import annotations

import json

import pytest

from src.evaluation.ground_truth import (
    GroundTruthCompletenessError,
    GroundTruthInputError,
    build_annotation_package,
    load_complete_annotations,
)
from src.evaluation.ground_truth.schemas import AnnotationCategory


@pytest.mark.parametrize("value", [True, False, 0.0, 1.0, "0", "1", -1, 2, None])
def test_relevance_requires_json_integer_zero_or_one(value) -> None:
    with pytest.raises(GroundTruthInputError):
        AnnotationCategory(category="focus", relevant=value)
from ground_truth_test_support import submission_value, write_json


def _setup(tmp_path):
    package = build_annotation_package().value
    root = tmp_path / "annotations"
    for annotator_id in ("gpt5", "claude_sonnet_5", "gemini_1_5_pro"):
        write_json(root / f"{annotator_id}.json", submission_value(package, annotator_id))
    return package, root


def test_complete_manual_annotation_matrix_is_exactly_sixty(tmp_path) -> None:
    package, root = _setup(tmp_path)
    submissions = load_complete_annotations(root, package=package)
    assert len(submissions) == 3
    assert sum(len(item.results) for item in submissions) == 60


@pytest.mark.parametrize("mutation", ["missing_unit", "stale_hash", "non_binary", "unknown_field"])
def test_invalid_manual_annotation_fails_closed(tmp_path, mutation: str) -> None:
    package, root = _setup(tmp_path)
    path = root / "gpt5.json"
    value = json.loads(path.read_text())
    if mutation == "missing_unit":
        value["results"].pop()
    elif mutation == "stale_hash":
        value["results"][0]["input_sha256"] = "0" * 64
    elif mutation == "non_binary":
        value["results"][0]["categories"][0]["relevant"] = 2
    else:
        value["results"][0]["experiment_id"] = "forbidden"
    write_json(path, value)
    with pytest.raises((GroundTruthInputError, GroundTruthCompletenessError)):
        load_complete_annotations(root, package=package)


def test_missing_or_unknown_annotator_file_is_rejected(tmp_path) -> None:
    package, root = _setup(tmp_path)
    (root / "gpt5.json").unlink()
    with pytest.raises(GroundTruthCompletenessError):
        load_complete_annotations(root, package=package)


def test_duplicate_annotator_identity_and_unknown_agent_event_are_rejected(tmp_path) -> None:
    package, root = _setup(tmp_path)
    claude = root / "claude_sonnet_5.json"
    value = json.loads(claude.read_text())
    value["annotator_id"] = "gpt5"
    value["provider_model_label"] = "GPT-5"
    write_json(claude, value)
    with pytest.raises(GroundTruthInputError):
        load_complete_annotations(root, package=package)
    package, root = _setup(tmp_path / "second")
    gpt = root / "gpt5.json"
    value = json.loads(gpt.read_text())
    value["results"][0]["agent_id"] = "unknown_agent"
    value["results"][0]["target_event_id"] = "event_99"
    write_json(gpt, value)
    with pytest.raises(GroundTruthInputError):
        load_complete_annotations(root, package=package)
    (root / "unexpected.json").write_text("{}")
    with pytest.raises(GroundTruthCompletenessError):
        load_complete_annotations(root, package=package)


def test_duplicate_keys_nan_and_symlink_are_rejected(tmp_path) -> None:
    package, root = _setup(tmp_path)
    path = root / "gpt5.json"
    path.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(GroundTruthInputError):
        load_complete_annotations(root, package=package)


def test_stale_package_hash_is_rejected(tmp_path) -> None:
    package, root = _setup(tmp_path)
    path = root / "gemini_1_5_pro.json"
    value = json.loads(path.read_text())
    value["annotation_package_sha256"] = "f" * 64
    write_json(path, value)
    with pytest.raises(GroundTruthInputError, match="stale"):
        load_complete_annotations(root, package=package)
    path.unlink()
    path.write_text('{"value":NaN}')
    with pytest.raises(GroundTruthInputError):
        load_complete_annotations(root, package=package)
    path.unlink()
    path.symlink_to(root / "claude_sonnet_5.json")
    with pytest.raises(GroundTruthInputError):
        load_complete_annotations(root, package=package)
