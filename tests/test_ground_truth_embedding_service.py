from __future__ import annotations

import pytest

from src.evaluation.ground_truth import (
    build_annotation_package,
    build_embedding_input,
    generate_embedding_evidence,
    load_embedding_evidence,
)
from src.evaluation.ground_truth.errors import GroundTruthInputError
from ground_truth_test_support import submission_value, write_json


def _input(tmp_path):
    package = build_annotation_package().value
    values = tuple(
        submission_value(package, annotator_id)
        for annotator_id in ("gpt5_6_sol", "claude_sonnet_5", "qwen3_8")
    )
    from src.evaluation.ground_truth.readers import parse_submission

    embedding_input = build_embedding_input(
        package=package,
        submissions=tuple(parse_submission(value) for value in values),
    )
    path = tmp_path / "embedding_input.json"
    write_json(path, embedding_input)
    return path, embedding_input


def test_dummy_embedding_evidence_is_structural_but_not_finalization_eligible(tmp_path) -> None:
    input_path, embedding_input = _input(tmp_path)
    output = tmp_path / "embedding_evidence.json"
    value = generate_embedding_evidence(
        input_path=input_path,
        output_path=output,
        mode="dummy",
        confirmation="GENERATE_GT_EMBEDDINGS",
    )
    assert value["acquisition_mode"] == "dummy"
    with pytest.raises(GroundTruthInputError, match="dummy"):
        load_embedding_evidence(output, expected_input=embedding_input)


def test_real_embedding_preflight_rejects_without_explicit_authorization(tmp_path) -> None:
    input_path, _ = _input(tmp_path)
    with pytest.raises(GroundTruthInputError, match="allow-real"):
        generate_embedding_evidence(
            input_path=input_path,
            output_path=tmp_path / "out.json",
            mode="real",
            confirmation="GENERATE_GT_EMBEDDINGS",
            real_authorized=False,
        )
