from src.evaluation.ground_truth import (
    authenticated_ground_truth_identity,
    build_annotation_package,
    finalize_ground_truth,
)
from ground_truth_test_support import complete_evidence


def test_authenticated_ground_truth_identity_is_complete_and_deterministic(tmp_path):
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    output = tmp_path / "finalization"
    bundle = finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=output,
    )
    first = authenticated_ground_truth_identity(
        bundle.final_ground_truth, evidence_directory=str(output)
    )
    second = authenticated_ground_truth_identity(
        bundle.final_ground_truth, evidence_directory=str(output)
    )
    assert first == second
    assert first.finalization_fingerprint == bundle.fingerprint_sha256
    assert first.embedding_model == "openai:text-embedding-3-small"
    assert first.embedding_dimension == 1536
    assert first.cosine_threshold == 0.85
    assert first.annotators == ("gpt5_6_sol", "claude_sonnet_5", "qwen3_8")
    assert first.to_dict() == second.to_dict()
