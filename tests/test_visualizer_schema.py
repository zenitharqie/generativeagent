from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from src.visualizer import build_replay, serialize_replay
from src.visualizer.schemas import ReplayCandidate, ReplaySchemaError, thaw_json


def test_replay_schema_is_frozen_and_deeply_detached(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document

    with pytest.raises(FrozenInstanceError):
        document.run.source_status = "completed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        document.source["ground_truth_included"] = True  # type: ignore[index]

    detached = document.to_dict()
    detached["run"]["source_status"] = "forged"
    assert document.run.source_status == "excluded"


def test_replay_schema_contains_fixed_run_identity(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document

    assert document.schema_version == "1.0"
    assert document.run.experiment_id == "phase08_tests"
    assert document.run.run_id == "balanced_rep01_pilot"
    assert document.run.retrieval_config_id == "balanced"
    assert document.run.repetition_index == 1
    assert document.run.source_status == "excluded"
    assert document.run.primary_research_data is False


def test_base_replay_explicitly_excludes_sensitive_domains(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    value = build_replay(run_dir=result.run_paths.run_dir).document.to_dict()

    assert value["source"]["ground_truth_included"] is False
    assert value["source"]["replay_is_research_data"] is False
    assert value["source"]["embeddings_included"] is False
    assert value["source"]["prompts_included"] is False
    assert value["source"]["provider_responses_included"] is False
    assert value["evaluation_overlay"] is None
    assert value["pilot_overlay"] is None


def test_serialization_is_strict_canonical_json(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document
    first = serialize_replay(document)
    second = serialize_replay(document)

    assert first == second
    assert first.endswith(b"\n")
    assert b"NaN" not in first and b"Infinity" not in first
    assert json.loads(first)["schema_version"] == "1.0"


def test_candidate_schema_rejects_non_finite_score() -> None:
    with pytest.raises(ReplaySchemaError, match="finite number"):
        ReplayCandidate(
            rank=1,
            memory_id="mem_teacher_000001",
            category="classroom_norms",
            raw_recency=1.0,
            raw_importance=1.0,
            raw_relevance=1.0,
            normalized_recency=1.0,
            normalized_importance=1.0,
            normalized_relevance=1.0,
            final_score=float("inf"),
            selected=True,
        )


def test_thaw_json_returns_mutable_copy() -> None:
    original = ("one", {"nested": (1, 2)})
    thawed = thaw_json(original)
    assert thawed == ["one", {"nested": [1, 2]}]
    thawed[1]["nested"].append(3)
    assert original == ("one", {"nested": (1, 2)})
