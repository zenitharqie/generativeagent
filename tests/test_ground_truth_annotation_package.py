from __future__ import annotations

import json

from src.evaluation.ground_truth import build_annotation_package


def test_annotation_package_has_exact_deterministic_condition_blind_units() -> None:
    first = build_annotation_package()
    second = build_annotation_package()
    assert first.payload == second.payload
    assert first.package_sha256 == second.package_sha256
    assert len(first.value["units"]) == 20
    assert [unit["annotation_unit_id"] for unit in first.value["units"]][:4] == [
        "gt_event_01_teacher",
        "gt_event_01_active_student",
        "gt_event_01_quiet_student",
        "gt_event_01_academic_student",
    ]
    forbidden = {
        "retrieval_config_id",
        "retrieval_weights",
        "repetition",
        "experiment_id",
        "retrieved_memories",
        "retrieval_scores",
        "top_k",
        "mra",
        "bcr",
        "ground_truth",
        "majority_vote",
        "expected_memory_themes",
        "evaluation_role",
    }
    serialized = json.loads(first.payload)
    for unit in serialized["units"]:
        assert forbidden.isdisjoint(unit)
        assert unit["necessity_test"]["relevant_value"] == 1
        assert unit["necessity_test"]["not_relevant_value"] == 0
        assert set(unit["canonical_persona"]) == {
            "agent_id",
            "display_name",
            "role",
            "archetype",
            "persona_summary",
            "traits",
            "goals",
            "communication_style",
            "motivations",
        }


def test_annotation_package_hash_changes_if_methodological_input_changes() -> None:
    package = build_annotation_package()
    value = dict(package.value)
    value["units"] = list(value["units"])
    value["units"][0] = dict(value["units"][0])
    value["units"][0]["canonical_event_description"] = "changed"
    assert value != package.value
