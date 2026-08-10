"""Neutral, target-only Phase 09 BCR export construction."""

from __future__ import annotations

import json

from src.evaluation import build_bcr_export, load_evaluation_run


def _full_records(phase08_full_artifacts):
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    return loaded, build_bcr_export(loaded)


def test_full_run_exports_exactly_twenty_target_event_rows(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)

    assert len(records) == 20
    assert all("target" in record.identity.step_id for record in records)
    assert not any("natural" in record.identity.step_id for record in records)


def test_prior_context_is_all_earlier_same_agent_target_behavior_only(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)

    for agent_id in (
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
    ):
        agent_records = tuple(
            record for record in records if record.identity.agent_id == agent_id
        )
        assert tuple(len(record.prior_behaviors) for record in agent_records) == (
            0,
            1,
            2,
            3,
            4,
        )
        for index, record in enumerate(agent_records):
            assert tuple(prior.response for prior in record.prior_behaviors) == tuple(
                earlier.judge_input.response for earlier in agent_records[:index]
            )


def test_first_target_event_prior_context_is_empty_for_every_agent(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)

    event_one = tuple(
        record for record in records if record.identity.event_id == "event_01"
    )

    assert len(event_one) == 4
    assert all(record.judge_input.prior_target_behaviors == () for record in event_one)


def test_selected_memory_texts_preserve_retrieval_order(
    phase08_full_artifacts,
) -> None:
    loaded, records = _full_records(phase08_full_artifacts)

    for unit, record in zip(loaded.target_units, records, strict=True):
        assert record.judge_input.selected_memory_texts == tuple(
            memory.text for memory in unit.selected_memories
        )
        assert tuple(
            evidence.memory_id for evidence in record.selected_memories
        ) == tuple(memory.memory_id for memory in unit.selected_memories)


def test_judge_input_contains_exact_persona_event_and_response(
    phase08_full_artifacts,
) -> None:
    loaded, records = _full_records(phase08_full_artifacts)
    unit = loaded.target_units[0]
    record = records[0]
    payload = record.judge_input.to_dict()

    assert payload["persona"] == {
        "display_name": unit.agent.display_name,
        "role": unit.agent.role,
        "archetype": unit.agent.archetype,
        "persona_summary": unit.agent.persona_summary,
        "traits": list(unit.agent.traits),
        "goals": list(unit.agent.goals),
        "communication_style": unit.agent.communication_style,
        "motivations": list(unit.agent.motivations),
    }
    assert payload["target_event"] == {
        "title": unit.event.title,
        "description": unit.event.description,
    }
    assert payload["response"] == unit.response


def test_outer_audit_metadata_contains_identity_memory_ids_and_categories(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)
    payload = records[0].to_dict()

    for field in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
        "event_id",
        "step_id",
        "turn_id",
        "turn_index",
        "timestamp",
        "agent_id",
    ):
        assert field in payload["audit_metadata"]
    assert set(payload["selected_memory_evidence"][0]) == {
        "retrieval_rank",
        "memory_id",
        "category",
    }


def test_judge_input_is_blind_to_condition_mra_ground_truth_and_memory_labels(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)

    for record in records:
        payload = json.dumps(record.judge_input.to_dict(), sort_keys=True).lower()
        for forbidden in (
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "repetition_index",
            "memory_id",
            "category",
            "raw_recency",
            "normalized_recency",
            "final_score",
            "weights",
            "mra",
            "ground_truth",
            "expected_categories",
            "matched",
        ):
            assert forbidden not in payload


def test_future_evaluator_metadata_is_outside_judge_input_and_unpopulated(
    phase08_full_artifacts,
) -> None:
    _, records = _full_records(phase08_full_artifacts)
    payload = records[0].to_dict()

    assert payload["model_metadata"] == {
        "simulator_model_identity": None,
        "evaluator_model_identity": None,
        "evaluator_prompt_version": None,
        "evaluator_rubric_version": None,
    }
    assert "model_metadata" not in payload["judge_input"]


def test_diagnostic_pilot_exports_four_non_primary_rows(
    phase08_pilot_artifacts,
) -> None:
    loaded = load_evaluation_run(
        phase08_pilot_artifacts[0].run_paths.run_dir,
        diagnostic=True,
    )

    records = build_bcr_export(loaded)

    assert len(records) == 4
    assert all(record.evaluation_mode == "diagnostic" for record in records)
    assert all(record.primary_eligible is False for record in records)
    assert all(
        record.included_in_final_research_aggregation is False for record in records
    )


def test_repeated_bcr_export_construction_is_deterministic(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    assert build_bcr_export(loaded) == build_bcr_export(loaded)
    assert tuple(record.to_dict() for record in build_bcr_export(loaded)) == tuple(
        record.to_dict() for record in build_bcr_export(loaded)
    )
