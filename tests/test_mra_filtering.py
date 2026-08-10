"""Target-event-only MRA selection on real Phase 08 artifacts."""

from __future__ import annotations

from src.evaluation import calculate_run_mra, load_evaluation_run


def test_full_run_produces_exactly_twenty_target_event_mra_records(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    records = calculate_run_mra(loaded)

    assert len(records) == 20
    assert all(record.evaluation_mode == "primary" for record in records)
    assert all(record.primary_eligible is True for record in records)
    assert all(
        record.included_in_final_research_aggregation is True for record in records
    )
    assert {record.identity.event_id for record in records} == {
        "event_01",
        "event_02",
        "event_03",
        "event_04",
        "event_05",
    }


def test_natural_blocks_never_enter_mra_records(phase08_full_artifacts) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    records = calculate_run_mra(loaded)

    assert all("target" in record.identity.step_id for record in records)
    assert not any("natural" in record.identity.step_id for record in records)


def test_mra_order_follows_snapshot_timeline_then_participant_order(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    identities = tuple(
        (record.identity.event_id, record.identity.agent_id)
        for record in calculate_run_mra(loaded)
    )

    assert identities == tuple(
        (event_id, agent_id)
        for event_id in (
            "event_01",
            "event_02",
            "event_03",
            "event_04",
            "event_05",
        )
        for agent_id in (
            "teacher",
            "active_student",
            "quiet_student",
            "academic_student",
        )
    )


def test_mra_preserves_exact_retrieval_and_ground_truth_evidence(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    unit = loaded.target_units[0]

    result = calculate_run_mra(loaded)[0]

    assert result.identity == unit.identity
    assert result.selected_memory_ids == tuple(
        memory.memory_id for memory in unit.selected_memories
    )
    assert result.retrieved_categories == tuple(
        memory.category for memory in unit.selected_memories
    )
    assert result.expected_categories == unit.expected_categories
    assert result.top_k == result.denominator == 5


def test_diagnostic_pilot_produces_four_non_primary_target_records(
    phase08_pilot_artifacts,
) -> None:
    loaded = load_evaluation_run(
        phase08_pilot_artifacts[0].run_paths.run_dir,
        diagnostic=True,
    )

    records = calculate_run_mra(loaded)

    assert len(records) == 4
    assert all(record.evaluation_mode == "diagnostic" for record in records)
    assert all(record.primary_eligible is False for record in records)
    assert all(
        record.included_in_final_research_aggregation is False for record in records
    )
    assert {record.identity.event_id for record in records} == {"event_01"}
    assert all("natural" not in record.identity.step_id for record in records)


def test_repeated_run_mra_calculation_is_identical(phase08_full_artifacts) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    assert calculate_run_mra(loaded) == calculate_run_mra(loaded)
    assert tuple(record.to_dict() for record in calculate_run_mra(loaded)) == tuple(
        record.to_dict() for record in calculate_run_mra(loaded)
    )
