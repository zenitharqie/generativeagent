"""Run and three-repetition macro aggregation contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.evaluation import (
    EvaluationAggregationError,
    aggregate_configuration_mra,
    aggregate_run_mra,
    calculate_run_mra,
    load_evaluation_run,
)


def _as_run(base, run_id: str, repetition_index: int, mean_mra=None, **changes):
    run_mean = base.run.mean_mra if mean_mra is None else mean_mra
    return replace(
        base,
        run_id=run_id,
        repetition_index=repetition_index,
        run=replace(base.run, key=run_id, mean_mra=run_mean),
        **changes,
    )


def test_primary_run_aggregate_uses_arithmetic_mean_of_twenty_units(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    records = calculate_run_mra(loaded)

    aggregate = aggregate_run_mra(records, evaluation_mode="primary")

    assert aggregate.unit_count == 20
    assert aggregate.run.mean_mra == sum(record.mra for record in records) / 20


def test_run_aggregate_contains_event_and_agent_breakdowns(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)

    aggregate = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="primary",
    )

    assert tuple(item.key for item in aggregate.by_event) == (
        "event_01",
        "event_02",
        "event_03",
        "event_04",
        "event_05",
    )
    assert all(item.unit_count == 4 for item in aggregate.by_event)
    assert tuple(item.key for item in aggregate.by_agent) == (
        "teacher",
        "active_student",
        "quiet_student",
        "academic_student",
    )
    assert all(item.unit_count == 5 for item in aggregate.by_agent)


def test_primary_run_aggregate_rejects_incomplete_twenty_unit_scope(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    records = calculate_run_mra(loaded)

    with pytest.raises(EvaluationAggregationError, match="20"):
        aggregate_run_mra(records[:-1], evaluation_mode="primary")


def test_diagnostic_run_aggregate_accepts_complete_pilot_target_scope(
    phase08_pilot_artifacts,
) -> None:
    loaded = load_evaluation_run(
        phase08_pilot_artifacts[0].run_paths.run_dir,
        diagnostic=True,
    )

    aggregate = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="diagnostic",
    )

    assert aggregate.unit_count == 4
    assert aggregate.evaluation_mode == "diagnostic"


def test_configuration_aggregate_macro_averages_three_run_means(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    base = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="primary",
    )
    runs = (
        _as_run(base, "balanced_rep01", 1),
        _as_run(base, "balanced_rep02", 2, 0.2),
        _as_run(base, "balanced_rep03", 3, 0.8),
    )

    aggregate = aggregate_configuration_mra(runs)

    assert aggregate.repetition_indices == (1, 2, 3)
    assert aggregate.run_mra_values == (base.run.mean_mra, 0.2, 0.8)
    assert aggregate.mean_mra == sum(aggregate.run_mra_values) / 3


@pytest.mark.parametrize("invalid", ["missing", "duplicate"])
def test_configuration_aggregate_requires_exactly_three_distinct_repetitions(
    phase08_full_artifacts,
    invalid: str,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    base = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="primary",
    )
    runs = (
        _as_run(base, "balanced_rep01", 1),
        _as_run(base, "balanced_rep02", 2),
        _as_run(
            base,
            "balanced_rep03",
            2 if invalid == "duplicate" else 3,
        ),
    )
    if invalid == "missing":
        runs = runs[:2]

    with pytest.raises(EvaluationAggregationError, match="three|repetition"):
        aggregate_configuration_mra(runs)


def test_configuration_aggregate_rejects_mixed_retrieval_configs(
    phase08_full_artifacts,
) -> None:
    loaded = load_evaluation_run(phase08_full_artifacts[0].run_paths.run_dir)
    base = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="primary",
    )
    runs = (
        _as_run(base, "balanced_rep01", 1),
        _as_run(base, "balanced_rep02", 2),
        _as_run(
            base,
            "foreign_rep03",
            3,
            retrieval_config_id="recency_heavy",
        ),
    )

    with pytest.raises(EvaluationAggregationError, match="retrieval_config"):
        aggregate_configuration_mra(runs)


def test_configuration_aggregate_rejects_diagnostic_runs(
    phase08_pilot_artifacts,
) -> None:
    loaded = load_evaluation_run(
        phase08_pilot_artifacts[0].run_paths.run_dir,
        diagnostic=True,
    )
    base = aggregate_run_mra(
        calculate_run_mra(loaded),
        evaluation_mode="diagnostic",
    )

    with pytest.raises(EvaluationAggregationError, match="primary"):
        aggregate_configuration_mra(
            (
                _as_run(base, "rep01", 1),
                _as_run(base, "rep02", 2),
                _as_run(base, "rep03", 3),
            )
        )
