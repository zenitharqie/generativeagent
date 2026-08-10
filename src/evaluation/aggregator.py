"""Deterministic MRA aggregation without statistical hypothesis testing."""

from __future__ import annotations

from collections import defaultdict

from src.evaluation.errors import EvaluationAggregationError
from src.evaluation.schemas import (
    ConfigurationMRAAggregate,
    MRAEvaluationResult,
    MRAGroupAggregate,
    RunMRAAggregates,
)


def _mean(records: tuple[MRAEvaluationResult, ...]) -> float:
    if not records:
        raise EvaluationAggregationError("cannot aggregate an empty MRA group")
    return sum(record.mra for record in records) / len(records)


def _group(
    records: tuple[MRAEvaluationResult, ...],
    *,
    dimension: str,
    key_field: str,
) -> tuple[MRAGroupAggregate, ...]:
    order: list[str] = []
    grouped: dict[str, list[MRAEvaluationResult]] = defaultdict(list)
    for record in records:
        key = getattr(record.identity, key_field)
        if key not in grouped:
            order.append(key)
        grouped[key].append(record)
    return tuple(
        MRAGroupAggregate(
            dimension=dimension,
            key=key,
            unit_count=len(grouped[key]),
            mean_mra=sum(item.mra for item in grouped[key]) / len(grouped[key]),
        )
        for key in order
    )


def aggregate_run_mra(
    records: tuple[MRAEvaluationResult, ...],
    *,
    evaluation_mode: str,
) -> RunMRAAggregates:
    """Aggregate one validated run by run, event, and agent."""
    if evaluation_mode not in {"primary", "diagnostic"}:
        raise EvaluationAggregationError(
            f"unsupported evaluation_mode: {evaluation_mode!r}"
        )
    if not isinstance(records, tuple) or not records or any(
        not isinstance(record, MRAEvaluationResult) for record in records
    ):
        raise EvaluationAggregationError(
            "records must be a non-empty tuple of MRAEvaluationResult values"
        )
    first = records[0].identity
    expected_run_identity = (
        first.experiment_id,
        first.run_id,
        first.retrieval_config_id,
        first.repetition_index,
    )
    seen_units: set[tuple[str, str, str]] = set()
    for record in records:
        identity = record.identity
        actual_run_identity = (
            identity.experiment_id,
            identity.run_id,
            identity.retrieval_config_id,
            identity.repetition_index,
        )
        if actual_run_identity != expected_run_identity:
            raise EvaluationAggregationError(
                "MRA records from different runs/configurations cannot be combined"
            )
        unit_key = (identity.event_id, identity.agent_id, identity.turn_id)
        if unit_key in seen_units:
            raise EvaluationAggregationError(
                f"duplicate MRA evaluation unit: {unit_key}"
            )
        seen_units.add(unit_key)
    if evaluation_mode == "primary" and len(records) != 20:
        raise EvaluationAggregationError(
            f"primary run aggregation requires exactly 20 units, got {len(records)}"
        )

    by_event = _group(records, dimension="event", key_field="event_id")
    by_agent = _group(records, dimension="agent", key_field="agent_id")
    if evaluation_mode == "primary":
        if len(by_event) != 5 or any(item.unit_count != 4 for item in by_event):
            raise EvaluationAggregationError(
                "primary run requires five events with four units each"
            )
        if len(by_agent) != 4 or any(item.unit_count != 5 for item in by_agent):
            raise EvaluationAggregationError(
                "primary run requires four agents with five units each"
            )
    return RunMRAAggregates(
        experiment_id=first.experiment_id,
        run_id=first.run_id,
        retrieval_config_id=first.retrieval_config_id,
        repetition_index=first.repetition_index,
        evaluation_mode=evaluation_mode,
        unit_count=len(records),
        run=MRAGroupAggregate(
            dimension="run",
            key=first.run_id,
            unit_count=len(records),
            mean_mra=_mean(records),
        ),
        by_event=by_event,
        by_agent=by_agent,
    )


def aggregate_configuration_mra(
    run_aggregates: tuple[RunMRAAggregates, ...],
) -> ConfigurationMRAAggregate:
    """Macro-average exactly three primary repetition/run means."""
    if (
        not isinstance(run_aggregates, tuple)
        or len(run_aggregates) != 3
        or any(
            not isinstance(item, RunMRAAggregates) for item in run_aggregates
        )
    ):
        raise EvaluationAggregationError(
            "configuration aggregation requires exactly three run aggregates"
        )
    if any(item.evaluation_mode != "primary" for item in run_aggregates):
        raise EvaluationAggregationError(
            "configuration aggregation accepts primary runs only"
        )
    experiments = {item.experiment_id for item in run_aggregates}
    if len(experiments) != 1:
        raise EvaluationAggregationError(
            "configuration aggregation cannot cross experiment_id"
        )
    configs = {item.retrieval_config_id for item in run_aggregates}
    if len(configs) != 1:
        raise EvaluationAggregationError(
            "configuration aggregation cannot mix retrieval_config_id values"
        )
    ordered = tuple(sorted(run_aggregates, key=lambda item: item.repetition_index))
    repetitions = tuple(item.repetition_index for item in ordered)
    if repetitions != (1, 2, 3):
        raise EvaluationAggregationError(
            "configuration aggregation requires distinct repetitions 1, 2, and 3"
        )
    run_ids = tuple(item.run_id for item in ordered)
    if len(run_ids) != len(set(run_ids)):
        raise EvaluationAggregationError(
            "configuration aggregation requires distinct run IDs"
        )
    values = tuple(item.run.mean_mra for item in ordered)
    return ConfigurationMRAAggregate(
        experiment_id=ordered[0].experiment_id,
        retrieval_config_id=ordered[0].retrieval_config_id,
        repetition_indices=repetitions,
        run_ids=run_ids,
        run_mra_values=values,
        mean_mra=sum(values) / 3,
    )


__all__ = ["aggregate_configuration_mra", "aggregate_run_mra"]
