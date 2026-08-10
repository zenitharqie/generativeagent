"""Strict read-only loading and indexing of completed Phase 08 run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.config_loader import (
    ConfigLoadError,
    ConfigValidationError,
    ExperimentConfig,
    load_experiment_config,
)
from src.config_loader.loader import CONFIG_FILES
from src.config_loader.validators import FIXED_EVENT_IDS
from src.domain.agents import AgentProfile, FIXED_AGENT_IDS
from src.domain.simulation import TargetEvent, TargetEventStep
from src.logging_system import RunSummary, RunSummaryError, read_run_summary
from src.logging_system import jsonl_logger as canonical_jsonl

from src.evaluation.errors import (
    EvaluationEligibilityError,
    EvaluationGroundTruthError,
    EvaluationInputError,
    EvaluationJoinError,
)
from src.evaluation.schemas import EvaluationUnitIdentity, SelectedMemoryEvidence


_MANDATORY_LOGS = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)

_TURN_IDENTITY_FIELDS = (
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "step_id",
    "step_type",
    "is_target_event",
    "evaluation_enabled",
    "day",
    "period",
    "timestamp",
    "event_id",
    "turn_id",
    "turn_index",
    "agent_id",
)


class _DuplicateJsonKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TargetEvaluationUnit:
    """Validated joined persisted inputs for one target-event response."""

    identity: EvaluationUnitIdentity
    top_k: int
    candidate_count: int
    selected_memories: tuple[SelectedMemoryEvidence, ...]
    expected_categories: tuple[str, ...]
    response: str
    agent: AgentProfile
    event: TargetEvent


@dataclass(frozen=True)
class LoadedEvaluationRun:
    """Immutable, timeline-ordered index consumed by Phase 09 calculations."""

    run_dir: Path
    summary: RunSummary
    config: ExperimentConfig
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    target_units: tuple[TargetEvaluationUnit, ...]
    simulator_model_identity: str | None


def _load_summary(run_dir: Path) -> RunSummary:
    if run_dir.is_symlink():
        raise EvaluationInputError(
            f"source run directory must not be a symbolic link: {run_dir}"
        )
    if not run_dir.is_dir():
        raise EvaluationInputError(f"source run directory does not exist: {run_dir}")
    try:
        return read_run_summary(run_dir / "run_summary.json")
    except RunSummaryError as exc:
        raise EvaluationInputError(f"invalid run_summary.json: {exc}") from exc


def _eligibility(summary: RunSummary, diagnostic: bool) -> tuple[str, bool, bool]:
    if not isinstance(diagnostic, bool):
        raise EvaluationEligibilityError("diagnostic must be boolean")
    if summary.status == "completed":
        if (
            summary.completed_natural_blocks,
            summary.completed_target_events,
            summary.completed_main_turns,
        ) != (7, 5, 104):
            raise EvaluationEligibilityError(
                "completed run does not satisfy fixed 7/5/104 invariants"
            )
        if summary.error_count != 0:
            raise EvaluationEligibilityError(
                "completed primary run must have error_count equal to zero"
            )
        return "primary", True, True
    if summary.status == "excluded":
        if not diagnostic:
            raise EvaluationEligibilityError(
                "excluded pilot requires explicit diagnostic mode"
            )
        if summary.error_count != 0:
            raise EvaluationEligibilityError(
                "excluded diagnostic pilot must have error_count equal to zero"
            )
        return "diagnostic", False, False
    raise EvaluationEligibilityError(
        f"run status {summary.status!r} is never eligible for evaluation"
    )


def _read_ground_truth(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationGroundTruthError(
            f"missing or invalid canonical config_snapshot/ground_truth.json: {path}"
        )
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_strict_object)
    except _DuplicateJsonKeyError as exc:
        raise EvaluationGroundTruthError(
            f"invalid config_snapshot/ground_truth.json: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationGroundTruthError(
            "invalid JSON in config_snapshot/ground_truth.json: "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise EvaluationGroundTruthError(
            f"could not read config_snapshot/ground_truth.json: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise EvaluationGroundTruthError("ground truth must be a JSON object")
    if value.get("evaluation_step_type") != "target_event":
        raise EvaluationGroundTruthError(
            "ground truth evaluation_step_type must be target_event"
        )
    targets = value.get("targets")
    if not isinstance(targets, list):
        raise EvaluationGroundTruthError("ground_truth.targets must be a list")

    expected_pairs = {
        (event_id, agent_id)
        for event_id in FIXED_EVENT_IDS
        for agent_id in FIXED_AGENT_IDS
    }
    indexed: dict[tuple[str, str], tuple[str, ...]] = {}
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise EvaluationGroundTruthError(
                f"ground truth target {index} must be an object"
            )
        optional = target.get("optional", False)
        if not isinstance(optional, bool):
            raise EvaluationGroundTruthError(
                f"ground truth target {index} optional must be boolean"
            )
        event_id = target.get("event_id")
        agent_id = target.get("agent_id")
        categories = target.get("expected_categories")
        if not isinstance(event_id, str) or not event_id.strip():
            raise EvaluationGroundTruthError(
                f"ground truth target {index} event_id is invalid"
            )
        if agent_id not in FIXED_AGENT_IDS:
            raise EvaluationGroundTruthError(
                f"ground truth target {index} agent_id is invalid"
            )
        if (
            not isinstance(categories, list)
            or not categories
            or any(
                not isinstance(category, str) or not category.strip()
                for category in categories
            )
        ):
            raise EvaluationGroundTruthError(
                f"ground truth target {index} expected_categories must contain strings"
            )
        category_tuple = tuple(categories)
        if len(category_tuple) != len(set(category_tuple)):
            raise EvaluationGroundTruthError(
                f"ground truth target {index} contains duplicate expected categories"
            )
        pair = (event_id, agent_id)
        if optional and pair not in expected_pairs:
            continue
        if pair in indexed:
            raise EvaluationGroundTruthError(
                f"duplicate ground truth event/agent pair: {event_id}/{agent_id}"
            )
        indexed[pair] = category_tuple

    actual_pairs = set(indexed)
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        details: list[str] = []
        if missing:
            details.append(f"missing pairs: {missing}")
        if extra:
            details.append(f"unexpected pairs: {extra}")
        raise EvaluationGroundTruthError(
            "ground truth must cover every target-event/agent pair; "
            + "; ".join(details)
        )
    return indexed


def _load_snapshot(
    run_dir: Path,
) -> tuple[ExperimentConfig, dict[tuple[str, str], tuple[str, ...]]]:
    snapshot = run_dir / "config_snapshot"
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise EvaluationInputError(
            f"missing or invalid config_snapshot directory: {snapshot}"
        )
    expected_names = set(CONFIG_FILES.values())
    try:
        actual_names = {path.name for path in snapshot.iterdir()}
    except OSError as exc:
        raise EvaluationInputError(f"could not inspect config_snapshot: {exc}") from exc
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if "ground_truth.json" in missing:
            raise EvaluationGroundTruthError(
                "missing canonical config_snapshot/ground_truth.json"
            )
        raise EvaluationInputError(
            f"config_snapshot must contain exact canonical files; "
            f"missing={missing}, extra={extra}"
        )
    ground_truth = _read_ground_truth(snapshot / "ground_truth.json")
    try:
        config = load_experiment_config(snapshot)
    except (ConfigLoadError, ConfigValidationError) as exc:
        if "ground truth" in str(exc).lower() or "ground_truth" in str(exc).lower():
            raise EvaluationGroundTruthError(
                f"invalid snapshotted ground truth: {exc}"
            ) from exc
        raise EvaluationInputError(f"invalid config_snapshot: {exc}") from exc
    return config, ground_truth


def _read_logs(run_dir: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    streams: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename in _MANDATORY_LOGS:
        path = run_dir / filename
        if path.is_symlink() or not path.is_file():
            raise EvaluationInputError(f"missing mandatory log file: {filename}")
        try:
            # The Phase 07 reader performs strict UTF-8/LF/JSON/schema validation
            # and typed canonical reconstruction without modifying the stream.
            streams[filename] = canonical_jsonl._read_existing_records(path)
        except Exception as exc:
            raise EvaluationInputError(f"invalid {filename}: {exc}") from exc
    return streams


def _require_run_identity(
    summary: RunSummary,
    streams: Mapping[str, tuple[dict[str, Any], ...]],
) -> None:
    expected = (
        summary.experiment_id,
        summary.run_id,
        summary.retrieval_config_id,
        summary.repetition_index,
    )
    for filename, records in streams.items():
        for index, record in enumerate(records):
            actual = tuple(
                record[field]
                for field in (
                    "experiment_id",
                    "run_id",
                    "retrieval_config_id",
                    "repetition_index",
                )
            )
            if actual != expected:
                raise EvaluationJoinError(
                    f"{filename} record {index} has cross-run identity mismatch"
                )


def _unique_index(
    records: tuple[dict[str, Any], ...],
    field_name: str,
    stream_name: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record[field_name]
        if key in result:
            raise EvaluationJoinError(
                f"duplicate {stream_name} identity {field_name}={key!r}"
            )
        result[key] = record
    return result


def _validate_step_record(record: Mapping[str, Any], configured: object) -> None:
    expected_event_id = getattr(configured, "event_id", None)
    expected = (
        configured.step_id,
        configured.step_type,
        configured.is_target_event,
        configured.evaluation_enabled,
        configured.day,
        configured.period,
        expected_event_id,
    )
    actual = tuple(
        record[field]
        for field in (
            "step_id",
            "step_type",
            "is_target_event",
            "evaluation_enabled",
            "day",
            "period",
            "event_id",
        )
    )
    if actual != expected:
        raise EvaluationJoinError(
            f"turn {record.get('turn_id')!r} does not match snapshotted step metadata"
        )


def _validate_timeline(
    summary: RunSummary,
    config: ExperimentConfig,
    streams: Mapping[str, tuple[dict[str, Any], ...]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    retrievals = streams["retrieval_logs.jsonl"]
    behaviors = streams["behavior_logs.jsonl"]
    events = streams["event_summaries.jsonl"]
    natural = streams["natural_block_summaries.jsonl"]
    if len(retrievals) != summary.completed_main_turns:
        raise EvaluationInputError(
            "retrieval log count does not match completed_main_turns"
        )
    if len(behaviors) != summary.completed_main_turns:
        raise EvaluationInputError(
            "behavior log count does not match completed_main_turns"
        )
    if len(events) != summary.completed_target_events:
        raise EvaluationInputError(
            "event summary count does not match completed_target_events"
        )
    if len(natural) != summary.completed_natural_blocks:
        raise EvaluationInputError(
            "natural summary count does not match completed_natural_blocks"
        )
    if streams["errors.jsonl"]:
        raise EvaluationInputError("eligible run must have an empty errors.jsonl")

    retrieval_by_turn = _unique_index(retrievals, "turn_id", "retrieval")
    behavior_by_turn = _unique_index(behaviors, "turn_id", "behavior")
    event_by_step = _unique_index(events, "step_id", "target-event summary")
    natural_by_step = _unique_index(natural, "step_id", "natural-block summary")
    _unique_index(streams["memory_logs.jsonl"], "memory_id", "memory")
    _unique_index(streams["reflection_logs.jsonl"], "turn_id", "reflection")

    executed_summary_ids = set(event_by_step) | set(natural_by_step)
    if set(event_by_step) & set(natural_by_step):
        raise EvaluationJoinError("a step appears in both summary streams")
    executed_steps = config.steps[: len(executed_summary_ids)]
    if {step.step_id for step in executed_steps} != executed_summary_ids:
        raise EvaluationJoinError(
            "completed step summaries must form an exact snapshotted timeline prefix"
        )

    expected_turn_ids: list[str] = []
    for step in executed_steps:
        summary_record = (
            event_by_step.get(step.step_id) or natural_by_step.get(step.step_id)
        )
        if summary_record is None:
            raise EvaluationJoinError(f"missing step summary for {step.step_id!r}")
        _validate_step_record(summary_record, step)
        expected_turn_ids.extend(summary_record["ordered_main_turn_ids"])
    if tuple(record["turn_id"] for record in retrievals) != tuple(expected_turn_ids):
        raise EvaluationJoinError(
            "retrieval records do not follow exact snapshotted timeline/turn order"
        )
    if tuple(record["turn_id"] for record in behaviors) != tuple(expected_turn_ids):
        raise EvaluationJoinError(
            "behavior records do not follow exact snapshotted timeline/turn order"
        )
    expected_timestamps = tuple(range(1, summary.completed_main_turns + 1))
    if tuple(record["timestamp"] for record in retrievals) != expected_timestamps:
        raise EvaluationJoinError("retrieval logical timestamps are not exact and contiguous")
    if tuple(record["timestamp"] for record in behaviors) != expected_timestamps:
        raise EvaluationJoinError("behavior logical timestamps are not exact and contiguous")

    configured_by_step = {step.step_id: step for step in config.steps}
    for retrieval, behavior in zip(retrievals, behaviors, strict=True):
        if any(retrieval[field] != behavior[field] for field in _TURN_IDENTITY_FIELDS):
            raise EvaluationJoinError(
                f"retrieval/behavior identity mismatch for {retrieval['turn_id']!r}"
            )
        if retrieval["selected_memory_ids"] != behavior["retrieved_memory_ids"]:
            raise EvaluationJoinError(
                "behavior retrieved_memory_ids do not match retrieval selected_memory_ids"
            )
        configured = configured_by_step.get(retrieval["step_id"])
        if configured is None:
            raise EvaluationJoinError(
                f"log references unknown step_id {retrieval['step_id']!r}"
            )
        _validate_step_record(retrieval, configured)

    return event_by_step, natural_by_step


def _selected_evidence(
    retrieval: Mapping[str, Any],
    memory_by_id: Mapping[str, dict[str, Any]],
) -> tuple[SelectedMemoryEvidence, ...]:
    selected_ids = tuple(retrieval["selected_memory_ids"])
    candidates = retrieval["candidates"]
    selected_candidates = tuple(candidates[: len(selected_ids)])
    if tuple(candidate["memory_id"] for candidate in selected_candidates) != selected_ids:
        raise EvaluationJoinError("selected candidate order is inconsistent")
    evidence: list[SelectedMemoryEvidence] = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        memory_id = candidate["memory_id"]
        memory = memory_by_id.get(memory_id)
        if memory is None:
            raise EvaluationJoinError(
                f"selected memory {memory_id!r} is missing from memory_logs.jsonl"
            )
        if memory["agent_id"] != retrieval["agent_id"]:
            raise EvaluationJoinError(
                f"selected memory {memory_id!r} belongs to a different agent"
            )
        if memory["category"] != candidate["category"]:
            raise EvaluationJoinError(
                f"selected memory {memory_id!r} category mismatch"
            )
        if memory["created_at"] >= retrieval["timestamp"]:
            raise EvaluationJoinError(
                f"selected memory {memory_id!r} was not available before retrieval"
            )
        evidence.append(
            SelectedMemoryEvidence(
                retrieval_rank=rank,
                memory_id=memory_id,
                category=candidate["category"],
                text=memory["text"],
            )
        )
    return tuple(evidence)


def _build_target_units(
    summary: RunSummary,
    config: ExperimentConfig,
    ground_truth: Mapping[tuple[str, str], tuple[str, ...]],
    streams: Mapping[str, tuple[dict[str, Any], ...]],
    event_by_step: Mapping[str, dict[str, Any]],
) -> tuple[TargetEvaluationUnit, ...]:
    retrieval_by_turn = {
        record["turn_id"]: record for record in streams["retrieval_logs.jsonl"]
    }
    behavior_by_turn = {
        record["turn_id"]: record for record in streams["behavior_logs.jsonl"]
    }
    memory_by_id = {
        record["memory_id"]: record for record in streams["memory_logs.jsonl"]
    }
    agents = {agent.agent_id: agent for agent in config.agents}
    events = {event.event_id: event for event in config.target_events}
    units: list[TargetEvaluationUnit] = []
    for step in config.steps:
        if not isinstance(step, TargetEventStep) or step.step_id not in event_by_step:
            continue
        event_summary = event_by_step[step.step_id]
        for turn_index, (turn_id, agent_id) in enumerate(
            zip(
                event_summary["ordered_main_turn_ids"],
                step.participants,
                strict=True,
            ),
            start=1,
        ):
            retrieval = retrieval_by_turn.get(turn_id)
            behavior = behavior_by_turn.get(turn_id)
            if retrieval is None or behavior is None:
                raise EvaluationJoinError(
                    f"missing retrieval or behavior record for target turn {turn_id!r}"
                )
            if (
                retrieval["agent_id"] != agent_id
                or retrieval["turn_index"] != turn_index
                or retrieval["event_id"] != step.event_id
                or retrieval["step_type"] != "target_event"
                or retrieval["is_target_event"] is not True
                or retrieval["evaluation_enabled"] is not True
            ):
                raise EvaluationJoinError(
                    f"target turn {turn_id!r} identity/order is inconsistent"
                )
            categories = ground_truth.get((step.event_id, agent_id))
            if categories is None:
                raise EvaluationGroundTruthError(
                    f"missing ground truth for {step.event_id}/{agent_id}"
                )
            identity = EvaluationUnitIdentity(
                experiment_id=summary.experiment_id,
                run_id=summary.run_id,
                retrieval_config_id=summary.retrieval_config_id,
                repetition_index=summary.repetition_index,
                event_id=step.event_id,
                step_id=step.step_id,
                turn_id=turn_id,
                turn_index=turn_index,
                timestamp=retrieval["timestamp"],
                agent_id=agent_id,
            )
            units.append(
                TargetEvaluationUnit(
                    identity=identity,
                    top_k=retrieval["top_k"],
                    candidate_count=retrieval["candidate_count"],
                    selected_memories=_selected_evidence(retrieval, memory_by_id),
                    expected_categories=categories,
                    response=behavior["response"],
                    agent=agents[agent_id],
                    event=events[step.event_id],
                )
            )
    return tuple(units)


def _simulator_model_identity(
    usage_records: tuple[dict[str, Any], ...],
) -> str | None:
    models = tuple(sorted({record["model"] for record in usage_records}))
    if not models:
        return None
    if len(models) != 1:
        raise EvaluationInputError(
            "multiple simulator model identities cannot be represented unambiguously"
        )
    return models[0]


def load_evaluation_run(
    run_dir: str | Path,
    *,
    diagnostic: bool = False,
) -> LoadedEvaluationRun:
    """Load one eligible run entirely from persisted, validated artifacts."""
    try:
        source_dir = Path(run_dir)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"invalid source run directory: {run_dir!r}") from exc
    summary = _load_summary(source_dir)
    mode, primary_eligible, aggregate_eligible = _eligibility(summary, diagnostic)
    config, ground_truth = _load_snapshot(source_dir)
    streams = _read_logs(source_dir)
    _require_run_identity(summary, streams)
    event_by_step, _ = _validate_timeline(summary, config, streams)
    units = _build_target_units(
        summary,
        config,
        ground_truth,
        streams,
        event_by_step,
    )
    expected_units = 20 if mode == "primary" else summary.completed_target_events * 4
    if not units:
        raise EvaluationInputError("evaluation scope is empty")
    if len(units) != expected_units:
        raise EvaluationInputError(
            f"target evaluation unit count must be {expected_units}, got {len(units)}"
        )
    return LoadedEvaluationRun(
        run_dir=source_dir.resolve(),
        summary=summary,
        config=config,
        evaluation_mode=mode,
        primary_eligible=primary_eligible,
        included_in_final_research_aggregation=aggregate_eligible,
        target_units=units,
        simulator_model_identity=_simulator_model_identity(
            streams["api_usage_logs.jsonl"]
        ),
    )


__all__ = ["LoadedEvaluationRun", "TargetEvaluationUnit", "load_evaluation_run"]
