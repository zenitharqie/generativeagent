"""Deterministic Phase 12 timeline reconstruction from persisted artifacts only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any

from src.visualizer.errors import (
    ReplayIdentityError,
    ReplayJoinError,
    ReplayOrderingError,
    ReplaySchemaError,
)
from src.visualizer.readers import LoadedRunArtifacts
from src.visualizer.schemas import (
    ReplayAgent,
    ReplayCandidate,
    ReplayDocument,
    ReplayFailure,
    ReplayFrame,
    ReplayIssue,
    ReplayMemory,
    ReplayReflection,
    ReplayRetrieval,
    ReplayRun,
    ReplayStep,
    ReplayTurn,
    ReplayUsageRecord,
)


# This is a projection of the frozen Phase 08 turn-policy contract. Importing
# the execution policy would pull replay into the simulation dependency graph.
FIXED_NATURAL_TURN_ORDER = (
    "teacher",
    "active_student",
    "academic_student",
    "quiet_student",
    "active_student",
    "teacher",
    "academic_student",
    "quiet_student",
    "academic_student",
    "active_student",
    "quiet_student",
    "teacher",
)

_SAFE_PILOT_MESSAGES = frozenset(
    {
        "Pilot run interrupted.",
        "Pilot run failed; inspect the canonical error type and attempt records.",
    }
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


@dataclass(frozen=True)
class _ExpectedTurn:
    global_index: int
    step_index: int
    step_id: str
    step_type: str
    event_id: str | None
    day: int
    period: str
    turn_id: str
    turn_index: int
    agent_id: str
    timestamp: int


@dataclass(frozen=True)
class _ExpectedReflection:
    global_index: int
    step_index: int
    step_id: str
    step_type: str
    event_id: str | None
    day: int
    period: str
    operation_id: str
    participant_index: int
    agent_id: str
    timestamp: int


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ReplaySchemaError(f"{field} is not a safe identifier")
    return value


def _require_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplaySchemaError(f"{field} must be an integer >= {minimum}")
    return value


def _require_number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplaySchemaError(f"{field} must be numeric")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReplaySchemaError(f"{field} must be boolean")
    return value


def _metadata_tuple(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record[name]
        for name in (
            "step_id",
            "step_type",
            "event_id",
            "day",
            "period",
            "turn_id",
            "turn_index",
            "agent_id",
            "timestamp",
        )
    )


def _expected_metadata(turn: _ExpectedTurn) -> tuple[Any, ...]:
    return (
        turn.step_id,
        turn.step_type,
        turn.event_id,
        turn.day,
        turn.period,
        turn.turn_id,
        turn.turn_index,
        turn.agent_id,
        turn.timestamp,
    )


def _expected_reflection_metadata(item: _ExpectedReflection) -> tuple[Any, ...]:
    return (
        item.step_id,
        item.step_type,
        item.event_id,
        item.day,
        item.period,
        item.operation_id,
        item.participant_index,
        item.agent_id,
        item.timestamp,
    )


def _build_expected_schedule(
    configured_steps: Sequence[Mapping[str, Any]],
) -> tuple[tuple[_ExpectedTurn, ...], tuple[_ExpectedReflection, ...]]:
    turns: list[_ExpectedTurn] = []
    reflections: list[_ExpectedReflection] = []
    timestamp = 0
    for step_index, step in enumerate(configured_steps):
        step_type = step["step_type"]
        if step_type == "natural_interaction_block":
            speakers = FIXED_NATURAL_TURN_ORDER
        else:
            speakers = tuple(step["participants"])
        for turn_index, agent_id in enumerate(speakers, start=1):
            timestamp += 1
            turns.append(
                _ExpectedTurn(
                    global_index=len(turns),
                    step_index=step_index,
                    step_id=step["step_id"],
                    step_type=step_type,
                    event_id=step.get("event_id"),
                    day=step["day"],
                    period=step["period"],
                    turn_id=f"{step['step_id']}_turn_{turn_index:02d}",
                    turn_index=turn_index,
                    agent_id=agent_id,
                    timestamp=timestamp,
                )
            )
        final_timestamp = timestamp
        for participant_index, agent_id in enumerate(step["participants"], start=1):
            reflections.append(
                _ExpectedReflection(
                    global_index=len(reflections),
                    step_index=step_index,
                    step_id=step["step_id"],
                    step_type=step_type,
                    event_id=step.get("event_id"),
                    day=step["day"],
                    period=step["period"],
                    operation_id=f"{step['step_id']}_reflection_{participant_index:02d}",
                    participant_index=participant_index,
                    agent_id=agent_id,
                    timestamp=final_timestamp,
                )
            )
    if len(turns) != 104 or timestamp != 104 or len(reflections) != 48:
        raise ReplaySchemaError("snapshotted schedule does not derive 104 turns and 48 reflections")
    return tuple(turns), tuple(reflections)


def _validate_prefix(
    records: Sequence[Mapping[str, Any]],
    expected: Sequence[_ExpectedTurn],
    *,
    filename: str,
) -> None:
    if len(records) > len(expected):
        raise ReplayOrderingError(f"{filename} exceeds the configured turn schedule")
    for index, record in enumerate(records):
        if _metadata_tuple(record) != _expected_metadata(expected[index]):
            raise ReplayOrderingError(
                f"{filename} is not a deterministic prefix at record {index + 1}"
            )
        if record["is_target_event"] is not (expected[index].step_type == "target_event"):
            raise ReplayIdentityError(f"{filename} target-event flag is inconsistent")
        if record["evaluation_enabled"] is not (
            expected[index].step_type == "target_event"
        ):
            raise ReplayIdentityError(f"{filename} evaluation flag is inconsistent")


def _validate_reflection_prefix(
    records: Sequence[Mapping[str, Any]],
    expected: Sequence[_ExpectedReflection],
) -> None:
    if len(records) > len(expected):
        raise ReplayOrderingError("reflection logs exceed the configured schedule")
    for index, record in enumerate(records):
        if _metadata_tuple(record) != _expected_reflection_metadata(expected[index]):
            raise ReplayOrderingError(
                f"reflection logs are not a deterministic prefix at record {index + 1}"
            )


def _validate_step_metadata(
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    *,
    timestamp: int,
) -> None:
    expected = (
        step["step_id"],
        step["step_type"],
        step.get("event_id"),
        step["day"],
        step["period"],
        timestamp,
        step["step_type"] == "target_event",
        step["step_type"] == "target_event",
    )
    actual = tuple(
        record[name]
        for name in (
            "step_id",
            "step_type",
            "event_id",
            "day",
            "period",
            "timestamp",
            "is_target_event",
            "evaluation_enabled",
        )
    )
    if actual != expected:
        raise ReplayIdentityError("structural summary metadata differs from its configured step")


def _validate_summary_streams(
    streams: Mapping[str, Sequence[Mapping[str, Any]]],
    steps: Sequence[Mapping[str, Any]],
    turns: Sequence[_ExpectedTurn],
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    natural_expected = [item for item in steps if item["step_type"] == "natural_interaction_block"]
    target_expected = [item for item in steps if item["step_type"] == "target_event"]
    for filename, expected_steps in (
        ("natural_block_summaries.jsonl", natural_expected),
        ("event_summaries.jsonl", target_expected),
    ):
        records = streams[filename]
        if len(records) > len(expected_steps):
            raise ReplayOrderingError(f"{filename} has too many records")
        for index, record in enumerate(records):
            step = expected_steps[index]
            step_turns = [item for item in turns if item.step_id == step["step_id"]]
            final_timestamp = step_turns[-1].timestamp
            _validate_step_metadata(record, step, timestamp=final_timestamp)
            expected_ids = [item.turn_id for item in step_turns]
            if record["ordered_main_turn_ids"] != expected_ids:
                raise ReplayOrderingError("structural summary main-turn order is corrupt")
            if record["participant_ids"] != step["participants"]:
                raise ReplayOrderingError("structural summary participant order is corrupt")
            expected_count = 12 if step["step_type"] == "natural_interaction_block" else 4
            expected_per_agent = (
                {agent_id: 3 for agent_id in step["participants"]}
                if expected_count == 12
                else {agent_id: 1 for agent_id in step["participants"]}
            )
            if (
                record["completed_main_turn_count"] != expected_count
                or record["reflection_count"] != 4
                or record["final_logical_timestamp"] != final_timestamp
                or record["per_agent_main_turn_counts"] != expected_per_agent
            ):
                raise ReplaySchemaError("structural summary counts are inconsistent")
            if step["step_type"] == "target_event" and (
                record["ordered_responding_agent_ids"] != step["participants"]
                or record["response_count"] != 4
            ):
                raise ReplayOrderingError("target-event response order is corrupt")
            if step["step_id"] in by_id:
                raise ReplayJoinError("duplicate structural step summary")
            by_id[step["step_id"]] = record
    configured_prefix = tuple(
        step["step_id"] for step in steps if step["step_id"] in by_id
    )
    if configured_prefix != tuple(step["step_id"] for step in steps[: len(by_id)]):
        raise ReplayOrderingError("structural summaries do not preserve configured step order")
    return by_id


def _build_agents(config: Mapping[str, Any]) -> tuple[ReplayAgent, ...]:
    return tuple(
        ReplayAgent(
            agent_id=item["agent_id"],
            display_name=item["display_name"],
            role=item["role"],
            archetype=item["archetype"],
            persona_summary=item["persona_summary"],
            traits=tuple(item["traits"]),
            goals=tuple(item["goals"]),
            communication_style=item["communication_style"],
            motivations=tuple(item["motivations"]),
        )
        for item in config["agents"]
    )


def _validate_memory_records(
    records: Sequence[Mapping[str, Any]],
    *,
    agents: Sequence[Mapping[str, Any]],
    turns: Sequence[_ExpectedTurn],
    reflections: Sequence[_ExpectedReflection],
    status: str,
) -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    memory_index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        memory_id = _require_identifier(record["memory_id"], "memory_id")
        if memory_id in memory_index:
            raise ReplayJoinError("duplicate memory ID in persisted memory history")
        memory_index[memory_id] = record
        if record["created_at"] != record["timestamp"]:
            raise ReplayIdentityError("memory created_at differs from persisted timestamp")
        if record["last_accessed_at"] < record["created_at"]:
            raise ReplaySchemaError("memory access time precedes its creation")
        if record["embedding"] is not None:
            # Embeddings are deliberately validated only as opaque numeric arrays
            # and are never copied into the replay domain.
            if not record["embedding"] or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in record["embedding"]
            ):
                raise ReplaySchemaError("persisted memory embedding is malformed")

    seeds = tuple(record for record in records if record["memory_type"] == "seed_persona")
    actions = tuple(record for record in records if record["memory_type"] == "action")
    reflection_memories = tuple(
        record for record in records if record["memory_type"] == "reflection"
    )
    if len(seeds) + len(actions) + len(reflection_memories) != len(records):
        raise ReplaySchemaError("memory history contains an unsupported memory type")

    expected_seeds = [
        (agent["agent_id"], seed_index, seed)
        for agent in agents
        for seed_index, seed in enumerate(agent["seed_memories"], start=1)
    ]
    if len(seeds) > len(expected_seeds):
        raise ReplayOrderingError("seed memory history exceeds configured seeds")
    for index, record in enumerate(seeds):
        agent_id, seed_index, seed = expected_seeds[index]
        expected_id = f"mem_{agent_id}_{seed_index:06d}"
        if (
            record["memory_id"] != expected_id
            or record["agent_id"] != agent_id
            or record["text"] != seed["text"]
            or record["category"] != seed["category"]
            or record["importance"] != seed["importance"]
            or record["created_at"] != 0
            or record["last_accessed_at"] != 0
            or record["step_id"] != "seed_memory_initialization"
            or record["step_type"] != "agent_initialization"
            or record["turn_id"] is not None
            or record["turn_index"] is not None
            or record["metadata"] != {"seed_index": seed_index}
        ):
            raise ReplayIdentityError("seed memory history is not the configured prefix")
    if status in {"completed", "excluded"} and len(seeds) != 12:
        raise ReplayJoinError("successful source run does not contain all seed memories")

    if len(actions) > len(turns):
        raise ReplayOrderingError("action memory history exceeds configured turns")
    for index, record in enumerate(actions):
        expected = turns[index]
        if _metadata_tuple(record) != _expected_metadata(expected):
            raise ReplayOrderingError("action memory history is not a main-turn prefix")
        if record["agent_id"] != expected.agent_id or record["step_type"] != expected.step_type:
            raise ReplayIdentityError("action memory ownership is inconsistent")

    if len(reflection_memories) > len(reflections):
        raise ReplayOrderingError("reflection memory history exceeds configured reflections")
    for index, record in enumerate(reflection_memories):
        expected = reflections[index]
        if _metadata_tuple(record) != _expected_reflection_metadata(expected):
            raise ReplayOrderingError("reflection memory history is not a reflection prefix")
        if record["agent_id"] != expected.agent_id:
            raise ReplayIdentityError("reflection memory ownership is inconsistent")

    if len(seeds) == 12:
        expected_tail: list[tuple[str, tuple[Any, ...]]] = []
        for step_index in range(12):
            expected_tail.extend(
                ("action", _expected_metadata(item))
                for item in turns
                if item.step_index == step_index
            )
            expected_tail.extend(
                ("reflection", _expected_reflection_metadata(item))
                for item in reflections
                if item.step_index == step_index
            )
        actual_tail = records[12:]
        if len(actual_tail) > len(expected_tail):
            raise ReplayOrderingError("memory history exceeds the fixed operation schedule")
        for index, record in enumerate(actual_tail):
            expected_type, expected_metadata = expected_tail[index]
            if record["memory_type"] != expected_type or _metadata_tuple(record) != expected_metadata:
                raise ReplayOrderingError(
                    "memory history does not preserve action/reflection persistence order"
                )
    elif len(records) != len(seeds):
        raise ReplayOrderingError("simulation memory appears before seed initialization completed")

    return memory_index, seeds, actions, reflection_memories


def _validate_retrieval(
    record: Mapping[str, Any],
    *,
    memory_index: Mapping[str, Mapping[str, Any]],
) -> ReplayRetrieval:
    if set(record["weights"]) != {"recency", "importance", "relevance"}:
        raise ReplaySchemaError("retrieval weight fields are invalid")
    weights = {
        name: _require_number(record["weights"][name], f"weights.{name}")
        for name in ("recency", "importance", "relevance")
    }
    top_k = _require_integer(record["top_k"], "top_k", minimum=1)
    if top_k != 5:
        raise ReplaySchemaError("persisted Top-K differs from the fixed value 5")
    candidates: list[ReplayCandidate] = []
    candidate_ids: list[str] = []
    for rank, candidate in enumerate(record["candidates"], start=1):
        memory_id = _require_identifier(candidate["memory_id"], "candidate memory_id")
        if memory_id in candidate_ids:
            raise ReplayJoinError("retrieval candidates contain a duplicate memory ID")
        source_memory = memory_index.get(memory_id)
        if source_memory is None:
            raise ReplayJoinError("retrieval candidate references a missing memory")
        if (
            source_memory["agent_id"] != record["agent_id"]
            or source_memory["category"] != candidate["category"]
            or source_memory["created_at"] >= record["timestamp"]
        ):
            raise ReplayJoinError("retrieval candidate identity is inconsistent with memory history")
        for name in (
            "normalized_recency",
            "normalized_importance",
            "normalized_relevance",
        ):
            normalized = _require_number(candidate[name], name)
            if not 0 <= normalized <= 1:
                raise ReplaySchemaError("persisted normalized retrieval score is out of range")
        candidate_ids.append(memory_id)
        candidates.append(
            ReplayCandidate(
                rank=rank,
                memory_id=memory_id,
                category=candidate["category"],
                raw_recency=_require_number(candidate["raw_recency"], "raw_recency"),
                raw_importance=_require_number(candidate["raw_importance"], "raw_importance"),
                raw_relevance=_require_number(candidate["raw_relevance"], "raw_relevance"),
                normalized_recency=candidate["normalized_recency"],
                normalized_importance=candidate["normalized_importance"],
                normalized_relevance=candidate["normalized_relevance"],
                final_score=_require_number(candidate["final_score"], "final_score"),
                selected=_require_bool(candidate["selected"], "selected"),
            )
        )
    selected = tuple(record["selected_memory_ids"])
    if any(
        current.final_score < following.final_score
        for current, following in zip(candidates, candidates[1:])
    ):
        raise ReplayOrderingError("persisted retrieval candidate scores contradict rank order")
    expected_selected_count = min(top_k, len(candidate_ids))
    if len(selected) != expected_selected_count or len(set(selected)) != len(selected):
        raise ReplayOrderingError("persisted retrieval Top-K selection is inconsistent")
    rank_position = {item.memory_id: index for index, item in enumerate(candidates)}
    if any(memory_id not in rank_position for memory_id in selected):
        raise ReplayOrderingError("persisted retrieval Top-K selection is inconsistent")
    selected_positions = [rank_position[memory_id] for memory_id in selected]
    if selected_positions != sorted(selected_positions):
        raise ReplayOrderingError("persisted retrieval Top-K selection is inconsistent")
    flagged = tuple(item.memory_id for item in candidates if item.selected)
    if selected != flagged:
        raise ReplayOrderingError("persisted retrieval Top-K selection is inconsistent")
    accessed = tuple(record["accessed_memory_ids"])
    if accessed != selected:
        raise ReplayOrderingError("persisted retrieval access order differs from selected Top-K")
    return ReplayRetrieval(
        query=record["query"],
        weights=weights,
        top_k=top_k,
        candidate_count=record["candidate_count"],
        candidates=tuple(candidates),
        selected_memory_ids=selected,
        accessed_memory_ids=accessed,
    )


def _usage_records(
    records: Sequence[Mapping[str, Any]],
    expected_turn_ids: Mapping[str, tuple[Any, ...]],
) -> tuple[tuple[ReplayUsageRecord, ...], Mapping[str, tuple[ReplayUsageRecord, ...]]]:
    all_usage: list[ReplayUsageRecord] = []
    grouped: dict[str, list[ReplayUsageRecord]] = defaultdict(list)
    for record in records:
        turn_id = record["turn_id"]
        expected = expected_turn_ids.get(turn_id)
        if expected is None or _metadata_tuple(record) != expected:
            raise ReplayJoinError("API usage references an unknown or inconsistent operation")
        input_tokens = _require_integer(record["input_tokens"], "input_tokens")
        output_tokens = _require_integer(record["output_tokens"], "output_tokens")
        total_tokens = _require_integer(record["total_tokens"], "total_tokens")
        if total_tokens != input_tokens + output_tokens:
            raise ReplaySchemaError("persisted API total tokens do not equal input plus output")
        item = ReplayUsageRecord(
            turn_id=turn_id,
            call_type=record["call_type"],
            model=record["model"],
            prompt_template_id=record["prompt_template"],
            status=record["status"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            error_type=record["error_type"],
        )
        all_usage.append(item)
        grouped[turn_id].append(item)
    return tuple(all_usage), MappingProxyType(
        {key: tuple(value) for key, value in grouped.items()}
    )


def _safe_failure_message(record: Mapping[str, Any] | None, status: str) -> str:
    if record is not None and record["message"] in _SAFE_PILOT_MESSAGES:
        return record["message"]
    if status == "interrupted":
        return "Run interrupted; inspect the canonical error type and persisted prefix."
    return "Run failed; inspect the canonical error type and persisted prefix."


def _build_failure(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[ReplayFailure | None, tuple[ReplayIssue, ...]]:
    status = summary["status"]
    if status not in {"failed", "interrupted"}:
        if records or summary["error_count"] != 0:
            raise ReplaySchemaError("successful source status contains terminal error evidence")
        return None, ()
    if len(records) > 1:
        raise ReplaySchemaError("source run contains multiple terminal error records")
    record = records[0] if records else None
    if record is None:
        return (
            ReplayFailure(
                error_type="UnrecordedTerminalFailure",
                safe_message=_safe_failure_message(None, status),
                recoverable=False,
                retry_count=None,
                step_id=None,
                turn_id=None,
                logical_timestamp=None,
            ),
            (
                ReplayIssue(
                    code="missing_terminal_error_record",
                    severity="incomplete",
                    scope="run",
                    message="The terminal run summary exists, but errors.jsonl has no record.",
                ),
            ),
        )
    return (
        ReplayFailure(
            error_type=_require_identifier(record["error_type"], "error_type"),
            safe_message=_safe_failure_message(record, status),
            recoverable=record["recoverable"],
            retry_count=record["retry_count"],
            step_id=record["step_id"],
            turn_id=record["turn_id"],
            logical_timestamp=record["timestamp"],
        ),
        (),
    )


def _memory_projection(
    records: Sequence[Mapping[str, Any]],
    access_timestamps: Mapping[str, list[int]],
) -> tuple[ReplayMemory, ...]:
    result: list[ReplayMemory] = []
    for record in records:
        seed_index = None
        if record["memory_type"] == "seed_persona":
            seed_index = record["metadata"]["seed_index"]
        result.append(
            ReplayMemory(
                memory_id=record["memory_id"],
                owner_agent_id=record["agent_id"],
                memory_type=record["memory_type"],
                category=record["category"],
                text=record["text"],
                importance=record["importance"],
                created_at=record["created_at"],
                last_accessed_at_at_persistence=record["last_accessed_at"],
                source_step_id=record["step_id"],
                source_step_type=record["step_type"],
                source_turn_id=record["turn_id"],
                source_turn_index=record["turn_index"],
                seed_index=seed_index,
                persisted_access_timestamps=tuple(access_timestamps.get(record["memory_id"], ())),
            )
        )
    return tuple(result)


def _scope(summary: Mapping[str, Any], evidence_step_indexes: set[int]) -> tuple[str, int]:
    if summary["status"] == "completed":
        return "day_1_through_6", 12
    if summary["status"] == "excluded":
        return "day_1_through_2_excluded_pilot", 4
    if any(index >= 4 for index in evidence_step_indexes) or (
        summary["completed_natural_blocks"] > 3
        or summary["completed_target_events"] > 1
        or summary["completed_main_turns"] > 40
    ):
        return "day_1_through_6_partial", 12
    return "unknown_full_or_day_1_through_2_partial", 12


def reconstruct_replay(
    run: LoadedRunArtifacts,
    *,
    evaluation_overlay: Mapping[str, Any] | None = None,
    pilot_overlay: Mapping[str, Any] | None = None,
) -> ReplayDocument:
    """Build one immutable replay without invoking any execution component."""
    steps_config = tuple(run.config["simulation_steps.json"]["steps"])
    agents_config = tuple(run.config["agents.json"]["agents"])
    expected_turns, expected_reflections = _build_expected_schedule(steps_config)
    streams = run.streams

    retrieval_records = streams["retrieval_logs.jsonl"]
    behavior_records = streams["behavior_logs.jsonl"]
    reflection_records = streams["reflection_logs.jsonl"]
    _validate_prefix(retrieval_records, expected_turns, filename="retrieval_logs.jsonl")
    _validate_prefix(behavior_records, expected_turns, filename="behavior_logs.jsonl")
    _validate_reflection_prefix(reflection_records, expected_reflections)
    summary_by_step = _validate_summary_streams(streams, steps_config, expected_turns)

    memory_index, seeds, action_records, reflection_memory_records = _validate_memory_records(
        streams["memory_logs.jsonl"],
        agents=agents_config,
        turns=expected_turns,
        reflections=expected_reflections,
        status=run.summary["status"],
    )
    if not (
        0 <= len(retrieval_records) - len(behavior_records) <= 1
        and 0 <= len(behavior_records) - len(action_records) <= 1
        and run.summary["completed_main_turns"] <= len(action_records)
        and len(action_records) - run.summary["completed_main_turns"] <= 1
    ):
        raise ReplayJoinError("main-turn persistence prefixes contradict the run summary")
    if not 0 <= len(reflection_memory_records) - len(reflection_records) <= 1:
        raise ReplayJoinError("reflection log and reflection-memory persistence order is invalid")

    for index, record in enumerate(reflection_records):
        memory = reflection_memory_records[index]
        if (
            record["memory_id"] != memory["memory_id"]
            or record["reflection"] != memory["text"]
            or record["category"] != memory["category"]
        ):
            raise ReplayJoinError("reflection log does not match its persisted memory")

    expected_operation_metadata = {
        item.turn_id: _expected_metadata(item) for item in expected_turns
    }
    expected_operation_metadata.update(
        {
            item.operation_id: _expected_reflection_metadata(item)
            for item in expected_reflections
        }
    )
    usage, usage_by_operation = _usage_records(
        streams["api_usage_logs.jsonl"], expected_operation_metadata
    )

    access_timestamps: dict[str, list[int]] = defaultdict(list)
    replay_retrievals: list[ReplayRetrieval] = []
    active_configs = [
        item
        for item in run.config["retrieval_configs.json"]["configurations"]
        if item["config_id"] == run.summary["retrieval_config_id"]
    ]
    if len(active_configs) != 1:
        raise ReplayIdentityError("source retrieval configuration is not uniquely snapshotted")
    active_weights = active_configs[0]["weights"]
    for record in retrieval_records:
        if record["weights"] != active_weights:
            raise ReplayIdentityError(
                "persisted retrieval weights differ from the active snapshotted configuration"
            )
        projected = _validate_retrieval(record, memory_index=memory_index)
        replay_retrievals.append(projected)
        for memory_id in projected.accessed_memory_ids:
            access_timestamps[memory_id].append(record["timestamp"])

    for index, behavior in enumerate(behavior_records):
        if tuple(behavior["retrieved_memory_ids"]) != replay_retrievals[index].selected_memory_ids:
            raise ReplayJoinError("behavior selected memories differ from persisted retrieval")

    max_turn_evidence = max(
        (len(retrieval_records), len(behavior_records), len(action_records)),
        default=0,
    )
    if run.summary["status"] in {"completed", "excluded"} and max_turn_evidence != run.summary[
        "completed_main_turns"
    ]:
        raise ReplayJoinError("successful source run has incomplete turn artifacts")

    turns_by_step: dict[str, list[ReplayTurn]] = defaultdict(list)
    evidence_step_indexes: set[int] = set()
    for index in range(max_turn_evidence):
        expected = expected_turns[index]
        retrieval = replay_retrievals[index] if index < len(replay_retrievals) else None
        behavior = behavior_records[index] if index < len(behavior_records) else None
        action = action_records[index] if index < len(action_records) else None
        if action is not None and behavior is None:
            raise ReplayJoinError("action memory exists without persisted behavior")
        complete = (
            retrieval is not None
            and behavior is not None
            and action is not None
            and index < run.summary["completed_main_turns"]
        )
        turns_by_step[expected.step_id].append(
            ReplayTurn(
                turn_id=expected.turn_id,
                turn_index=expected.turn_index,
                agent_id=expected.agent_id,
                logical_timestamp=expected.timestamp,
                completeness="complete" if complete else "partial",
                response=None if behavior is None else behavior["response"],
                plan_summary=None if behavior is None else behavior["plan_summary"],
                retrieval=retrieval,
                action_memory_id=None if action is None else action["memory_id"],
                usage_records=usage_by_operation.get(expected.turn_id, ()),
            )
        )
        evidence_step_indexes.add(expected.step_index)

    reflections_by_step: dict[str, list[ReplayReflection]] = defaultdict(list)
    for index, memory in enumerate(reflection_memory_records):
        expected = expected_reflections[index]
        record = reflection_records[index] if index < len(reflection_records) else None
        reflections_by_step[expected.step_id].append(
            ReplayReflection(
                operation_id=expected.operation_id,
                participant_index=expected.participant_index,
                agent_id=expected.agent_id,
                logical_timestamp=expected.timestamp,
                completeness="complete" if record is not None else "partial",
                text=memory["text"] if record is None else record["reflection"],
                category=memory["category"] if record is None else record["category"],
                memory_id=memory["memory_id"] if record is None else record["memory_id"],
            )
        )
        evidence_step_indexes.add(expected.step_index)
    for record in streams["api_usage_logs.jsonl"]:
        operation_id = record["turn_id"]
        for item in expected_turns:
            if item.turn_id == operation_id:
                evidence_step_indexes.add(item.step_index)
                break
        else:
            for item in expected_reflections:
                if item.operation_id == operation_id:
                    evidence_step_indexes.add(item.step_index)
                    break

    failure, failure_issues = _build_failure(run.summary, streams["errors.jsonl"])
    if failure is not None and failure.step_id is not None:
        for index, step in enumerate(steps_config):
            if step["step_id"] == failure.step_id:
                evidence_step_indexes.add(index)
                break

    intended_scope, scope_step_count = _scope(run.summary, evidence_step_indexes)
    replay_steps: list[ReplayStep] = []
    for step_index, step in enumerate(steps_config):
        turns = tuple(turns_by_step.get(step["step_id"], ()))
        reflections = tuple(reflections_by_step.get(step["step_id"], ()))
        structural = summary_by_step.get(step["step_id"])
        if structural is not None:
            state = "complete"
            expected_turn_count = 12 if step["step_type"] == "natural_interaction_block" else 4
            if (
                len(turns) != expected_turn_count
                or any(turn.completeness != "complete" for turn in turns)
                or len(reflections) != 4
                or any(item.completeness != "complete" for item in reflections)
            ):
                raise ReplayJoinError("complete step summary has incomplete constituent artifacts")
        elif turns or reflections or step_index in evidence_step_indexes:
            state = "partial"
        elif run.summary["status"] == "excluded" and step_index >= scope_step_count:
            state = "out_of_scope"
        else:
            state = "unobserved"
        step_issues: tuple[ReplayIssue, ...] = ()
        if state == "partial":
            step_issues = (
                ReplayIssue(
                    code="partial_step",
                    severity="incomplete",
                    scope=step["step_id"],
                    message="Only the persisted prefix of this step is available.",
                ),
            )
        step_turn_schedule = [item for item in expected_turns if item.step_id == step["step_id"]]
        final_timestamp = (
            structural["final_logical_timestamp"]
            if structural is not None
            else (turns[-1].logical_timestamp if turns else None)
        )
        replay_steps.append(
            ReplayStep(
                step_id=step["step_id"],
                step_type=step["step_type"],
                day=step["day"],
                period=step["period"],
                title=step["title"],
                description=step["description"],
                participants=tuple(step["participants"]),
                event_id=step.get("event_id"),
                evaluation_enabled=step["evaluation_enabled"],
                state=state,
                turns=turns,
                reflections=reflections,
                final_logical_timestamp=final_timestamp,
                structural_summary=structural,
                issues=step_issues,
            )
        )
        if state == "complete" and final_timestamp != step_turn_schedule[-1].timestamp:
            raise ReplayOrderingError("complete step final timestamp is inconsistent")

    complete_steps = [step for step in replay_steps if step.state == "complete"]
    actual_natural = sum(step.step_type == "natural_interaction_block" for step in complete_steps)
    actual_target = sum(step.step_type == "target_event" for step in complete_steps)
    if (
        actual_natural != run.summary["completed_natural_blocks"]
        or actual_target != run.summary["completed_target_events"]
    ):
        raise ReplayJoinError("completed structural summaries contradict aggregate counts")

    if evaluation_overlay is not None:
        units = evaluation_overlay.get("units")
        if not isinstance(units, (list, tuple)):
            raise ReplayJoinError("evaluation overlay units are malformed")
        target_turns = {
            turn.turn_id: (step.event_id, step.step_id, turn.agent_id, turn.logical_timestamp)
            for step in replay_steps
            if step.step_type == "target_event"
            for turn in step.turns
        }
        seen_units: set[str] = set()
        for unit in units:
            if not isinstance(unit, Mapping):
                raise ReplayJoinError("evaluation overlay unit must be an object")
            turn_id = unit.get("turn_id")
            expected = target_turns.get(turn_id)
            actual = (
                unit.get("event_id"),
                unit.get("step_id"),
                unit.get("agent_id"),
                unit.get("logical_timestamp"),
            )
            if turn_id in seen_units or expected is None or actual != expected:
                raise ReplayJoinError("evaluation overlay does not join the source target turns")
            seen_units.add(turn_id)

    playback: list[ReplayFrame] = [
        ReplayFrame(0, "initialization", 0, None, "seed_memory_initialization")
    ]
    for step in replay_steps:
        for turn in step.turns:
            playback.append(
                ReplayFrame(
                    len(playback), "main_turn", turn.logical_timestamp, step.step_id, turn.turn_id
                )
            )
        for reflection in step.reflections:
            playback.append(
                ReplayFrame(
                    len(playback),
                    "reflection",
                    reflection.logical_timestamp,
                    step.step_id,
                    reflection.operation_id,
                )
            )
    if failure is not None:
        playback.append(
            ReplayFrame(
                len(playback),
                "failure",
                failure.logical_timestamp,
                failure.step_id,
                failure.error_type,
            )
        )

    source = run.snapshot.to_source_dict()
    source.update(
        {
            "artifact_kind": "phase12_read_only_replay_projection",
            "replay_is_research_data": False,
            "ground_truth_included": False,
            "embeddings_included": False,
            "prompts_included": False,
            "provider_responses_included": False,
        }
    )
    usage_value = {
        "record_count": len(usage),
        "known_input_tokens": sum(item.input_tokens for item in usage),
        "known_output_tokens": sum(item.output_tokens for item in usage),
        "known_total_tokens": sum(item.total_tokens for item in usage),
        "missing_usage_count": 0,
        "records": [item.to_dict() for item in usage],
        "note": "Canonical source usage contains token integers only; optional pilot evidence preserves unavailable usage as null.",
    }
    issues = list(failure_issues)
    if run.summary["status"] in {"failed", "interrupted"}:
        issues.append(
            ReplayIssue(
                code="partial_terminal_run",
                severity="incomplete",
                scope="run",
                message="This replay is a persisted prefix and is not a completed run.",
            )
        )

    return ReplayDocument(
        source=source,
        run=ReplayRun(
            experiment_id=run.summary["experiment_id"],
            run_id=run.summary["run_id"],
            retrieval_config_id=run.summary["retrieval_config_id"],
            repetition_index=run.summary["repetition_index"],
            source_status=run.summary["status"],
            replay_completeness=(
                "complete_source_scope"
                if run.summary["status"] in {"completed", "excluded"}
                else "partial_persisted_prefix"
            ),
            completed_natural_blocks=run.summary["completed_natural_blocks"],
            completed_target_events=run.summary["completed_target_events"],
            completed_main_turns=run.summary["completed_main_turns"],
            error_count=run.summary["error_count"],
            intended_scope=intended_scope,
            primary_research_data=run.summary["status"] == "completed",
        ),
        agents=_build_agents(run.config["agents.json"]),
        steps=tuple(replay_steps),
        memories=_memory_projection(streams["memory_logs.jsonl"], access_timestamps),
        playback=tuple(playback),
        usage=usage_value,
        failure=failure,
        evaluation_overlay=evaluation_overlay,
        pilot_overlay=pilot_overlay,
        issues=tuple(issues),
    )


__all__ = ["FIXED_NATURAL_TURN_ORDER", "reconstruct_replay"]
