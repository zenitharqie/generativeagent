"""Focused tests for Phase 08 structural step-summary persistence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

import src.logging_system as logging_system
from src.logging_system import (
    CorruptJsonlError,
    DuplicateRecordError,
    JsonlValidationError,
    LogSchemaError,
    NaturalBlockSummaryRecord,
    RunLogMetadata,
    SCHEMA_VERSION,
    StepLogMetadata,
    TargetEventSummaryRecord,
    append_jsonl,
    record_to_dict,
    serialize_record,
)
from src.storage import create_run_directory


PARTICIPANT_IDS = (
    "teacher",
    "active_student",
    "quiet_student",
    "academic_student",
)


def run_metadata() -> RunLogMetadata:
    return RunLogMetadata(
        experiment_id="classroom_experiment",
        run_id="balanced_rep01",
        retrieval_config_id="balanced",
        repetition_index=1,
    )


def natural_step(
    *,
    step_id: str = "day_01_natural_01",
    timestamp: int = 12,
) -> StepLogMetadata:
    return StepLogMetadata(
        run=run_metadata(),
        step_id=step_id,
        step_type="natural_interaction_block",
        is_target_event=False,
        evaluation_enabled=False,
        day=1,
        period="morning",
        timestamp=timestamp,
    )


def target_step(
    *,
    step_id: str = "day_02_target_01",
    timestamp: int = 16,
) -> StepLogMetadata:
    return StepLogMetadata(
        run=run_metadata(),
        step_id=step_id,
        step_type="target_event",
        is_target_event=True,
        evaluation_enabled=True,
        day=2,
        period="afternoon",
        timestamp=timestamp,
        event_id="event_01",
    )


def natural_summary(
    *,
    step: StepLogMetadata | None = None,
) -> NaturalBlockSummaryRecord:
    summary_step = step or natural_step()
    return NaturalBlockSummaryRecord(
        step=summary_step,
        ordered_main_turn_ids=tuple(
            f"{summary_step.step_id}_turn_{index:02d}" for index in range(1, 13)
        ),
        participant_ids=PARTICIPANT_IDS,
        per_agent_main_turn_counts={agent_id: 3 for agent_id in PARTICIPANT_IDS},
        completed_main_turn_count=12,
        reflection_count=4,
        final_logical_timestamp=summary_step.timestamp,
    )


def target_summary(
    *,
    step: StepLogMetadata | None = None,
) -> TargetEventSummaryRecord:
    summary_step = step or target_step()
    return TargetEventSummaryRecord(
        step=summary_step,
        ordered_main_turn_ids=tuple(
            f"{summary_step.step_id}_turn_{index:02d}" for index in range(1, 5)
        ),
        participant_ids=PARTICIPANT_IDS,
        per_agent_main_turn_counts={agent_id: 1 for agent_id in PARTICIPANT_IDS},
        completed_main_turn_count=4,
        reflection_count=4,
        final_logical_timestamp=summary_step.timestamp,
        ordered_responding_agent_ids=PARTICIPANT_IDS,
        response_count=4,
    )


def canonical_path(tmp_path: Path, filename: str) -> Path:
    return tmp_path / "classroom_experiment" / "balanced_rep01" / filename


def test_logging_package_exports_phase08_structural_summary_records() -> None:
    assert {
        "NaturalBlockSummaryRecord",
        "TargetEventSummaryRecord",
    } <= set(logging_system.__all__)


def test_natural_summary_is_immutable_and_serializes_only_structural_fields() -> None:
    source_counts = {agent_id: 3 for agent_id in PARTICIPANT_IDS}
    record = NaturalBlockSummaryRecord(
        step=natural_step(),
        ordered_main_turn_ids=tuple(
            f"day_01_natural_01_turn_{index:02d}" for index in range(1, 13)
        ),
        participant_ids=PARTICIPANT_IDS,
        per_agent_main_turn_counts=source_counts,
        completed_main_turn_count=12,
        reflection_count=4,
        final_logical_timestamp=12,
    )
    source_counts["teacher"] = 99

    data = record_to_dict(record)

    assert data == {
        "schema_version": SCHEMA_VERSION,
        **natural_step().to_dict(),
        "ordered_main_turn_ids": [
            f"day_01_natural_01_turn_{index:02d}" for index in range(1, 13)
        ],
        "participant_ids": list(PARTICIPANT_IDS),
        "per_agent_main_turn_counts": {
            agent_id: 3 for agent_id in PARTICIPANT_IDS
        },
        "completed_main_turn_count": 12,
        "reflection_count": 4,
        "final_logical_timestamp": 12,
    }
    assert "summary" not in data
    assert "response" not in data
    assert json.loads(serialize_record(record)) == data
    with pytest.raises(FrozenInstanceError):
        record.reflection_count = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.per_agent_main_turn_counts["teacher"] = 99  # type: ignore[index]


def test_target_summary_persists_event_and_configured_response_order() -> None:
    data = record_to_dict(target_summary())

    assert data["event_id"] == "event_01"
    assert data["ordered_main_turn_ids"] == [
        f"day_02_target_01_turn_{index:02d}" for index in range(1, 5)
    ]
    assert data["participant_ids"] == list(PARTICIPANT_IDS)
    assert data["ordered_responding_agent_ids"] == list(PARTICIPANT_IDS)
    assert data["per_agent_main_turn_counts"] == {
        agent_id: 1 for agent_id in PARTICIPANT_IDS
    }
    assert data["completed_main_turn_count"] == data["response_count"] == 4
    assert data["reflection_count"] == 4
    assert data["final_logical_timestamp"] == 16


@pytest.mark.parametrize(
    ("filename", "record_factory"),
    [
        ("natural_block_summaries.jsonl", natural_summary),
        ("event_summaries.jsonl", target_summary),
    ],
)
def test_reserved_summary_stream_accepts_only_canonical_typed_record(
    tmp_path: Path,
    filename: str,
    record_factory,
) -> None:
    path = canonical_path(tmp_path, filename)
    record = record_factory()

    append_jsonl(path, record)

    assert json.loads(path.read_text(encoding="utf-8")) == record.to_dict()
    assert path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("filename", "wrong_record", "expected_type"),
    [
        (
            "natural_block_summaries.jsonl",
            target_summary(),
            "NaturalBlockSummaryRecord",
        ),
        (
            "event_summaries.jsonl",
            natural_summary(),
            "TargetEventSummaryRecord",
        ),
    ],
)
def test_summary_stream_rejects_wrong_typed_record_before_filesystem_mutation(
    tmp_path: Path,
    filename: str,
    wrong_record: object,
    expected_type: str,
) -> None:
    path = canonical_path(tmp_path, filename)

    with pytest.raises(JsonlValidationError, match=expected_type):
        append_jsonl(path, wrong_record)

    assert not path.exists()


@pytest.mark.parametrize(
    "filename",
    ["natural_block_summaries.jsonl", "event_summaries.jsonl"],
)
def test_summary_stream_still_rejects_untyped_mapping(
    tmp_path: Path,
    filename: str,
) -> None:
    path = canonical_path(tmp_path, filename)
    untyped = {
        "schema_version": SCHEMA_VERSION,
        **run_metadata().to_dict(),
    }

    with pytest.raises(
        JsonlValidationError,
        match="simulation controller stream.*requires",
    ):
        append_jsonl(path, untyped)

    assert not path.exists()


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        (
            "ordered_main_turn_ids",
            tuple(f"turn_{index:02d}" for index in range(1, 12)),
            "exactly 12",
        ),
        (
            "per_agent_main_turn_counts",
            {**{agent_id: 3 for agent_id in PARTICIPANT_IDS}, "teacher": 2},
            "three main turns",
        ),
        ("completed_main_turn_count", 11, "exactly 12"),
        ("reflection_count", 3, "participant count"),
        ("final_logical_timestamp", 11, "match step metadata"),
    ],
)
def test_natural_summary_rejects_incomplete_or_inconsistent_completion(
    field_name: str,
    value: object,
    match: str,
) -> None:
    values = {
        "step": natural_step(),
        "ordered_main_turn_ids": tuple(
            f"day_01_natural_01_turn_{index:02d}" for index in range(1, 13)
        ),
        "participant_ids": PARTICIPANT_IDS,
        "per_agent_main_turn_counts": {
            agent_id: 3 for agent_id in PARTICIPANT_IDS
        },
        "completed_main_turn_count": 12,
        "reflection_count": 4,
        "final_logical_timestamp": 12,
    }
    values[field_name] = value

    with pytest.raises(LogSchemaError, match=match):
        NaturalBlockSummaryRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        (
            "ordered_responding_agent_ids",
            tuple(reversed(PARTICIPANT_IDS)),
            "equal participant_ids in order",
        ),
        ("response_count", 3, "exactly four"),
        ("reflection_count", 3, "participant count"),
        (
            "per_agent_main_turn_counts",
            {**{agent_id: 1 for agent_id in PARTICIPANT_IDS}, "teacher": 2},
            "one main turn",
        ),
    ],
)
def test_target_summary_rejects_incomplete_or_reordered_responses(
    field_name: str,
    value: object,
    match: str,
) -> None:
    values = {
        "step": target_step(),
        "ordered_main_turn_ids": tuple(
            f"day_02_target_01_turn_{index:02d}" for index in range(1, 5)
        ),
        "participant_ids": PARTICIPANT_IDS,
        "per_agent_main_turn_counts": {
            agent_id: 1 for agent_id in PARTICIPANT_IDS
        },
        "completed_main_turn_count": 4,
        "reflection_count": 4,
        "final_logical_timestamp": 16,
        "ordered_responding_agent_ids": PARTICIPANT_IDS,
        "response_count": 4,
    }
    values[field_name] = value

    with pytest.raises(LogSchemaError, match=match):
        TargetEventSummaryRecord(**values)  # type: ignore[arg-type]


def test_summary_schema_rejects_wrong_step_type() -> None:
    with pytest.raises(LogSchemaError, match="natural-interaction"):
        natural_summary(step=target_step())
    with pytest.raises(LogSchemaError, match="target-event"):
        target_summary(step=natural_step())


@pytest.mark.parametrize(
    ("record_factory", "turn_count"),
    [
        (natural_summary, 12),
        (target_summary, 4),
    ],
)
def test_summary_schema_rejects_foreign_or_noncanonical_turn_ids(
    record_factory,
    turn_count: int,
) -> None:
    with pytest.raises(LogSchemaError, match="deterministic one-based step turn IDs"):
        replace(
            record_factory(),
            ordered_main_turn_ids=tuple(
                f"foreign_turn_{index:02d}"
                for index in range(1, turn_count + 1)
            ),
        )


@pytest.mark.parametrize("summary_kind", ["natural", "target"])
def test_published_run_summary_rejects_participant_order_not_in_snapshot(
    tmp_path: Path,
    summary_kind: str,
) -> None:
    paths = create_run_directory(
        tmp_path,
        "classroom_experiment",
        "balanced_rep01",
        retrieval_config_id="balanced",
        repetition_index=1,
    )
    reversed_participants = tuple(reversed(PARTICIPANT_IDS))
    if summary_kind == "natural":
        source = natural_summary()
        record = replace(
            source,
            participant_ids=reversed_participants,
            per_agent_main_turn_counts={
                agent_id: 3 for agent_id in reversed_participants
            },
        )
        path = paths.natural_block_summaries
    else:
        source = target_summary()
        record = replace(
            source,
            participant_ids=reversed_participants,
            per_agent_main_turn_counts={
                agent_id: 1 for agent_id in reversed_participants
            },
            ordered_responding_agent_ids=reversed_participants,
        )
        path = paths.event_summaries

    with pytest.raises(JsonlValidationError, match="configured participant order"):
        append_jsonl(path, record)

    assert path.read_bytes() == b""


@pytest.mark.parametrize(
    ("filename", "record_factory"),
    [
        ("natural_block_summaries.jsonl", natural_summary),
        ("event_summaries.jsonl", target_summary),
    ],
)
def test_summary_stream_has_non_bypassable_step_duplicate_key(
    tmp_path: Path,
    filename: str,
    record_factory,
) -> None:
    path = canonical_path(tmp_path, filename)
    record = record_factory()
    append_jsonl(path, record)
    before = path.read_bytes()

    with pytest.raises(DuplicateRecordError, match=record.step.step_id):
        append_jsonl(path, record)

    assert path.read_bytes() == before


def test_summary_stream_rejects_physical_run_identity_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "classroom_experiment" / "other_run" / "event_summaries.jsonl"

    with pytest.raises(JsonlValidationError, match="path does not match"):
        append_jsonl(path, target_summary())

    assert not path.exists()


def test_existing_semantically_corrupt_summary_is_not_extended(
    tmp_path: Path,
) -> None:
    path = canonical_path(tmp_path, "natural_block_summaries.jsonl")
    append_jsonl(path, natural_summary())
    corrupt = json.loads(path.read_text(encoding="utf-8"))
    corrupt["reflection_count"] = 3
    path.write_text(
        json.dumps(corrupt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    second_step = natural_step(step_id="day_01_natural_02", timestamp=24)

    with pytest.raises(CorruptJsonlError, match="reflection_count"):
        append_jsonl(path, natural_summary(step=second_step))

    assert path.read_bytes() == before
