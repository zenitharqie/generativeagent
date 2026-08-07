"""Focused tests for append-only JSONL and run-summary persistence."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from src.config_loader import load_experiment_config
from src.logging_system.jsonl_logger import (
    CorruptJsonlError,
    DuplicateRecordError,
    JsonlLogger,
    JsonlValidationError,
    JsonlWriteError,
    append_jsonl,
)
from src.logging_system.run_summary import (
    CorruptRunSummaryError,
    RunSummary,
    RunSummaryStateError,
    RunSummaryValidationError,
    RunSummaryWriteError,
    read_run_summary,
    write_run_summary,
)
from src.logging_system.schemas import (
    BehaviorLogRecord,
    ErrorLogRecord,
    MemoryLogRecord,
    RetrievalCandidateRecord,
    RetrievalLogRecord,
    RetrievalWeightsRecord,
    RunLogMetadata,
    SCHEMA_VERSION,
    StepLogMetadata,
    TurnLogMetadata,
)
from src.storage.config_snapshot import copy_config_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "config"


def record(turn_id: str, *, text: str = "response") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-01",
        "turn_id": turn_id,
        "text": text,
        "scores": [0.25, 1.0],
        "optional": None,
    }


def summary(status: str = "initialized", **overrides: object) -> RunSummary:
    values: dict[str, object] = {
        "experiment_id": "experiment-01",
        "run_id": "balanced-rep01",
        "retrieval_config_id": "balanced",
        "repetition_index": 0,
        "status": status,
    }
    if status != "initialized":
        values.update(
            completed_natural_blocks=7,
            completed_target_events=5,
            completed_main_turns=104,
            error_count=0,
        )
    values.update(overrides)
    return RunSummary(**values)  # type: ignore[arg-type]


def turn_metadata(
    *,
    run: RunLogMetadata | None = None,
    turn_id: str = "turn-01",
) -> TurnLogMetadata:
    return TurnLogMetadata(
        step=StepLogMetadata(
            run=run or RunLogMetadata("experiment-01", "run-01", "balanced", 0),
            step_id="day_01_natural_01",
            step_type="natural_interaction_block",
            is_target_event=False,
            evaluation_enabled=False,
            day=1,
            period="morning",
            timestamp=1,
        ),
        turn_id=turn_id,
        turn_index=0,
        agent_id="teacher",
    )


def behavior_record(
    *,
    run: RunLogMetadata | None = None,
    turn_id: str = "turn-01",
    response: str = "Respons terkontrol.",
) -> BehaviorLogRecord:
    return BehaviorLogRecord(
        turn=turn_metadata(run=run, turn_id=turn_id),
        response=response,
        retrieved_memory_ids=("mem_teacher_000001",),
        plan_summary="Menjawab sesuai konteks.",
    )


def retrieval_record(
    *,
    run: RunLogMetadata | None = None,
    turn: TurnLogMetadata | None = None,
    turn_id: str = "turn-01",
    memory_id: str = "mem_teacher_000001",
    category: str = "classroom_norms",
) -> RetrievalLogRecord:
    candidate = RetrievalCandidateRecord(
        memory_id=memory_id,
        category=category,
        raw_recency=1.0,
        raw_importance=8.0,
        raw_relevance=0.5,
        normalized_recency=1.0,
        normalized_importance=1.0,
        normalized_relevance=1.0,
        final_score=1.0,
        selected=True,
    )
    return RetrievalLogRecord(
        turn=turn or turn_metadata(run=run, turn_id=turn_id),
        query="Apa aturan kelas yang relevan?",
        query_embedding=(1.0, 0.0),
        weights=RetrievalWeightsRecord(0.33, 0.33, 0.33),
        top_k=5,
        candidate_count=1,
        candidates=(candidate,),
        selected_memory_ids=(memory_id,),
        accessed_memory_ids=(memory_id,),
    )


def canonical_summary_path(tmp_path: Path) -> Path:
    return tmp_path / "experiment-01" / "balanced-rep01" / "run_summary.json"


def seed_memory_record(
    *,
    run: RunLogMetadata,
    agent_id: str = "teacher",
    seed_index: int = 1,
) -> MemoryLogRecord:
    config = load_experiment_config(CONFIG_DIRECTORY)
    agent = next(agent for agent in config.agents if agent.agent_id == agent_id)
    seed = agent.seed_memories[seed_index - 1]
    return MemoryLogRecord(
        run=run,
        memory_id=f"mem_{agent_id}_{seed_index:06d}",
        agent_id=agent_id,
        memory_type="seed_persona",
        category=seed.category,
        text=seed.text,
        importance=seed.importance,
        created_at=0,
        last_accessed_at=0,
        step_id="seed_memory_initialization",
        step_type="agent_initialization",
        embedding=(1.0, 0.0),
        metadata={"seed_index": seed_index},
    )


def create_persisted_config_snapshot(run_directory: Path) -> None:
    copy_config_snapshot(CONFIG_DIRECTORY, run_directory / "config_snapshot")


def test_jsonl_append_is_deterministic_utf8_and_lf_terminated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    value = record("turn-01", text="Halo, 世界")

    append_jsonl(path, value)

    expected = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    assert path.read_text(encoding="utf-8") == expected
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()


def test_jsonl_records_append_in_call_order(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    logger = JsonlLogger(path)

    logger.append(record("turn-01"))
    logger.append(record("turn-02"))

    parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["turn_id"] for item in parsed] == ["turn-01", "turn-02"]


def test_jsonl_accepts_explicit_to_dict_serializer_without_mutating_it(
    tmp_path: Path,
) -> None:
    class ExplicitRecord:
        def __init__(self) -> None:
            self.value = record("turn-01")

        def to_dict(self) -> dict[str, object]:
            return self.value

    source = ExplicitRecord()
    before = json.loads(json.dumps(source.value))

    append_jsonl(tmp_path / "records.jsonl", source)

    assert source.value == before


def test_jsonl_appends_a_validated_phase_07_record_without_reinterpreting_it(
    tmp_path: Path,
) -> None:
    phase_record = ErrorLogRecord(
        run=RunLogMetadata("experiment-01", "run-01", "balanced", 0),
        error_type="ProviderUnavailable",
        message="Operasi dihentikan; tidak dicoba ulang.",
        recoverable=False,
        retry_count=None,
    )
    path = tmp_path / "experiment-01" / "run-01" / "errors.jsonl"

    append_jsonl(path, phase_record)

    assert json.loads(path.read_text(encoding="utf-8")) == phase_record.to_dict()


@pytest.mark.parametrize(
    "invalid_record",
    [
        object(),
        {"schema_version": "0.9", "turn_id": "turn-01"},
        {"schema_version": SCHEMA_VERSION, "score": float("nan")},
        {"schema_version": SCHEMA_VERSION, "score": float("inf")},
        {"schema_version": SCHEMA_VERSION, "metadata": {1: "invalid-key"}},
        {"schema_version": SCHEMA_VERSION, "unsupported": {"set"}},
    ],
)
def test_invalid_jsonl_records_fail_before_filesystem_mutation(
    tmp_path: Path,
    invalid_record: object,
) -> None:
    path = tmp_path / "not-created" / "records.jsonl"

    with pytest.raises(JsonlValidationError):
        append_jsonl(path, invalid_record)

    assert not path.parent.exists()


def test_jsonl_duplicate_detection_uses_caller_strategy_for_generic_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    logger = JsonlLogger(
        path,
        duplicate_key=lambda value: (value["run_id"], value["turn_id"]),
    )
    logger.append(record("turn-01"))
    before = path.read_bytes()

    with pytest.raises(DuplicateRecordError, match="run-01"):
        logger.append(record("turn-01", text="different"))

    assert path.read_bytes() == before


def test_generic_jsonl_allows_duplicates_when_caller_supplies_no_strategy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"

    append_jsonl(path, record("turn-01"))
    append_jsonl(path, record("turn-01"))

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_canonical_stream_rejects_untyped_or_wrong_schema_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "run-01" / "behavior_logs.jsonl"

    with pytest.raises(JsonlValidationError, match="requires BehaviorLogRecord"):
        append_jsonl(path, record("turn-01"))

    assert not path.exists()


@pytest.mark.parametrize(
    "filename",
    ["event_summaries.jsonl", "natural_block_summaries.jsonl"],
)
def test_controller_summary_streams_remain_empty_until_typed_schemas_exist(
    tmp_path: Path,
    filename: str,
) -> None:
    path = tmp_path / "experiment-01" / "run-01" / filename
    untyped_summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "experiment-01",
        "run_id": "run-01",
        "retrieval_config_id": "balanced",
        "repetition_index": 0,
    }

    with pytest.raises(JsonlValidationError, match="simulation controller"):
        append_jsonl(path, untyped_summary)

    assert not path.exists()


def test_canonical_stream_uses_non_bypassable_duplicate_turn_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "run-01" / "behavior_logs.jsonl"
    append_jsonl(path, behavior_record())
    before = path.read_bytes()

    with pytest.raises(DuplicateRecordError, match="turn-01"):
        append_jsonl(path, behavior_record(response="Respons berbeda."))

    assert path.read_bytes() == before


def test_canonical_stream_rejects_physical_run_identity_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "other-run" / "behavior_logs.jsonl"

    with pytest.raises(JsonlValidationError, match="path does not match"):
        append_jsonl(path, behavior_record())

    assert not path.exists()


def test_canonical_stream_rejects_symlinked_experiment_directory(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    (outside / "run-01").mkdir(parents=True)
    linked_experiment = tmp_path / "experiment-01"
    linked_experiment.symlink_to(outside, target_is_directory=True)
    path = linked_experiment / "run-01" / "behavior_logs.jsonl"

    with pytest.raises(JsonlValidationError, match="must not be symlinks"):
        append_jsonl(path, behavior_record())

    assert not (outside / "run-01" / "behavior_logs.jsonl").exists()


def test_canonical_stream_rejects_sibling_summary_configuration_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "run-01" / "behavior_logs.jsonl"
    summary_path = path.parent / "run_summary.json"
    write_run_summary(
        summary_path,
        RunSummary("experiment-01", "run-01", "balanced", 0),
    )
    mismatched_run = RunLogMetadata(
        "experiment-01",
        "run-01",
        "recency_heavy",
        0,
    )

    with pytest.raises(JsonlValidationError, match="retrieval_config_id"):
        append_jsonl(path, behavior_record(run=mismatched_run))

    assert not path.exists()


def test_canonical_existing_semantic_corruption_is_not_extended(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "run-01" / "behavior_logs.jsonl"
    append_jsonl(path, behavior_record())
    corrupted = json.loads(path.read_text(encoding="utf-8"))
    del corrupted["response"]
    path.write_text(
        json.dumps(corrupted, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    with pytest.raises(CorruptJsonlError, match="missing fields: response"):
        append_jsonl(path, behavior_record(turn_id="turn-02"))

    assert path.read_bytes() == before


def test_seed_persona_record_matches_persisted_agent_snapshot(tmp_path: Path) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = seed_memory_record(run=run)

    append_jsonl(memory_path, memory)

    assert json.loads(memory_path.read_text(encoding="utf-8"))["text"] == memory.text


def test_seed_persona_text_must_match_persisted_agent_snapshot(tmp_path: Path) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = replace(seed_memory_record(run=run), text="Mismatched seed text.")

    with pytest.raises(JsonlValidationError, match="seed_persona text"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()


def test_seed_persona_category_must_match_persisted_agent_snapshot(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = replace(seed_memory_record(run=run), category="mismatched_category")

    with pytest.raises(JsonlValidationError, match="seed_persona category"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()


def test_seed_persona_importance_must_match_persisted_agent_snapshot(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = replace(seed_memory_record(run=run), importance=8)

    with pytest.raises(JsonlValidationError, match="seed_persona importance"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()


def test_seed_persona_agent_must_match_deterministic_memory_identity(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = replace(seed_memory_record(run=run), agent_id="active_student")

    with pytest.raises(JsonlValidationError, match="agent_id"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()


def test_seed_persona_index_must_match_deterministic_seed_order(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = replace(seed_memory_record(run=run), metadata={"seed_index": 2})

    with pytest.raises(JsonlValidationError, match="deterministic seed order"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()


def test_reopened_logger_revalidates_seed_persona_from_persisted_snapshot(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    JsonlLogger(memory_path).append(seed_memory_record(run=run, seed_index=1))

    reopened = JsonlLogger(memory_path)
    reopened.append(seed_memory_record(run=run, seed_index=2))

    records = memory_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(record)["metadata"]["seed_index"] for record in records] == [
        1,
        2,
    ]


def test_retrieval_write_cross_validates_available_memory_category(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    retrieval_path = run_directory / "retrieval_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory = seed_memory_record(run=run)
    append_jsonl(memory_path, memory)
    memory_before = memory_path.read_bytes()
    with pytest.raises(DuplicateRecordError, match=memory.memory_id):
        append_jsonl(memory_path, memory)
    assert memory_path.read_bytes() == memory_before

    candidate = RetrievalCandidateRecord(
        memory_id=memory.memory_id,
        category="different_category",
        raw_recency=1.0,
        raw_importance=8.0,
        raw_relevance=0.5,
        normalized_recency=1.0,
        normalized_importance=1.0,
        normalized_relevance=1.0,
        final_score=1.0,
        selected=True,
    )
    retrieval = RetrievalLogRecord(
        turn=turn_metadata(run=run),
        query="Apa aturan kelas yang relevan?",
        query_embedding=(1.0, 0.0),
        weights=RetrievalWeightsRecord(0.33, 0.33, 0.33),
        top_k=5,
        candidate_count=1,
        candidates=(candidate,),
        selected_memory_ids=(memory.memory_id,),
        accessed_memory_ids=(memory.memory_id,),
    )

    with pytest.raises(JsonlValidationError, match="category does not match"):
        append_jsonl(retrieval_path, retrieval)

    assert not retrieval_path.exists()


def test_memory_write_cross_validates_previously_persisted_retrieval_category(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    memory_path = run_directory / "memory_logs.jsonl"
    retrieval_path = run_directory / "retrieval_logs.jsonl"
    create_persisted_config_snapshot(run_directory)
    memory_id = "mem_teacher_000001"
    candidate = RetrievalCandidateRecord(
        memory_id=memory_id,
        category="different_category",
        raw_recency=1.0,
        raw_importance=8.0,
        raw_relevance=0.5,
        normalized_recency=1.0,
        normalized_importance=1.0,
        normalized_relevance=1.0,
        final_score=1.0,
        selected=True,
    )
    retrieval = RetrievalLogRecord(
        turn=turn_metadata(run=run),
        query="Apa aturan kelas yang relevan?",
        query_embedding=(1.0, 0.0),
        weights=RetrievalWeightsRecord(0.33, 0.33, 0.33),
        top_k=5,
        candidate_count=1,
        candidates=(candidate,),
        selected_memory_ids=(memory_id,),
        accessed_memory_ids=(memory_id,),
    )
    append_jsonl(retrieval_path, retrieval)
    retrieval_before = retrieval_path.read_bytes()
    memory = seed_memory_record(run=run)

    with pytest.raises(JsonlValidationError, match="category does not match"):
        append_jsonl(memory_path, memory)

    assert not memory_path.exists()
    assert retrieval_path.read_bytes() == retrieval_before


@pytest.mark.parametrize("first_stream", ["retrieval", "behavior"])
def test_retrieval_and_behavior_selected_ids_must_match_in_either_write_order(
    tmp_path: Path,
    first_stream: str,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    retrieval_path = run_directory / "retrieval_logs.jsonl"
    behavior_path = run_directory / "behavior_logs.jsonl"
    memory_id = "mem_teacher_000001"
    candidate = RetrievalCandidateRecord(
        memory_id=memory_id,
        category="classroom_norms",
        raw_recency=1.0,
        raw_importance=8.0,
        raw_relevance=0.5,
        normalized_recency=1.0,
        normalized_importance=1.0,
        normalized_relevance=1.0,
        final_score=1.0,
        selected=True,
    )
    retrieval = RetrievalLogRecord(
        turn=turn_metadata(run=run),
        query="Apa aturan kelas yang relevan?",
        query_embedding=(1.0, 0.0),
        weights=RetrievalWeightsRecord(0.33, 0.33, 0.33),
        top_k=5,
        candidate_count=1,
        candidates=(candidate,),
        selected_memory_ids=(memory_id,),
        accessed_memory_ids=(memory_id,),
    )
    behavior = BehaviorLogRecord(
        turn=turn_metadata(run=run),
        response="Respons terkontrol.",
        retrieved_memory_ids=("mem_teacher_000002",),
        plan_summary="Menjawab sesuai konteks.",
    )
    if first_stream == "retrieval":
        first_path, first_record = retrieval_path, retrieval
        rejected_path, rejected_record = behavior_path, behavior
    else:
        first_path, first_record = behavior_path, behavior
        rejected_path, rejected_record = retrieval_path, retrieval
    append_jsonl(first_path, first_record)
    first_before = first_path.read_bytes()

    with pytest.raises(JsonlValidationError, match="retrieved_memory_ids do not match"):
        append_jsonl(rejected_path, rejected_record)

    assert not rejected_path.exists()
    assert first_path.read_bytes() == first_before


@pytest.mark.parametrize("first_stream", ["retrieval", "behavior"])
def test_matching_retrieval_and_behavior_are_accepted_in_either_write_order(
    tmp_path: Path,
    first_stream: str,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    retrieval_path = run_directory / "retrieval_logs.jsonl"
    behavior_path = run_directory / "behavior_logs.jsonl"
    retrieval = retrieval_record(run=run)
    behavior = behavior_record(run=run)
    ordered = (
        ((retrieval_path, retrieval), (behavior_path, behavior))
        if first_stream == "retrieval"
        else ((behavior_path, behavior), (retrieval_path, retrieval))
    )

    for path, value in ordered:
        append_jsonl(path, value)

    assert json.loads(retrieval_path.read_text(encoding="utf-8"))[
        "selected_memory_ids"
    ] == ["mem_teacher_000001"]
    assert json.loads(behavior_path.read_text(encoding="utf-8"))[
        "retrieved_memory_ids"
    ] == ["mem_teacher_000001"]


@pytest.mark.parametrize("first_stream", ["retrieval", "behavior"])
def test_retrieval_and_behavior_turn_metadata_must_match_in_either_write_order(
    tmp_path: Path,
    first_stream: str,
) -> None:
    run = RunLogMetadata("experiment-01", "run-01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    retrieval_path = run_directory / "retrieval_logs.jsonl"
    behavior_path = run_directory / "behavior_logs.jsonl"
    retrieval_turn = turn_metadata(run=run)
    behavior_turn = TurnLogMetadata(
        step=retrieval_turn.step,
        turn_id=retrieval_turn.turn_id,
        turn_index=1,
        agent_id=retrieval_turn.agent_id,
    )
    retrieval = retrieval_record(turn=retrieval_turn)
    behavior = BehaviorLogRecord(
        turn=behavior_turn,
        response="Respons terkontrol.",
        retrieved_memory_ids=retrieval.selected_memory_ids,
        plan_summary="Menjawab sesuai konteks.",
    )
    if first_stream == "retrieval":
        first_path, first_record = retrieval_path, retrieval
        rejected_path, rejected_record = behavior_path, behavior
    else:
        first_path, first_record = behavior_path, behavior
        rejected_path, rejected_record = retrieval_path, retrieval
    append_jsonl(first_path, first_record)
    first_before = first_path.read_bytes()

    with pytest.raises(JsonlValidationError, match="inconsistent turn metadata"):
        append_jsonl(rejected_path, rejected_record)

    assert not rejected_path.exists()
    assert first_path.read_bytes() == first_before


def test_jsonl_rejects_invalid_new_duplicate_key_before_file_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    logger = JsonlLogger(path, duplicate_key=lambda value: value["missing"])

    with pytest.raises(JsonlValidationError, match="new record"):
        logger.append(record("turn-01"))

    assert not path.exists()


@pytest.mark.parametrize(
    "existing",
    [
        b'{"schema_version":"1.0"',
        b'{"schema_version":"1.0"}\n\n',
        b'{"schema_version":"1.0"}\r\n',
        b'[]\n',
        b'{"schema_version":"0.9"}\n',
        b'{"schema_version":"1.0","score":NaN}\n',
        b'{"schema_version":"1.0","schema_version":"1.0"}\n',
        b"\xff\n",
    ],
)
def test_jsonl_rejects_corrupt_existing_data_without_changing_it(
    tmp_path: Path,
    existing: bytes,
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(existing)

    with pytest.raises((CorruptJsonlError, JsonlWriteError)):
        append_jsonl(path, record("turn-02"))

    assert path.read_bytes() == existing


def test_jsonl_duplicate_strategy_failure_on_existing_record_is_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"schema_version": SCHEMA_VERSION, "legacy": True})
    before = path.read_bytes()
    logger = JsonlLogger(path, duplicate_key=lambda value: value["turn_id"])

    with pytest.raises(CorruptJsonlError, match="existing record"):
        logger.append(record("turn-01"))

    assert path.read_bytes() == before


def test_jsonl_parent_creation_error_is_visible(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")

    with pytest.raises(JsonlWriteError, match="parent directory"):
        append_jsonl(blocking_file / "records.jsonl", record("turn-01"))

    assert blocking_file.read_text(encoding="utf-8") == "occupied"


def test_jsonl_fsync_failure_is_visible_and_preserves_prior_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "records.jsonl"
    append_jsonl(path, record("turn-01"))
    prior_line = path.read_bytes()

    def fail_fsync(_file_descriptor: int) -> None:
        raise OSError("durability unavailable")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(JsonlWriteError, match="durability unavailable"):
        append_jsonl(path, record("turn-02"))

    assert path.read_bytes().startswith(prior_line)


def test_run_summary_initialized_write_has_exact_versioned_schema(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)

    write_run_summary(path, summary())

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "experiment-01",
        "run_id": "balanced-rep01",
        "retrieval_config_id": "balanced",
        "repetition_index": 0,
        "status": "initialized",
        "completed_natural_blocks": 0,
        "completed_target_events": 0,
        "completed_main_turns": 0,
        "error_count": 0,
    }
    assert path.read_bytes().endswith(b"\n")
    assert read_run_summary(path) == summary()


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "interrupted", "excluded"])
def test_run_summary_allows_one_initialized_to_terminal_transition(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    path = canonical_summary_path(tmp_path)
    write_run_summary(path, summary())

    write_run_summary(path, summary(terminal_status))

    assert read_run_summary(path).status == terminal_status


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "interrupted", "excluded"])
def test_terminal_run_summary_is_read_only(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    path = canonical_summary_path(tmp_path)
    write_run_summary(path, summary())
    write_run_summary(path, summary(terminal_status))
    before = path.read_bytes()

    with pytest.raises(RunSummaryStateError, match="terminal and read-only"):
        write_run_summary(path, summary("failed"))

    assert path.read_bytes() == before


def test_terminal_run_summary_also_protects_sibling_jsonl(
    tmp_path: Path,
) -> None:
    run = RunLogMetadata("experiment-01", "balanced-rep01", "balanced", 0)
    run_directory = tmp_path / run.experiment_id / run.run_id
    summary_path = run_directory / "run_summary.json"
    log_path = run_directory / "behavior_logs.jsonl"
    append_jsonl(log_path, behavior_record(run=run))
    write_run_summary(summary_path, summary())
    write_run_summary(summary_path, summary("failed"))
    before = log_path.read_bytes()

    with pytest.raises(RunSummaryStateError, match="terminal and read-only"):
        append_jsonl(log_path, behavior_record(run=run, turn_id="turn-02"))

    assert log_path.read_bytes() == before


def test_run_summary_cannot_be_created_terminal(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)

    with pytest.raises(RunSummaryStateError, match="created with initialized"):
        write_run_summary(path, summary("failed"))

    assert not path.exists()


def test_run_summary_cannot_rewrite_initialized_state(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)
    write_run_summary(path, summary())
    before = path.read_bytes()

    with pytest.raises(RunSummaryStateError, match="only to a terminal"):
        write_run_summary(path, summary())

    assert path.read_bytes() == before


def test_run_summary_identity_cannot_change(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)
    write_run_summary(path, summary())
    before = path.read_bytes()

    with pytest.raises(RunSummaryStateError, match="run_id"):
        write_run_summary(path, summary("failed", run_id="foreign-run"))

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("repetition_index", True),
        ("completed_natural_blocks", -1),
        ("completed_target_events", 1.5),
        ("completed_main_turns", True),
        ("error_count", -1),
    ],
)
def test_run_summary_rejects_invalid_non_negative_integer_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(RunSummaryValidationError, match=field_name):
        summary("failed", **{field_name: invalid_value})


def test_initialized_run_summary_requires_zero_counts() -> None:
    with pytest.raises(RunSummaryValidationError, match="zero aggregate counts"):
        summary(completed_main_turns=1)


def test_completed_run_summary_requires_all_fixed_run_totals() -> None:
    with pytest.raises(RunSummaryValidationError, match="fixed completed totals"):
        summary("completed", completed_main_turns=103)


def test_partial_terminal_summary_preserves_valid_partial_counts() -> None:
    partial = summary(
        "interrupted",
        completed_natural_blocks=6,
        completed_target_events=4,
        completed_main_turns=90,
        error_count=1,
    )

    assert partial.completed_natural_blocks == 6
    assert partial.completed_target_events == 4
    assert partial.completed_main_turns == 90
    assert partial.error_count == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("completed_natural_blocks", 8),
        ("completed_target_events", 6),
        ("completed_main_turns", 105),
    ],
)
def test_run_summary_counts_cannot_exceed_fixed_run_totals(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(RunSummaryValidationError, match="fixed run total"):
        summary("failed", **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("experiment_id", "../escape"),
        ("run_id", "/absolute"),
        ("retrieval_config_id", "contains space"),
    ],
)
def test_run_summary_validates_all_caller_supplied_identifiers(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(RunSummaryValidationError, match=field_name):
        summary(**{field_name: value})


@pytest.mark.parametrize("status", ["running", "partial", "", None])
def test_run_summary_rejects_unknown_status(status: object) -> None:
    with pytest.raises(RunSummaryValidationError, match="unsupported run status"):
        summary(status)  # type: ignore[arg-type]


def test_corrupt_existing_run_summary_is_not_repaired(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)
    path.parent.mkdir(parents=True)
    original = b'{"schema_version":"1.0","status":"initialized"}\n'
    path.write_bytes(original)

    with pytest.raises(CorruptRunSummaryError):
        write_run_summary(path, summary("failed"))

    assert path.read_bytes() == original


def test_run_summary_rejects_duplicate_json_keys_without_repair(tmp_path: Path) -> None:
    path = canonical_summary_path(tmp_path)
    path.parent.mkdir(parents=True)
    original = b'{"schema_version":"1.0","schema_version":"1.0"}\n'
    path.write_bytes(original)

    with pytest.raises(CorruptRunSummaryError, match="duplicate JSON key"):
        write_run_summary(path, summary("failed"))

    assert path.read_bytes() == original


def test_atomic_replace_failure_preserves_previous_summary_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = canonical_summary_path(tmp_path)
    write_run_summary(path, summary())
    before = path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("replace unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RunSummaryWriteError, match="replace unavailable"):
        write_run_summary(path, summary("failed"))

    assert path.read_bytes() == before
    assert list(path.parent.glob(".run_summary.json.*.tmp")) == []


def test_run_summary_rejects_physical_identity_mismatch_before_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment-01" / "other-run" / "run_summary.json"

    with pytest.raises(RunSummaryValidationError, match="path does not match"):
        write_run_summary(path, summary())

    assert not path.parent.exists()


def test_run_summary_rejects_symbolic_link_target(tmp_path: Path) -> None:
    actual = tmp_path / "actual.json"
    write_run_summary(actual, summary())
    linked = tmp_path / "linked.json"
    linked.symlink_to(actual)
    before = actual.read_bytes()

    with pytest.raises(RunSummaryStateError, match="symbolic link"):
        write_run_summary(linked, summary("failed"))

    assert actual.read_bytes() == before
