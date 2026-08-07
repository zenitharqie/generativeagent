"""Persistence ordering and no-retry failure tests for Phase 08 execution."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import src.cognition.pipeline as pipeline_module
import src.simulation.natural_block_runner as natural_runner_module
import src.simulation.target_event_runner as target_runner_module
import src.simulation.turn_executor as turn_executor_module
from src.llm import DummyEmbeddingProvider
from src.logging_system import MemoryLogRecord, read_run_summary
from src.simulation import (
    RunController,
    RunExecutionError,
    build_deterministic_dummy_llm_client,
)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def _run_pilot(output_root: Path, *, run_id: str, client=None):
    llm_client = client or build_deterministic_dummy_llm_client()
    result = RunController(
        llm_client=llm_client,
        embedding_provider=DummyEmbeddingProvider(),
    ).run(
        output_root=output_root,
        experiment_id="phase08_persistence",
        run_id=run_id,
        retrieval_config_id="balanced",
        repetition_index=1,
        days=(1, 2),
    )
    return result, llm_client


def test_successful_turn_and_reflection_append_order_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_append = turn_executor_module.append_jsonl
    writes: list[tuple[str, object]] = []

    def record_write(path, record):
        writes.append((Path(path).name, record))
        return real_append(path, record)

    monkeypatch.setattr(turn_executor_module, "append_jsonl", record_write)
    result, _ = _run_pilot(tmp_path, run_id="ordered_writes")
    cursor = 0

    for step in result.step_results:
        for turn in step.main_turn_results:
            group = writes[cursor : cursor + 3]
            assert [filename for filename, _ in group] == [
                "retrieval_logs.jsonl",
                "behavior_logs.jsonl",
                "memory_logs.jsonl",
            ]
            assert [type(record).__name__ for _, record in group] == [
                "RetrievalLogRecord",
                "BehaviorLogRecord",
                "MemoryLogRecord",
            ]
            assert all(record.turn.turn_id == turn.turn_id for _, record in group)
            assert group[2][1].memory_type == "action"
            cursor += 3
        for reflection in step.reflection_results:
            group = writes[cursor : cursor + 2]
            assert [filename for filename, _ in group] == [
                "memory_logs.jsonl",
                "reflection_logs.jsonl",
            ]
            assert group[0][1].memory_type == "reflection"
            assert group[0][1].memory_id == reflection.reflection_memory.memory_id
            assert group[1][1].memory_id == reflection.reflection_memory.memory_id
            cursor += 2

    assert cursor == len(writes) == (40 * 3) + (16 * 2)


@pytest.mark.parametrize(
    ("failed_filename", "retrieval_count", "behavior_count", "memory_count"),
    [
        ("retrieval_logs.jsonl", 0, 0, 12),
        ("behavior_logs.jsonl", 1, 0, 12),
        ("memory_logs.jsonl", 1, 1, 12),
    ],
)
def test_main_turn_logging_failure_never_retries_cognition_or_counts_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_filename: str,
    retrieval_count: int,
    behavior_count: int,
    memory_count: int,
) -> None:
    real_append = turn_executor_module.append_jsonl
    failed_once = False

    def fail_selected_write(path, record):
        nonlocal failed_once
        if Path(path).name == failed_filename and not failed_once:
            failed_once = True
            raise OSError(f"injected {failed_filename} failure")
        return real_append(path, record)

    monkeypatch.setattr(turn_executor_module, "append_jsonl", fail_selected_write)
    client = build_deterministic_dummy_llm_client()

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(
            tmp_path,
            run_id=f"failure_{failed_filename.removesuffix('.jsonl')}",
            client=client,
        )

    paths = raised.value.run_paths
    summary = read_run_summary(paths.run_summary)
    assert summary.status == "failed"
    assert summary.completed_main_turns == 0
    assert summary.completed_natural_blocks == 0
    assert len(client.invocations) == 3
    assert len(_read_jsonl(paths.retrieval_logs)) == retrieval_count
    assert len(_read_jsonl(paths.behavior_logs)) == behavior_count
    assert len(_read_jsonl(paths.memory_logs)) == memory_count
    assert paths.natural_block_summaries.read_bytes() == b""


@pytest.mark.parametrize(
    ("fail_after_memory", "expected_reflection_memories"),
    [(False, 0), (True, 1)],
)
def test_reflection_persistence_failure_leaves_step_incomplete_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_after_memory: bool,
    expected_reflection_memories: int,
) -> None:
    real_append = turn_executor_module.append_jsonl
    failed_once = False

    def fail_reflection_boundary(path, record):
        nonlocal failed_once
        is_reflection_memory = (
            isinstance(record, MemoryLogRecord)
            and record.memory_type == "reflection"
        )
        should_fail = (
            Path(path).name == "reflection_logs.jsonl"
            if fail_after_memory
            else is_reflection_memory
        )
        if should_fail and not failed_once:
            failed_once = True
            raise OSError("injected post-step reflection persistence failure")
        return real_append(path, record)

    monkeypatch.setattr(
        turn_executor_module,
        "append_jsonl",
        fail_reflection_boundary,
    )
    client = build_deterministic_dummy_llm_client()

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(
            tmp_path,
            run_id=f"reflection_failure_{int(fail_after_memory)}",
            client=client,
        )

    paths = raised.value.run_paths
    summary = read_run_summary(paths.run_summary)
    memories = _read_jsonl(paths.memory_logs)
    assert summary.status == "failed"
    assert summary.completed_main_turns == 12
    assert summary.completed_natural_blocks == 0
    assert len(client.invocations) == 38
    assert sum(record["memory_type"] == "action" for record in memories) == 12
    assert (
        sum(record["memory_type"] == "reflection" for record in memories)
        == expected_reflection_memories
    )
    assert paths.reflection_logs.read_bytes() == b""
    assert paths.natural_block_summaries.read_bytes() == b""


def test_structural_summary_failure_leaves_step_incomplete_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_summary(path, record):
        raise OSError("injected natural summary failure")

    monkeypatch.setattr(natural_runner_module, "append_jsonl", fail_summary)
    client = build_deterministic_dummy_llm_client()

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(tmp_path, run_id="natural_summary_failure", client=client)

    paths = raised.value.run_paths
    summary = read_run_summary(paths.run_summary)
    assert summary.status == "failed"
    assert summary.completed_main_turns == 12
    assert summary.completed_natural_blocks == 0
    assert len(_read_jsonl(paths.reflection_logs)) == 4
    assert paths.natural_block_summaries.read_bytes() == b""
    assert len(client.invocations) == 44


def test_target_summary_failure_leaves_event_incomplete_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_summary(path, record):
        raise OSError("injected target summary failure")

    monkeypatch.setattr(target_runner_module, "append_jsonl", fail_summary)
    client = build_deterministic_dummy_llm_client()

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(tmp_path, run_id="target_summary_failure", client=client)

    paths = raised.value.run_paths
    summary = read_run_summary(paths.run_summary)
    assert summary.status == "failed"
    assert (
        summary.completed_natural_blocks,
        summary.completed_target_events,
        summary.completed_main_turns,
    ) == (3, 0, 40)
    assert len(_read_jsonl(paths.natural_block_summaries)) == 3
    assert paths.event_summaries.read_bytes() == b""
    assert len(_read_jsonl(paths.reflection_logs)) == 16
    assert len(client.invocations) == 152


def test_unexpected_main_result_identity_is_rejected_before_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_run_turn = pipeline_module.CognitivePipeline.run_turn
    calls = 0

    def return_mismatched_result(self, turn_input):
        nonlocal calls
        calls += 1
        result = real_run_turn(self, turn_input)
        return replace(result, turn_id="mismatched_turn")

    monkeypatch.setattr(
        pipeline_module.CognitivePipeline,
        "run_turn",
        return_mismatched_result,
    )

    with pytest.raises(RunExecutionError, match="identity") as raised:
        _run_pilot(tmp_path, run_id="mismatched_main_result")

    paths = raised.value.run_paths
    assert calls == 1
    assert read_run_summary(paths.run_summary).completed_main_turns == 0
    assert paths.retrieval_logs.read_bytes() == b""
    assert paths.behavior_logs.read_bytes() == b""
    assert len(_read_jsonl(paths.memory_logs)) == 12


def test_unexpected_reflection_result_identity_is_rejected_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_reflect = pipeline_module.CognitivePipeline.run_post_step_reflection
    calls = 0

    def return_mismatched_result(self, reflection_input):
        nonlocal calls
        calls += 1
        result = real_reflect(self, reflection_input)
        return replace(result, retrieval_config_id="recency_heavy")

    monkeypatch.setattr(
        pipeline_module.CognitivePipeline,
        "run_post_step_reflection",
        return_mismatched_result,
    )

    with pytest.raises(RunExecutionError, match="reflection result") as raised:
        _run_pilot(tmp_path, run_id="mismatched_reflection_result")

    paths = raised.value.run_paths
    assert calls == 1
    summary = read_run_summary(paths.run_summary)
    assert summary.completed_main_turns == 12
    assert summary.completed_natural_blocks == 0
    assert paths.reflection_logs.read_bytes() == b""
    assert paths.natural_block_summaries.read_bytes() == b""
