"""One-run lifecycle and artifact tests for the deterministic Phase 08 controller."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import src.simulation.run_controller as run_controller_module
from src.config_loader import load_experiment_config
from src.llm import DummyEmbeddingProvider
from src.logging_system import read_run_summary
from src.simulation import (
    RunController,
    RunControllerError,
    RunExecutionError,
    build_deterministic_dummy_llm_client,
)
from src.storage import RunConflictError


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


class _FailingLLMClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.call_count = 0

    def generate(self, request):
        self.call_count += 1
        raise self.error


def _run_pilot(
    output_root: Path,
    *,
    run_id: str,
    llm_client=None,
):
    client = llm_client or build_deterministic_dummy_llm_client()
    result = RunController(
        llm_client=client,
        embedding_provider=DummyEmbeddingProvider(),
    ).run(
        output_root=output_root,
        experiment_id="phase08_lifecycle",
        run_id=run_id,
        retrieval_config_id="balanced",
        repetition_index=1,
        days=(1, 2),
    )
    return result, client


def test_successful_pilot_returns_success_but_persists_excluded_status(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    summary = result.run_summary

    assert result.scope_succeeded is True
    assert result.requested_days == (1, 2)
    assert summary.status == "excluded"
    assert (
        summary.completed_natural_blocks,
        summary.completed_target_events,
        summary.completed_main_turns,
        summary.error_count,
    ) == (3, 1, 40, 0)
    assert read_run_summary(result.run_paths.run_summary) == summary
    with pytest.raises(FrozenInstanceError):
        result.scope_succeeded = False  # type: ignore[misc]


def test_pilot_persists_exact_artifact_counts_and_leaves_dummy_usage_empty(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    paths = result.run_paths

    assert len(_read_jsonl(paths.retrieval_logs)) == 40
    assert len(_read_jsonl(paths.behavior_logs)) == 40
    assert len(_read_jsonl(paths.memory_logs)) == 68
    assert len(_read_jsonl(paths.reflection_logs)) == 16
    assert len(_read_jsonl(paths.natural_block_summaries)) == 3
    assert len(_read_jsonl(paths.event_summaries)) == 1
    assert paths.api_usage_logs.read_bytes() == b""
    assert paths.errors.read_bytes() == b""


def test_seed_memories_are_embedded_at_time_zero_in_snapshot_agent_order(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    memories = _read_jsonl(result.run_paths.memory_logs)
    seeds = tuple(record for record in memories if record["memory_type"] == "seed_persona")
    expected = tuple(
        (agent.agent_id, index, seed.text, seed.category, seed.importance)
        for agent in config.agents
        for index, seed in enumerate(agent.seed_memories, start=1)
    )

    assert len(seeds) == len(expected) == 12
    assert tuple(
        (
            record["agent_id"],
            record["metadata"]["seed_index"],
            record["text"],
            record["category"],
            record["importance"],
        )
        for record in seeds
    ) == expected
    assert all(record["created_at"] == record["last_accessed_at"] == 0 for record in seeds)
    assert all(record["timestamp"] == 0 and record["turn_id"] is None for record in seeds)
    assert all(record["embedding"] == [1.0, 1.0] for record in seeds)


def test_embedding_provider_receives_exact_queries_and_created_memory_texts(
    phase08_pilot_artifacts,
) -> None:
    result, _, provider = phase08_pilot_artifacts
    config = load_experiment_config(result.run_paths.config_snapshot)
    expected_texts: list[str] = [
        seed.text for agent in config.agents for seed in agent.seed_memories
    ]
    for step in result.step_results:
        for turn in step.main_turn_results:
            expected_texts.extend(
                (turn.retrieval_query.text, turn.action.memory_to_store)
            )
        expected_texts.extend(
            reflection.reflection.reflection
            for reflection in step.reflection_results
        )

    assert provider.received_texts == tuple(expected_texts)


@pytest.mark.parametrize(
    "retrieval_config_id",
    ["recency_heavy", "importance_heavy", "relevance_heavy"],
)
def test_selected_nonbalanced_retrieval_configuration_propagates_unchanged(
    tmp_path: Path,
    retrieval_config_id: str,
) -> None:
    result = RunController(
        llm_client=build_deterministic_dummy_llm_client(),
        embedding_provider=DummyEmbeddingProvider(),
    ).run(
        output_root=tmp_path,
        experiment_id="phase08_config_propagation",
        run_id=f"{retrieval_config_id}_rep02",
        retrieval_config_id=retrieval_config_id,
        repetition_index=2,
        days=(1, 2),
    )
    config = load_experiment_config(result.run_paths.config_snapshot)
    configured = next(
        item
        for item in config.retrieval_configs
        if item.config_id == retrieval_config_id
    )
    expected_weights = {
        "recency": configured.weights.recency,
        "importance": configured.weights.importance,
        "relevance": configured.weights.relevance,
    }
    cognitive_results = tuple(
        turn
        for step in result.step_results
        for turn in step.main_turn_results
    )

    assert result.run_summary.retrieval_config_id == retrieval_config_id
    assert result.run_summary.repetition_index == 2
    assert all(
        turn.retrieval_config_id == retrieval_config_id
        and turn.retrieval_result.retrieval_config_id == retrieval_config_id
        and turn.retrieval_result.applied_weights == configured.weights
        for turn in cognitive_results
    )
    for path in result.run_paths.jsonl_files:
        assert all(
            record["retrieval_config_id"] == retrieval_config_id
            and record["repetition_index"] == 2
            for record in _read_jsonl(path)
        )
    assert all(
        record["weights"] == expected_weights
        for record in _read_jsonl(result.run_paths.retrieval_logs)
    )
    assert len(_read_jsonl(result.run_paths.reflection_logs)) == 16


@pytest.mark.parametrize(
    ("retrieval_config_id", "repetition_index", "days", "match"),
    [
        ("unknown", 1, (1, 2), "unknown retrieval_config_id"),
        ("balanced", 0, (1, 2), "repetition_index"),
        ("balanced", 4, (1, 2), "repetition_index"),
        ("balanced", 1, (1,), "supports only"),
        ("balanced", 1, (2, 1), "supports only"),
    ],
)
def test_invalid_run_request_is_rejected_before_publication(
    tmp_path: Path,
    retrieval_config_id: str,
    repetition_index: int,
    days: tuple[int, ...],
    match: str,
) -> None:
    controller = RunController(
        llm_client=build_deterministic_dummy_llm_client(),
        embedding_provider=DummyEmbeddingProvider(),
    )

    with pytest.raises(RunControllerError, match=match):
        controller.run(
            output_root=tmp_path,
            experiment_id="phase08_invalid",
            run_id="invalid_request",
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
            days=days,
        )

    assert not (tmp_path / "phase08_invalid" / "invalid_request").exists()


def test_existing_terminal_run_is_a_conflict_and_is_not_resumed(
    phase08_pilot_artifacts,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    summary_before = result.run_paths.run_summary.read_bytes()

    with pytest.raises(RunConflictError, match="already exists"):
        RunController(
            llm_client=build_deterministic_dummy_llm_client(),
            embedding_provider=DummyEmbeddingProvider(),
        ).run(
            output_root=result.run_paths.output_root,
            experiment_id="phase08_tests",
            run_id="balanced_rep01_pilot",
            retrieval_config_id="balanced",
            repetition_index=1,
            days=(1, 2),
        )

    assert result.run_paths.run_summary.read_bytes() == summary_before


def test_cognition_failure_is_terminalized_failed_without_retry(tmp_path: Path) -> None:
    client = _FailingLLMClient(RuntimeError("originating cognition failure"))
    controller = RunController(
        llm_client=client,
        embedding_provider=DummyEmbeddingProvider(),
    )

    with pytest.raises(RunExecutionError) as raised:
        controller.run(
            output_root=tmp_path,
            experiment_id="phase08_lifecycle",
            run_id="cognition_failure",
            retrieval_config_id="balanced",
            repetition_index=1,
            days=(1, 2),
        )

    assert isinstance(raised.value.original_error, RuntimeError)
    assert client.call_count == 1
    summary = read_run_summary(raised.value.run_paths.run_summary)
    assert summary.status == "failed"
    assert summary.completed_main_turns == 0
    assert summary.error_count == 1
    errors = _read_jsonl(raised.value.run_paths.errors)
    assert len(errors) == 1
    assert errors[0]["error_type"] == "RuntimeError"
    assert errors[0]["message"] == "originating cognition failure"


def test_keyboard_interrupt_is_terminalized_interrupted_and_reraised(
    tmp_path: Path,
) -> None:
    client = _FailingLLMClient(KeyboardInterrupt("operator stop"))
    run_dir = tmp_path / "phase08_lifecycle" / "interrupted_run"

    with pytest.raises(KeyboardInterrupt, match="operator stop"):
        RunController(
            llm_client=client,
            embedding_provider=DummyEmbeddingProvider(),
        ).run(
            output_root=tmp_path,
            experiment_id="phase08_lifecycle",
            run_id="interrupted_run",
            retrieval_config_id="balanced",
            repetition_index=1,
            days=(1, 2),
        )

    assert client.call_count == 1
    summary = read_run_summary(run_dir / "run_summary.json")
    assert summary.status == "interrupted"
    assert summary.completed_main_turns == 0
    assert summary.error_count == 1


def test_terminal_summary_failure_transitions_run_to_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = run_controller_module.write_run_summary
    failed_once = False

    def fail_success_terminal_once(path, summary):
        nonlocal failed_once
        if summary.status == "excluded" and not failed_once:
            failed_once = True
            raise OSError("terminal summary failure")
        return real_write(path, summary)

    monkeypatch.setattr(
        run_controller_module,
        "write_run_summary",
        fail_success_terminal_once,
    )

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(tmp_path, run_id="terminal_summary_failure")

    assert isinstance(raised.value.original_error, OSError)
    assert raised.value.summary_error is None
    summary = read_run_summary(raised.value.run_paths.run_summary)
    assert summary.status == "failed"
    assert (
        summary.completed_natural_blocks,
        summary.completed_target_events,
        summary.completed_main_turns,
        summary.error_count,
    ) == (3, 1, 40, 1)
    assert len(_read_jsonl(raised.value.run_paths.errors)) == 1


def test_keyboard_interrupt_during_success_terminalization_marks_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = run_controller_module.write_run_summary
    interrupted_once = False

    def interrupt_success_terminal_once(path, summary):
        nonlocal interrupted_once
        if summary.status == "excluded" and not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt("interrupt during terminal write")
        return real_write(path, summary)

    monkeypatch.setattr(
        run_controller_module,
        "write_run_summary",
        interrupt_success_terminal_once,
    )
    run_dir = tmp_path / "phase08_lifecycle" / "terminal_interrupt"

    with pytest.raises(KeyboardInterrupt, match="terminal write"):
        _run_pilot(tmp_path, run_id="terminal_interrupt")

    summary = read_run_summary(run_dir / "run_summary.json")
    assert summary.status == "interrupted"
    assert summary.completed_main_turns == 40
    assert summary.error_count == 1


def test_error_record_failure_does_not_hide_origin_or_inflate_error_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_append = run_controller_module.append_jsonl

    def fail_error_record(path, record):
        if Path(path).name == "errors.jsonl":
            raise OSError("secondary error logging failure")
        return real_append(path, record)

    monkeypatch.setattr(run_controller_module, "append_jsonl", fail_error_record)
    client = _FailingLLMClient(RuntimeError("primary execution failure"))

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(
            tmp_path,
            run_id="error_record_failure",
            llm_client=client,
        )

    assert str(raised.value.original_error) == "primary execution failure"
    assert isinstance(raised.value.error_log_error, OSError)
    summary = read_run_summary(raised.value.run_paths.run_summary)
    assert summary.status == "failed"
    assert summary.error_count == 1
    assert raised.value.run_paths.errors.read_bytes() == b""


def test_seed_log_failure_preserves_prior_seeds_and_terminalizes_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_append = run_controller_module.append_jsonl
    seed_writes = 0

    def fail_fifth_seed(path, record):
        nonlocal seed_writes
        if Path(path).name == "memory_logs.jsonl":
            seed_writes += 1
            if seed_writes == 5:
                raise OSError("injected seed log failure")
        return real_append(path, record)

    monkeypatch.setattr(run_controller_module, "append_jsonl", fail_fifth_seed)
    client = build_deterministic_dummy_llm_client()

    with pytest.raises(RunExecutionError) as raised:
        _run_pilot(tmp_path, run_id="seed_log_failure", llm_client=client)

    paths = raised.value.run_paths
    summary = read_run_summary(paths.run_summary)
    assert summary.status == "failed"
    assert summary.completed_main_turns == 0
    assert summary.error_count == 1
    assert len(client.invocations) == 0
    assert len(_read_jsonl(paths.memory_logs)) == 4
    assert len(_read_jsonl(paths.errors)) == 1
