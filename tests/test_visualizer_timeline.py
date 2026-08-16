from __future__ import annotations

from pathlib import Path

import pytest

from phase12_test_support import copy_run, make_partial_run, read_json, read_jsonl, write_json, write_jsonl
from src.visualizer import ReplayEligibilityError, ReplayIdentityError, ReplayOrderingError, build_replay


def test_excluded_pilot_reconstructs_exact_day_1_2_timeline(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document

    assert len(document.steps) == 12
    assert [step.state for step in document.steps[:4]] == ["complete"] * 4
    assert [step.state for step in document.steps[4:]] == ["out_of_scope"] * 8
    assert sum(len(step.turns) for step in document.steps) == 40
    assert sum(len(step.reflections) for step in document.steps) == 16
    assert [turn.logical_timestamp for step in document.steps for turn in step.turns] == list(range(1, 41))
    assert len(document.playback) == 1 + 40 + 16


def test_complete_run_reconstructs_exact_104_turns(phase08_full_artifacts) -> None:
    result, _, _ = phase08_full_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document
    turns = [turn for step in document.steps for turn in step.turns]

    assert document.run.source_status == "completed"
    assert document.run.primary_research_data is True
    assert [step.state for step in document.steps] == ["complete"] * 12
    assert len(turns) == 104
    assert [turn.logical_timestamp for turn in turns] == list(range(1, 105))
    assert len(document.memories) == 164
    assert len(document.playback) == 1 + 104 + 48


def test_natural_speaker_and_target_participant_orders_are_persisted(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document

    assert [turn.agent_id for turn in document.steps[0].turns] == [
        "teacher", "active_student", "academic_student", "quiet_student",
        "active_student", "teacher", "academic_student", "quiet_student",
        "academic_student", "active_student", "quiet_student", "teacher",
    ]
    assert [turn.agent_id for turn in document.steps[3].turns] == list(document.steps[3].participants)


def test_reflections_follow_main_turns_and_reuse_step_final_time(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document

    for step in document.steps[:4]:
        assert len(step.reflections) == 4
        assert {item.logical_timestamp for item in step.reflections} == {step.turns[-1].logical_timestamp}
        frames = [frame for frame in document.playback if frame.step_id == step.step_id]
        assert [frame.frame_kind for frame in frames[-4:]] == ["reflection"] * 4


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_terminal_run_is_partial_and_never_upgraded(
    tmp_path: Path, phase08_pilot_artifacts, status: str
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / status)
    make_partial_run(run_dir, status=status, completed_turns=5, retrieval_extra=1)

    document = build_replay(run_dir=run_dir).document

    assert document.run.source_status == status
    assert document.run.replay_completeness == "partial_persisted_prefix"
    assert document.steps[0].state == "partial"
    assert len(document.steps[0].turns) == 6
    assert document.steps[0].turns[-1].completeness == "partial"
    assert document.steps[0].turns[-1].response is None
    assert document.failure is not None
    assert document.failure.safe_message.startswith("Run ")
    assert document.playback[-1].frame_kind == "failure"


def test_provider_error_text_is_not_projected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "safe_error")
    sentinel = "UNSAFE_PROVIDER_SENTINEL_RESPONSE_BODY_AUTHORIZATION"
    make_partial_run(run_dir, completed_turns=5, message=sentinel)

    document = build_replay(run_dir=run_dir).document
    assert document.failure is not None
    assert document.failure.error_type == "ProviderTimeoutError"
    assert sentinel not in str(document.to_dict())


def test_initialized_run_is_not_replay_eligible(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "initialized")
    summary = read_json(run_dir / "run_summary.json")
    summary.update(
        status="initialized",
        completed_natural_blocks=0,
        completed_target_events=0,
        completed_main_turns=0,
        error_count=0,
    )
    write_json(run_dir / "run_summary.json", summary)

    with pytest.raises(ReplayEligibilityError, match="initialized"):
        build_replay(run_dir=run_dir)


def test_cross_run_record_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "foreign")
    records = read_jsonl(run_dir / "behavior_logs.jsonl")
    records[0]["run_id"] = "foreign_run"
    write_jsonl(run_dir / "behavior_logs.jsonl", records)

    with pytest.raises(ReplayIdentityError, match="cross-run"):
        build_replay(run_dir=run_dir)


def test_out_of_order_turn_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "order")
    records = read_jsonl(run_dir / "retrieval_logs.jsonl")
    records[0], records[1] = records[1], records[0]
    write_jsonl(run_dir / "retrieval_logs.jsonl", records)

    with pytest.raises(ReplayOrderingError, match="deterministic prefix"):
        build_replay(run_dir=run_dir)


def test_seed_memories_are_timestamp_zero_and_typed(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    document = build_replay(run_dir=result.run_paths.run_dir).document
    seeds = [item for item in document.memories if item.memory_type == "seed_persona"]

    assert len(seeds) == 12
    assert {item.created_at for item in seeds} == {0}
    assert {item.source_step_type for item in seeds} == {"agent_initialization"}
    assert all(item.source_turn_id is None for item in seeds)


def test_retrieval_persisted_score_order_is_not_recomputed(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "scores")
    records = read_jsonl(run_dir / "retrieval_logs.jsonl")
    persisted_second = records[0]["candidates"][1]["final_score"]
    records[0]["candidates"][0]["final_score"] = 987.654
    write_jsonl(run_dir / "retrieval_logs.jsonl", records)

    first = build_replay(run_dir=run_dir).document.steps[0].turns[0].retrieval
    assert first is not None
    assert [item.final_score for item in first.candidates[:2]] == [987.654, persisted_second]
    assert [item.rank for item in first.candidates[:2]] == [1, 2]


def test_partial_post_step_reflections_reuse_final_main_timestamp(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "partial_reflection")
    make_partial_run(run_dir, completed_turns=12, reflection_count=2)

    step = build_replay(run_dir=run_dir).document.steps[0]
    assert step.state == "partial"
    assert len(step.turns) == 12
    assert len(step.reflections) == 2
    assert {item.logical_timestamp for item in step.reflections} == {12}
    assert {item.completeness for item in step.reflections} == {"complete"}


def test_retrieval_score_rank_corruption_is_not_silently_repaired(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "corrupt_score_order")
    records = read_jsonl(run_dir / "retrieval_logs.jsonl")
    records[0]["candidates"][0]["final_score"] = -999.0
    write_jsonl(run_dir / "retrieval_logs.jsonl", records)

    with pytest.raises(ReplayOrderingError, match="rank order"):
        build_replay(run_dir=run_dir)


def test_reflection_memory_without_final_log_is_visible_as_partial(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "partial_reflection_log")
    make_partial_run(run_dir, completed_turns=12, reflection_count=2)
    records = read_jsonl(run_dir / "reflection_logs.jsonl")
    write_jsonl(run_dir / "reflection_logs.jsonl", records[:1])

    reflections = build_replay(run_dir=run_dir).document.steps[0].reflections
    assert [item.completeness for item in reflections] == ["complete", "partial"]
    assert reflections[1].memory_id is not None
    assert reflections[1].text is not None
