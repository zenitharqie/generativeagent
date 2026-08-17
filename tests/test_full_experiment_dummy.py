"""Complete offline Phase 13 12-run source-plus-Phase-09 proof."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Iterator

import pytest

import src.evaluation.judgment_service as judgment_module
import src.simulation.experiment_controller as controller_module
import src.simulation.experiment_validator as experiment_validator_module
from src.llm.openai_client import OpenAIProviderAdapter
from src.simulation import (
    ExperimentExecutionResult,
    ExperimentValidationError,
    FULL_EXPERIMENT_CONFIRMATION,
    execute_full_experiment,
    fingerprint_directory as pilot_fingerprint_directory,
    hash_config_directory,
    validate_complete_chains,
    validate_experiment_manifest,
    validate_phase09_evaluation,
    validate_run_chain_unchanged,
    validate_source_run,
)
from src.storage import RunConflictError


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


@pytest.fixture(scope="module")
def two_complete_dummy_experiments(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[ExperimentExecutionResult, ExperimentExecutionResult]]:
    patcher = pytest.MonkeyPatch()
    touched = {"provider": 0, "judge": 0, "dummy_llm": 0, "dummy_embedding": 0}
    original_dummy_llm = controller_module.build_deterministic_dummy_llm_client
    original_dummy_embedding = controller_module.DummyEmbeddingProvider

    def provider_forbidden(*args: object, **kwargs: object) -> object:
        touched["provider"] += 1
        raise AssertionError("a real provider boundary was constructed")

    def judge_forbidden(*args: object, **kwargs: object) -> object:
        touched["judge"] += 1
        raise AssertionError("BCR judgment was invoked")

    def fresh_dummy_llm(*args: object, **kwargs: object) -> object:
        touched["dummy_llm"] += 1
        return original_dummy_llm(*args, **kwargs)

    def fresh_dummy_embedding(*args: object, **kwargs: object) -> object:
        touched["dummy_embedding"] += 1
        return original_dummy_embedding(*args, **kwargs)

    patcher.setattr(controller_module, "_default_provider_factory", provider_forbidden)
    patcher.setattr(OpenAIProviderAdapter, "from_environment", provider_forbidden)
    patcher.setattr(judgment_module, "judge_bcr_export", judge_forbidden)
    patcher.setattr(judgment_module.BCRJudgeService, "__init__", judge_forbidden)
    patcher.setattr(
        controller_module, "build_deterministic_dummy_llm_client", fresh_dummy_llm
    )
    patcher.setattr(controller_module, "DummyEmbeddingProvider", fresh_dummy_embedding)
    first_root = tmp_path_factory.mktemp("phase13_dummy_one")
    second_root = tmp_path_factory.mktemp("phase13_dummy_two")
    try:
        first = execute_full_experiment(
            mode="dummy",
            output_root=first_root,
            experiment_id="phase13_mocked_proof",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )
        second = execute_full_experiment(
            mode="dummy",
            output_root=second_root,
            experiment_id="phase13_mocked_proof",
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )
        assert touched == {
            "provider": 0,
            "judge": 0,
            "dummy_llm": 24,
            "dummy_embedding": 24,
        }
        yield first, second
    finally:
        patcher.undo()


def test_dummy_controller_completes_exact_twelve_unique_chains(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]

    assert len(result.run_chains) == 12
    assert len({chain.run.run_id for chain in result.run_chains}) == 12
    assert all(chain.eligible for chain in result.run_chains)
    assert all(chain.source.source_status == "completed" for chain in result.run_chains)
    assert [chain.run for chain in result.run_chains] == list(result.plan.runs)


def test_dummy_controller_persists_separate_source_and_evaluation_directories(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    sources = tuple(result.paths.source_run_dir(run.run_id) for run in result.plan.runs)
    evaluations = tuple(result.paths.evaluation_dir(run.run_id) for run in result.plan.runs)

    assert len(set(sources)) == len(set(evaluations)) == 12
    assert all(path.is_dir() for path in sources)
    assert all(path.is_dir() for path in evaluations)
    assert all(evaluation.parent == result.paths.evaluations_dir for evaluation in evaluations)
    assert all(evaluation not in source.parents for source, evaluation in zip(sources, evaluations))


def test_every_dummy_source_has_exact_full_structure_and_fresh_state(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    seed_id_sequences: list[tuple[str, ...]] = []
    for chain in result.run_chains:
        source = result.paths.source_run_dir(chain.run.run_id)
        retrievals = _read_jsonl(source / "retrieval_logs.jsonl")
        behaviors = _read_jsonl(source / "behavior_logs.jsonl")
        memories = _read_jsonl(source / "memory_logs.jsonl")
        reflections = _read_jsonl(source / "reflection_logs.jsonl")
        assert len(retrievals) == len(behaviors) == 104
        assert len(memories) == 164
        assert len(reflections) == 48
        assert len(_read_jsonl(source / "natural_block_summaries.jsonl")) == 7
        assert len(_read_jsonl(source / "event_summaries.jsonl")) == 5
        assert (source / "errors.jsonl").read_bytes() == b""
        assert (source / "api_usage_logs.jsonl").read_bytes() == b""
        assert all(record["run_id"] == chain.run.run_id for record in memories)
        seeds = tuple(
            record["memory_id"]
            for record in memories
            if record["memory_type"] == "seed_persona"
        )
        assert len(seeds) == 12
        seed_id_sequences.append(seeds)
    assert all(sequence == seed_id_sequences[0] for sequence in seed_id_sequences)


def test_only_planned_retrieval_identity_and_weights_differ_across_runs(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    snapshot_hashes = {
        tuple(
            (item.filename, item.sha256)
            for item in hash_config_directory(
                result.paths.source_run_dir(run.run_id) / "config_snapshot"
            )
        )
        for run in result.plan.runs
    }
    assert len(snapshot_hashes) == 1
    assert {
        (
            run.retrieval_config_id,
            run.recency_weight,
            run.importance_weight,
            run.relevance_weight,
        )
        for run in result.plan.runs
    } == {
        ("balanced", 0.33, 0.33, 0.33),
        ("recency_heavy", 0.60, 0.20, 0.20),
        ("relevance_heavy", 0.20, 0.20, 0.60),
        ("importance_heavy", 0.20, 0.60, 0.20),
    }


def test_every_phase09_output_has_20_mra_and_20_condition_blind_exports(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    total_mra = 0
    total_bcr = 0
    forbidden = {
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
        "weights",
        "retrieval_score",
        "normalized_retrieval_score",
        "mra",
        "ground_truth",
        "expected_categories",
        "memory_id",
    }
    for run in result.plan.runs:
        evaluation = result.paths.evaluation_dir(run.run_id)
        mra = _read_jsonl(evaluation / "mra_records.jsonl")
        bcr = _read_jsonl(evaluation / "bcr_export.jsonl")
        summary = json.loads((evaluation / "evaluation_summary.json").read_text())
        assert len(mra) == len(bcr) == 20
        assert summary["bcr_export_count"] == 20
        assert summary["mra_aggregates"]["unit_count"] == 20
        assert summary["bcr_judgment"]["decision_type"] == "binary"
        assert summary["bcr_judgment"]["allowed_final_values"] == [0, 1]
        for row in bcr:
            judge_input = row["judge_input"]
            assert forbidden.isdisjoint(judge_input)
            serialized = json.dumps(judge_input, sort_keys=True)
            assert "balanced_rep" not in serialized
            assert "recency_heavy_rep" not in serialized
            assert "relevance_heavy_rep" not in serialized
            assert "importance_heavy_rep" not in serialized
        total_mra += len(mra)
        total_bcr += len(bcr)
    assert total_mra == total_bcr == 240


def test_phase13_creates_no_judgment_artifact_and_no_evaluator_attempt(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    names = {path.name for path in result.paths.experiment_dir.rglob("*")}
    assert "bcr_judged_records.jsonl" not in names
    assert "bcr_judgment_summary.json" not in names
    assert result.usage.to_dict()["evaluator_logical_calls"] == 0
    assert result.usage.to_dict()["evaluator_physical_attempts"] == 0
    assert result.manifest["actual_totals"]["bcr_judgment_count"] == 0


def test_dummy_usage_accounts_for_all_logical_operations_without_fake_attempts(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    usage = result.usage

    assert usage.logical_call_count == 8112
    assert usage.physical_attempt_count == 0
    assert usage.retry_count == 0
    assert usage.missing_usage_attempt_count == 0
    assert usage.actual_total_tokens == 0
    assert usage.budget_state == "not_applicable"
    assert result.paths.attempts_file.read_bytes() == b""


def test_final_manifest_requires_and_reports_all_twelve_chains(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    manifest = json.loads(result.paths.manifest_file.read_text())

    assert manifest["status"] == "completed"
    assert manifest["completion_contract"] == "source_plus_phase09_evaluation_ready"
    assert manifest["actual_totals"] == {
        "run_count": 12,
        "source_run_count": 12,
        "evaluation_count": 12,
        "step_count": 144,
        "main_turn_count": 1248,
        "post_step_reflection_count": 576,
        "mra_record_count": 240,
        "bcr_export_record_count": 240,
        "bcr_judgment_count": 0,
    }
    assert len(manifest["runs"]) == 12
    assert manifest["final_validation_state"] == "valid"


def test_standalone_manifest_validation_rejects_nested_ineligibility_or_missing_usage(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    valid = json.loads(result.paths.manifest_file.read_text())
    validate_experiment_manifest(valid, plan=result.plan)

    invalid_evaluation = deepcopy(valid)
    invalid_evaluation["runs"][0]["evaluation"]["mra_record_count"] = 19
    with pytest.raises(ExperimentValidationError, match="evaluation completion"):
        validate_experiment_manifest(invalid_evaluation, plan=result.plan)

    invalid_usage = deepcopy(valid)
    invalid_usage["usage"]["missing_usage_attempt_count"] = 1
    invalid_usage["usage"]["usage_complete"] = False
    invalid_usage["usage"]["actual_total_tokens"] = None
    with pytest.raises(ExperimentValidationError, match="usage completion"):
        validate_experiment_manifest(invalid_usage, plan=result.plan)


def test_eleven_of_twelve_chains_cannot_validate_completion(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    with pytest.raises(ExperimentValidationError, match="exactly 12"):
        validate_complete_chains(result.plan, result.run_chains[:11])


def test_two_complete_dummy_manifests_and_plans_are_byte_identical(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    first, second = two_complete_dummy_experiments
    assert first.paths.plan_file.read_bytes() == second.paths.plan_file.read_bytes()
    assert first.paths.manifest_file.read_bytes() == second.paths.manifest_file.read_bytes()
    assert experiment_validator_module.fingerprint_directory(
        first.paths.experiment_dir
    ) == experiment_validator_module.fingerprint_directory(second.paths.experiment_dir)


def test_sealed_source_and_evaluation_artifacts_remain_unchanged(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    before = {
        chain.run.run_id: (
            pilot_fingerprint_directory(result.paths.source_run_dir(chain.run.run_id)),
            pilot_fingerprint_directory(result.paths.evaluation_dir(chain.run.run_id)),
        )
        for chain in result.run_chains
    }
    for chain in result.run_chains:
        validate_run_chain_unchanged(
            chain,
            source=result.paths.source_run_dir(chain.run.run_id),
            evaluation=result.paths.evaluation_dir(chain.run.run_id),
        )
    after = {
        chain.run.run_id: (
            pilot_fingerprint_directory(result.paths.source_run_dir(chain.run.run_id)),
            pilot_fingerprint_directory(result.paths.evaluation_dir(chain.run.run_id)),
        )
        for chain in result.run_chains
    }
    assert after == before


def test_existing_experiment_is_never_reused_or_overwritten(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    original = result.paths.manifest_file.read_bytes()

    with pytest.raises(RunConflictError):
        execute_full_experiment(
            mode="dummy",
            output_root=result.paths.output_root,
            experiment_id=result.plan.experiment_id,
            confirmation=FULL_EXPERIMENT_CONFIRMATION,
        )

    assert result.paths.manifest_file.read_bytes() == original


def test_manifest_contains_no_absolute_output_paths_or_research_conclusions(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
) -> None:
    result = two_complete_dummy_experiments[0]
    payload = result.paths.manifest_file.read_text(encoding="utf-8")
    assert str(result.paths.output_root) not in payload
    for forbidden in (
        "configuration_ranking",
        "winning_configuration",
        "statistical_significance",
        "confidence_interval",
        "research_conclusion",
        "raw_response",
        "api_key",
    ):
        assert forbidden not in payload.lower()


def test_corrupt_source_and_evaluation_are_rejected_without_repair(
    two_complete_dummy_experiments: tuple[ExperimentExecutionResult, ExperimentExecutionResult],
    tmp_path: Path,
) -> None:
    result = two_complete_dummy_experiments[0]
    spec = result.plan.runs[0]
    source_copy = tmp_path / spec.experiment_id / spec.run_id
    evaluation_copy = tmp_path / "evaluation"
    source_copy.parent.mkdir()
    shutil.copytree(result.paths.source_run_dir(spec.run_id), source_copy)
    shutil.copytree(result.paths.evaluation_dir(spec.run_id), evaluation_copy)
    (source_copy / "reflection_logs.jsonl").write_bytes(b"")

    with pytest.raises(ExperimentValidationError, match="count"):
        validate_source_run(
            source_copy,
            spec=spec,
            scope=result.plan.run_scope,
            mode="dummy",
        )

    bcr_path = evaluation_copy / "bcr_export.jsonl"
    rows = list(_read_jsonl(bcr_path))
    rows[0]["judge_input"]["retrieval_config_id"] = "balanced"
    bcr_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    unsafe_bytes = bcr_path.read_bytes()
    with pytest.raises(ExperimentValidationError, match="BCR export"):
        validate_phase09_evaluation(evaluation_copy, spec=spec)
    assert bcr_path.read_bytes() == unsafe_bytes
