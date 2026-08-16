"""Copied-artifact helpers for offline Phase 12 replay tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Any


def copy_run(source: Path, root: Path) -> Path:
    destination = root / source.parent.name / source.name
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return destination


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="",
    )


def directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def terminal_error(
    summary: dict[str, Any],
    *,
    status: str,
    message: str = "UNSAFE_PROVIDER_SENTINEL_RESPONSE_BODY",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": summary["experiment_id"],
        "run_id": summary["run_id"],
        "retrieval_config_id": summary["retrieval_config_id"],
        "repetition_index": summary["repetition_index"],
        "step_id": None,
        "step_type": None,
        "is_target_event": None,
        "evaluation_enabled": None,
        "day": None,
        "period": None,
        "timestamp": None,
        "event_id": None,
        "turn_id": None,
        "turn_index": None,
        "agent_id": None,
        "error_type": "KeyboardInterrupt" if status == "interrupted" else "ProviderTimeoutError",
        "message": message,
        "recoverable": False,
        "retry_count": 1,
    }


def make_partial_run(
    run_dir: Path,
    *,
    status: str = "failed",
    completed_turns: int = 5,
    retrieval_extra: int = 0,
    reflection_count: int = 0,
    message: str = "UNSAFE_PROVIDER_SENTINEL_RESPONSE_BODY",
) -> Path:
    summary = read_json(run_dir / "run_summary.json")
    summary.update(
        {
            "status": status,
            "completed_natural_blocks": 0,
            "completed_target_events": 0,
            "completed_main_turns": completed_turns,
            "error_count": 1,
        }
    )
    write_json(run_dir / "run_summary.json", summary)
    retrieval = read_jsonl(run_dir / "retrieval_logs.jsonl")[: completed_turns + retrieval_extra]
    behavior = read_jsonl(run_dir / "behavior_logs.jsonl")[:completed_turns]
    memories = read_jsonl(run_dir / "memory_logs.jsonl")
    seeds = [item for item in memories if item["memory_type"] == "seed_persona"]
    actions = [item for item in memories if item["memory_type"] == "action"][:completed_turns]
    reflection_memories = [
        item for item in memories if item["memory_type"] == "reflection"
    ][:reflection_count]
    reflections = read_jsonl(run_dir / "reflection_logs.jsonl")[:reflection_count]
    retained_ids = {
        item["memory_id"] for item in (*seeds, *actions, *reflection_memories)
    }
    write_jsonl(run_dir / "retrieval_logs.jsonl", retrieval)
    write_jsonl(run_dir / "behavior_logs.jsonl", behavior)
    write_jsonl(
        run_dir / "memory_logs.jsonl",
        [item for item in memories if item["memory_id"] in retained_ids],
    )
    write_jsonl(run_dir / "reflection_logs.jsonl", reflections)
    write_jsonl(run_dir / "natural_block_summaries.jsonl", [])
    write_jsonl(run_dir / "event_summaries.jsonl", [])
    write_jsonl(run_dir / "api_usage_logs.jsonl", [])
    write_jsonl(run_dir / "errors.jsonl", [terminal_error(summary, status=status, message=message)])
    return run_dir


def make_diagnostic_evaluation(run_dir: Path, output: Path) -> Path:
    output.mkdir()
    summary = read_json(run_dir / "run_summary.json")
    mode = "primary" if summary["status"] == "completed" else "diagnostic"
    eligible = mode == "primary"
    target_rows = [
        row
        for row in read_jsonl(run_dir / "behavior_logs.jsonl")
        if row["step_type"] == "target_event"
    ]
    retrieval_by_turn = {
        row["turn_id"]: row for row in read_jsonl(run_dir / "retrieval_logs.jsonl")
    }
    mra_rows: list[dict[str, Any]] = []
    bcr_rows: list[dict[str, Any]] = []
    for row in target_rows:
        retrieval = retrieval_by_turn[row["turn_id"]]
        selected = list(retrieval["selected_memory_ids"])
        audit = {
            name: row[name]
            for name in (
                "experiment_id",
                "run_id",
                "retrieval_config_id",
                "repetition_index",
                "event_id",
                "step_id",
                "turn_id",
                "turn_index",
                "agent_id",
            )
        }
        audit["timestamp"] = row["timestamp"]
        mra_rows.append(
            {
                "schema_version": "1.0",
                "evaluation_mode": mode,
                "primary_eligible": eligible,
                "included_in_final_research_aggregation": eligible,
                **audit,
                "top_k": 5,
                "retrieved_count": len(selected),
                "candidate_count": retrieval["candidate_count"],
                "selected_memory_ids": selected,
                "retrieved_categories": [
                    item["category"] for item in retrieval["candidates"] if item["selected"]
                ],
                "expected_categories": ["classroom_norms"],
                "matched_memory_ids": selected[:1],
                "unmatched_memory_ids": selected[1:],
                "matched_categories": ["classroom_norms"] if selected else [],
                "matched_count": 1 if selected else 0,
                "denominator": 5,
                "mra": 0.2 if selected else 0.0,
            }
        )
        bcr_rows.append(
            {
                "schema_version": "1.0",
                "evaluation_mode": mode,
                "primary_eligible": eligible,
                "included_in_final_research_aggregation": eligible,
                "audit_metadata": audit,
                "selected_memory_evidence": [],
                "prior_behavior_evidence": [],
                "model_metadata": {"simulator_model_identity": None},
                "judge_input": {"diagnostic_neutral_export": True},
            }
        )
    write_jsonl(output / "mra_records.jsonl", mra_rows)
    write_jsonl(output / "bcr_export.jsonl", bcr_rows)
    write_json(
        output / "evaluation_summary.json",
        {
            "schema_version": "1.0",
            "source_schema_version": "1.0",
            "experiment_id": summary["experiment_id"],
            "run_id": summary["run_id"],
            "retrieval_config_id": summary["retrieval_config_id"],
            "repetition_index": summary["repetition_index"],
            "source_run_status": summary["status"],
            "evaluation_mode": mode,
            "primary_eligible": eligible,
            "included_in_final_research_aggregation": eligible,
            "mra_formula": "persisted_fixture_value_only",
            "mra_aggregates": {},
            "bcr_export_count": len(bcr_rows),
            "bcr_judgment": {"status": "not_judged"},
            "model_metadata": {"simulator_model_identity": None},
            "output_files": ["mra_records.jsonl", "bcr_export.jsonl"],
        },
    )
    return output


def make_primary_judgment(evaluation: Path, output: Path) -> Path:
    output.mkdir()
    evaluation_summary = read_json(evaluation / "evaluation_summary.json")
    bcr_lines = (evaluation / "bcr_export.jsonl").read_bytes().splitlines(keepends=True)
    bcr_rows = read_jsonl(evaluation / "bcr_export.jsonl")
    judged_rows: list[dict[str, Any]] = []
    for bcr_row, line in zip(bcr_rows, bcr_lines, strict=True):
        judged_rows.append(
            {
                "schema_version": "1.0",
                "evaluation_mode": "primary",
                "primary_eligible": True,
                "included_in_final_research_aggregation": True,
                "audit_metadata": bcr_row["audit_metadata"],
                "source_record_hash": sha256(line).hexdigest(),
                "judgment": {
                    "communication_style_score": 0.8,
                    "focus_consistency_score": 0.7,
                    "temporal_consistency_score": 0.9,
                    "communication_style_reason": "Persisted safe fixture reason.",
                    "focus_consistency_reason": "Persisted safe fixture reason.",
                    "temporal_consistency_reason": "Persisted safe fixture reason.",
                    "bcr": 0.8,
                },
                "model_metadata": {"evaluator_model_identity": "fixture:evaluator"},
                "prompt_identity": {"template_id": "bcr_judge", "prompt_hash": "2" * 64},
                "provider_attempts": [],
            }
        )
    write_jsonl(output / "bcr_judged_records.jsonl", judged_rows)
    write_json(
        output / "bcr_judgment_summary.json",
        {
            "schema_version": "1.0",
            "source_evaluation_schema_version": "1.0",
            "experiment_id": evaluation_summary["experiment_id"],
            "run_id": evaluation_summary["run_id"],
            "retrieval_config_id": evaluation_summary["retrieval_config_id"],
            "repetition_index": evaluation_summary["repetition_index"],
            "evaluation_mode": "primary",
            "primary_eligible": True,
            "included_in_final_research_aggregation": True,
            "source_bcr_export_hash": sha256(
                (evaluation / "bcr_export.jsonl").read_bytes()
            ).hexdigest(),
            "judged_unit_count": len(judged_rows),
            "bcr_formula": "persisted_fixture_value_only",
            "run_aggregation": {},
            "run_mean_bcr": 0.8,
            "model_metadata": {"evaluator_model_identity": "fixture:evaluator"},
            "prompt_identity": {"template_id": "bcr_judge", "prompt_hash": "2" * 64},
            "runtime_settings": {},
            "provider_attempt_count": 0,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "missing_total_usage_attempt_count": 0,
            },
            "output_files": ["bcr_judged_records.jsonl"],
        },
    )
    return output


def content_hash(value: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def directory_fingerprint(path: Path) -> str:
    digest = sha256()
    for item in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def make_dummy_pilot_evidence(run_dir: Path, output: Path) -> Path:
    output.mkdir()
    summary = read_json(run_dir / "run_summary.json")
    config_hashes = [
        {"filename": item.name, "sha256": sha256(item.read_bytes()).hexdigest()}
        for item in sorted((run_dir / "config_snapshot").iterdir())
    ]
    prompt_ids = (
        "plan",
        "natural_interaction",
        "target_event_response",
        "reflect",
        "importance_scoring",
        "bcr_judge",
    )
    prompts = [
        {
            "template_id": name,
            "filename": f"{name}.txt",
            "prompt_version": "1.0",
            "prompt_hash": sha256(name.encode("utf-8")).hexdigest(),
            "model_role": "evaluator" if name == "bcr_judge" else "simulator",
            "output_contract": f"{name}_output",
        }
        for name in prompt_ids
    ]
    call_limits = {
        "simulator_logical_calls": 152,
        "embedding_logical_calls": 108,
        "evaluator_logical_calls": 0,
        "total_logical_calls": 260,
        "simulator_physical_attempts": 304,
        "embedding_physical_attempts": 216,
        "evaluator_physical_attempts": 0,
        "total_physical_attempts": 520,
    }
    plan = {
        "mode": "dummy",
        "retrieval_config_id": summary["retrieval_config_id"],
        "repetition_index": summary["repetition_index"],
        "days": [1, 2],
        "step_ids": [
            "day_01_natural_01",
            "day_01_natural_02",
            "day_02_natural_03",
            "day_02_target_01",
        ],
        "natural_block_count": 3,
        "target_event_count": 1,
        "main_turn_count": 40,
        "post_step_reflection_count": 16,
        "target_evaluation_unit_count": 4,
        "call_limits": call_limits,
    }
    manifest_base = {
        "schema_version": "1.0",
        "pilot_id": summary["run_id"],
        "experiment_id": summary["experiment_id"],
        "run_id": summary["run_id"],
        "pilot_only": True,
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
        "expected_source_run_status": "excluded",
        "plan": plan,
        "max_total_tokens": None,
        "configuration_hashes": config_hashes,
        "prompts": prompts,
        "simulator_settings": {"model_identity": "dummy:offline"},
        "evaluator_settings": {"model_identity": "openai:gpt-5-2025-08-07"},
        "embedding_settings": {"model_identity": "dummy:offline", "dimension": 2},
        "retry_policy": {"maximum_total_attempts": 2, "deterministic_delay_seconds": 0, "jitter": False},
        "evaluation_schema_version": "1.0",
        "research_log_schema_version": "1.0",
        "bcr_rubric_version": "1.0",
    }
    manifest = {**manifest_base, "manifest_hash": content_hash(manifest_base)}
    role_usage = [
        {
            "service_role": role,
            "logical_call_count": 0,
            "physical_attempt_count": 0,
            "retry_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "missing_total_usage_attempt_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "invalid_output_count": 0,
        }
        for role in ("simulator", "embedding", "evaluator")
    ]
    usage = {
        "schema_version": "1.0",
        "max_total_tokens": None,
        "reported_total_tokens": 0,
        "logical_call_count": 0,
        "physical_attempt_count": 0,
        "retry_count": 0,
        "missing_total_usage_attempt_count": 0,
        "budget_exceeded": False,
        "role_usage": role_usage,
    }
    source_fingerprint = directory_fingerprint(run_dir)
    validation = {
        "source_run_status": "excluded",
        "natural_block_count": 3,
        "target_event_count": 1,
        "main_turn_count": 40,
        "retrieval_record_count": 40,
        "behavior_record_count": 40,
        "memory_record_count": 68,
        "reflection_count": 16,
        "error_record_count": 0,
        "api_usage_record_count": 0,
        "target_evaluation_unit_count": 4,
        "simulator_model_identity": None,
        "top_k_validated": True,
        "secret_scan_passed": True,
        "source_fingerprint": source_fingerprint,
    }
    report = {
        "schema_version": "1.0",
        "pilot_id": summary["run_id"],
        "experiment_id": summary["experiment_id"],
        "run_id": summary["run_id"],
        "mode": "dummy",
        "manifest_hash": manifest["manifest_hash"],
        "acceptance_state": "pending",
        "technical_success": True,
        "eligible_for_manual_acceptance": False,
        "source_run_status": "excluded",
        "source_run_directory_name": summary["run_id"],
        "diagnostic_evaluation_directory_name": f"{summary['run_id']}_diagnostic_evaluation",
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
        "usage": usage,
        "validation": validation,
        "source_fingerprint": source_fingerprint,
        "diagnostic_evaluation_fingerprint": "1" * 64,
        "mra_diagnostic_record_count": 4,
        "mra_diagnostic_mean": 0.2,
        "bcr_export_record_count": 4,
        "bcr_judgment_count": 0,
        "technical_checks": [
            "authorization",
            "fixed_scope",
            "model_and_settings_identity",
            "prompt_inventory_and_hashes",
            "configuration_snapshot",
            "token_and_call_budget",
            "source_run_artifact_completeness",
            "target_only_diagnostic_evaluation",
            "secret_and_payload_isolation",
            "source_run_immutability_after_evaluation",
        ],
        "rejection_reasons": [],
        "terminal_error_type": None,
    }
    write_json(output / "pilot_manifest.json", manifest)
    write_jsonl(output / "pilot_attempts.jsonl", [])
    write_json(output / "pilot_usage_summary.json", usage)
    write_json(output / "pilot_report.json", report)
    return output


def make_rejected_pilot_review(evidence: Path, output: Path) -> Path:
    output.mkdir()
    report = read_json(evidence / "pilot_report.json")
    manifest = read_json(evidence / "pilot_manifest.json")
    acceptance_base = {
        "schema_version": "1.0",
        "pilot_id": report["pilot_id"],
        "experiment_id": report["experiment_id"],
        "run_id": report["run_id"],
        "acceptance_state": "rejected",
        "manual_review": True,
        "manifest_hash": manifest["manifest_hash"],
        "evidence_hash": directory_fingerprint(evidence),
        "source_run_status": report["source_run_status"],
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
    }
    write_json(
        output / "pilot_acceptance.json",
        {**acceptance_base, "acceptance_hash": content_hash(acceptance_base)},
    )
    return output


def make_failed_real_pilot_evidence(run_dir: Path, output: Path) -> Path:
    make_dummy_pilot_evidence(run_dir, output)
    manifest = read_json(output / "pilot_manifest.json")
    manifest["plan"]["mode"] = "real"
    manifest["max_total_tokens"] = 100
    manifest_base = dict(manifest)
    manifest_base.pop("manifest_hash")
    manifest["manifest_hash"] = content_hash(manifest_base)
    attempt = {
        "schema_version": "1.0",
        "service_role": "embedding",
        "logical_call_id": "embedding_000001_" + "3" * 64,
        "attempt_index": 1,
        "call_type": "embedding",
        "model": "openai:text-embedding-3-small",
        "prompt_template": None,
        "prompt_version": None,
        "prompt_hash": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "status": "failed",
        "error_type": "ProviderTimeoutError",
    }
    usage = read_json(output / "pilot_usage_summary.json")
    usage.update(
        max_total_tokens=100,
        logical_call_count=1,
        physical_attempt_count=1,
        retry_count=0,
        missing_total_usage_attempt_count=1,
    )
    embedding = usage["role_usage"][1]
    embedding.update(
        logical_call_count=1,
        physical_attempt_count=1,
        missing_total_usage_attempt_count=1,
        failed_count=1,
    )
    report = read_json(output / "pilot_report.json")
    report.update(
        mode="real",
        manifest_hash=manifest["manifest_hash"],
        technical_success=False,
        eligible_for_manual_acceptance=False,
        source_run_status=read_json(run_dir / "run_summary.json")["status"],
        usage=usage,
        validation=None,
        diagnostic_evaluation_fingerprint=None,
        mra_diagnostic_record_count=0,
        mra_diagnostic_mean=None,
        bcr_export_record_count=0,
        technical_checks=[],
        rejection_reasons=["ProviderTimeoutError"],
        terminal_error_type="ProviderTimeoutError",
    )
    write_json(output / "pilot_manifest.json", manifest)
    write_jsonl(output / "pilot_attempts.jsonl", [attempt])
    write_json(output / "pilot_usage_summary.json", usage)
    write_json(output / "pilot_report.json", report)
    return output


def make_forged_accepted_review(evidence: Path, output: Path) -> Path:
    output.mkdir()
    report = read_json(evidence / "pilot_report.json")
    manifest = read_json(evidence / "pilot_manifest.json")
    acceptance_base = {
        "schema_version": "1.0",
        "pilot_id": report["pilot_id"],
        "experiment_id": report["experiment_id"],
        "run_id": report["run_id"],
        "acceptance_state": "accepted",
        "manual_review": True,
        "manifest_hash": manifest["manifest_hash"],
        "evidence_hash": directory_fingerprint(evidence),
        "source_run_status": report["source_run_status"],
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
    }
    write_json(
        output / "pilot_acceptance.json",
        {**acceptance_base, "acceptance_hash": content_hash(acceptance_base)},
    )
    return output
