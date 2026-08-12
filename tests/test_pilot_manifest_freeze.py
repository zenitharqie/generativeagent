"""Prompt/settings manifest and explicit offline review/freeze contracts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from phase11_test_support import Phase11MockProvider
from src.llm.prompt_loader import DEFAULT_PROMPT_DIRECTORY, PromptLoader
from src.simulation.pilot import (
    PilotReviewError,
    execute_pilot,
    review_pilot,
    validate_accepted_freeze,
)
from src.simulation.safeguards import APPROVED_PHASE10_PROMPT_HASHES


def test_real_manifest_records_exact_phase10_candidate_identity_and_freeze(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "phase11-offline-review-key")
    result = execute_pilot(
        mode="real",
        output_root=tmp_path,
        experiment_id="phase11_review",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
        real_authorized=True,
        max_total_tokens=3000,
        provider=Phase11MockProvider(),
        sleeper=lambda _: None,
    )
    manifest = json.loads(
        (result.paths.evidence_dir / "pilot_manifest.json").read_text(encoding="utf-8")
    )

    assert {item["template_id"]: item["prompt_hash"] for item in manifest["prompts"]} == dict(
        APPROVED_PHASE10_PROMPT_HASHES
    )
    assert {item["prompt_version"] for item in manifest["prompts"]} == {
        "phase10_pilot_v1"
    }
    assert manifest["included_in_final_research_matrix"] is False

    reviewed = review_pilot(
        evidence_dir=result.paths.evidence_dir,
        decision="accepted",
    )

    assert reviewed.acceptance_state == "accepted"
    assert (reviewed.review_dir / "pilot_acceptance.json").is_file()
    freeze = json.loads(
        (reviewed.review_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    assert freeze["prompt_version"] == "phase10_pilot_v1"
    assert freeze["included_in_final_research_matrix"] is False
    assert len(freeze["configuration_hashes"]) == 6
    assert len(freeze["prompts"]) == 6
    assert freeze["simulator_settings"]["model_identity"] == (
        "openai:gpt-5-nano-2025-08-07"
    )
    assert freeze["evaluator_settings"]["model_identity"] == (
        "openai:gpt-5-2025-08-07"
    )
    assert freeze["embedding_settings"]["model_identity"] == (
        "openai:text-embedding-3-small"
    )
    assert freeze["embedding_settings"]["expected_dimension"] == 1536
    assert freeze["bcr_rubric_version"] == "phase10_bcr_rubric_v1"
    assert freeze["phase13_source_and_git_verification_required"] is True
    validation = validate_accepted_freeze(reviewed.review_dir)
    assert validation.pilot_id == "pilot"
    assert validation.prompt_version == "phase10_pilot_v1"

    with pytest.raises(PilotReviewError, match="already exists"):
        review_pilot(
            evidence_dir=result.paths.evidence_dir,
            decision="rejected",
        )


def test_prompt_content_change_invalidates_an_accepted_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "phase11-offline-freeze-drift-key")
    result = execute_pilot(
        mode="real",
        output_root=tmp_path,
        experiment_id="phase11_freeze_drift",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
        real_authorized=True,
        max_total_tokens=3000,
        provider=Phase11MockProvider(),
        sleeper=lambda _: None,
    )
    reviewed = review_pilot(
        evidence_dir=result.paths.evidence_dir,
        decision="accepted",
    )
    changed_prompts = tmp_path / "changed_prompts"
    shutil.copytree(DEFAULT_PROMPT_DIRECTORY, changed_prompts)
    plan = changed_prompts / "plan.txt"
    plan.write_text(
        plan.read_text(encoding="utf-8") + "\nChanged after acceptance.\n",
        encoding="utf-8",
    )

    with pytest.raises(PilotReviewError, match="invalidate"):
        validate_accepted_freeze(
            reviewed.review_dir,
            prompt_loader=PromptLoader(changed_prompts),
        )


def test_execution_success_never_creates_review_or_freeze_output(
    tmp_path,
) -> None:
    result = execute_pilot(
        mode="dummy",
        output_root=tmp_path,
        experiment_id="phase11_pending",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
    )

    assert result.report.acceptance_state == "pending"
    assert not result.paths.review_dir.exists()


def test_manual_rejection_is_offline_terminal_and_never_creates_a_freeze(
    tmp_path: Path,
) -> None:
    result = execute_pilot(
        mode="dummy",
        output_root=tmp_path,
        experiment_id="phase11_manual_reject",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
    )

    reviewed = review_pilot(
        evidence_dir=result.paths.evidence_dir,
        decision="rejected",
    )

    assert reviewed.acceptance_state == "rejected"
    assert reviewed.freeze_hash is None
    assert (reviewed.review_dir / "pilot_acceptance.json").is_file()
    assert not (reviewed.review_dir / "freeze_manifest.json").exists()
    acceptance = json.loads(
        (reviewed.review_dir / "pilot_acceptance.json").read_text(encoding="utf-8")
    )
    assert acceptance["manual_review"] is True
    assert acceptance["primary_eligible"] is False
    assert acceptance["included_in_final_research_matrix"] is False


def test_review_reconstructs_usage_instead_of_trusting_self_reported_counts(
    tmp_path: Path,
) -> None:
    result = execute_pilot(
        mode="dummy",
        output_root=tmp_path,
        experiment_id="phase11_usage_review",
        run_id="pilot",
        retrieval_config_id="balanced",
        repetition_index=1,
    )
    usage_path = result.paths.evidence_dir / "pilot_usage_summary.json"
    report_path = result.paths.evidence_dir / "pilot_report.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    usage["logical_call_count"] = 1
    report["usage"] = usage
    usage_path.write_text(json.dumps(usage, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PilotReviewError, match="cross-file"):
        review_pilot(
            evidence_dir=result.paths.evidence_dir,
            decision="rejected",
        )

    assert not result.paths.review_dir.exists()
