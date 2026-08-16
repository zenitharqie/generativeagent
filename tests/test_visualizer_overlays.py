from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase12_test_support import (
    directory_bytes,
    make_diagnostic_evaluation,
    make_dummy_pilot_evidence,
    make_failed_real_pilot_evidence,
    make_forged_accepted_review,
    make_partial_run,
    make_primary_judgment,
    make_rejected_pilot_review,
    read_json,
    write_json,
)
from src.visualizer import ReplayInputError, ReplayOverlayError, export_visualizer_data


def test_explicit_evaluation_overlay_is_display_only(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evaluation = make_diagnostic_evaluation(result.run_paths.run_dir, tmp_path / "evaluation")
    source_before = directory_bytes(result.run_paths.run_dir)
    evaluation_before = directory_bytes(evaluation)
    output = tmp_path / "with-evaluation.json"

    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        evaluation_dir=evaluation,
        output=output,
    )
    value = json.loads(output.read_text(encoding="utf-8"))

    assert value["evaluation_overlay"]["evaluation_mode"] == "diagnostic"
    assert len(value["evaluation_overlay"]["units"]) == 4
    assert value["evaluation_overlay"]["ground_truth_visibility"] == (
        "evaluation_overlay_only_not_agent_observation"
    )
    assert directory_bytes(result.run_paths.run_dir) == source_before
    assert directory_bytes(evaluation) == evaluation_before


def test_base_replay_does_not_auto_discover_evaluation(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    make_diagnostic_evaluation(result.run_paths.run_dir, tmp_path / "evaluation")
    output = tmp_path / "base.json"

    export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)
    assert json.loads(output.read_text(encoding="utf-8"))["evaluation_overlay"] is None


def test_judgment_requires_explicit_evaluation(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    judgment = tmp_path / "judgment"
    judgment.mkdir()

    with pytest.raises(ReplayInputError, match="requires"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            judgment_dir=judgment,
            output=tmp_path / "invalid.json",
        )


def test_pilot_review_requires_explicit_evidence(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    review = tmp_path / "review"
    review.mkdir()

    with pytest.raises(ReplayInputError, match="requires"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            pilot_review_dir=review,
            output=tmp_path / "invalid.json",
        )


def test_evaluation_cross_run_identity_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evaluation = make_diagnostic_evaluation(result.run_paths.run_dir, tmp_path / "evaluation")
    summary = read_json(evaluation / "evaluation_summary.json")
    summary["run_id"] = "foreign_run"
    write_json(evaluation / "evaluation_summary.json", summary)

    with pytest.raises(ReplayOverlayError, match="identity"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            evaluation_dir=evaluation,
            output=tmp_path / "invalid.json",
        )


def test_evaluation_overlay_never_changes_source_research_eligibility(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evaluation = make_diagnostic_evaluation(result.run_paths.run_dir, tmp_path / "evaluation")
    output = tmp_path / "pilot.json"
    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        evaluation_dir=evaluation,
        output=output,
    )
    run = json.loads(output.read_text(encoding="utf-8"))["run"]
    assert run["source_status"] == "excluded"
    assert run["primary_research_data"] is False


def test_explicit_pilot_overlay_preserves_pending_manual_state(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evidence = make_dummy_pilot_evidence(
        result.run_paths.run_dir, tmp_path / "pilot_evidence"
    )
    source_before = directory_bytes(result.run_paths.run_dir)
    evidence_before = directory_bytes(evidence)
    output = tmp_path / "pilot-overlay.json"

    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        pilot_evidence_dir=evidence,
        output=output,
    )
    pilot = json.loads(output.read_text(encoding="utf-8"))["pilot_overlay"]

    assert pilot["technical_success"] is True
    assert pilot["acceptance_state"] == "pending"
    assert pilot["primary_eligible"] is False
    assert pilot["included_in_final_research_matrix"] is False
    assert directory_bytes(result.run_paths.run_dir) == source_before
    assert directory_bytes(evidence) == evidence_before


def test_manual_rejection_overlay_does_not_change_technical_outcome(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evidence = make_dummy_pilot_evidence(
        result.run_paths.run_dir, tmp_path / "pilot_evidence"
    )
    review = make_rejected_pilot_review(evidence, tmp_path / "pilot_review")
    output = tmp_path / "review-overlay.json"

    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        pilot_evidence_dir=evidence,
        pilot_review_dir=review,
        output=output,
    )
    pilot = json.loads(output.read_text(encoding="utf-8"))["pilot_overlay"]

    assert pilot["technical_success"] is True
    assert pilot["acceptance_state"] == "rejected"
    assert pilot["review"]["manual_review"] is True


def test_tampered_pilot_manifest_hash_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evidence = make_dummy_pilot_evidence(
        result.run_paths.run_dir, tmp_path / "pilot_evidence"
    )
    manifest = read_json(evidence / "pilot_manifest.json")
    manifest["manifest_hash"] = "0" * 64
    write_json(evidence / "pilot_manifest.json", manifest)

    with pytest.raises(ReplayOverlayError, match="content hash"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            pilot_evidence_dir=evidence,
            output=tmp_path / "invalid-pilot.json",
        )


def test_tampered_manual_review_hash_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    evidence = make_dummy_pilot_evidence(
        result.run_paths.run_dir, tmp_path / "pilot_evidence"
    )
    review = make_rejected_pilot_review(evidence, tmp_path / "pilot_review")
    acceptance = read_json(review / "pilot_acceptance.json")
    acceptance["acceptance_hash"] = "0" * 64
    write_json(review / "pilot_acceptance.json", acceptance)

    with pytest.raises(ReplayOverlayError, match="manual review"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            pilot_evidence_dir=evidence,
            pilot_review_dir=review,
            output=tmp_path / "invalid-review.json",
        )


def test_explicit_primary_judgment_overlay_displays_persisted_bcr_only(
    tmp_path: Path, phase08_full_artifacts
) -> None:
    result, _, _ = phase08_full_artifacts
    evaluation = make_diagnostic_evaluation(
        result.run_paths.run_dir, tmp_path / "primary_evaluation"
    )
    judgment = make_primary_judgment(evaluation, tmp_path / "judgment")
    source_before = directory_bytes(result.run_paths.run_dir)
    evaluation_before = directory_bytes(evaluation)
    judgment_before = directory_bytes(judgment)
    output = tmp_path / "judged-replay.json"

    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        evaluation_dir=evaluation,
        judgment_dir=judgment,
        output=output,
    )
    overlay = json.loads(output.read_text(encoding="utf-8"))["evaluation_overlay"]

    assert overlay["evaluation_mode"] == "primary"
    assert overlay["judgment"]["judged_unit_count"] == 20
    assert {unit["bcr"] for unit in overlay["units"]} == {0.8}
    assert directory_bytes(result.run_paths.run_dir) == source_before
    assert directory_bytes(evaluation) == evaluation_before
    assert directory_bytes(judgment) == judgment_before


def test_missing_pilot_usage_remains_null_and_technical_failure_pending(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    from phase12_test_support import copy_run

    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "failed_source")
    make_partial_run(run_dir, status="failed", completed_turns=5)
    evidence = make_failed_real_pilot_evidence(run_dir, tmp_path / "failed_evidence")
    output = tmp_path / "failed-pilot-replay.json"

    export_visualizer_data(
        run_dir=run_dir,
        pilot_evidence_dir=evidence,
        output=output,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    pilot = value["pilot_overlay"]

    assert value["run"]["source_status"] == "failed"
    assert pilot["technical_success"] is False
    assert pilot["acceptance_state"] == "pending"
    assert pilot["usage"]["missing_total_usage_attempt_count"] == 1
    assert pilot["usage"]["actual_total_tokens"] is None
    assert pilot["attempts"][0]["total_tokens"] is None


def test_forged_acceptance_with_missing_usage_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    from phase12_test_support import copy_run

    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "failed_source")
    make_partial_run(run_dir, status="failed", completed_turns=5)
    evidence = make_failed_real_pilot_evidence(run_dir, tmp_path / "failed_evidence")
    review = make_forged_accepted_review(evidence, tmp_path / "forged_review")

    with pytest.raises(ReplayOverlayError, match="ineligible"):
        export_visualizer_data(
            run_dir=run_dir,
            pilot_evidence_dir=evidence,
            pilot_review_dir=review,
            output=tmp_path / "must-not-exist.json",
        )
