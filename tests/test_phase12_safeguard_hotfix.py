from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from phase12_test_support import (
    copy_run,
    directory_bytes,
    make_diagnostic_evaluation,
    make_dummy_pilot_evidence,
    make_failed_real_pilot_evidence,
    make_partial_run,
    make_primary_judgment,
    read_json,
    write_json,
)
from src.visualizer import (
    ReplayExportError,
    ReplayOverlayError,
    ReplaySecurityError,
    export_visualizer_data,
)
from src.visualizer import exporter as replay_exporter


FIXED_FRONTEND_ERROR = "Replay rejected: invalid or unsupported replay document."
UNSAFE_SENTINEL = "UNSAFE_PROVIDER_SENTINEL response body Authorization: Bearer secret"


def _assert_no_publication(output: Path, parent: Path) -> None:
    assert not output.exists()
    assert list(parent.glob(".*.staging")) == []


def test_staging_and_final_bytes_are_reopened_and_validated(
    tmp_path: Path,
    phase08_pilot_artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "validated.json"
    calls: list[Path] = []
    original = replay_exporter._read_and_validate_replay_file

    def recording_readback(path: Path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(
        replay_exporter,
        "_read_and_validate_replay_file",
        recording_readback,
    )
    exported = export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    assert calls[0].name.endswith(".staging")
    assert calls[1] == output.resolve()
    assert output.read_bytes() == replay_exporter.serialize_replay(exported.document)


def test_staging_byte_mutation_blocks_publication_and_preserves_source(
    tmp_path: Path,
    phase08_pilot_artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "mutated.json"
    before = directory_bytes(result.run_paths.run_dir)
    original = replay_exporter._read_and_validate_replay_file
    mutated = False

    def mutate_then_read(path: Path, **kwargs):
        nonlocal mutated
        if path.name.endswith(".staging") and not mutated:
            mutated = True
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"[")
                stream.flush()
        return original(path, **kwargs)

    monkeypatch.setattr(
        replay_exporter,
        "_read_and_validate_replay_file",
        mutate_then_read,
    )
    with pytest.raises(ReplaySecurityError, match="changed"):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    _assert_no_publication(output, tmp_path)
    assert directory_bytes(result.run_paths.run_dir) == before


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (b'{"schema_version":"1.0"\n', ReplayExportError, "strict JSON"),
        (b'{"schema_version":"1.0","schema_version":"1.0"}\n', ReplayExportError, "strict JSON"),
        (b'{"schema_version":"1.0"}\n', ReplayExportError, "ReplayDocument schema"),
    ],
    ids=["malformed-json", "duplicate-key", "schema-invalid"],
)
def test_invalid_staged_json_never_publishes(
    tmp_path: Path,
    phase08_pilot_artifacts,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    error_type: type[Exception],
    message: str,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "invalid-staged.json"
    before = directory_bytes(result.run_paths.run_dir)
    monkeypatch.setattr(replay_exporter, "serialize_replay", lambda document: payload)

    with pytest.raises(error_type, match=message):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    _assert_no_publication(output, tmp_path)
    assert directory_bytes(result.run_paths.run_dir) == before


def test_staging_replaced_by_symlink_never_publishes(
    tmp_path: Path,
    phase08_pilot_artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "symlink-staged.json"
    target = tmp_path / "attacker-owned"
    target.write_bytes(b"not replay data")
    before = directory_bytes(result.run_paths.run_dir)
    original = replay_exporter._read_and_validate_replay_file
    replaced = False

    def replace_then_read(path: Path, **kwargs):
        nonlocal replaced
        if path.name.endswith(".staging") and not replaced:
            replaced = True
            path.unlink()
            try:
                path.symlink_to(target)
            except (NotImplementedError, OSError):
                pytest.skip("symbolic links are unavailable")
        return original(path, **kwargs)

    monkeypatch.setattr(
        replay_exporter,
        "_read_and_validate_replay_file",
        replace_then_read,
    )
    with pytest.raises(ReplaySecurityError, match="identity"):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    _assert_no_publication(output, tmp_path)
    assert target.read_bytes() == b"not replay data"
    assert directory_bytes(result.run_paths.run_dir) == before


def _run_frontend(value: object) -> dict[str, object]:
    completed = subprocess.run(
        ["node", "tests/phase12_frontend_harness.js"],
        input=json.dumps(value, ensure_ascii=False, allow_nan=False),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _valid_replay_value(tmp_path: Path, phase08_pilot_artifacts) -> dict[str, object]:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "browser-valid.json"
    export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)
    return json.loads(output.read_text(encoding="utf-8"))


def _mutate_frontend_case(value: dict[str, object], case: str) -> None:
    if case == "invalid-run-status":
        value["run"]["source_status"] = "impossible"  # type: ignore[index]
    elif case == "empty-agents":
        value["agents"] = []
    elif case == "empty-steps":
        value["steps"] = []
    elif case == "invalid-frame-kind":
        value["playback"][1]["frame_kind"] = "invented"  # type: ignore[index]
    elif case == "missing-frame-entity":
        value["playback"][1]["entity_id"] = "missing_turn"  # type: ignore[index]
    elif case == "invalid-turn-field":
        value["steps"][0]["turns"][0]["logical_timestamp"] = "one"  # type: ignore[index]
    elif case == "invalid-candidate-field":
        value["steps"][0]["turns"][0]["retrieval"]["candidates"][0]["final_score"] = "high"  # type: ignore[index]
    elif case == "invalid-memory-type":
        value["memories"][0]["memory_type"] = "observation"  # type: ignore[index]
    elif case == "invalid-usage-token":
        value["usage"]["known_total_tokens"] = "zero"  # type: ignore[index]
    elif case == "invalid-usage-status":
        turn = value["steps"][0]["turns"][0]  # type: ignore[index]
        record = {
            "turn_id": turn["turn_id"],
            "call_type": "plan",
            "model": "fixture:model",
            "prompt_template_id": "plan",
            "status": "invented",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "error_type": None,
        }
        turn["usage_records"] = [record]
        value["usage"].update({"record_count": 1, "records": [record]})  # type: ignore[union-attr]
    elif case == "invalid-failure":
        value["failure"] = {"error_type": UNSAFE_SENTINEL}
    elif case == "invalid-evaluation-overlay":
        value["evaluation_overlay"] = {"schema_version": "1.0", "units": []}
    elif case == "invalid-pilot-overlay":
        value["pilot_overlay"] = {"schema_version": "1.0", "terminal_error_type": UNSAFE_SENTINEL}
    elif case == "unknown-schema":
        value["schema_version"] = "99.0"
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(case)


def test_valid_exported_replay_renders_in_static_frontend(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    value = _valid_replay_value(tmp_path, phase08_pilot_artifacts)
    result = _run_frontend(value)

    assert result["rendered"] is True
    assert result["run_id"] == value["run"]["run_id"]  # type: ignore[index]


@pytest.mark.parametrize("overlay_kind", ["evaluation", "pilot", "failed-pilot"])
def test_valid_optional_overlay_exports_render_in_static_frontend(
    tmp_path: Path,
    phase08_pilot_artifacts,
    overlay_kind: str,
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = result.run_paths.run_dir
    options: dict[str, Path] = {}
    if overlay_kind == "evaluation":
        options["evaluation_dir"] = make_diagnostic_evaluation(
            run_dir,
            tmp_path / "evaluation",
        )
    elif overlay_kind == "pilot":
        options["pilot_evidence_dir"] = make_dummy_pilot_evidence(
            run_dir,
            tmp_path / "pilot_evidence",
        )
    else:
        run_dir = copy_run(run_dir, tmp_path / "failed_browser_source")
        make_partial_run(run_dir, status="failed", completed_turns=5)
        options["pilot_evidence_dir"] = make_failed_real_pilot_evidence(
            run_dir,
            tmp_path / "failed_browser_evidence",
        )
    output = tmp_path / f"browser-{overlay_kind}.json"
    export_visualizer_data(run_dir=run_dir, output=output, **options)

    frontend = _run_frontend(json.loads(output.read_text(encoding="utf-8")))
    assert frontend["rendered"] is True


def test_valid_judgment_overlay_export_renders_in_static_frontend(
    tmp_path: Path,
    phase08_full_artifacts,
) -> None:
    result, _, _ = phase08_full_artifacts
    evaluation = make_diagnostic_evaluation(
        result.run_paths.run_dir,
        tmp_path / "primary_evaluation",
    )
    judgment = make_primary_judgment(evaluation, tmp_path / "judgment")
    output = tmp_path / "browser-judgment.json"
    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        evaluation_dir=evaluation,
        judgment_dir=judgment,
        output=output,
    )

    frontend = _run_frontend(json.loads(output.read_text(encoding="utf-8")))
    assert frontend["rendered"] is True


@pytest.mark.parametrize("mutation", ["non-binary", "obsolete-fields"])
def test_frontend_rejects_malformed_binary_judgment_without_rendering(
    tmp_path: Path,
    phase08_full_artifacts,
    mutation: str,
) -> None:
    result, _, _ = phase08_full_artifacts
    evaluation = make_diagnostic_evaluation(
        result.run_paths.run_dir,
        tmp_path / "primary_evaluation",
    )
    judgment = make_primary_judgment(evaluation, tmp_path / "judgment")
    output = tmp_path / "browser-judgment.json"
    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        evaluation_dir=evaluation,
        judgment_dir=judgment,
        output=output,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    unit = value["evaluation_overlay"]["units"][0]
    if mutation == "non-binary":
        unit["consistent"] = 2
        unit["reason"] = UNSAFE_SENTINEL
    else:
        unit.pop("consistent")
        unit.pop("communication_style_assessment")
        unit.pop("focus_consistency_assessment")
        unit.pop("temporal_consistency_assessment")
        unit.pop("reason")
        unit["communication_style_score"] = 5
        unit["focus_consistency_score"] = 4
        unit["temporal_consistency_score"] = 3
        unit["communication_style_reason"] = UNSAFE_SENTINEL
        unit["focus_consistency_reason"] = "Obsolete."
        unit["temporal_consistency_reason"] = "Obsolete."
        unit["bcr"] = 4.0

    frontend = _run_frontend(value)

    assert frontend["rendered"] is False
    assert frontend["error_text"] == FIXED_FRONTEND_ERROR
    assert UNSAFE_SENTINEL not in frontend["all_text"]


@pytest.mark.parametrize(
    "case",
    [
        "invalid-run-status",
        "empty-agents",
        "empty-steps",
        "invalid-frame-kind",
        "missing-frame-entity",
        "invalid-turn-field",
        "invalid-candidate-field",
        "invalid-memory-type",
        "invalid-usage-token",
        "invalid-usage-status",
        "invalid-failure",
        "invalid-evaluation-overlay",
        "invalid-pilot-overlay",
        "unknown-schema",
    ],
)
def test_frontend_rejects_invalid_nested_replay_without_rendering(
    tmp_path: Path,
    phase08_pilot_artifacts,
    case: str,
) -> None:
    value = deepcopy(_valid_replay_value(tmp_path, phase08_pilot_artifacts))
    _mutate_frontend_case(value, case)
    result = _run_frontend(value)

    assert result["rendered"] is False
    assert result["error_text"] == FIXED_FRONTEND_ERROR
    assert UNSAFE_SENTINEL not in result["all_text"]


def _failed_pilot_inputs(
    tmp_path: Path,
    phase08_pilot_artifacts,
    *,
    status: str = "failed",
) -> tuple[Path, Path]:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / f"{status}_source")
    make_partial_run(run_dir, status=status, completed_turns=5)
    evidence = make_failed_real_pilot_evidence(run_dir, tmp_path / f"{status}_evidence")
    return run_dir, evidence


def test_canonical_pilot_failure_identifier_is_preserved(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    run_dir, evidence = _failed_pilot_inputs(tmp_path, phase08_pilot_artifacts)
    output = tmp_path / "canonical-failure.json"
    export_visualizer_data(
        run_dir=run_dir,
        pilot_evidence_dir=evidence,
        output=output,
    )

    pilot = json.loads(output.read_text(encoding="utf-8"))["pilot_overlay"]
    assert pilot["terminal_error_type"] == "ProviderTimeoutError"
    assert pilot["rejection_reasons"] == ["ProviderTimeoutError"]
    assert pilot["usage"]["actual_total_tokens"] is None
    assert pilot["attempts"][0]["total_tokens"] is None


def test_canonical_keyboard_interrupt_identifier_is_preserved(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    run_dir, evidence = _failed_pilot_inputs(
        tmp_path,
        phase08_pilot_artifacts,
        status="interrupted",
    )
    report = read_json(evidence / "pilot_report.json")
    report["terminal_error_type"] = "KeyboardInterrupt"
    report["rejection_reasons"] = ["KeyboardInterrupt"]
    write_json(evidence / "pilot_report.json", report)
    output = tmp_path / "keyboard-interrupt.json"

    export_visualizer_data(
        run_dir=run_dir,
        pilot_evidence_dir=evidence,
        output=output,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["pilot_overlay"][
        "terminal_error_type"
    ] == "KeyboardInterrupt"


@pytest.mark.parametrize(
    "unsafe",
    [
        "",
        "provider returned a raw response body",
        "Provider/Timeout",
        "Provider\\Timeout",
        "ProviderTimeout\nresponse",
        "ProviderTimeout\x1b[31m",
        UNSAFE_SENTINEL,
    ],
)
def test_unsafe_pilot_rejection_reason_is_rejected(
    tmp_path: Path,
    phase08_pilot_artifacts,
    unsafe: str,
) -> None:
    run_dir, evidence = _failed_pilot_inputs(tmp_path, phase08_pilot_artifacts)
    report = read_json(evidence / "pilot_report.json")
    report["rejection_reasons"] = [unsafe]
    write_json(evidence / "pilot_report.json", report)
    output = tmp_path / "unsafe-reason.json"

    with pytest.raises(ReplayOverlayError, match="safe identifier"):
        export_visualizer_data(
            run_dir=run_dir,
            pilot_evidence_dir=evidence,
            output=output,
        )
    assert not output.exists()


def test_unsafe_pilot_terminal_error_type_is_rejected_without_projection(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    run_dir, evidence = _failed_pilot_inputs(tmp_path, phase08_pilot_artifacts)
    report = read_json(evidence / "pilot_report.json")
    report["terminal_error_type"] = UNSAFE_SENTINEL
    write_json(evidence / "pilot_report.json", report)
    output = tmp_path / "unsafe-terminal.json"

    with pytest.raises(ReplayOverlayError, match="safe identifier"):
        export_visualizer_data(
            run_dir=run_dir,
            pilot_evidence_dir=evidence,
            output=output,
        )
    assert not output.exists()


def test_conflicting_pilot_and_source_terminal_types_are_rejected(
    tmp_path: Path,
    phase08_pilot_artifacts,
) -> None:
    run_dir, evidence = _failed_pilot_inputs(tmp_path, phase08_pilot_artifacts)
    report = read_json(evidence / "pilot_report.json")
    report["terminal_error_type"] = "KeyboardInterrupt"
    report["rejection_reasons"] = ["KeyboardInterrupt"]
    write_json(evidence / "pilot_report.json", report)
    output = tmp_path / "mismatch.json"

    with pytest.raises(ReplayOverlayError, match="contradicts"):
        export_visualizer_data(
            run_dir=run_dir,
            pilot_evidence_dir=evidence,
            output=output,
        )
    assert not output.exists()
