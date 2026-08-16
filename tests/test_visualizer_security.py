from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from phase12_test_support import copy_run, read_json, read_jsonl, write_jsonl
from src.visualizer import (
    ReplayInputError,
    ReplaySchemaError,
    ReplaySecurityError,
    build_replay,
    export_visualizer_data,
)
from src.visualizer import readers as replay_readers


def test_duplicate_json_key_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "duplicate")
    summary = (run_dir / "run_summary.json").read_text(encoding="utf-8")
    duplicate = summary.replace('"schema_version": "1.0",', '"schema_version": "1.0",\n  "schema_version": "1.0",')
    (run_dir / "run_summary.json").write_text(duplicate, encoding="utf-8")

    with pytest.raises(ReplaySchemaError, match="strict JSON"):
        build_replay(run_dir=run_dir)


def test_malformed_jsonl_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "malformed")
    (run_dir / "behavior_logs.jsonl").write_bytes(b'{"schema_version":"1.0"')

    with pytest.raises(ReplaySchemaError, match="LF-terminated"):
        build_replay(run_dir=run_dir)


def test_missing_canonical_artifact_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "missing")
    (run_dir / "behavior_logs.jsonl").unlink()

    with pytest.raises(ReplayInputError, match="canonical"):
        build_replay(run_dir=run_dir)


def test_unexpected_source_artifact_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "extra")
    (run_dir / "unexpected.txt").write_text("not canonical", encoding="utf-8")

    with pytest.raises(ReplayInputError, match="canonical"):
        build_replay(run_dir=run_dir)


def test_unsupported_schema_version_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "schema")
    summary = read_json(run_dir / "run_summary.json")
    summary["schema_version"] = "99.0"
    (run_dir / "run_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ReplaySchemaError, match="unsupported"):
        build_replay(run_dir=run_dir)


def test_input_path_traversal_is_rejected(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    traversal = result.run_paths.run_dir.parent / ".." / result.run_paths.run_dir.parent.name / result.run_paths.run_dir.name
    with pytest.raises(ReplaySecurityError, match="traversal"):
        build_replay(run_dir=traversal)


def test_output_path_traversal_is_rejected(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "nested" / ".." / "replay.json"
    with pytest.raises(ReplaySecurityError, match="traversal"):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)


def test_symlinked_input_directory_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    link = tmp_path / "linked-run"
    try:
        link.symlink_to(result.run_paths.run_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ReplaySecurityError, match="symbolic"):
        build_replay(run_dir=link)


def test_symlinked_output_parent_is_rejected(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ReplaySecurityError, match="symbolic"):
        export_visualizer_data(
            run_dir=result.run_paths.run_dir,
            output=link / "replay.json",
        )


def test_bounded_reader_rejects_oversized_artifact(
    tmp_path: Path, phase08_pilot_artifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "bounded")
    monkeypatch.setattr(replay_readers, "MAX_INPUT_FILE_BYTES", 32)

    with pytest.raises(ReplaySecurityError, match="size limit"):
        build_replay(run_dir=run_dir)


def test_html_like_agent_text_remains_data_not_markup(
    tmp_path: Path, phase08_pilot_artifacts
) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "html")
    rows = read_jsonl(run_dir / "behavior_logs.jsonl")
    sentinel = '<img src=x onerror="globalThis.compromised=true">'
    rows[0]["response"] = sentinel
    write_jsonl(run_dir / "behavior_logs.jsonl", rows)
    output = tmp_path / "html-text.json"

    export_visualizer_data(run_dir=run_dir, output=output)

    assert json.loads(output.read_text(encoding="utf-8"))["steps"][0]["turns"][0]["response"] == sentinel
    app = Path("visualizer/app.js").read_text(encoding="utf-8")
    assert ".textContent" in app
    assert "innerHTML" not in app
    assert "outerHTML" not in app
    assert "insertAdjacentHTML" not in app


def test_terminal_control_characters_are_json_escaped_and_never_cli_rendered(
    tmp_path: Path, phase08_pilot_artifacts, capsys
) -> None:
    from main import main

    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "controls")
    rows = read_jsonl(run_dir / "behavior_logs.jsonl")
    sentinel = "safe-prefix\x1b[31mCONTROL-SENTINEL\x1b[0m"
    rows[0]["response"] = sentinel
    write_jsonl(run_dir / "behavior_logs.jsonl", rows)
    output = tmp_path / "controls.json"

    assert main([
        "export-visualizer-data",
        "--run-dir",
        str(run_dir),
        "--output",
        str(output),
    ]) == 0
    captured = capsys.readouterr()

    assert "CONTROL-SENTINEL" not in captured.out + captured.err
    assert b"\x1b" not in output.read_bytes()
    assert b"\\u001b" in output.read_bytes()
    assert json.loads(output.read_text(encoding="utf-8"))["steps"][0]["turns"][0]["response"] == sentinel


def test_static_visualizer_has_no_network_execution_primitive() -> None:
    app = Path("visualizer/app.js").read_text(encoding="utf-8")
    html = Path("visualizer/index.html").read_text(encoding="utf-8")
    forbidden = ("fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "http://", "https://")
    assert all(value not in app for value in forbidden)
    assert "connect-src 'none'" in html
    assert all(value not in html for value in ("http://", "https://", "cdn"))
    assert "state.data.usage.records" in app
    assert "Persisted provider usage" in app


def test_replay_dependency_graph_does_not_import_execution_packages() -> None:
    forbidden_roots = {
        "src.cognition",
        "src.evaluation",
        "src.llm",
        "src.memory",
        "src.retrieval",
        "src.simulation",
    }
    for path in Path("src/visualizer").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not {name for name in imports if any(name == root or name.startswith(root + ".") for root in forbidden_roots)}


def test_visualizer_source_does_not_read_environment_or_open_network() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/visualizer").glob("*.py"))
    assert "os.environ" not in source
    assert "getenv(" not in source
    assert "socket" not in source
    assert "urllib" not in source
    assert "requests" not in source
    assert "OpenAIProviderAdapter" not in source
    assert "LLMService" not in source
