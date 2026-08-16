from __future__ import annotations

import json
from pathlib import Path

from main import build_parser, main
from phase12_test_support import directory_bytes


def test_phase12_parser_exposes_additive_export_command() -> None:
    args = build_parser().parse_args(
        [
            "export-visualizer-data",
            "--run-dir",
            "source-run",
            "--output",
            "replay.json",
        ]
    )
    assert args.command == "export-visualizer-data"
    assert args.evaluation_dir is None
    assert args.judgment_dir is None
    assert args.pilot_evidence_dir is None
    assert args.pilot_review_dir is None


def test_phase12_cli_exports_replay_offline(
    tmp_path: Path, phase08_pilot_artifacts, capsys
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "cli.json"
    before = directory_bytes(result.run_paths.run_dir)

    exit_code = main(
        [
            "export-visualizer-data",
            "--run-dir",
            str(result.run_paths.run_dir),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "exported successfully" in captured.out
    assert captured.err == ""
    assert json.loads(output.read_text(encoding="utf-8"))["run"]["source_status"] == "excluded"
    assert directory_bytes(result.run_paths.run_dir) == before


def test_phase12_cli_conflict_is_safely_reported(
    tmp_path: Path, phase08_pilot_artifacts, capsys
) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "existing.json"
    output.write_text("owned", encoding="utf-8")

    exit_code = main(
        [
            "export-visualizer-data",
            "--run-dir",
            str(result.run_paths.run_dir),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ReplayConflictError" in captured.err
    assert output.read_text(encoding="utf-8") == "owned"


def test_existing_no_argument_behavior_is_unchanged(capsys) -> None:
    assert main([]) == 0
    assert "repository scaffold is ready" in capsys.readouterr().out


def test_static_visualizer_files_exist() -> None:
    assert Path("visualizer/index.html").is_file()
    assert Path("visualizer/styles.css").is_file()
    assert Path("visualizer/app.js").is_file()
    assert Path("visualizer/sample_data/README.md").is_file()
