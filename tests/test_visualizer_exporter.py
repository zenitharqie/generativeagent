from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from phase12_test_support import copy_run, directory_bytes, read_jsonl, write_jsonl
from src.visualizer import ReplayConflictError, ReplaySecurityError, export_visualizer_data


def test_exporter_writes_valid_replay_json(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "pilot-replay.json"

    exported = export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)
    value = json.loads(output.read_text(encoding="utf-8"))

    assert exported.output_path == output.resolve()
    assert exported.output_sha256 == sha256(output.read_bytes()).hexdigest()
    assert exported.output_bytes == output.stat().st_size
    assert value["run"]["source_status"] == "excluded"
    assert len(value["steps"][0]["turns"]) == 12
    assert value["steps"][0]["turns"][0]["retrieval"]["selected_memory_ids"]


def test_exporter_does_not_mutate_source_bytes(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    before = directory_bytes(result.run_paths.run_dir)

    export_visualizer_data(
        run_dir=result.run_paths.run_dir,
        output=tmp_path / "immutable.json",
    )

    assert directory_bytes(result.run_paths.run_dir) == before


def test_two_fresh_exports_are_byte_identical(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"

    first_result = export_visualizer_data(run_dir=result.run_paths.run_dir, output=first)
    second_result = export_visualizer_data(run_dir=result.run_paths.run_dir, output=second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.output_sha256 == second_result.output_sha256


def test_output_conflict_is_never_overwritten(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "existing.json"
    output.write_bytes(b"user-owned\n")

    with pytest.raises(ReplayConflictError, match="overwrite"):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    assert output.read_bytes() == b"user-owned\n"
    assert list(tmp_path.glob(".*.staging")) == []


def test_output_inside_source_run_is_rejected(phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = result.run_paths.run_dir / "replay.json"

    with pytest.raises(ReplaySecurityError, match="outside"):
        export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)

    assert not output.exists()


def test_embeddings_are_omitted_recursively(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "no-embeddings.json"
    export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)
    value = json.loads(output.read_text(encoding="utf-8"))

    def keys(item):
        if isinstance(item, dict):
            for key, nested in item.items():
                yield key
                yield from keys(nested)
        elif isinstance(item, list):
            for nested in item:
                yield from keys(nested)

    assert "embedding" not in set(keys(value))
    assert "query_embedding" not in set(keys(value))


def test_secret_like_agent_text_blocks_export(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    run_dir = copy_run(result.run_paths.run_dir, tmp_path / "secret_source")
    records = read_jsonl(run_dir / "behavior_logs.jsonl")
    sentinel = "sk-" + "THIS_IS_A_CONSPICUOUS_FAKE_SECRET_123456"
    records[0]["response"] = sentinel
    write_jsonl(run_dir / "behavior_logs.jsonl", records)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ReplaySecurityError, match="secret-like"):
        export_visualizer_data(run_dir=run_dir, output=output)

    assert not output.exists()


def test_export_contains_source_hashes_not_source_paths(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    output = tmp_path / "portable.json"
    export_visualizer_data(run_dir=result.run_paths.run_dir, output=output)
    value = json.loads(output.read_text(encoding="utf-8"))

    assert len(value["source"]["aggregate_sha256"]) == 64
    assert value["source"]["artifact_hashes"]
    assert str(result.run_paths.run_dir) not in output.read_text(encoding="utf-8")


def test_export_leaves_no_staging_file(tmp_path: Path, phase08_pilot_artifacts) -> None:
    result, _, _ = phase08_pilot_artifacts
    export_visualizer_data(run_dir=result.run_paths.run_dir, output=tmp_path / "clean.json")
    assert [item for item in tmp_path.iterdir() if item.name.startswith(".")] == []
