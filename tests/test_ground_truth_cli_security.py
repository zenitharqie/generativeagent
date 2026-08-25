from __future__ import annotations

import json
import os
import socket

import pytest

import main as cli
from src.evaluation.ground_truth import build_annotation_package
from src.evaluation.ground_truth import GroundTruthPersistenceError
from src.evaluation.ground_truth.service import generate_annotation_package
from src.evaluation.ground_truth import finalize_ground_truth
import src.evaluation.ground_truth.persistence as persistence_module
from ground_truth_test_support import complete_evidence


def test_package_cli_writes_no_provider_or_network(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setenv("OPENAI_API_KEY", "credential-sentinel-must-not-be-read")
    output = tmp_path / "package.json"
    assert cli.main([
        "generate-ground-truth-annotation-package",
        "--output",
        str(output),
    ]) == 0
    assert output.is_file()
    assert "credential-sentinel" not in output.read_text()
    assert "20" in capsys.readouterr().out


def test_finalize_cli_uses_persisted_embeddings_only(monkeypatch, tmp_path, capsys) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setenv("OPENAI_API_KEY", "credential-sentinel-must-not-be-read")
    output = tmp_path / "final"
    assert cli.main([
        "finalize-ground-truth",
        "--package",
        str(package_path),
        "--annotations-dir",
        str(annotations),
        "--embedding-evidence",
        str(embeddings),
        "--output-dir",
        str(output),
    ]) == 0
    all_bytes = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"credential-sentinel" not in all_bytes
    assert not any("judgment" in path.name for path in output.rglob("*"))
    assert "finalized successfully" in capsys.readouterr().out


def test_cli_suppresses_untrusted_annotation_text(tmp_path, capsys) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    sentinel = "RAW_PROVIDER_SENTINEL should never persist"
    path = annotations / "gpt5_6_sol.json"
    value = json.loads(path.read_text())
    value["results"][0]["categories"][0]["category"] = sentinel + "\n\x1b[31m"
    path.write_text(json.dumps(value))
    assert cli.main([
        "finalize-ground-truth",
        "--package",
        str(package_path),
        "--annotations-dir",
        str(annotations),
        "--embedding-evidence",
        str(embeddings),
        "--output-dir",
        str(tmp_path / "final"),
    ]) == 1
    captured = capsys.readouterr()
    assert sentinel not in captured.err
    assert sentinel not in captured.out
    assert not (tmp_path / "final").exists()


def test_output_conflict_traversal_and_symlink_fail_without_overwrite(tmp_path) -> None:
    conflict = tmp_path / "package.json"
    conflict.write_text("user data")
    with pytest.raises(GroundTruthPersistenceError):
        generate_annotation_package(output=conflict)
    assert conflict.read_text() == "user data"
    with pytest.raises(GroundTruthPersistenceError):
        generate_annotation_package(output=tmp_path / "nested" / ".." / "unsafe.json")
    safe = tmp_path / "safe"
    safe.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(safe, target_is_directory=True)
    with pytest.raises(GroundTruthPersistenceError):
        generate_annotation_package(output=linked / "package.json")


def test_finalization_post_rename_fsync_failure_rolls_back_owned_output(
    monkeypatch, tmp_path
) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)
    output = tmp_path / "final"
    original = persistence_module._fsync_directory

    def fail_parent(path):
        if path == output.parent and output.exists():
            raise OSError("synthetic durability failure")
        return original(path)

    monkeypatch.setattr(persistence_module, "_fsync_directory", fail_parent)
    with pytest.raises(GroundTruthPersistenceError):
        finalize_ground_truth(
            package_path=package_path,
            annotations_directory=annotations,
            embedding_evidence_path=embeddings,
            output_directory=output,
        )
    assert not output.exists()
    assert not (tmp_path / ".final.staging").exists()


def test_ground_truth_runtime_never_touches_credentials_or_execution_services(
    monkeypatch, tmp_path
) -> None:
    package = build_annotation_package().value
    package_path, annotations, embeddings = complete_evidence(tmp_path, package)

    def forbidden(*args, **kwargs):
        pytest.fail("provider, judge, experiment, or credential boundary was touched")

    import src.llm.embedding as embedding_module
    import src.evaluation.judgment_service as judgment_module
    import src.simulation.experiment_controller as experiment_module

    monkeypatch.setattr(embedding_module.OpenAIEmbeddingProvider, "__init__", forbidden)
    monkeypatch.setattr(judgment_module, "judge_bcr_export", forbidden)
    monkeypatch.setattr(experiment_module, "execute_full_experiment", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    original_get = os._Environ.get

    def guarded_get(self, key, default=None):
        if key == "OPENAI_API_KEY":
            return forbidden()
        return original_get(self, key, default)

    monkeypatch.setattr(os._Environ, "get", guarded_get)
    bundle = finalize_ground_truth(
        package_path=package_path,
        annotations_directory=annotations,
        embedding_evidence_path=embeddings,
        output_directory=tmp_path / "final-safe",
    )
    assert bundle.final_ground_truth["status"] == "final"
