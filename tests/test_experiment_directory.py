"""Phase 13 durable staging, readback, conflict, and source-recheck tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.storage.experiment_directory as directory_module
from src.storage import (
    ExperimentStorageError,
    RunConflictError,
    build_experiment_paths,
    create_experiment_directory,
    publish_experiment_manifest,
)


def _payload(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_manifest_staging_is_reread_validated_and_source_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="readback")
    create_experiment_directory(paths)
    value = {"schema_version": "1.0", "status": "completed"}
    checks: list[str] = []
    fsync_paths: list[Path] = []
    original_fsync_directory = directory_module._fsync_directory

    def observed_fsync(path: Path) -> None:
        fsync_paths.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(directory_module, "_fsync_directory", observed_fsync)

    publish_experiment_manifest(
        paths,
        _payload(value),
        validator=lambda parsed: checks.append("validate")
        if dict(parsed) == value
        else (_ for _ in ()).throw(AssertionError("wrong content")),
        source_recheck=lambda: checks.append("source_recheck"),
    )

    assert paths.manifest_file.read_bytes() == _payload(value)
    assert checks == ["validate", "source_recheck", "validate"]
    assert fsync_paths == [paths.experiment_dir]
    assert not paths.manifest_staging_file.exists()


def test_staged_manifest_byte_mutation_is_detected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="mutated")
    create_experiment_directory(paths)
    original_read = directory_module._read_regular_file_with_identity

    def altered(path: Path) -> tuple[bytes, tuple[int, int]]:
        payload, identity = original_read(path)
        if path == paths.manifest_staging_file:
            return payload + b" ", identity
        return payload, identity

    monkeypatch.setattr(directory_module, "_read_regular_file_with_identity", altered)

    with pytest.raises(ExperimentStorageError, match="differs"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_malformed_or_schema_invalid_staging_never_publishes(
    tmp_path: Path,
) -> None:
    malformed_paths = build_experiment_paths(tmp_path, experiment_id="malformed")
    create_experiment_directory(malformed_paths)
    with pytest.raises(ExperimentStorageError, match="malformed"):
        publish_experiment_manifest(
            malformed_paths,
            b"{bad}\n",
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )
    assert not malformed_paths.manifest_file.exists()

    invalid_paths = build_experiment_paths(tmp_path, experiment_id="invalid_schema")
    create_experiment_directory(invalid_paths)

    def reject_schema(parsed: object) -> None:
        raise ExperimentStorageError("invalid experiment schema")

    with pytest.raises(ExperimentStorageError, match="invalid experiment schema"):
        publish_experiment_manifest(
            invalid_paths,
            _payload({"schema_version": "unsupported"}),
            validator=reject_schema,
            source_recheck=lambda: None,
        )
    assert not invalid_paths.manifest_file.exists()
    assert not invalid_paths.manifest_staging_file.exists()


def test_source_recheck_failure_blocks_manifest_publication(tmp_path: Path) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="source_changed")
    create_experiment_directory(paths)

    def reject_changed_source() -> None:
        raise ExperimentStorageError("sealed source changed")

    with pytest.raises(ExperimentStorageError, match="sealed source"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0"}),
            validator=lambda parsed: None,
            source_recheck=reject_changed_source,
        )

    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_manifest_never_overwrites_existing_output(tmp_path: Path) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="conflict")
    create_experiment_directory(paths)
    paths.manifest_file.write_bytes(b"preserve")

    with pytest.raises(RunConflictError):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert paths.manifest_file.read_bytes() == b"preserve"


def test_fsync_failure_cleans_manifest_staging_and_preserves_originating_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="fsync_failure")
    create_experiment_directory(paths)

    def fail_fsync(descriptor: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(directory_module.os, "fsync", fail_fsync)
    with pytest.raises(ExperimentStorageError, match="publication failed"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )
    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_parent_fsync_failure_rolls_back_owned_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="post_rename_fsync")
    create_experiment_directory(paths)
    source = paths.source_run_dir("balanced_rep01")
    source.mkdir()
    source_file = source / "source.bin"
    source_file.write_bytes(b"sealed source")
    evaluation = paths.evaluation_dir("balanced_rep01")
    evaluation.mkdir()
    evaluation_file = evaluation / "evaluation.bin"
    evaluation_file.write_bytes(b"sealed evaluation")
    sealed = (source_file.read_bytes(), evaluation_file.read_bytes())
    original_fsync_directory = directory_module._fsync_directory
    calls = 0

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic post-rename directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(directory_module, "_fsync_directory", fail_once)

    with pytest.raises(ExperimentStorageError, match="publication failed") as captured:
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == (
        "synthetic post-rename directory fsync failure"
    )
    assert calls == 2
    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()
    assert (source_file.read_bytes(), evaluation_file.read_bytes()) == sealed


def test_interrupt_immediately_after_rename_rolls_back_owned_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="rename_interrupt")
    create_experiment_directory(paths)
    original_replace = directory_module.os.replace

    def replace_then_interrupt(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(directory_module.os, "replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_final_reopen_failure_rolls_back_owned_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="final_reopen")
    create_experiment_directory(paths)
    original_read = directory_module._read_regular_file_with_identity

    def fail_final(path: Path) -> tuple[bytes, tuple[int, int]]:
        if path == paths.manifest_file:
            raise OSError("synthetic final reopen failure")
        return original_read(path)

    monkeypatch.setattr(directory_module, "_read_regular_file_with_identity", fail_final)

    with pytest.raises(ExperimentStorageError, match="publication failed"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_final_schema_failure_rolls_back_owned_final(
    tmp_path: Path,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="final_schema")
    create_experiment_directory(paths)
    validation_count = 0

    def reject_final(parsed: object) -> None:
        nonlocal validation_count
        validation_count += 1
        if validation_count == 2:
            raise ExperimentStorageError("synthetic final schema failure")

    with pytest.raises(ExperimentStorageError, match="final schema failure"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=reject_final,
            source_recheck=lambda: None,
        )

    assert validation_count == 2
    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_final_byte_mismatch_rolls_back_owned_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="final_mismatch")
    create_experiment_directory(paths)
    original_read = directory_module._read_regular_file_with_identity

    def alter_final(path: Path) -> tuple[bytes, tuple[int, int]]:
        payload, identity = original_read(path)
        if path == paths.manifest_file:
            return payload + b" ", identity
        return payload, identity

    monkeypatch.setattr(directory_module, "_read_regular_file_with_identity", alter_final)

    with pytest.raises(ExperimentStorageError, match="differs"):
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert not paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_cleanup_failure_preserves_publication_error_as_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="cleanup_failure")
    create_experiment_directory(paths)

    def fail_publication(path: Path) -> None:
        raise OSError("originating post-rename fsync failure")

    def fail_cleanup(path: Path, identity: tuple[int, int]) -> bool:
        raise PermissionError("synthetic rollback unlink failure")

    monkeypatch.setattr(directory_module, "_fsync_directory", fail_publication)
    monkeypatch.setattr(directory_module, "_remove_owned_regular_file", fail_cleanup)

    with pytest.raises(ExperimentStorageError, match="publication failed") as captured:
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "originating post-rename fsync failure"
    assert any(
        "owned final rollback also failed: PermissionError" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert paths.manifest_file.exists()
    assert not paths.manifest_staging_file.exists()


def test_post_rename_cleanup_does_not_remove_external_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = build_experiment_paths(tmp_path, experiment_id="external_replacement")
    create_experiment_directory(paths)
    replacement = b"external file must remain untouched"

    def replace_before_final_readback(path: Path) -> None:
        paths.manifest_file.unlink()
        paths.manifest_file.write_bytes(replacement)

    monkeypatch.setattr(
        directory_module,
        "_fsync_directory",
        replace_before_final_readback,
    )

    with pytest.raises(ExperimentStorageError, match="ownership changed") as captured:
        publish_experiment_manifest(
            paths,
            _payload({"schema_version": "1.0", "status": "completed"}),
            validator=lambda parsed: None,
            source_recheck=lambda: None,
        )

    assert any(
        "owned final rollback also failed: ExperimentStorageError" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert paths.manifest_file.read_bytes() == replacement
    assert not paths.manifest_staging_file.exists()
