from pathlib import Path

import pytest

import src.storage as storage
from src.config_loader import loader as config_loader
from src.logging_system.run_summary import read_run_summary
from src.storage import config_snapshot, run_directory
from src.storage.config_snapshot import ConfigSnapshotError
from src.storage.run_directory import (
    InvalidIdentifierError,
    InvalidStoragePathError,
    RunConflictError,
    StorageError,
    create_run_directory,
)


EXPECTED_JSONL_FILENAMES = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)


def test_storage_package_exports_the_phase_07_public_api() -> None:
    assert {
        "RunPaths",
        "create_run_directory",
        "copy_config_snapshot",
        "StorageError",
        "RunConflictError",
        "ConfigSnapshotError",
    } <= set(storage.__all__)


def test_create_run_directory_publishes_exact_approved_layout(tmp_path: Path) -> None:
    output_root = tmp_path / "logs"

    paths = create_run_directory(output_root, "experiment_01", "balanced_rep01")

    assert paths.output_root == output_root.resolve()
    assert paths.experiment_dir == output_root.resolve() / "experiment_01"
    assert paths.run_dir == paths.experiment_dir / "balanced_rep01"
    assert paths.config_snapshot.is_dir()
    assert {path.name for path in paths.config_snapshot.iterdir()} == set(
        config_loader.CONFIG_FILES.values()
    )
    summary = read_run_summary(paths.run_summary)
    assert summary.status == "initialized"
    assert summary.run_id == "balanced_rep01"
    assert summary.retrieval_config_id == "balanced"
    assert summary.repetition_index == 0
    assert tuple(path.name for path in paths.jsonl_files) == EXPECTED_JSONL_FILENAMES
    assert all(path.is_file() and path.read_bytes() == b"" for path in paths.jsonl_files)
    assert sorted(path.name for path in paths.run_dir.iterdir()) == sorted(
        (*EXPECTED_JSONL_FILENAMES, "config_snapshot", "run_summary.json")
    )
    assert not (paths.experiment_dir / ".balanced_rep01.staging").exists()


def test_create_run_directory_creates_parent_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "nested" / "logs"

    paths = create_run_directory(output_root, "experiment_01", "run_01")

    assert paths.run_dir.is_dir()


def test_existing_run_always_fails_without_changing_prior_data(tmp_path: Path) -> None:
    paths = create_run_directory(tmp_path / "logs", "experiment_01", "run_01")
    marker = paths.run_dir / "existing-research-data.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(RunConflictError, match="run directory already exists"):
        create_run_directory(tmp_path / "logs", "experiment_01", "run_01")

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_existing_empty_run_directory_is_still_a_conflict(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "experiment_01" / "run_01"
    run_dir.mkdir(parents=True)

    with pytest.raises(RunConflictError, match="run directory already exists"):
        create_run_directory(tmp_path / "logs", "experiment_01", "run_01")

    assert list(run_dir.iterdir()) == []


def test_stale_staging_directory_is_not_repaired_or_removed(tmp_path: Path) -> None:
    staging = tmp_path / "logs" / "experiment_01" / ".run_01.staging"
    staging.mkdir(parents=True)
    marker = staging / "partial"
    marker.write_text("retain for inspection", encoding="utf-8")

    with pytest.raises(RunConflictError, match="staging directory already exists"):
        create_run_directory(tmp_path / "logs", "experiment_01", "run_01")

    assert marker.read_text(encoding="utf-8") == "retain for inspection"


@pytest.mark.parametrize(
    "invalid_id",
    (
        "",
        ".",
        "..",
        "../run",
        "run/child",
        "run\\child",
        "/absolute",
        "-leading-hyphen",
        "contains space",
        "contains.dot",
        "line\nbreak",
        "a" * 129,
    ),
)
def test_invalid_run_id_is_rejected_before_paths_are_created(
    tmp_path: Path, invalid_id: str
) -> None:
    output_root = tmp_path / "logs"

    with pytest.raises(InvalidIdentifierError, match="run_id"):
        create_run_directory(output_root, "experiment_01", invalid_id)

    assert not output_root.exists()


@pytest.mark.parametrize("invalid_id", ("", "../experiment", "experiment/child", "bad id"))
def test_invalid_experiment_id_is_rejected_before_paths_are_created(
    tmp_path: Path, invalid_id: str
) -> None:
    output_root = tmp_path / "logs"

    with pytest.raises(InvalidIdentifierError, match="experiment_id"):
        create_run_directory(output_root, invalid_id, "run_01")

    assert not output_root.exists()


def test_output_root_must_not_be_an_existing_file(tmp_path: Path) -> None:
    output_root = tmp_path / "logs"
    output_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(InvalidStoragePathError, match="output_root"):
        create_run_directory(output_root, "experiment_01", "run_01")

    assert output_root.read_text(encoding="utf-8") == "not a directory"


def test_output_root_symlink_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real-logs"
    real_root.mkdir()
    linked_root = tmp_path / "logs"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(InvalidStoragePathError, match="must not be a symlink"):
        create_run_directory(linked_root, "experiment_01", "run_01")

    assert list(real_root.iterdir()) == []


def test_experiment_directory_symlink_is_rejected(tmp_path: Path) -> None:
    output_root = tmp_path / "logs"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "experiment_01").symlink_to(outside, target_is_directory=True)

    with pytest.raises(InvalidStoragePathError, match="experiment directory"):
        create_run_directory(output_root, "experiment_01", "run_01")

    assert list(outside.iterdir()) == []


def test_jsonl_creation_failure_does_not_publish_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "logs"
    original_create = run_directory._create_empty_file
    calls = 0

    def fail_during_creation(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated disk failure")
        original_create(path)

    monkeypatch.setattr(run_directory, "_create_empty_file", fail_during_creation)

    with pytest.raises(StorageError, match="simulated disk failure"):
        create_run_directory(output_root, "experiment_01", "run_01")

    experiment_dir = output_root / "experiment_01"
    assert not (experiment_dir / "run_01").exists()
    assert not (experiment_dir / ".run_01.staging").exists()


def test_config_snapshot_failure_does_not_publish_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "logs"

    def fail_snapshot(_source: object, _destination: object) -> tuple[Path, ...]:
        raise ConfigSnapshotError("simulated snapshot failure")

    monkeypatch.setattr(config_snapshot, "copy_config_snapshot", fail_snapshot)

    with pytest.raises(ConfigSnapshotError, match="simulated snapshot failure"):
        create_run_directory(output_root, "experiment_01", "run_01")

    experiment_dir = output_root / "experiment_01"
    assert not (experiment_dir / "run_01").exists()
    assert not (experiment_dir / ".run_01.staging").exists()


def test_initialized_summary_failure_does_not_publish_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "logs"

    def fail_summary(_path: object, _summary: object) -> None:
        raise OSError("simulated summary failure")

    monkeypatch.setattr(run_directory, "write_run_summary", fail_summary)

    with pytest.raises(StorageError, match="simulated summary failure"):
        create_run_directory(output_root, "experiment_01", "run_01")

    experiment_dir = output_root / "experiment_01"
    assert not (experiment_dir / "run_01").exists()
    assert not (experiment_dir / ".run_01.staging").exists()


def test_final_publication_failure_does_not_publish_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "logs"

    def fail_publication(_staging: Path, _final: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(
        run_directory, "_publish_staging_directory", fail_publication
    )

    with pytest.raises(StorageError, match="simulated publication failure"):
        create_run_directory(output_root, "experiment_01", "run_01")

    experiment_dir = output_root / "experiment_01"
    assert not (experiment_dir / "run_01").exists()
    assert not (experiment_dir / ".run_01.staging").exists()


def test_separate_run_ids_are_isolated(tmp_path: Path) -> None:
    output_root = tmp_path / "logs"

    first = create_run_directory(output_root, "experiment_01", "run_01")
    second = create_run_directory(output_root, "experiment_01", "run_02")
    first.behavior_logs.write_text('{"run":"one"}\n', encoding="utf-8")

    assert first.run_dir != second.run_dir
    assert second.behavior_logs.read_bytes() == b""
