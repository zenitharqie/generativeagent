from pathlib import Path
import shutil

import pytest

from src.config_loader import loader as config_loader
from src.logging_system.run_summary import RunSummary, write_run_summary
from src.storage import config_snapshot
from src.storage.config_snapshot import ConfigSnapshotError, copy_config_snapshot
from src.storage.run_directory import create_run_directory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "config"


def copied_config(tmp_path: Path) -> Path:
    destination = tmp_path / "source-config"
    shutil.copytree(CONFIG_DIRECTORY, destination)
    return destination


def canonical_filenames() -> tuple[str, ...]:
    return tuple(config_loader.CONFIG_FILES.values())


def test_snapshot_copies_only_canonical_files_byte_for_byte(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    (source / ".env").write_text("API_KEY=secret", encoding="utf-8")
    (source / ".env.example").write_text("API_KEY=", encoding="utf-8")
    (source / "unrelated.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "snapshot"

    copied = copy_config_snapshot(source, destination)

    assert tuple(path.name for path in copied) == canonical_filenames()
    assert {path.name for path in destination.iterdir()} == set(canonical_filenames())
    for filename in canonical_filenames():
        assert (destination / filename).read_bytes() == (source / filename).read_bytes()
    assert not (destination / ".env").exists()
    assert not (destination / ".env.example").exists()
    assert not (destination / "unrelated.json").exists()


def test_run_initialization_populates_complete_config_snapshot(tmp_path: Path) -> None:
    paths = create_run_directory(tmp_path / "logs", "experiment_01", "run_01")

    assert {path.name for path in paths.config_snapshot.iterdir()} == set(
        canonical_filenames()
    )
    for filename in canonical_filenames():
        assert (paths.config_snapshot / filename).read_bytes() == (
            CONFIG_DIRECTORY / filename
        ).read_bytes()


def test_initialized_run_allows_its_first_config_snapshot(tmp_path: Path) -> None:
    run_directory = tmp_path / "logs" / "experiment_01" / "run_01"
    run_directory.mkdir(parents=True)
    write_run_summary(
        run_directory / "run_summary.json",
        RunSummary("experiment_01", "run_01", "balanced", 0),
    )

    copied = copy_config_snapshot(
        CONFIG_DIRECTORY,
        run_directory / "config_snapshot",
    )

    assert tuple(path.name for path in copied) == canonical_filenames()


def test_snapshot_does_not_modify_source_files(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    before = {name: (source / name).read_bytes() for name in canonical_filenames()}

    copy_config_snapshot(source, tmp_path / "snapshot")

    assert {name: (source / name).read_bytes() for name in canonical_filenames()} == before


def test_snapshot_uses_dynamic_canonical_file_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copied_config(tmp_path)
    extension_name = "research_extension.json"
    (source / extension_name).write_text('{"extension":true}\n', encoding="utf-8")
    dynamic_files = {**config_loader.CONFIG_FILES, "extension": extension_name}
    monkeypatch.setattr(config_loader, "CONFIG_FILES", dynamic_files)

    copied = copy_config_snapshot(source, tmp_path / "snapshot")

    assert tuple(path.name for path in copied) == tuple(dynamic_files.values())
    assert (tmp_path / "snapshot" / extension_name).read_bytes() == (
        source / extension_name
    ).read_bytes()


def test_missing_canonical_file_prevents_snapshot_publication(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    (source / canonical_filenames()[0]).unlink()
    destination = tmp_path / "snapshot"

    with pytest.raises(ConfigSnapshotError, match="missing or unreadable"):
        copy_config_snapshot(source, destination)

    assert not destination.exists()


def test_invalid_config_prevents_snapshot_publication(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    settings = source / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    destination = tmp_path / "snapshot"

    with pytest.raises(ConfigSnapshotError, match="configuration validation failed"):
        copy_config_snapshot(source, destination)

    assert not destination.exists()


def test_external_symlink_source_is_rejected(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    filename = canonical_filenames()[0]
    outside = tmp_path / filename
    outside.write_bytes((source / filename).read_bytes())
    (source / filename).unlink()
    (source / filename).symlink_to(outside)

    with pytest.raises(ConfigSnapshotError, match="escapes its source directory"):
        copy_config_snapshot(source, tmp_path / "snapshot")

    assert outside.is_file()
    assert not (tmp_path / "snapshot").exists()


def test_unsafe_filename_in_canonical_mapping_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copied_config(tmp_path)
    unsafe_files = {**config_loader.CONFIG_FILES, "escape": "../outside.json"}
    monkeypatch.setattr(config_loader, "CONFIG_FILES", unsafe_files)

    with pytest.raises(ConfigSnapshotError, match="unsafe canonical"):
        copy_config_snapshot(source, tmp_path / "snapshot")

    assert not (tmp_path / "snapshot").exists()


def test_existing_nonempty_snapshot_is_never_overwritten(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    marker = destination / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ConfigSnapshotError, match="not empty"):
        copy_config_snapshot(source, destination)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_snapshot_destination_symlink_is_rejected(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "snapshot"
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConfigSnapshotError, match="must not be a symlink"):
        copy_config_snapshot(source, destination)

    assert list(outside.iterdir()) == []


def test_snapshot_rejects_symlinked_experiment_directory(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    outside = tmp_path / "outside"
    (outside / "run_01").mkdir(parents=True)
    linked_experiment = tmp_path / "experiment_01"
    linked_experiment.symlink_to(outside, target_is_directory=True)
    destination = linked_experiment / "run_01" / "config_snapshot"

    with pytest.raises(ConfigSnapshotError, match="must not be symlinks"):
        copy_config_snapshot(source, destination)

    assert not (outside / "run_01" / "config_snapshot").exists()


def test_copy_failure_preserves_empty_destination_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copied_config(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    original_copy = config_snapshot._copy_file_durably
    calls = 0

    def fail_during_copy(source_path: Path, destination_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        original_copy(source_path, destination_path)

    monkeypatch.setattr(config_snapshot, "_copy_file_durably", fail_during_copy)

    with pytest.raises(ConfigSnapshotError, match="simulated copy failure"):
        copy_config_snapshot(source, destination)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not (tmp_path / ".snapshot.staging").exists()


def test_existing_snapshot_staging_is_not_repaired(tmp_path: Path) -> None:
    source = copied_config(tmp_path)
    staging = tmp_path / ".snapshot.staging"
    staging.mkdir()
    marker = staging / "partial"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ConfigSnapshotError, match="staging directory already exists"):
        copy_config_snapshot(source, tmp_path / "snapshot")

    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "excluded"])
def test_terminal_run_prevents_config_snapshot_mutation(
    tmp_path: Path,
    status: str,
) -> None:
    source = copied_config(tmp_path)
    paths = create_run_directory(tmp_path / "logs", "experiment_01", "run_01")
    terminal = RunSummary(
        experiment_id="experiment_01",
        run_id="run_01",
        retrieval_config_id="balanced",
        repetition_index=0,
        status=status,
        completed_natural_blocks=7,
        completed_target_events=5,
        completed_main_turns=104,
    )
    write_run_summary(paths.run_summary, terminal)

    with pytest.raises(ConfigSnapshotError, match="terminal and read-only"):
        copy_config_snapshot(source, paths.config_snapshot)

    assert {path.name for path in paths.config_snapshot.iterdir()} == set(
        canonical_filenames()
    )
