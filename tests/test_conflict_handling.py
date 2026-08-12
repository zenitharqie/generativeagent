"""Fresh-path and no-reuse contracts for Phase 11 artifacts."""

from __future__ import annotations

import pytest

from src.storage import RunConflictError
from src.simulation.safeguards import (
    PilotPathError,
    build_pilot_paths,
    validate_fresh_pilot_paths,
)


@pytest.mark.parametrize(
    "attribute",
    [
        "source_run_dir",
        "source_staging_dir",
        "diagnostic_evaluation_dir",
        "diagnostic_evaluation_staging_dir",
        "evidence_dir",
        "evidence_staging_dir",
        "review_dir",
        "review_staging_dir",
    ],
)
def test_any_existing_final_or_staging_path_is_a_conflict(tmp_path, attribute: str) -> None:
    paths = build_pilot_paths(
        tmp_path,
        experiment_id="phase11_conflict",
        run_id="pilot",
    )
    conflict = getattr(paths, attribute)
    conflict.mkdir(parents=True)

    with pytest.raises(RunConflictError):
        validate_fresh_pilot_paths(paths)


def test_output_root_traversal_is_rejected_without_creating_paths(tmp_path) -> None:
    requested = tmp_path / "nested" / ".." / "escape"

    with pytest.raises(PilotPathError, match="traversal"):
        build_pilot_paths(
            requested,
            experiment_id="phase11_path",
            run_id="pilot",
        )

    assert not (tmp_path / "nested").exists()
    assert not (tmp_path / "escape").exists()


def test_symbolic_link_output_root_is_rejected_without_target_mutation(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-output"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(PilotPathError, match="symbolic link"):
        build_pilot_paths(
            link,
            experiment_id="phase11_path",
            run_id="pilot",
        )

    assert tuple(target.iterdir()) == ()
