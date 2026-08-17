"""Offline public API for the controlled Ground Truth workflow."""

from __future__ import annotations

from pathlib import Path

from src.config_loader import DEFAULT_CONFIG_DIRECTORY

from .annotation_package import AnnotationPackage, build_annotation_package
from .finalizer import GroundTruthFinalizationBundle, build_finalization_bundle
from .persistence import (
    fingerprint_source_files,
    publish_finalization_bundle,
    publish_json_file,
)
from .readers import (
    build_embedding_input,
    load_annotation_package,
    load_complete_annotations,
)
from .schemas import ANNOTATOR_MODELS, canonical_json_bytes


def _resolved(path: str | Path, *, must_exist: bool) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts:
        from .errors import GroundTruthPersistenceError

        raise GroundTruthPersistenceError("Ground Truth paths must not contain traversal")
    try:
        return candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError, ValueError) as exc:
        from .errors import GroundTruthPersistenceError

        raise GroundTruthPersistenceError("could not resolve Ground Truth path safely") from exc


def _require_separate_output(
    output: str | Path,
    *,
    protected: tuple[str | Path, ...],
    output_is_directory: bool,
) -> None:
    from .errors import GroundTruthPersistenceError

    resolved_output = _resolved(output, must_exist=False)
    for source in protected:
        resolved_source = _resolved(source, must_exist=True)
        overlaps = resolved_output == resolved_source
        if resolved_source.is_dir():
            overlaps = overlaps or resolved_source in resolved_output.parents
            if output_is_directory:
                overlaps = overlaps or resolved_output in resolved_source.parents
        elif output_is_directory:
            overlaps = overlaps or resolved_output in resolved_source.parents
        if overlaps:
            raise GroundTruthPersistenceError(
                "Ground Truth output must be separate from canonical inputs"
            )


def generate_annotation_package(
    *,
    output: str | Path,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> AnnotationPackage:
    _require_separate_output(
        output,
        protected=(config_directory,),
        output_is_directory=False,
    )
    package = build_annotation_package(config_directory)
    publish_json_file(output, package.payload)
    return package


def prepare_embedding_input(
    *,
    package_path: str | Path,
    annotations_directory: str | Path,
    output: str | Path,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> dict[str, object]:
    _require_separate_output(
        output,
        protected=(package_path, annotations_directory, config_directory),
        output_is_directory=False,
    )
    package = load_annotation_package(package_path, config_directory=config_directory)
    submissions = load_complete_annotations(annotations_directory, package=package)
    value = build_embedding_input(package=package, submissions=submissions)
    publish_json_file(output, canonical_json_bytes(value, pretty=True))
    return value


def finalize_ground_truth(
    *,
    package_path: str | Path,
    annotations_directory: str | Path,
    embedding_evidence_path: str | Path,
    output_directory: str | Path,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> GroundTruthFinalizationBundle:
    _require_separate_output(
        output_directory,
        protected=(
            package_path,
            annotations_directory,
            embedding_evidence_path,
            config_directory,
        ),
        output_is_directory=True,
    )
    annotation_root = Path(annotations_directory)
    source_paths = (
        Path(package_path),
        *(annotation_root / f"{annotator_id}.json" for annotator_id in ANNOTATOR_MODELS),
        Path(embedding_evidence_path),
        Path(config_directory) / "agents.json",
        Path(config_directory) / "target_events.json",
    )
    before = fingerprint_source_files(source_paths)
    bundle = build_finalization_bundle(
        package_path=package_path,
        annotations_directory=annotations_directory,
        embedding_evidence_path=embedding_evidence_path,
        config_directory=config_directory,
    )
    publish_finalization_bundle(
        output_directory,
        bundle,
        source_fingerprints=before,
        recheck_sources=lambda: fingerprint_source_files(source_paths),
    )
    return bundle


__all__ = [
    "finalize_ground_truth",
    "generate_annotation_package",
    "prepare_embedding_input",
]
