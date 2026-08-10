"""End-to-end deterministic Phase 09 evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.aggregator import aggregate_run_mra
from src.evaluation.bcr_export import build_bcr_export
from src.evaluation.errors import EvaluationPersistenceError
from src.evaluation.mra import calculate_run_mra
from src.evaluation.persistence import publish_evaluation
from src.evaluation.readers import load_evaluation_run
from src.evaluation.schemas import (
    BCRExportRecord,
    BCRModelMetadata,
    MRAEvaluationResult,
    RunEvaluationSummary,
)


@dataclass(frozen=True)
class EvaluationRunResult:
    """Immutable in-process result of one atomically published evaluation."""

    output_dir: Path
    mra_records: tuple[MRAEvaluationResult, ...]
    bcr_records: tuple[BCRExportRecord, ...]
    summary: RunEvaluationSummary


def _require_separate_output(source_run: Path, output_dir: str | Path) -> None:
    try:
        candidate = Path(output_dir).resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        raise EvaluationPersistenceError(f"invalid output_dir: {output_dir!r}") from exc
    if candidate == source_run or source_run in candidate.parents:
        raise EvaluationPersistenceError(
            "evaluation output must be separate from the source run directory"
        )


def evaluate_run(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    diagnostic: bool = False,
) -> EvaluationRunResult:
    """Evaluate one persisted run without executing or mutating simulation code."""
    loaded = load_evaluation_run(run_dir, diagnostic=diagnostic)
    _require_separate_output(loaded.run_dir, output_dir)
    mra_records = calculate_run_mra(loaded)
    bcr_records = build_bcr_export(loaded)
    aggregates = aggregate_run_mra(
        mra_records,
        evaluation_mode=loaded.evaluation_mode,
    )
    model_metadata = BCRModelMetadata(
        simulator_model_identity=loaded.simulator_model_identity,
    )
    summary = RunEvaluationSummary(
        experiment_id=loaded.summary.experiment_id,
        run_id=loaded.summary.run_id,
        retrieval_config_id=loaded.summary.retrieval_config_id,
        repetition_index=loaded.summary.repetition_index,
        source_run_status=loaded.summary.status,
        source_schema_version=loaded.summary.schema_version,
        evaluation_mode=loaded.evaluation_mode,
        primary_eligible=loaded.primary_eligible,
        included_in_final_research_aggregation=(
            loaded.included_in_final_research_aggregation
        ),
        mra_aggregates=aggregates,
        bcr_export_count=len(bcr_records),
        model_metadata=model_metadata,
    )
    published = publish_evaluation(
        output_dir=output_dir,
        mra_records=mra_records,
        bcr_records=bcr_records,
        summary=summary,
    )
    return EvaluationRunResult(
        output_dir=published,
        mra_records=mra_records,
        bcr_records=bcr_records,
        summary=summary,
    )


__all__ = ["EvaluationRunResult", "evaluate_run"]
