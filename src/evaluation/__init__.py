"""Public APIs for deterministic post-run Phase 09 evaluation."""

from src.evaluation.aggregator import (
    aggregate_configuration_mra,
    aggregate_run_mra,
)
from src.evaluation.bcr_export import build_bcr_export
from src.evaluation.errors import (
    EvaluationAggregationError,
    EvaluationCalculationError,
    EvaluationConflictError,
    EvaluationEligibilityError,
    EvaluationError,
    EvaluationGroundTruthError,
    EvaluationInputError,
    EvaluationJoinError,
    EvaluationPersistenceError,
    EvaluationSchemaError,
)
from src.evaluation.mra import calculate_mra, calculate_run_mra
from src.evaluation.persistence import EVALUATION_FILENAMES, publish_evaluation
from src.evaluation.readers import (
    LoadedEvaluationRun,
    TargetEvaluationUnit,
    load_evaluation_run,
)
from src.evaluation.schemas import (
    BCRExportRecord,
    BCRJudgeInput,
    BCRModelMetadata,
    ConfigurationMRAAggregate,
    EVALUATION_MODES,
    EVALUATION_SCHEMA_VERSION,
    EvaluationUnitIdentity,
    FutureBCRJudgment,
    MRAEvaluationResult,
    MRAGroupAggregate,
    PersonaJudgeInput,
    PriorBehaviorEvidence,
    RunEvaluationSummary,
    RunMRAAggregates,
    SelectedMemoryEvidence,
    TargetEventJudgeInput,
    future_bcr_contract_dict,
)
from src.evaluation.service import EvaluationRunResult, evaluate_run

__all__ = [
    "BCRExportRecord",
    "BCRJudgeInput",
    "BCRModelMetadata",
    "ConfigurationMRAAggregate",
    "EVALUATION_FILENAMES",
    "EVALUATION_MODES",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationAggregationError",
    "EvaluationCalculationError",
    "EvaluationConflictError",
    "EvaluationEligibilityError",
    "EvaluationError",
    "EvaluationGroundTruthError",
    "EvaluationInputError",
    "EvaluationJoinError",
    "EvaluationPersistenceError",
    "EvaluationRunResult",
    "EvaluationSchemaError",
    "EvaluationUnitIdentity",
    "FutureBCRJudgment",
    "LoadedEvaluationRun",
    "MRAEvaluationResult",
    "MRAGroupAggregate",
    "PersonaJudgeInput",
    "PriorBehaviorEvidence",
    "RunEvaluationSummary",
    "RunMRAAggregates",
    "SelectedMemoryEvidence",
    "TargetEvaluationUnit",
    "TargetEventJudgeInput",
    "aggregate_configuration_mra",
    "aggregate_run_mra",
    "build_bcr_export",
    "calculate_mra",
    "calculate_run_mra",
    "evaluate_run",
    "future_bcr_contract_dict",
    "load_evaluation_run",
    "publish_evaluation",
]
