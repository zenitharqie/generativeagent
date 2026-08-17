"""Manual-independent Ground Truth annotation and automated finalization."""

from .annotation_package import (
    ANNOTATION_INSTRUCTIONS,
    ANNOTATION_OUTPUT_SCHEMA,
    NECESSITY_TEST,
    AnnotationPackage,
    build_annotation_package,
    validate_annotation_package,
)
from .errors import (
    GroundTruthCompletenessError,
    GroundTruthConsolidationError,
    GroundTruthFinalizationError,
    GroundTruthInputError,
    GroundTruthPersistenceError,
    GroundTruthWorkflowError,
)
from .embedding_service import CONFIRMATION as EMBEDDING_CONFIRMATION, generate_embedding_evidence
from .finalizer import (
    GroundTruthFinalizationBundle,
    build_finalization_bundle,
    calculate_fleiss_kappa,
)
from .readers import (
    build_embedding_input,
    load_annotation_package,
    load_complete_annotations,
    load_embedding_evidence,
    parse_submission,
    positive_phrase_inventory,
)
from .schemas import (
    ANNOTATOR_MODELS,
    EXPECTED_ANNOTATION_COUNT,
    EXPECTED_UNIT_COUNT,
    FINAL_GROUND_TRUTH_STATUS,
    GROUND_TRUTH_FINALIZATION_VERSION,
    GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
    MAJORITY_SUPPORT_REQUIRED,
    SEMANTIC_CONSOLIDATION_MODEL_IDENTITY,
    SEMANTIC_CONSOLIDATION_THRESHOLD,
    validate_final_ground_truth_identity,
)
from .service import (
    finalize_ground_truth,
    generate_annotation_package,
    prepare_embedding_input,
)

__all__ = [
    "ANNOTATION_INSTRUCTIONS",
    "ANNOTATION_OUTPUT_SCHEMA",
    "ANNOTATOR_MODELS",
    "AnnotationPackage",
    "EXPECTED_ANNOTATION_COUNT",
    "EXPECTED_UNIT_COUNT",
    "FINAL_GROUND_TRUTH_STATUS",
    "GROUND_TRUTH_FINALIZATION_VERSION",
    "GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION",
    "GroundTruthCompletenessError",
    "GroundTruthConsolidationError",
    "GroundTruthFinalizationBundle",
    "GroundTruthFinalizationError",
    "GroundTruthInputError",
    "GroundTruthPersistenceError",
    "GroundTruthWorkflowError",
    "EMBEDDING_CONFIRMATION",
    "MAJORITY_SUPPORT_REQUIRED",
    "NECESSITY_TEST",
    "SEMANTIC_CONSOLIDATION_MODEL_IDENTITY",
    "SEMANTIC_CONSOLIDATION_THRESHOLD",
    "build_annotation_package",
    "build_embedding_input",
    "build_finalization_bundle",
    "calculate_fleiss_kappa",
    "finalize_ground_truth",
    "generate_embedding_evidence",
    "generate_annotation_package",
    "load_annotation_package",
    "load_complete_annotations",
    "load_embedding_evidence",
    "parse_submission",
    "positive_phrase_inventory",
    "prepare_embedding_input",
    "validate_annotation_package",
    "validate_final_ground_truth_identity",
]
