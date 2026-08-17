"""Binding Phase 10 provider, model, prompt, and runtime contracts."""

from pathlib import Path

from src.llm.prompt_loader import DEFAULT_PROMPT_DIRECTORY, PROMPT_SPECS
from src.llm.settings import (
    BCR_RUBRIC_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_IDENTITY,
    EMBEDDING_PROFILE,
    EVALUATOR_MODEL,
    EVALUATOR_MODEL_IDENTITY,
    EVALUATOR_PROFILE,
    MAX_PROVIDER_ATTEMPTS,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    RETRY_DELAY_SECONDS,
    SIMULATOR_MODEL,
    SIMULATOR_MODEL_IDENTITY,
    SIMULATOR_PROFILE,
)


EXPECTED_PROMPT_FILES = {
    "plan.txt",
    "natural_interaction.txt",
    "target_event_response.txt",
    "reflect.txt",
    "importance_scoring.txt",
    "bcr_judge.txt",
}


def test_provider_model_and_embedding_contracts_are_exact() -> None:
    assert SIMULATOR_MODEL == "gpt-5-nano-2025-08-07"
    assert SIMULATOR_MODEL_IDENTITY == "openai:gpt-5-nano-2025-08-07"
    assert EVALUATOR_MODEL == "claude-sonnet-4-6"
    assert EVALUATOR_MODEL_IDENTITY == "anthropic:claude-sonnet-4-6"
    assert SIMULATOR_MODEL_IDENTITY != EVALUATOR_MODEL_IDENTITY
    assert EMBEDDING_MODEL == "text-embedding-3-small"
    assert EMBEDDING_MODEL_IDENTITY == "openai:text-embedding-3-small"
    assert EMBEDDING_DIMENSION == 1536


def test_runtime_profiles_freeze_approved_pilot_settings() -> None:
    assert SIMULATOR_PROFILE.reasoning_effort == "minimal"
    assert EVALUATOR_PROFILE.reasoning_effort == "medium"
    assert SIMULATOR_PROFILE.request_timeout_seconds == 60
    assert EVALUATOR_PROFILE.request_timeout_seconds == 60
    assert EMBEDDING_PROFILE.request_timeout_seconds == 30
    assert SIMULATOR_PROFILE.stream is False
    assert EVALUATOR_PROFILE.stream is False
    assert SIMULATOR_PROFILE.temperature is None
    assert SIMULATOR_PROFILE.top_p is None
    assert SIMULATOR_PROFILE.seed is None
    assert EVALUATOR_PROFILE.temperature is None
    assert EVALUATOR_PROFILE.top_p is None
    assert EVALUATOR_PROFILE.seed is None
    assert MAX_PROVIDER_ATTEMPTS == 2
    assert RETRY_DELAY_SECONDS == 1
    assert BCR_RUBRIC_VERSION == "phase10_binary_bcr_rubric_v2"
    assert OPENAI_API_KEY_ENVIRONMENT_VARIABLE == "OPENAI_API_KEY"


def test_call_type_output_token_limits_are_exact() -> None:
    assert SIMULATOR_PROFILE.output_token_limits == {
        "plan": 300,
        "natural_response": 500,
        "target_event_response": 500,
        "reflection": 400,
        "importance_score": 150,
    }
    assert EVALUATOR_PROFILE.output_token_limits == {"bcr_judge": 700}


def test_exact_six_production_prompt_files_and_specs() -> None:
    files = {
        path.name
        for path in Path(DEFAULT_PROMPT_DIRECTORY).iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    assert files == EXPECTED_PROMPT_FILES
    assert {spec.filename for spec in PROMPT_SPECS.values()} == EXPECTED_PROMPT_FILES
    assert len(PROMPT_SPECS) == 6


def test_provider_dependencies_are_declared_for_bounded_ranges() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "openai>=2,<3" in requirements
    assert "anthropic>=0.104,<0.121" in requirements
