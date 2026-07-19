"""Typed experiment settings models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseLimits:
    natural_interaction_sentences: int
    target_event_max_paragraphs: int
    reflection_sentences: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.natural_interaction_sentences,
                self.target_event_max_paragraphs,
                self.reflection_sentences,
            )
        ):
            raise ValueError("response limits must be positive integers")


@dataclass(frozen=True)
class ModelSettings:
    """Placeholders reserved for the later real-LLM phase."""

    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.model_name is not None and (
            not isinstance(self.model_name, str) or not self.model_name.strip()
        ):
            raise ValueError("model_name must be a non-empty string or None")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a non-negative number or None")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer or None")


@dataclass(frozen=True)
class ExperimentSettings:
    simulation_days: int
    natural_blocks_total: int
    target_events_total: int
    turns_per_agent_per_natural_block: int
    total_turns_per_natural_block: int
    teacher_opens_natural_block: bool
    top_k: int
    repetitions: int
    primary_mra_step_type: str
    primary_bcr_step_type: str
    natural_block_primary_evaluation: bool
    target_event_primary_evaluation: bool
    dummy_mode_default: bool
    allow_real_llm_default: bool
    allow_full_experiment_default: bool
    response_limits: ResponseLimits
    model_settings: ModelSettings = ModelSettings()

    def __post_init__(self) -> None:
        integer_fields = {
            "simulation_days": self.simulation_days,
            "natural_blocks_total": self.natural_blocks_total,
            "target_events_total": self.target_events_total,
            "turns_per_agent_per_natural_block": self.turns_per_agent_per_natural_block,
            "total_turns_per_natural_block": self.total_turns_per_natural_block,
            "top_k": self.top_k,
            "repetitions": self.repetitions,
        }
        for field_name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")

        boolean_fields = {
            "teacher_opens_natural_block": self.teacher_opens_natural_block,
            "natural_block_primary_evaluation": self.natural_block_primary_evaluation,
            "target_event_primary_evaluation": self.target_event_primary_evaluation,
            "dummy_mode_default": self.dummy_mode_default,
            "allow_real_llm_default": self.allow_real_llm_default,
            "allow_full_experiment_default": self.allow_full_experiment_default,
        }
        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")

        for field_name, value in (
            ("primary_mra_step_type", self.primary_mra_step_type),
            ("primary_bcr_step_type", self.primary_bcr_step_type),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.response_limits, ResponseLimits):
            raise ValueError("response_limits must be a ResponseLimits value")
        if not isinstance(self.model_settings, ModelSettings):
            raise ValueError("model_settings must be a ModelSettings value")
