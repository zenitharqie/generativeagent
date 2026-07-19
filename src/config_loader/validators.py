"""Cross-file validation for the fixed experiment architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from src.domain.agents import FIXED_AGENT_IDS
from src.domain.simulation import NaturalInteractionBlockStep, TargetEventStep

if TYPE_CHECKING:
    from src.config_loader.loader import ExperimentConfig


FIXED_EVENT_IDS = ("event_01", "event_02", "event_03", "event_04", "event_05")
FIXED_RETRIEVAL_WEIGHTS = {
    "balanced": (0.33, 0.33, 0.33),
    "recency_heavy": (0.60, 0.20, 0.20),
    "importance_heavy": (0.20, 0.60, 0.20),
    "relevance_heavy": (0.20, 0.20, 0.60),
}


class ConfigValidationError(ValueError):
    """Raised when parsed configuration violates schema or architecture rules."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate all fixed constants and relationships across config files."""
    _validate_agents(config)
    _validate_settings(config)
    _validate_steps(config)
    _validate_events(config)
    _validate_retrieval_configs(config)
    _validate_ground_truth(config)


def _validate_agents(config: ExperimentConfig) -> None:
    agent_ids = [agent.agent_id for agent in config.agents]
    _require(len(agent_ids) == 4, "agents.json must define exactly 4 agents")
    _require(len(set(agent_ids)) == 4, "agent IDs must be unique")
    _require(
        set(agent_ids) == FIXED_AGENT_IDS,
        "agent IDs must be exactly teacher, active_student, quiet_student, academic_student",
    )


def _validate_settings(config: ExperimentConfig) -> None:
    settings = config.settings
    fixed_values = {
        "simulation_days": (settings.simulation_days, 6),
        "natural_blocks_total": (settings.natural_blocks_total, 7),
        "target_events_total": (settings.target_events_total, 5),
        "turns_per_agent_per_natural_block": (
            settings.turns_per_agent_per_natural_block,
            3,
        ),
        "total_turns_per_natural_block": (
            settings.total_turns_per_natural_block,
            12,
        ),
        "top_k": (settings.top_k, 5),
        "repetitions": (settings.repetitions, 3),
    }
    for field_name, (actual, expected) in fixed_values.items():
        _require(actual == expected, f"settings.{field_name} must be {expected}")

    _require(
        settings.teacher_opens_natural_block is True,
        "settings.teacher_opens_natural_block must be true",
    )
    _require(
        settings.natural_block_primary_evaluation is False,
        "natural blocks must not be primary evaluation targets",
    )
    _require(
        settings.target_event_primary_evaluation is True,
        "target events must be primary evaluation targets",
    )
    _require(
        settings.primary_mra_step_type == "target_event",
        "primary MRA step type must be target_event",
    )
    _require(
        settings.primary_bcr_step_type == "target_event",
        "primary BCR step type must be target_event",
    )
    _require(settings.dummy_mode_default is True, "dummy_mode_default must be true")
    _require(
        settings.allow_real_llm_default is False,
        "allow_real_llm_default must be false",
    )
    _require(
        settings.allow_full_experiment_default is False,
        "allow_full_experiment_default must be false",
    )


def _validate_steps(config: ExperimentConfig) -> None:
    natural_steps = [
        step for step in config.steps if isinstance(step, NaturalInteractionBlockStep)
    ]
    target_steps = [step for step in config.steps if isinstance(step, TargetEventStep)]

    _require(len(config.steps) == 12, "simulation_steps.json must define exactly 12 steps")
    _require(len(natural_steps) == 7, "timeline must contain exactly 7 natural blocks")
    _require(len(target_steps) == 5, "timeline must contain exactly 5 target events")

    for step in config.steps:
        _require(
            len(step.participants) == 4 and set(step.participants) == FIXED_AGENT_IDS,
            f"step {step.step_id} participants must be exactly the four fixed agents",
        )
        _require(
            step.memory_generation_enabled is True,
            f"step {step.step_id} must enable memory generation",
        )

    for step in natural_steps:
        _require(
            step.turns_per_agent == 3,
            f"natural block {step.step_id} turns_per_agent must be 3",
        )
        _require(
            step.total_turns == 12,
            f"natural block {step.step_id} total_turns must be 12",
        )
        _require(
            step.teacher_opens_block is True,
            f"natural block {step.step_id} teacher_opens_block must be true",
        )
        _require(
            step.evaluation_enabled is False,
            f"natural block {step.step_id} must not enable evaluation",
        )

    for step in target_steps:
        _require(
            step.evaluation_enabled is True,
            f"target event {step.step_id} must enable evaluation",
        )

    day_1_types = [step.step_type for step in config.steps if step.day == 1]
    _require(
        day_1_types == ["natural_interaction_block", "natural_interaction_block"],
        "Day 1 must contain exactly 2 natural blocks and no target event",
    )
    for day in range(2, 7):
        day_types = [step.step_type for step in config.steps if step.day == day]
        _require(
            day_types == ["natural_interaction_block", "target_event"],
            f"Day {day} must contain one natural block followed by one target event",
        )
    _require(
        all(1 <= step.day <= 6 for step in config.steps),
        "all simulation steps must occur during Days 1-6",
    )


def _validate_events(config: ExperimentConfig) -> None:
    event_ids = tuple(event.event_id for event in config.target_events)
    _require(
        event_ids == FIXED_EVENT_IDS,
        "target events must use the five fixed event IDs in order",
    )

    defined_ids = set(event_ids)
    events_by_id = {event.event_id: event for event in config.target_events}
    target_steps = [step for step in config.steps if isinstance(step, TargetEventStep)]
    for step in target_steps:
        _require(
            step.event_id in defined_ids,
            f"target event step {step.step_id} references missing event_id {step.event_id!r}",
        )
        event = events_by_id[step.event_id]
        _require(
            (step.title, step.description) == (event.title, event.description),
            f"target event step {step.step_id} must match its event definition",
        )
    _require(
        tuple(step.event_id for step in target_steps) == FIXED_EVENT_IDS,
        "timeline must reference each target event once in fixed order",
    )


def _validate_retrieval_configs(config: ExperimentConfig) -> None:
    by_id = {item.config_id: item for item in config.retrieval_configs}
    _require(
        len(config.retrieval_configs) == 4 and set(by_id) == set(FIXED_RETRIEVAL_WEIGHTS),
        (
            "retrieval configs must be exactly balanced, recency_heavy, "
            "importance_heavy, relevance_heavy"
        ),
    )
    for config_id, expected in FIXED_RETRIEVAL_WEIGHTS.items():
        weights = by_id[config_id].weights
        actual = (weights.recency, weights.importance, weights.relevance)
        _require(actual == expected, f"retrieval weights for {config_id} must remain fixed")


def _validate_ground_truth(config: ExperimentConfig) -> None:
    ground_truth = config.ground_truth
    _require(
        ground_truth.get("evaluation_step_type") == "target_event",
        "ground truth must apply to target_event by default",
    )
    targets = ground_truth.get("targets")
    _require(isinstance(targets, list), "ground_truth.targets must be a list")
    event_ids = {event.event_id for event in config.target_events}
    required_pairs = {
        (event_id, agent_id) for event_id in event_ids for agent_id in FIXED_AGENT_IDS
    }
    actual_required_pairs: list[tuple[str, str]] = []

    for index, target in enumerate(targets):
        _require(isinstance(target, Mapping), f"ground truth target {index} must be an object")
        event_id = target.get("event_id")
        optional = target.get("optional", False)
        _require(
            isinstance(optional, bool),
            f"ground truth target {index} optional must be boolean",
        )
        _require(
            event_id in event_ids or optional,
            f"ground truth target {index} references non-target event {event_id!r}",
        )
        agent_id = target.get("agent_id")
        _require(
            agent_id in FIXED_AGENT_IDS,
            f"ground truth target {index} references invalid agent_id {agent_id!r}",
        )
        categories = target.get("expected_categories")
        _require(
            isinstance(categories, list)
            and categories
            and all(isinstance(category, str) and category.strip() for category in categories),
            f"ground truth target {index} expected_categories must contain strings",
        )
        if event_id in event_ids and not optional:
            actual_required_pairs.append((event_id, agent_id))

    _require(
        len(actual_required_pairs) == len(set(actual_required_pairs)),
        "ground truth target-event/agent pairs must be unique",
    )
    _require(
        set(actual_required_pairs) == required_pairs,
        "ground truth must cover every target-event/agent pair",
    )
