"""Load JSON configuration into typed experiment structures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.config_loader.validators import ConfigValidationError, validate_experiment_config
from src.domain.agents import AgentProfile, PersonaProfile, SeedMemoryConfig
from src.domain.retrieval import RETRIEVAL_COMPONENTS, RetrievalConfig, RetrievalWeights
from src.domain.settings import ExperimentSettings, ModelSettings, ResponseLimits
from src.domain.simulation import (
    NaturalInteractionBlockStep,
    SimulationStep,
    TargetEvent,
    TargetEventStep,
)


DEFAULT_CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "config"
CONFIG_FILES = {
    "agents": "agents.json",
    "steps": "simulation_steps.json",
    "events": "target_events.json",
    "retrieval": "retrieval_configs.json",
    "settings": "settings.json",
    "ground_truth": "ground_truth.json",
}


class ConfigLoadError(ValueError):
    """Raised when a configuration file cannot be read or decoded."""


@dataclass(frozen=True)
class ExperimentConfig:
    """All parsed configuration required to initialize an experiment."""

    agents: tuple[AgentProfile, ...]
    steps: tuple[SimulationStep, ...]
    target_events: tuple[TargetEvent, ...]
    retrieval_configs: tuple[RetrievalConfig, ...]
    settings: ExperimentSettings
    ground_truth: Mapping[str, Any]


def load_experiment_config(
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> ExperimentConfig:
    """Load, parse, and validate every required experiment config file."""
    directory = Path(config_directory)
    raw = {
        name: _load_json_object(directory / filename)
        for name, filename in CONFIG_FILES.items()
    }

    config = ExperimentConfig(
        agents=_parse_agents(raw["agents"]),
        steps=_parse_steps(raw["steps"]),
        target_events=_parse_target_events(raw["events"]),
        retrieval_configs=_parse_retrieval_configs(raw["retrieval"]),
        settings=_parse_settings(raw["settings"]),
        ground_truth=raw["ground_truth"],
    )
    validate_experiment_config(config)
    return config


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigLoadError(f"missing configuration file: {path}")
    try:
        with path.open(encoding="utf-8") as config_file:
            value = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ConfigLoadError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ConfigLoadError(f"could not read configuration file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{path.name} must contain a JSON object")
    return value


def _require_keys(data: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - data.keys())
    if missing:
        raise ConfigValidationError(f"{context} is missing required keys: {', '.join(missing)}")


def _require_list(data: Mapping[str, Any], key: str, context: str) -> list[Any]:
    _require_keys(data, {key}, context)
    value = data[key]
    if not isinstance(value, list):
        raise ConfigValidationError(f"{context}.{key} must be a list")
    return value


def _require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{context} must be an object")
    return value


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigValidationError(f"{context} must be a list of strings")
    return tuple(value)


def _construct(model_name: str, factory: Any) -> Any:
    try:
        return factory()
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"invalid {model_name}: {exc}") from exc


def _parse_agents(data: Mapping[str, Any]) -> tuple[AgentProfile, ...]:
    records = _require_list(data, "agents", "agents.json")
    agents: list[AgentProfile] = []
    required = {
        "agent_id",
        "display_name",
        "role",
        "archetype",
        "persona_summary",
        "traits",
        "goals",
        "communication_style",
        "motivations",
        "seed_memories",
    }
    for index, raw_record in enumerate(records):
        context = f"agents.json agents[{index}]"
        record = _require_object(raw_record, context)
        _require_keys(record, required, context)
        seed_records = _require_list(record, "seed_memories", context)
        seed_memories: list[SeedMemoryConfig] = []
        for seed_index, raw_seed in enumerate(seed_records):
            seed_context = f"{context}.seed_memories[{seed_index}]"
            seed = _require_object(raw_seed, seed_context)
            _require_keys(seed, {"text", "category", "importance"}, seed_context)
            seed_memories.append(
                _construct(
                    seed_context,
                    lambda seed=seed: SeedMemoryConfig(
                        text=seed["text"],
                        category=seed["category"],
                        importance=seed["importance"],
                    ),
                )
            )
        persona = _construct(
            f"{context} persona",
            lambda record=record: PersonaProfile(
                summary=record["persona_summary"],
                traits=_string_tuple(record["traits"], f"{context}.traits"),
                goals=_string_tuple(record["goals"], f"{context}.goals"),
                communication_style=record["communication_style"],
                motivations=_string_tuple(record["motivations"], f"{context}.motivations"),
            ),
        )
        agents.append(
            _construct(
                context,
                lambda record=record, persona=persona: AgentProfile(
                    agent_id=record["agent_id"],
                    display_name=record["display_name"],
                    role=record["role"],
                    archetype=record["archetype"],
                    persona=persona,
                    seed_memories=tuple(seed_memories),
                ),
            )
        )
    return tuple(agents)


def _parse_steps(data: Mapping[str, Any]) -> tuple[SimulationStep, ...]:
    records = _require_list(data, "steps", "simulation_steps.json")
    steps: list[SimulationStep] = []
    common_keys = {
        "step_id",
        "day",
        "period",
        "step_type",
        "title",
        "description",
        "participants",
        "memory_generation_enabled",
        "evaluation_enabled",
    }
    for index, raw_record in enumerate(records):
        context = f"simulation_steps.json steps[{index}]"
        record = _require_object(raw_record, context)
        _require_keys(record, common_keys, context)
        common = {
            "step_id": record["step_id"],
            "day": record["day"],
            "period": record["period"],
            "step_type": record["step_type"],
            "title": record["title"],
            "description": record["description"],
            "participants": _string_tuple(record["participants"], f"{context}.participants"),
            "memory_generation_enabled": record["memory_generation_enabled"],
            "evaluation_enabled": record["evaluation_enabled"],
        }
        if record["step_type"] == "natural_interaction_block":
            natural_keys = {
                "turns_per_agent",
                "total_turns",
                "teacher_opens_block",
                "turn_policy",
            }
            _require_keys(record, natural_keys, context)
            step = _construct(
                context,
                lambda: NaturalInteractionBlockStep(
                    **common,
                    turns_per_agent=record["turns_per_agent"],
                    total_turns=record["total_turns"],
                    teacher_opens_block=record["teacher_opens_block"],
                    turn_policy=record["turn_policy"],
                ),
            )
        elif record["step_type"] == "target_event":
            _require_keys(record, {"event_id"}, context)
            step = _construct(
                context,
                lambda: TargetEventStep(**common, event_id=record["event_id"]),
            )
        else:
            raise ConfigValidationError(
                f"{context} has unsupported step_type {record['step_type']!r}"
            )
        steps.append(step)
    return tuple(steps)


def _parse_target_events(data: Mapping[str, Any]) -> tuple[TargetEvent, ...]:
    records = _require_list(data, "events", "target_events.json")
    events: list[TargetEvent] = []
    required = {
        "event_id",
        "title",
        "description",
        "expected_memory_themes",
        "evaluation_role",
    }
    for index, raw_record in enumerate(records):
        context = f"target_events.json events[{index}]"
        record = _require_object(raw_record, context)
        _require_keys(record, required, context)
        events.append(
            _construct(
                context,
                lambda record=record: TargetEvent(
                    event_id=record["event_id"],
                    title=record["title"],
                    description=record["description"],
                    expected_memory_themes=_string_tuple(
                        record["expected_memory_themes"],
                        f"{context}.expected_memory_themes",
                    ),
                    evaluation_role=record["evaluation_role"],
                ),
            )
        )
    return tuple(events)


def _parse_retrieval_configs(data: Mapping[str, Any]) -> tuple[RetrievalConfig, ...]:
    records = _require_list(data, "configurations", "retrieval_configs.json")
    configurations: list[RetrievalConfig] = []
    for index, raw_record in enumerate(records):
        context = f"retrieval_configs.json configurations[{index}]"
        record = _require_object(raw_record, context)
        _require_keys(record, {"config_id", "weights"}, context)
        weights = _require_object(record["weights"], f"{context}.weights")
        _require_keys(weights, set(RETRIEVAL_COMPONENTS), f"{context}.weights")
        if set(weights) != set(RETRIEVAL_COMPONENTS):
            raise ConfigValidationError(
                f"{context}.weights may contain only recency, importance, relevance"
            )
        parsed_weights = _construct(
            f"{context}.weights",
            lambda weights=weights: RetrievalWeights(
                recency=weights["recency"],
                importance=weights["importance"],
                relevance=weights["relevance"],
            ),
        )
        configurations.append(
            _construct(
                context,
                lambda record=record, parsed_weights=parsed_weights: RetrievalConfig(
                    config_id=record["config_id"], weights=parsed_weights
                ),
            )
        )
    return tuple(configurations)


def _parse_settings(data: Mapping[str, Any]) -> ExperimentSettings:
    required = {
        "simulation_days",
        "natural_blocks_total",
        "target_events_total",
        "turns_per_agent_per_natural_block",
        "total_turns_per_natural_block",
        "teacher_opens_natural_block",
        "top_k",
        "repetitions",
        "primary_mra_step_type",
        "primary_bcr_step_type",
        "natural_block_primary_evaluation",
        "target_event_primary_evaluation",
        "dummy_mode_default",
        "allow_real_llm_default",
        "allow_full_experiment_default",
        "response_limits",
    }
    _require_keys(data, required, "settings.json")
    response_data = _require_object(data["response_limits"], "settings.response_limits")
    response_keys = {
        "natural_interaction_sentences",
        "target_event_max_paragraphs",
        "reflection_sentences",
    }
    _require_keys(response_data, response_keys, "settings.response_limits")
    response_limits = _construct(
        "settings.response_limits",
        lambda: ResponseLimits(
            natural_interaction_sentences=response_data["natural_interaction_sentences"],
            target_event_max_paragraphs=response_data["target_event_max_paragraphs"],
            reflection_sentences=response_data["reflection_sentences"],
        ),
    )

    model_data = data.get("model_settings", {})
    model_data = _require_object(model_data, "settings.model_settings")
    model_settings = _construct(
        "settings.model_settings",
        lambda: ModelSettings(
            model_name=model_data.get("model_name"),
            temperature=model_data.get("temperature"),
            max_tokens=model_data.get("max_tokens"),
        ),
    )
    return _construct(
        "settings.json",
        lambda: ExperimentSettings(
            simulation_days=data["simulation_days"],
            natural_blocks_total=data["natural_blocks_total"],
            target_events_total=data["target_events_total"],
            turns_per_agent_per_natural_block=data[
                "turns_per_agent_per_natural_block"
            ],
            total_turns_per_natural_block=data["total_turns_per_natural_block"],
            teacher_opens_natural_block=data["teacher_opens_natural_block"],
            top_k=data["top_k"],
            repetitions=data["repetitions"],
            primary_mra_step_type=data["primary_mra_step_type"],
            primary_bcr_step_type=data["primary_bcr_step_type"],
            natural_block_primary_evaluation=data["natural_block_primary_evaluation"],
            target_event_primary_evaluation=data["target_event_primary_evaluation"],
            dummy_mode_default=data["dummy_mode_default"],
            allow_real_llm_default=data["allow_real_llm_default"],
            allow_full_experiment_default=data["allow_full_experiment_default"],
            response_limits=response_limits,
            model_settings=model_settings,
        ),
    )
