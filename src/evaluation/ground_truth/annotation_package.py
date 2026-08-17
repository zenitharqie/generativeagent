"""Deterministic, condition-blind manual annotation package generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Mapping

from src.config_loader import DEFAULT_CONFIG_DIRECTORY, ExperimentConfig, load_experiment_config
from src.config_loader.loader import CONFIG_FILES
from src.domain.agents import FIXED_AGENT_IDS, AgentProfile

from .errors import GroundTruthInputError
from .schemas import (
    EXPECTED_UNIT_COUNT,
    GROUND_TRUTH_FINALIZATION_VERSION,
    GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
    canonical_json_bytes,
    content_hash,
    require_exact_keys,
    require_hash,
)


NECESSITY_TEST = {
    "decision": "binary_relevance",
    "relevant_value": 1,
    "not_relevant_value": 0,
    "criteria": [
        "directly related to the communication style defined in the agent Base Prompt",
        "directly related to the agent's primary focus or topic of attention",
        "directly related to the agent's primary motivation or goal",
    ],
    "context_requirement": "the category must also be relevant to the target-event context",
}

ANNOTATION_INSTRUCTIONS = (
    "Propose concise memory-category phrases that may be necessary for this agent when "
    "responding to this target event. Apply the Necessity Test to every proposed category. "
    "Label each category with integer relevance 1 or 0. Work independently. Do not infer "
    "experimental results, retrieval behavior, or another annotator's decisions. Return only "
    "the specified structured JSON; a concise evidence-based reason is optional and must not "
    "contain hidden chain-of-thought."
)

ANNOTATION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "annotation_unit_id",
        "agent_id",
        "target_event_id",
        "input_sha256",
        "categories",
    ],
    "properties": {
        "annotation_unit_id": {"type": "string"},
        "agent_id": {"type": "string"},
        "target_event_id": {"type": "string"},
        "input_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "categories": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "relevant"],
                "properties": {
                    "category": {"type": "string", "minLength": 1, "maxLength": 160},
                    "relevant": {"type": "integer", "enum": [0, 1]},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 500},
                },
            },
        },
    },
}


def _file_hash(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise GroundTruthInputError("canonical configuration input is missing or unsafe")
        return sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GroundTruthInputError("could not read canonical configuration input") from exc


def _reject_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if os.path.lexists(current) and current.is_symlink():
            raise GroundTruthInputError("configuration path components must not be symlinks")


def _persona(agent: AgentProfile) -> dict[str, object]:
    return {
        "agent_id": agent.agent_id,
        "display_name": agent.display_name,
        "role": agent.role,
        "archetype": agent.archetype,
        "persona_summary": agent.persona_summary,
        "traits": list(agent.traits),
        "goals": list(agent.goals),
        "communication_style": agent.communication_style,
        "motivations": list(agent.motivations),
    }


def _unit_input(
    *, agent: AgentProfile, event_id: str, event_description: str
) -> dict[str, object]:
    unit_id = f"gt_{event_id}_{agent.agent_id}"
    return {
        "annotation_unit_id": unit_id,
        "agent_id": agent.agent_id,
        "target_event_id": event_id,
        "canonical_persona": _persona(agent),
        "canonical_event_description": event_description,
        "necessity_test": NECESSITY_TEST,
        "annotation_instructions": ANNOTATION_INSTRUCTIONS,
        "expected_structured_output_schema": ANNOTATION_OUTPUT_SCHEMA,
    }


@dataclass(frozen=True)
class AnnotationPackage:
    value: Mapping[str, Any]
    payload: bytes
    package_sha256: str


def build_annotation_package(
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> AnnotationPackage:
    """Build the same 20 methodological inputs for all three manual annotators."""
    config_root = Path(config_directory)
    if ".." in config_root.parts or config_root.is_symlink():
        raise GroundTruthInputError("configuration directory is unsafe")
    _reject_symlink_components(config_root)
    config = load_experiment_config(config_root)
    agents = {agent.agent_id: agent for agent in config.agents}
    if set(agents) != FIXED_AGENT_IDS:
        raise GroundTruthInputError("canonical agent inventory drifted")
    units: list[dict[str, object]] = []
    for event in config.target_events:
        for agent_id in tuple(agent.agent_id for agent in config.agents):
            unit_input = _unit_input(
                agent=agents[agent_id],
                event_id=event.event_id,
                event_description=event.description,
            )
            units.append({**unit_input, "input_sha256": content_hash(unit_input)})
    if len(units) != EXPECTED_UNIT_COUNT:
        raise GroundTruthInputError("annotation package must contain exactly 20 units")
    content: dict[str, object] = {
        "schema_version": GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION,
        "methodology_id": GROUND_TRUTH_FINALIZATION_VERSION,
        "unit_count": EXPECTED_UNIT_COUNT,
        "canonical_source_hashes": {
            "agents.json": _file_hash(config_root / CONFIG_FILES["agents"]),
            "target_events.json": _file_hash(config_root / CONFIG_FILES["events"]),
        },
        "units": units,
    }
    package_sha256 = content_hash(content)
    value = {**content, "annotation_package_sha256": package_sha256}
    return AnnotationPackage(
        value=value,
        payload=canonical_json_bytes(value, pretty=True),
        package_sha256=package_sha256,
    )


def validate_annotation_package(
    value: object,
    *,
    config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
) -> Mapping[str, Any]:
    """Strictly validate a package and bind it to current persona/event bytes."""
    if not isinstance(value, Mapping):
        raise GroundTruthInputError("annotation package must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "methodology_id",
            "unit_count",
            "canonical_source_hashes",
            "units",
            "annotation_package_sha256",
        },
        "annotation package",
    )
    if (
        value["schema_version"] != GROUND_TRUTH_WORKFLOW_SCHEMA_VERSION
        or value["methodology_id"] != GROUND_TRUTH_FINALIZATION_VERSION
        or value["unit_count"] != EXPECTED_UNIT_COUNT
    ):
        raise GroundTruthInputError("annotation package contract is unsupported")
    supplied_hash = require_hash(value["annotation_package_sha256"], "package hash")
    content = dict(value)
    del content["annotation_package_sha256"]
    if content_hash(content) != supplied_hash:
        raise GroundTruthInputError("annotation package hash does not match its content")
    canonical = build_annotation_package(config_directory)
    if canonical.package_sha256 != supplied_hash or canonical.value != value:
        raise GroundTruthInputError("annotation package is stale or non-canonical")
    return value


__all__ = [
    "ANNOTATION_INSTRUCTIONS",
    "ANNOTATION_OUTPUT_SCHEMA",
    "AnnotationPackage",
    "NECESSITY_TEST",
    "build_annotation_package",
    "validate_annotation_package",
]
