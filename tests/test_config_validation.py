import json
from pathlib import Path
import shutil
from typing import Any, Callable

import pytest

from src.config_loader import ConfigValidationError, load_experiment_config


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def mutable_config(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    shutil.copytree(CONFIG_DIRECTORY, destination)
    return destination


def mutate_json(
    config_directory: Path,
    filename: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    path = config_directory / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_config_passes() -> None:
    load_experiment_config(CONFIG_DIRECTORY)


def test_invalid_agent_id_fails(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "agents.json",
        lambda data: data["agents"][0].update(agent_id="substitute_teacher"),
    )

    with pytest.raises(ConfigValidationError, match="unsupported agent_id"):
        load_experiment_config(mutable_config)


def test_invalid_day_1_structure_fails(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "simulation_steps.json",
        lambda data: data["steps"][1].update(day=2),
    )

    with pytest.raises(ConfigValidationError, match="Day 1"):
        load_experiment_config(mutable_config)


def test_invalid_day_2_to_6_structure_fails(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "simulation_steps.json",
        lambda data: data["steps"][3].update(day=3),
    )

    with pytest.raises(ConfigValidationError, match="Day 2"):
        load_experiment_config(mutable_config)


def test_invalid_top_k_fails(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "settings.json",
        lambda data: data.update(top_k=4),
    )

    with pytest.raises(ConfigValidationError, match=r"settings\.top_k must be 5"):
        load_experiment_config(mutable_config)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_value"),
    [
        ("simulation_days", 5, 6),
        ("natural_blocks_total", 6, 7),
        ("target_events_total", 4, 5),
        ("turns_per_agent_per_natural_block", 2, 3),
        ("total_turns_per_natural_block", 11, 12),
        ("repetitions", 2, 3),
    ],
)
def test_invalid_fixed_setting_fails(
    mutable_config: Path,
    field_name: str,
    invalid_value: int,
    expected_value: int,
) -> None:
    mutate_json(
        mutable_config,
        "settings.json",
        lambda data: data.update({field_name: invalid_value}),
    )

    with pytest.raises(
        ConfigValidationError,
        match=rf"settings\.{field_name} must be {expected_value}",
    ):
        load_experiment_config(mutable_config)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("turns_per_agent", 2),
        ("total_turns", 11),
        ("teacher_opens_block", False),
    ],
)
def test_invalid_natural_block_control_fails(
    mutable_config: Path, field_name: str, invalid_value: int | bool
) -> None:
    mutate_json(
        mutable_config,
        "simulation_steps.json",
        lambda data: data["steps"][0].update({field_name: invalid_value}),
    )

    with pytest.raises(ConfigValidationError, match=field_name):
        load_experiment_config(mutable_config)


@pytest.mark.parametrize(
    "weights",
    [
        {"recency": -0.1, "importance": 0.5, "relevance": 0.6},
        {"recency": 0.2, "importance": 0.2, "relevance": 0.2},
    ],
)
def test_invalid_retrieval_weights_fail(
    mutable_config: Path, weights: dict[str, float]
) -> None:
    mutate_json(
        mutable_config,
        "retrieval_configs.json",
        lambda data: data["configurations"][0].update(weights=weights),
    )

    with pytest.raises(ConfigValidationError, match="retrieval.*weights"):
        load_experiment_config(mutable_config)


def test_invalid_target_event_reference_fails(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "simulation_steps.json",
        lambda data: data["steps"][3].update(event_id="event_missing"),
    )

    with pytest.raises(ConfigValidationError, match="references missing event_id"):
        load_experiment_config(mutable_config)


def test_target_event_step_must_match_event_definition(mutable_config: Path) -> None:
    mutate_json(
        mutable_config,
        "simulation_steps.json",
        lambda data: data["steps"][3].update(description="Changed event text."),
    )

    with pytest.raises(ConfigValidationError, match="must match its event definition"):
        load_experiment_config(mutable_config)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("dummy_mode_default", False),
        ("allow_real_llm_default", True),
        ("allow_full_experiment_default", True),
    ],
)
def test_unsafe_execution_default_fails(
    mutable_config: Path, field_name: str, invalid_value: bool
) -> None:
    mutate_json(
        mutable_config,
        "settings.json",
        lambda data: data.update({field_name: invalid_value}),
    )

    with pytest.raises(ConfigValidationError, match=field_name):
        load_experiment_config(mutable_config)


def test_non_target_ground_truth_reference_fails_by_default(
    mutable_config: Path,
) -> None:
    mutate_json(
        mutable_config,
        "ground_truth.json",
        lambda data: data["targets"][0].update(event_id="day_01_natural_01"),
    )

    with pytest.raises(ConfigValidationError, match="references non-target event"):
        load_experiment_config(mutable_config)


def test_explicit_optional_ground_truth_reference_is_allowed(
    mutable_config: Path,
) -> None:
    def add_optional_target(data: dict[str, Any]) -> None:
        data["targets"].append(
            {
                "event_id": "optional_non_target_reference",
                "agent_id": "teacher",
                "expected_categories": ["teacher_guidance"],
                "optional": True,
            }
        )

    mutate_json(mutable_config, "ground_truth.json", add_optional_target)

    load_experiment_config(mutable_config)
