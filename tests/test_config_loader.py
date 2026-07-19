import json
from pathlib import Path
import shutil

import pytest

from src.config_loader import ConfigLoadError, ConfigValidationError, load_experiment_config
from src.domain import (
    AgentProfile,
    ExperimentSettings,
    NaturalInteractionBlockStep,
    RetrievalConfig,
    TargetEvent,
    TargetEventStep,
)


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "config"


def copied_config(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    shutil.copytree(CONFIG_DIRECTORY, destination)
    return destination


def test_load_valid_config_returns_typed_aggregate() -> None:
    config = load_experiment_config(CONFIG_DIRECTORY)

    assert len(config.agents) == 4
    assert all(isinstance(agent, AgentProfile) for agent in config.agents)
    assert sum(isinstance(step, NaturalInteractionBlockStep) for step in config.steps) == 7
    assert sum(isinstance(step, TargetEventStep) for step in config.steps) == 5
    assert all(isinstance(event, TargetEvent) for event in config.target_events)
    assert all(isinstance(item, RetrievalConfig) for item in config.retrieval_configs)
    assert isinstance(config.settings, ExperimentSettings)


def test_missing_config_file_has_clear_error(tmp_path: Path) -> None:
    config_directory = copied_config(tmp_path)
    (config_directory / "settings.json").unlink()

    with pytest.raises(ConfigLoadError, match="missing configuration file.*settings.json"):
        load_experiment_config(config_directory)


def test_invalid_json_has_file_and_location_error(tmp_path: Path) -> None:
    config_directory = copied_config(tmp_path)
    (config_directory / "settings.json").write_text("{invalid", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="invalid JSON in .*settings.json: line 1"):
        load_experiment_config(config_directory)


def test_missing_required_key_has_contextual_error(tmp_path: Path) -> None:
    config_directory = copied_config(tmp_path)
    settings_path = config_directory / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    del settings["top_k"]
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="settings.json.*top_k"):
        load_experiment_config(config_directory)
