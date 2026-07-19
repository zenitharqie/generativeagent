import json
from pathlib import Path

import pytest


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "config"
CONFIG_FILES = (
    "agents.json",
    "simulation_steps.json",
    "target_events.json",
    "retrieval_configs.json",
    "settings.json",
    "ground_truth.json",
)


@pytest.mark.parametrize("filename", CONFIG_FILES)
def test_config_file_exists(filename: str) -> None:
    assert (CONFIG_DIRECTORY / filename).is_file()


@pytest.mark.parametrize("filename", CONFIG_FILES)
def test_config_file_contains_parseable_json(filename: str) -> None:
    with (CONFIG_DIRECTORY / filename).open(encoding="utf-8") as config_file:
        parsed = json.load(config_file)

    assert isinstance(parsed, dict)
