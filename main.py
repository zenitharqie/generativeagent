"""Command-line entry point for the classroom experiment repository."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.config_loader import (
    DEFAULT_CONFIG_DIRECTORY,
    ConfigLoadError,
    ConfigValidationError,
    ExperimentConfig,
    load_experiment_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for implemented repository phases."""
    parser = argparse.ArgumentParser(
        description="Controlled classroom generative-agents experiment."
    )
    subparsers = parser.add_subparsers(dest="command")
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Load and validate all experiment configuration files.",
    )
    validate_parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
        help="Configuration directory (default: repository config directory).",
    )
    return parser


def format_validation_summary(config: ExperimentConfig) -> str:
    """Format validated architecture constants for supervisor review."""
    natural_blocks = sum(
        step.step_type == "natural_interaction_block" for step in config.steps
    )
    settings = config.settings
    primary_evaluation = settings.primary_mra_step_type
    natural_evaluation = (
        "enabled" if settings.natural_block_primary_evaluation else "disabled"
    )
    target_evaluation = (
        "enabled" if settings.target_event_primary_evaluation else "disabled"
    )
    return "\n".join(
        (
            "Configuration valid.",
            f"Simulation days: {settings.simulation_days}",
            f"Agents: {len(config.agents)}",
            f"Natural blocks: {natural_blocks}",
            f"Target events: {len(config.target_events)}",
            f"Retrieval configs: {len(config.retrieval_configs)}",
            f"Top-K: {settings.top_k}",
            f"Repetitions: {settings.repetitions}",
            f"Primary evaluation: {primary_evaluation} only",
            f"Natural block primary evaluation: {natural_evaluation}",
            f"Target event primary evaluation: {target_evaluation}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        try:
            config = load_experiment_config(args.config_dir)
        except (ConfigLoadError, ConfigValidationError) as exc:
            print(f"Configuration validation failed: {exc}", file=sys.stderr)
            return 1
        print(format_validation_summary(config))
        return 0

    print("Classroom generative-agents repository scaffold is ready.")
    print("Run 'python main.py --help' to view available commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
