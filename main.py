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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the placeholder CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        try:
            config = load_experiment_config(args.config_dir)
        except (ConfigLoadError, ConfigValidationError) as exc:
            print(f"Configuration validation failed: {exc}", file=sys.stderr)
            return 1
        natural_blocks = sum(
            step.step_type == "natural_interaction_block" for step in config.steps
        )
        print(
            "Configuration valid: "
            f"{len(config.agents)} agents, "
            f"{natural_blocks} natural blocks, "
            f"{len(config.target_events)} target events, "
            f"{len(config.retrieval_configs)} retrieval configurations."
        )
        return 0

    print("Classroom generative-agents repository scaffold is ready.")
    print("Run 'python main.py --help' to view available commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
