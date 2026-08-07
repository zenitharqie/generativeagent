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


DEFAULT_PILOT_EXPERIMENT_ID = "phase08_dummy_pilot"


def _parse_pilot_days(value: str) -> tuple[int, ...]:
    """Parse the only partial scope authorized for the Phase 08 pilot."""
    if value != "1-2":
        raise argparse.ArgumentTypeError(
            "Phase 08 run-pilot supports only the exact day range '1-2'"
        )
    return (1, 2)


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
    pilot_parser = subparsers.add_parser(
        "run-pilot",
        help="Run the deterministic offline Day 1-2 Phase 08 pilot.",
    )
    pilot_parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use deterministic offline LLM and embedding providers.",
    )
    pilot_parser.add_argument(
        "--config",
        dest="retrieval_config_id",
        default="balanced",
        help="One validated retrieval configuration ID (default: balanced).",
    )
    pilot_parser.add_argument(
        "--repetition",
        type=int,
        default=1,
        help="Human/research repetition number, 1 through 3 (default: 1).",
    )
    pilot_parser.add_argument(
        "--days",
        type=_parse_pilot_days,
        default=(1, 2),
        metavar="1-2",
        help="Approved pilot day range; must be 1-2.",
    )
    pilot_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Fresh output root for the isolated pilot run.",
    )
    pilot_parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
        help="Configuration directory (default: repository config directory).",
    )
    pilot_parser.add_argument(
        "--experiment-id",
        default=DEFAULT_PILOT_EXPERIMENT_ID,
        help=f"Output experiment ID (default: {DEFAULT_PILOT_EXPERIMENT_ID}).",
    )
    pilot_parser.add_argument(
        "--run-id",
        default=None,
        help="Fresh run ID; defaults deterministically from config and repetition.",
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

    if args.command == "run-pilot":
        if not args.dummy:
            print(
                "Pilot execution failed: Phase 08 run-pilot requires --dummy; "
                "real provider integration is out of scope.",
                file=sys.stderr,
            )
            return 1
        run_id = args.run_id or (
            f"{args.retrieval_config_id}_rep{args.repetition:02d}_pilot_day01_02"
        )
        try:
            from src.llm import DummyEmbeddingProvider
            from src.simulation import (
                RunController,
                build_deterministic_dummy_llm_client,
            )

            result = RunController(
                llm_client=build_deterministic_dummy_llm_client(),
                embedding_provider=DummyEmbeddingProvider(),
                config_directory=args.config_dir,
            ).run(
                output_root=args.output_root,
                experiment_id=args.experiment_id,
                run_id=run_id,
                retrieval_config_id=args.retrieval_config_id,
                repetition_index=args.repetition,
                days=args.days,
            )
        except Exception as exc:
            print(
                f"Pilot execution failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        summary = result.run_summary
        print("Pilot scope executed successfully.")
        print(f"Persisted run status: {summary.status}")
        print(f"Completed natural blocks: {summary.completed_natural_blocks}")
        print(f"Completed target events: {summary.completed_target_events}")
        print(f"Completed main turns: {summary.completed_main_turns}")
        print(f"Run directory: {result.run_paths.run_dir}")
        return 0

    print("Classroom generative-agents repository scaffold is ready.")
    print("Run 'python main.py --help' to view available commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
