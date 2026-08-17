"""Command-line entry point for the classroom experiment repository."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
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
    """Parse the only partial scope authorized for the guarded pilot."""
    if value != "1-2":
        raise argparse.ArgumentTypeError(
            "run-pilot supports only the exact day range '1-2'"
        )
    return (1, 2)


def _parse_explicit_true(value: str) -> bool:
    """Require the deliberate literal used by paid-call authorization flags."""
    if value != "true":
        raise argparse.ArgumentTypeError("value must be the exact literal 'true'")
    return True


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
        help="Plan or run the safeguarded, excluded Day 1-2 pilot.",
    )
    pilot_modes = pilot_parser.add_mutually_exclusive_group()
    pilot_modes.add_argument(
        "--dry-run",
        type=_parse_explicit_true,
        metavar="true",
        help="Validate and print the fixed plan without creating paths or clients.",
    )
    pilot_modes.add_argument(
        "--dummy",
        action="store_true",
        help="Use deterministic offline LLM and embedding providers.",
    )
    pilot_modes.add_argument(
        "--real",
        action="store_true",
        help="Request the fixed paid OpenAI pilot; also requires explicit authorization.",
    )
    pilot_parser.add_argument(
        "--allow-real-llm",
        type=_parse_explicit_true,
        metavar="true",
        help="Second deliberate authorization signal required with --real.",
    )
    pilot_parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Mandatory positive provider-reported token ceiling for --real.",
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
        default=None,
        help="Fresh output root; required for dummy or real execution.",
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
    pilot_parser.add_argument(
        "--on-conflict",
        choices=("fail",),
        default="fail",
        help="Conflict policy; Phase 11 permits only fail (no reuse or resume).",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one persisted run without rerunning simulation components.",
    )
    evaluate_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed source run directory containing persisted research artifacts.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh, separate directory for atomic evaluation publication.",
    )
    evaluate_parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Permit an excluded pilot as explicitly non-primary diagnostic input.",
    )
    smoke_parser = subparsers.add_parser(
        "llm-smoke-test",
        help="Run one minimal non-research real-provider structured request.",
    )
    smoke_parser.add_argument(
        "--real",
        action="store_true",
        required=True,
        help="Explicitly authorize the single real OpenAI smoke request.",
    )
    review_parser = subparsers.add_parser(
        "review-pilot",
        help="Publish one explicit offline pilot acceptance or rejection.",
    )
    review_parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Published, immutable Phase 11 pilot evidence directory.",
    )
    review_parser.add_argument(
        "--decision",
        choices=("accepted", "rejected"),
        required=True,
        help="Explicit manual review decision.",
    )
    review_parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
        help="Configuration directory used to verify an acceptance freeze.",
    )
    replay_parser = subparsers.add_parser(
        "export-visualizer-data",
        help="Export a deterministic read-only replay JSON from persisted artifacts.",
    )
    replay_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Canonical completed, excluded, failed, or interrupted source run directory.",
    )
    replay_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh .json output path outside the source and overlay directories.",
    )
    replay_parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=None,
        help="Optional explicit Phase 09 persisted evaluation overlay directory.",
    )
    replay_parser.add_argument(
        "--judgment-dir",
        type=Path,
        default=None,
        help="Optional explicit Phase 10 persisted judgment overlay directory.",
    )
    replay_parser.add_argument(
        "--pilot-evidence-dir",
        type=Path,
        default=None,
        help="Optional explicit Phase 11 pilot evidence overlay directory.",
    )
    replay_parser.add_argument(
        "--pilot-review-dir",
        type=Path,
        default=None,
        help="Optional explicit Phase 11 manual review/freeze overlay directory.",
    )
    annotation_package_parser = subparsers.add_parser(
        "generate-ground-truth-annotation-package",
        help="Publish the deterministic 20-unit manual annotation package.",
    )
    annotation_package_parser.add_argument(
        "--output", type=Path, required=True, help="Fresh annotation-package JSON path."
    )
    annotation_package_parser.add_argument(
        "--config-dir", type=Path, default=DEFAULT_CONFIG_DIRECTORY
    )
    embedding_input_parser = subparsers.add_parser(
        "prepare-ground-truth-embedding-input",
        help="Validate three manual submissions and publish candidate embedding input.",
    )
    embedding_input_parser.add_argument("--package", type=Path, required=True)
    embedding_input_parser.add_argument("--annotations-dir", type=Path, required=True)
    embedding_input_parser.add_argument("--output", type=Path, required=True)
    embedding_input_parser.add_argument(
        "--config-dir", type=Path, default=DEFAULT_CONFIG_DIRECTORY
    )
    finalization_parser = subparsers.add_parser(
        "finalize-ground-truth",
        help="Finalize Ground Truth from manual annotations and persisted embeddings.",
    )
    finalization_parser.add_argument("--package", type=Path, required=True)
    finalization_parser.add_argument("--annotations-dir", type=Path, required=True)
    finalization_parser.add_argument("--embedding-evidence", type=Path, required=True)
    finalization_parser.add_argument("--output-dir", type=Path, required=True)
    finalization_parser.add_argument(
        "--config-dir", type=Path, default=DEFAULT_CONFIG_DIRECTORY
    )
    judge_parser = subparsers.add_parser(
        "judge-bcr",
        help="Judge one Phase 09 BCR export with the independently authorized evaluator.",
    )
    judge_parser.add_argument("--evaluation-dir", type=Path, required=True)
    judge_parser.add_argument("--output-dir", type=Path, required=True)
    judge_parser.add_argument("--real", action="store_true", required=True)
    judge_parser.add_argument(
        "--allow-real-evaluator",
        type=_parse_explicit_true,
        metavar="true",
        required=True,
    )
    judge_parser.add_argument("--confirm-bcr-judgment", required=True)
    embedding_evidence_parser = subparsers.add_parser(
        "generate-ground-truth-embedding-evidence",
        help="Acquire only approved text-embedding-3-small vectors for Ground Truth consolidation.",
    )
    embedding_modes = embedding_evidence_parser.add_mutually_exclusive_group(required=True)
    embedding_modes.add_argument("--real", action="store_true")
    embedding_modes.add_argument("--dummy", action="store_true")
    embedding_evidence_parser.add_argument("--input", type=Path, required=True)
    embedding_evidence_parser.add_argument("--output", type=Path, required=True)
    embedding_evidence_parser.add_argument("--allow-real-embedding", type=_parse_explicit_true, metavar="true")
    embedding_evidence_parser.add_argument("--confirm-ground-truth-embedding", default=None)
    experiment_parser = subparsers.add_parser(
        "run-full-experiment",
        help="Plan or execute the fail-fast Phase 13 4x3 experiment.",
    )
    experiment_modes = experiment_parser.add_mutually_exclusive_group()
    experiment_modes.add_argument(
        "--dry-run",
        type=_parse_explicit_true,
        metavar="true",
        help="Print the deterministic 12-run plan without writing or executing.",
    )
    experiment_modes.add_argument(
        "--dummy",
        action="store_true",
        help="Execute all 12 source/evaluation chains with offline providers.",
    )
    experiment_modes.add_argument(
        "--real",
        action="store_true",
        help="Request explicitly authorized paid Phase 13 execution.",
    )
    experiment_parser.add_argument(
        "--allow-real-llm",
        type=_parse_explicit_true,
        metavar="true",
        help="Second explicit real-provider authorization signal.",
    )
    experiment_parser.add_argument(
        "--confirm-full-experiment",
        default=None,
        help="Exact confirmation token required for dummy or real execution.",
    )
    experiment_parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Positive global simulator/embedding token ceiling required for real mode.",
    )
    experiment_parser.add_argument(
        "--accepted-pilot-review-dir",
        type=Path,
        default=None,
        help="Current manually accepted Phase 11 review/freeze directory for real mode.",
    )
    experiment_parser.add_argument(
        "--ground-truth-finalization-dir",
        type=Path,
        default=None,
        help="Explicit evidence directory authenticating the installed final Ground Truth.",
    )
    experiment_parser.add_argument(
        "--experiment-id",
        default="phase13_full_experiment",
        help="Fresh safe experiment identity.",
    )
    experiment_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh output root required for dummy or real execution.",
    )
    experiment_parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIRECTORY,
        help="Configuration directory (default: repository config directory).",
    )
    experiment_parser.add_argument(
        "--on-conflict",
        choices=("fail",),
        default="fail",
        help="Only fail is supported; Phase 13 never resumes or overwrites.",
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
        mode = (
            "dry-run"
            if args.dry_run
            else "real"
            if args.real
            else "dummy"
            if args.dummy
            else None
        )
        if mode is None:
            print(
                "Pilot execution failed: run-pilot requires --dummy, --real, "
                "or --dry-run true.",
                file=sys.stderr,
            )
            return 1
        if mode == "dry-run":
            if (
                args.output_root is not None
                or args.allow_real_llm is not None
                or args.max_total_tokens is not None
            ):
                print(
                    "Pilot planning failed: dry-run does not accept output or real-call controls.",
                    file=sys.stderr,
                )
                return 1
            try:
                from src.simulation import dry_run_pilot

                plan = dry_run_pilot(
                    retrieval_config_id=args.retrieval_config_id,
                    repetition_index=args.repetition,
                    config_directory=args.config_dir,
                )
            except Exception as exc:
                print(
                    f"Pilot planning failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
            print("Pilot plan valid; no provider client or output path was created.")
            print(f"Planned steps: {len(plan.step_ids)}")
            print(f"Planned main turns: {plan.main_turn_count}")
            print(
                "Planned provider logical calls: "
                f"{plan.call_limits.total_logical_calls}"
            )
            print(
                "Maximum physical attempts: "
                f"{plan.call_limits.total_physical_attempts}"
            )
            return 0
        if args.output_root is None:
            print(
                "Pilot execution failed: --output-root is required for execution.",
                file=sys.stderr,
            )
            return 1
        run_id = args.run_id or (
            f"{args.retrieval_config_id}_rep{args.repetition:02d}_pilot_day01_02"
        )
        try:
            from src.simulation import execute_pilot

            result = execute_pilot(
                mode=mode,
                output_root=args.output_root,
                experiment_id=args.experiment_id,
                run_id=run_id,
                retrieval_config_id=args.retrieval_config_id,
                repetition_index=args.repetition,
                config_directory=args.config_dir,
                real_authorized=args.allow_real_llm is True,
                max_total_tokens=args.max_total_tokens,
            )
        except KeyboardInterrupt:
            print("Pilot execution interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(
                f"Pilot execution failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        summary = result.run_result.run_summary
        print("Pilot scope executed successfully.")
        print(f"Persisted run status: {summary.status}")
        print(f"Completed natural blocks: {summary.completed_natural_blocks}")
        print(f"Completed target events: {summary.completed_target_events}")
        print(f"Completed main turns: {summary.completed_main_turns}")
        print(f"Run directory: {result.run_result.run_paths.run_dir}")
        return 0

    if args.command == "evaluate":
        try:
            from src.evaluation import EvaluationError, evaluate_run

            result = evaluate_run(
                run_dir=args.run_dir,
                output_dir=args.output_dir,
                diagnostic=args.diagnostic,
            )
        except EvaluationError as exc:
            print(
                f"Evaluation failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        print("Evaluation completed successfully.")
        print(f"Evaluation mode: {result.summary.evaluation_mode}")
        print(
            "Primary eligible: "
            + ("yes" if result.summary.primary_eligible else "no")
        )
        print(f"MRA records: {len(result.mra_records)}")
        print(f"BCR export records: {len(result.bcr_records)}")
        print(f"Run MRA: {result.summary.mra_aggregates.run.mean_mra}")
        print(f"Evaluation directory: {result.output_dir}")
        return 0

    if args.command == "llm-smoke-test":
        try:
            from src.llm import run_real_llm_smoke_test

            result = run_real_llm_smoke_test()
        except Exception as exc:
            print(
                f"LLM smoke test failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        for line in result.sanitized_lines():
            print(line)
        return 0

    if args.command == "review-pilot":
        try:
            from src.simulation import review_pilot

            result = review_pilot(
                evidence_dir=args.evidence_dir,
                decision=args.decision,
                config_directory=args.config_dir,
            )
        except Exception as exc:
            print(
                f"Pilot review failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        print("Pilot review published successfully.")
        print(f"Acceptance state: {result.acceptance_state}")
        print(f"Review directory: {result.review_dir}")
        print(f"Acceptance hash: {result.acceptance_hash}")
        if result.freeze_hash is not None:
            print(f"Freeze hash: {result.freeze_hash}")
        return 0

    if args.command == "export-visualizer-data":
        try:
            from src.visualizer import VisualizerError, export_visualizer_data
        except Exception as exc:
            print(
                f"Visualizer export failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1

        try:
            result = export_visualizer_data(
                run_dir=args.run_dir,
                output=args.output,
                evaluation_dir=args.evaluation_dir,
                judgment_dir=args.judgment_dir,
                pilot_evidence_dir=args.pilot_evidence_dir,
                pilot_review_dir=args.pilot_review_dir,
            )
        except KeyboardInterrupt:
            print(
                "Visualizer export interrupted; no source artifact was changed.",
                file=sys.stderr,
            )
            return 130
        except VisualizerError as exc:
            print(
                f"Visualizer export failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:
            print(
                f"Visualizer export failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        print("Visualizer replay exported successfully.")
        print("Output published to the requested fresh path.")
        print(f"SHA-256: {result.output_sha256}")
        print(f"Bytes: {result.output_bytes}")
        print(f"Source status: {result.document.run.source_status}")
        return 0

    if args.command == "generate-ground-truth-annotation-package":
        try:
            from src.evaluation.ground_truth import generate_annotation_package

            package = generate_annotation_package(
                output=args.output,
                config_directory=args.config_dir,
            )
        except KeyboardInterrupt:
            print("Ground Truth package generation interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(
                f"Ground Truth package generation failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        print("Ground Truth annotation package published successfully.")
        print("Annotation units: 20")
        print(f"Package SHA-256: {package.package_sha256}")
        return 0

    if args.command == "prepare-ground-truth-embedding-input":
        try:
            from src.evaluation.ground_truth import prepare_embedding_input

            value = prepare_embedding_input(
                package_path=args.package,
                annotations_directory=args.annotations_dir,
                output=args.output,
                config_directory=args.config_dir,
            )
        except KeyboardInterrupt:
            print("Ground Truth embedding-input preparation interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(
                f"Ground Truth embedding-input preparation failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        print("Ground Truth embedding input published successfully.")
        print(f"Candidate phrases: {value['candidate_count']}")
        print(f"Candidate-set SHA-256: {value['candidate_set_sha256']}")
        return 0

    if args.command == "generate-ground-truth-embedding-evidence":
        try:
            from src.evaluation.ground_truth.embedding_service import generate_embedding_evidence

            evidence = generate_embedding_evidence(
                input_path=args.input,
                output_path=args.output,
                mode="real" if args.real else "dummy",
                confirmation=args.confirm_ground_truth_embedding,
                real_authorized=args.allow_real_embedding is True,
            )
        except KeyboardInterrupt:
            print("Ground Truth embedding acquisition interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Ground Truth embedding acquisition failed: {type(exc).__name__}", file=sys.stderr)
            return 1
        print("Ground Truth embedding evidence published successfully.")
        print(f"Candidate vectors: {len(evidence['embeddings'])}")
        print(f"Acquisition mode: {evidence['acquisition_mode']}")
        return 0

    if args.command == "finalize-ground-truth":
        try:
            from src.evaluation.ground_truth import finalize_ground_truth

            bundle = finalize_ground_truth(
                package_path=args.package,
                annotations_directory=args.annotations_dir,
                embedding_evidence_path=args.embedding_evidence,
                output_directory=args.output_dir,
                config_directory=args.config_dir,
            )
        except KeyboardInterrupt:
            print("Ground Truth finalization interrupted.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(
                f"Ground Truth finalization failed: {type(exc).__name__}",
                file=sys.stderr,
            )
            return 1
        print("Ground Truth finalized successfully from complete methodology evidence.")
        print("Final units: 20")
        print(f"Ground Truth fingerprint: {bundle.fingerprint_sha256}")
        return 0

    if args.command == "judge-bcr":
        if args.confirm_bcr_judgment != "JUDGE_BCR_WITH_CLAUDE":
            print("BCR judgment failed: invalid confirmation token", file=sys.stderr)
            return 1
        try:
            from src.evaluation.judgment_persistence import prepare_judgment_output
            from src.evaluation.judgment_reader import load_bcr_export

            source = load_bcr_export(args.evaluation_dir)
            if source.evaluation_mode == "primary" and len(source.records) != 20:
                raise ValueError("primary BCR judgment requires exactly 20 export rows")
            prepare_judgment_output(
                args.output_dir, source_directory=source.source_directory
            )
            if shutil.disk_usage(Path(args.output_dir).parent).free <= 0:
                raise OSError("insufficient filesystem space for judgment publication")
            from src.llm.anthropic_client import AnthropicProviderAdapter
            from src.llm.settings import EVALUATOR_PROFILE
            from src.evaluation.judgment_service import judge_bcr_export

            provider = AnthropicProviderAdapter.from_environment(
                timeout_seconds=EVALUATOR_PROFILE.request_timeout_seconds
            )
            result = judge_bcr_export(
                evaluation_dir=args.evaluation_dir,
                output_dir=args.output_dir,
                provider=provider,
            )
        except KeyboardInterrupt:
            print("BCR judgment interrupted; no successful bundle was published.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"BCR judgment failed: {type(exc).__name__}", file=sys.stderr)
            return 1
        print("BCR judgment completed successfully.")
        print(f"Judged units: {len(result.records)}")
        print(f"Judgment directory: {result.output_directory}")
        return 0

    if args.command == "run-full-experiment":
        mode = (
            "dry-run"
            if args.dry_run
            else "real"
            if args.real
            else "dummy"
            if args.dummy
            else None
        )
        if mode is None:
            print(
                "Full experiment failed: run-full-experiment requires --dry-run true, "
                "--dummy, or --real.",
                file=sys.stderr,
            )
            return 1
        if mode == "dry-run":
            if any(
                value is not None
                for value in (
                    args.output_root,
                    args.allow_real_llm,
                    args.max_total_tokens,
                    args.accepted_pilot_review_dir,
                    args.ground_truth_finalization_dir,
                    args.confirm_full_experiment,
                )
            ):
                print(
                    "Full experiment planning failed: dry-run accepts no output or "
                    "authorization controls.",
                    file=sys.stderr,
                )
                return 1
            try:
                from src.simulation import dry_run_full_experiment

                plan = dry_run_full_experiment(
                    experiment_id=args.experiment_id,
                    config_directory=args.config_dir,
                )
            except Exception as exc:
                print(
                    f"Full experiment planning failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
            print("Full experiment plan valid; no output, credential, or provider was used.")
            print(f"Planned runs: {len(plan.runs)}")
            for run in plan.runs:
                print(
                    f"{run.run_id}: {run.retrieval_config_id} "
                    f"repetition {run.repetition_index}"
                )
            print(f"Main turns per run: {plan.run_scope.main_turn_count}")
            print(f"Total main turns: {plan.total_main_turns}")
            print(
                "Simulator logical calls: "
                f"{plan.total_simulator_logical_calls}"
            )
            print(
                "Embedding logical calls: "
                f"{plan.total_embedding_logical_calls}"
            )
            print(f"Evaluator logical calls: {plan.total_evaluator_logical_calls}")
            print(f"Total logical calls: {plan.total_logical_calls}")
            print(f"Maximum physical attempts: {plan.maximum_physical_attempts}")
            return 0
        if args.output_root is None:
            print(
                "Full experiment failed: --output-root is required for execution.",
                file=sys.stderr,
            )
            return 1
        try:
            from src.simulation import execute_full_experiment

            result = execute_full_experiment(
                mode=mode,
                output_root=args.output_root,
                experiment_id=args.experiment_id,
                confirmation=args.confirm_full_experiment,
                config_directory=args.config_dir,
                real_authorized=args.allow_real_llm is True,
                max_total_tokens=args.max_total_tokens,
                accepted_pilot_review_dir=args.accepted_pilot_review_dir,
                ground_truth_finalization_dir=args.ground_truth_finalization_dir,
                on_conflict=args.on_conflict,
            )
        except KeyboardInterrupt:
            print("Full experiment interrupted; safe partial evidence was retained.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(
                f"Full experiment failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        print("Full experiment completed successfully.")
        print(f"Eligible source runs: {len(result.run_chains)}")
        print(f"Phase 09 evaluations: {len(result.run_chains)}")
        print("MRA records: 240")
        print("BCR export records: 240")
        print("BCR judgments: 0")
        print(f"Experiment directory: {result.paths.experiment_dir}")
        return 0

    print("Classroom generative-agents repository scaffold is ready.")
    print("Run 'python main.py --help' to view available commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
