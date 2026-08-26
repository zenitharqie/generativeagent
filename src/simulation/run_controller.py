"""Deterministic lifecycle controller for one retrieval condition and repetition."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from src.cognition import CognitivePipeline
from src.config_loader import (
    DEFAULT_CONFIG_DIRECTORY,
    ExperimentConfig,
    load_experiment_config,
)
from src.llm import DummyLLMClient, LLMCallType, LLMClient, TextEmbeddingProvider
from src.logging_system import (
    ErrorLogRecord,
    RunLogMetadata,
    RunSummary,
    append_jsonl,
    memory_record_from_entry,
    write_run_summary,
)
from src.memory import MemoryStream, initialize_memory_streams
from src.retrieval import validate_vector
from src.simulation.natural_block_runner import NaturalBlockRunner
from src.simulation.step_runner import StepExecutionResult, StepRunner
from src.simulation.target_event_runner import TargetEventRunner
from src.simulation.turn_executor import (
    SimulationExecutionError,
    SimulationProgress,
    TurnExecutor,
    validate_memory_stream_mapping,
)
from src.simulation.turn_policy import TurnPolicyManager
from src.storage import RunPaths, create_run_directory


FULL_RUN_DAYS = (1, 2, 3, 4, 5, 6)
PILOT_DAYS = (1, 2)
FULL_NATURAL_BLOCKS = 7
FULL_TARGET_EVENTS = 5
FULL_MAIN_TURNS = 104
PILOT_NATURAL_BLOCKS = 3
PILOT_TARGET_EVENTS = 1
PILOT_MAIN_TURNS = 40


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach secondary failure context when supported without hiding ``error``."""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


class RunControllerError(RuntimeError):
    """Base error for one-run Phase 08 lifecycle orchestration."""


class RunExecutionError(RunControllerError):
    """Expose the originating failure without hiding finalization failures."""

    def __init__(
        self,
        message: str,
        *,
        run_paths: RunPaths,
        original_error: BaseException,
        error_log_error: BaseException | None = None,
        summary_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.run_paths = run_paths
        self.original_error = original_error
        self.error_log_error = error_log_error
        self.summary_error = summary_error


@dataclass(frozen=True)
class RunExecutionResult:
    """Successful execution of the requested full or approved pilot scope."""

    run_paths: RunPaths
    run_summary: RunSummary
    requested_days: tuple[int, ...]
    step_results: tuple[StepExecutionResult, ...]
    scope_succeeded: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.run_paths, RunPaths):
            raise RunControllerError("run_paths must be RunPaths")
        if not isinstance(self.run_summary, RunSummary):
            raise RunControllerError("run_summary must be RunSummary")
        if self.run_summary.status not in {"completed", "excluded"}:
            raise RunControllerError(
                "successful run result requires completed or excluded summary status"
            )
        if not self.scope_succeeded:
            raise RunControllerError("successful run result must mark scope_succeeded")


class RunController:
    """Execute one snapshotted run with no retries, resume, or run-loop expansion."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_provider: TextEmbeddingProvider,
        config_directory: str | Path = DEFAULT_CONFIG_DIRECTORY,
        retrieval_engine: Any | None = None,
        sanitize_failure_messages: bool = False,
    ) -> None:
        if not callable(getattr(llm_client, "generate", None)):
            raise RunControllerError("llm_client must provide generate(request)")
        if not callable(getattr(embedding_provider, "embed", None)):
            raise RunControllerError(
                "embedding_provider must provide embed(text)"
            )
        if not isinstance(sanitize_failure_messages, bool):
            raise RunControllerError("sanitize_failure_messages must be boolean")
        self._llm_client = llm_client
        self._embedding_provider = embedding_provider
        self._config_directory = Path(config_directory)
        self._retrieval_engine = retrieval_engine
        self._sanitize_failure_messages = sanitize_failure_messages

    @staticmethod
    def _load_category_vocabulary(
        config_directory: Path,
    ) -> dict[str, tuple[str, ...]]:
        """Derive the per-agent labeling vocabulary from the frozen ground truth."""
        path = Path(config_directory) / "ground_truth.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            targets = value["targets"]
            vocabulary: dict[str, tuple[str, ...]] = {}
            for target in targets:
                agent_id = target["agent_id"]
                labels = vocabulary.setdefault(agent_id, ())
                merged = dict.fromkeys(labels)
                for label in target["expected_categories"]:
                    merged[label] = None
                vocabulary[agent_id] = tuple(merged)
            return vocabulary
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RunControllerError(
                "could not derive category labeling vocabulary from ground truth snapshot"
            ) from exc

    def run(
        self,
        *,
        output_root: str | Path,
        experiment_id: str,
        run_id: str,
        retrieval_config_id: str,
        repetition_index: int,
        days: tuple[int, ...] | None = None,
    ) -> RunExecutionResult:
        """Execute either the complete timeline or the approved Day 1-2 pilot."""
        source_config = load_experiment_config(self._config_directory)
        requested_days = self._validate_run_request(
            source_config,
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
            days=days,
        )
        run_metadata = RunLogMetadata(
            experiment_id=experiment_id,
            run_id=run_id,
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
        )
        progress = SimulationProgress()
        run_paths = create_run_directory(
            output_root,
            experiment_id,
            run_id,
            config_directory=self._config_directory,
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
        )

        try:
            config = load_experiment_config(run_paths.config_snapshot)
            category_vocabulary = self._load_category_vocabulary(
                run_paths.config_snapshot
            )
            selected_steps = tuple(
                step for step in config.steps if step.day in requested_days
            )
            self._validate_selected_steps(
                config,
                selected_steps=selected_steps,
                requested_days=requested_days,
            )
            streams = self._initialize_embedded_seed_streams(config, run_id=run_id)
            self._persist_seed_memories(
                streams,
                agent_ids=tuple(agent.agent_id for agent in config.agents),
                run_metadata=run_metadata,
                run_paths=run_paths,
            )

            pipeline = CognitivePipeline(
                config=config,
                llm_client=self._llm_client,
                embedding_provider=self._embedding_provider,
                retrieval_engine=self._retrieval_engine,
                category_vocabulary=category_vocabulary,
            )
            turn_executor = TurnExecutor(
                config=config,
                pipeline=pipeline,
                embedding_provider=self._embedding_provider,
                run_metadata=run_metadata,
                run_paths=run_paths,
            )
            step_runner = StepRunner(
                natural_block_runner=NaturalBlockRunner(
                    turn_executor=turn_executor,
                    turn_policy=TurnPolicyManager(),
                ),
                target_event_runner=TargetEventRunner(
                    turn_executor=turn_executor,
                ),
            )
            previous_responses: dict[str, str] = {}
            step_results: list[StepExecutionResult] = []
            for step in selected_steps:
                step_results.append(
                    step_runner.run_step(
                        step=step,
                        memory_streams=streams,
                        previous_same_agent_responses=previous_responses,
                        progress=progress,
                    )
                )

            terminal_status = self._validate_success_counts(
                progress,
                requested_days=requested_days,
            )
        except KeyboardInterrupt as exc:
            error_log_error, summary_error = self._finalize_failure(
                run_paths=run_paths,
                run_metadata=run_metadata,
                progress=progress,
                original_error=exc,
                status="interrupted",
            )
            if error_log_error is not None:
                _add_exception_note(
                    exc,
                    f"error logging also failed: {error_log_error}",
                )
            if summary_error is not None:
                _add_exception_note(
                    exc,
                    f"interrupted summary also failed: {summary_error}",
                )
            raise
        except Exception as exc:
            error_log_error, summary_error = self._finalize_failure(
                run_paths=run_paths,
                run_metadata=run_metadata,
                progress=progress,
                original_error=exc,
                status="failed",
            )
            details = [f"run failed due to {type(exc).__name__}: {exc}"]
            if error_log_error is not None:
                details.append(f"error logging also failed: {error_log_error}")
            if summary_error is not None:
                details.append(f"failed summary also failed: {summary_error}")
            raise RunExecutionError(
                "; ".join(details),
                run_paths=run_paths,
                original_error=exc,
                error_log_error=error_log_error,
                summary_error=summary_error,
            ) from exc

        summary = RunSummary(
            experiment_id=experiment_id,
            run_id=run_id,
            retrieval_config_id=retrieval_config_id,
            repetition_index=repetition_index,
            status=terminal_status,
            completed_natural_blocks=progress.completed_natural_blocks,
            completed_target_events=progress.completed_target_events,
            completed_main_turns=progress.completed_main_turns,
            error_count=0,
        )
        try:
            write_run_summary(run_paths.run_summary, summary)
        except KeyboardInterrupt as exc:
            error_log_error, summary_error = self._finalize_failure(
                run_paths=run_paths,
                run_metadata=run_metadata,
                progress=progress,
                original_error=exc,
                status="interrupted",
            )
            if error_log_error is not None:
                _add_exception_note(
                    exc,
                    f"error logging also failed: {error_log_error}",
                )
            if summary_error is not None:
                _add_exception_note(
                    exc,
                    f"interrupted summary also failed: {summary_error}",
                )
            raise
        except Exception as exc:
            error_log_error, failure_summary_error = self._finalize_failure(
                run_paths=run_paths,
                run_metadata=run_metadata,
                progress=progress,
                original_error=exc,
                status="failed",
            )
            details = [f"terminal summary failed due to {type(exc).__name__}: {exc}"]
            if error_log_error is not None:
                details.append(f"error logging also failed: {error_log_error}")
            if failure_summary_error is not None:
                details.append(
                    f"failed summary also failed: {failure_summary_error}"
                )
            raise RunExecutionError(
                "; ".join(details),
                run_paths=run_paths,
                original_error=exc,
                error_log_error=error_log_error,
                summary_error=failure_summary_error,
            ) from exc

        return RunExecutionResult(
            run_paths=run_paths,
            run_summary=summary,
            requested_days=requested_days,
            step_results=tuple(step_results),
        )

    @staticmethod
    def _validate_run_request(
        config: ExperimentConfig,
        *,
        retrieval_config_id: str,
        repetition_index: int,
        days: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        known_configs = {item.config_id for item in config.retrieval_configs}
        if retrieval_config_id not in known_configs:
            raise RunControllerError(
                f"unknown retrieval_config_id: {retrieval_config_id!r}"
            )
        if (
            isinstance(repetition_index, bool)
            or not isinstance(repetition_index, int)
            or not 1 <= repetition_index <= config.settings.repetitions
        ):
            raise RunControllerError(
                f"repetition_index must be between 1 and {config.settings.repetitions}"
            )
        requested = FULL_RUN_DAYS if days is None else days
        if not isinstance(requested, tuple) or any(
            isinstance(day, bool) or not isinstance(day, int) for day in requested
        ):
            raise RunControllerError("days must be a tuple of integer day numbers")
        if requested not in {PILOT_DAYS, FULL_RUN_DAYS}:
            raise RunControllerError(
                "Phase 08 supports only the Day 1-2 pilot or the complete Day 1-6 run"
            )
        return requested

    @staticmethod
    def _validate_selected_steps(
        config: ExperimentConfig,
        *,
        selected_steps: tuple[Any, ...],
        requested_days: tuple[int, ...],
    ) -> None:
        if not selected_steps:
            raise RunControllerError("selected simulation scope contains no steps")
        if tuple(step.day for step in selected_steps) != tuple(
            sorted(step.day for step in selected_steps)
        ):
            raise RunControllerError("simulation steps are not ordered by day")
        if len({step.step_id for step in selected_steps}) != len(selected_steps):
            raise RunControllerError("simulation step IDs must be unique")
        expected = tuple(step for step in config.steps if step.day in requested_days)
        if selected_steps != expected:
            raise RunControllerError(
                "selected simulation steps do not preserve snapshotted order"
            )

    def _initialize_embedded_seed_streams(
        self,
        config: ExperimentConfig,
        *,
        run_id: str,
    ) -> dict[str, MemoryStream]:
        raw_streams = initialize_memory_streams(config, run_id=run_id, created_at=0)
        embedded_streams: dict[str, MemoryStream] = {}
        expected_dimensions: int | None = None
        for agent in config.agents:
            embedded_stream = MemoryStream(agent.agent_id, run_id=run_id)
            for seed in raw_streams[agent.agent_id].list_memories():
                embedding = validate_vector(
                    self._embedding_provider.embed(seed.text),
                    field_name="seed_embedding",
                )
                if not any(value != 0.0 for value in embedding):
                    raise SimulationExecutionError("seed_embedding must be non-zero")
                if expected_dimensions is None:
                    expected_dimensions = len(embedding)
                elif len(embedding) != expected_dimensions:
                    raise SimulationExecutionError(
                        "all seed embeddings must have equal dimensions"
                    )
                embedded_stream.add_memory(replace(seed, embedding=embedding))
            embedded_streams[agent.agent_id] = embedded_stream
        validate_memory_stream_mapping(embedded_streams, run_id=run_id)
        return embedded_streams

    @staticmethod
    def _persist_seed_memories(
        streams: dict[str, MemoryStream],
        *,
        agent_ids: tuple[str, ...],
        run_metadata: RunLogMetadata,
        run_paths: RunPaths,
    ) -> None:
        for agent_id in agent_ids:
            for memory in streams[agent_id].list_memories():
                record = memory_record_from_entry(memory, run=run_metadata)
                append_jsonl(run_paths.memory_logs, record)

    @staticmethod
    def _validate_success_counts(
        progress: SimulationProgress,
        *,
        requested_days: tuple[int, ...],
    ) -> str:
        actual = (
            progress.completed_natural_blocks,
            progress.completed_target_events,
            progress.completed_main_turns,
        )
        if requested_days == FULL_RUN_DAYS:
            expected = (
                FULL_NATURAL_BLOCKS,
                FULL_TARGET_EVENTS,
                FULL_MAIN_TURNS,
            )
            if actual != expected or progress.final_logical_timestamp != 104:
                raise RunControllerError(
                    f"full run completion counts are {actual}, expected {expected}"
                )
            return "completed"
        expected = (
            PILOT_NATURAL_BLOCKS,
            PILOT_TARGET_EVENTS,
            PILOT_MAIN_TURNS,
        )
        if actual != expected or progress.final_logical_timestamp != 40:
            raise RunControllerError(
                f"pilot completion counts are {actual}, expected {expected}"
            )
        return "excluded"

    def _finalize_failure(
        self,
        *,
        run_paths: RunPaths,
        run_metadata: RunLogMetadata,
        progress: SimulationProgress,
        original_error: BaseException,
        status: str,
    ) -> tuple[BaseException | None, BaseException | None]:
        error_log_error = self._try_write_error_record(
            run_paths=run_paths,
            run_metadata=run_metadata,
            original_error=original_error,
            sanitize_message=self._sanitize_failure_messages,
        )
        summary_error: BaseException | None = None
        try:
            write_run_summary(
                run_paths.run_summary,
                RunSummary(
                    experiment_id=run_metadata.experiment_id,
                    run_id=run_metadata.run_id,
                    retrieval_config_id=run_metadata.retrieval_config_id,
                    repetition_index=run_metadata.repetition_index,
                    status=status,
                    completed_natural_blocks=progress.completed_natural_blocks,
                    completed_target_events=progress.completed_target_events,
                    completed_main_turns=progress.completed_main_turns,
                    error_count=1,
                ),
            )
        except Exception as exc:
            summary_error = exc
        return error_log_error, summary_error

    @staticmethod
    def _try_write_error_record(
        *,
        run_paths: RunPaths,
        run_metadata: RunLogMetadata,
        original_error: BaseException,
        sanitize_message: bool,
    ) -> BaseException | None:
        if sanitize_message:
            message = (
                "Pilot run interrupted."
                if isinstance(original_error, KeyboardInterrupt)
                else "Pilot run failed; inspect the canonical error type and attempt records."
            )
        else:
            message = str(original_error) or type(original_error).__name__
        try:
            append_jsonl(
                run_paths.errors,
                ErrorLogRecord(
                    run=run_metadata,
                    error_type=type(original_error).__name__,
                    message=message,
                    recoverable=False,
                    retry_count=0,
                ),
            )
        except Exception as exc:
            return exc
        return None


def build_deterministic_dummy_llm_client() -> DummyLLMClient:
    """Build distinct offline responses for the maximum one-run Phase 08 scope."""

    def payload(**values: str) -> str:
        return json.dumps(values, sort_keys=True, separators=(",", ":"))

    natural = tuple(
        payload(
            response=f"Deterministic dummy natural response {index}.",
            memory_to_store=f"Deterministic dummy natural memory {index}.",
            category="classroom_norms",
        )
        for index in range(1, 85)
    )
    target = tuple(
        payload(
            response=f"Deterministic dummy target response {index}.",
            memory_to_store=f"Deterministic dummy target memory {index}.",
            category="classroom_norms",
        )
        for index in range(1, 21)
    )
    reflections = tuple(
        payload(
            reflection=f"Deterministic post-step reflection {index}.",
            category="classroom_norms",
        )
        for index in range(1, 49)
    )
    return DummyLLMClient(
        {
            LLMCallType.NATURAL_RESPONSE: natural,
            LLMCallType.TARGET_EVENT_RESPONSE: target,
            LLMCallType.REFLECTION: reflections,
        }
    )
