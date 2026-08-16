"""Strict, bounded, read-only Phase 12 artifact readers.

This module intentionally imports no simulation, provider, retrieval, memory,
cognition, or evaluation execution package. Persisted bytes are the only input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any

from src.visualizer.errors import (
    ReplayEligibilityError,
    ReplayIdentityError,
    ReplayInputError,
    ReplayOverlayError,
    ReplaySchemaError,
    ReplaySecurityError,
)
from src.visualizer.schemas import SOURCE_SCHEMA_VERSION


MAX_INPUT_FILE_BYTES = 128 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 512 * 1024 * 1024

CONFIG_FILENAMES = (
    "agents.json",
    "ground_truth.json",
    "retrieval_configs.json",
    "settings.json",
    "simulation_steps.json",
    "target_events.json",
)
STREAM_FILENAMES = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)
RUN_ENTRIES = frozenset((*STREAM_FILENAMES, "run_summary.json", "config_snapshot"))
EVALUATION_FILENAMES = frozenset(
    {"mra_records.jsonl", "bcr_export.jsonl", "evaluation_summary.json"}
)
JUDGMENT_FILENAMES = frozenset(
    {"bcr_judged_records.jsonl", "bcr_judgment_summary.json"}
)
PILOT_EVIDENCE_FILENAMES = frozenset(
    {
        "pilot_manifest.json",
        "pilot_attempts.jsonl",
        "pilot_usage_summary.json",
        "pilot_report.json",
    }
)

_PILOT_MANIFEST_FIELDS = {
    "schema_version",
    "pilot_id",
    "experiment_id",
    "run_id",
    "pilot_only",
    "primary_eligible",
    "included_in_final_research_matrix",
    "expected_source_run_status",
    "plan",
    "max_total_tokens",
    "configuration_hashes",
    "prompts",
    "simulator_settings",
    "evaluator_settings",
    "embedding_settings",
    "retry_policy",
    "evaluation_schema_version",
    "research_log_schema_version",
    "bcr_rubric_version",
    "manifest_hash",
}
_PILOT_PLAN_FIELDS = {
    "mode",
    "retrieval_config_id",
    "repetition_index",
    "days",
    "step_ids",
    "natural_block_count",
    "target_event_count",
    "main_turn_count",
    "post_step_reflection_count",
    "target_evaluation_unit_count",
    "call_limits",
}
_PILOT_USAGE_FIELDS = {
    "schema_version",
    "max_total_tokens",
    "reported_total_tokens",
    "logical_call_count",
    "physical_attempt_count",
    "retry_count",
    "missing_total_usage_attempt_count",
    "budget_exceeded",
    "role_usage",
}
_PILOT_ROLE_USAGE_FIELDS = {
    "service_role",
    "logical_call_count",
    "physical_attempt_count",
    "retry_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "missing_total_usage_attempt_count",
    "success_count",
    "failed_count",
    "invalid_output_count",
}
_PILOT_REPORT_FIELDS = {
    "schema_version",
    "pilot_id",
    "experiment_id",
    "run_id",
    "mode",
    "manifest_hash",
    "acceptance_state",
    "technical_success",
    "eligible_for_manual_acceptance",
    "source_run_status",
    "source_run_directory_name",
    "diagnostic_evaluation_directory_name",
    "primary_eligible",
    "included_in_final_research_matrix",
    "usage",
    "validation",
    "source_fingerprint",
    "diagnostic_evaluation_fingerprint",
    "mra_diagnostic_record_count",
    "mra_diagnostic_mean",
    "bcr_export_record_count",
    "bcr_judgment_count",
    "technical_checks",
    "rejection_reasons",
    "terminal_error_type",
}
_PILOT_ACCEPTANCE_FIELDS = {
    "schema_version",
    "pilot_id",
    "experiment_id",
    "run_id",
    "acceptance_state",
    "manual_review",
    "manifest_hash",
    "evidence_hash",
    "source_run_status",
    "primary_eligible",
    "included_in_final_research_matrix",
    "acceptance_hash",
}
_PILOT_FREEZE_FIELDS = {
    "schema_version",
    "freeze_state",
    "pilot_id",
    "experiment_id",
    "run_id",
    "acceptance_hash",
    "evidence_hash",
    "manifest_hash",
    "prompt_version",
    "configuration_hashes",
    "prompts",
    "simulator_settings",
    "evaluator_settings",
    "embedding_settings",
    "retry_policy",
    "evaluation_schema_version",
    "research_log_schema_version",
    "bcr_rubric_version",
    "primary_eligible",
    "included_in_final_research_matrix",
    "phase13_source_and_git_verification_required",
    "freeze_hash",
}

STREAM_RECORD_LIMITS = {
    "retrieval_logs.jsonl": 104,
    "behavior_logs.jsonl": 104,
    "memory_logs.jsonl": 164,
    "reflection_logs.jsonl": 48,
    "errors.jsonl": 1,
    "event_summaries.jsonl": 5,
    "natural_block_summaries.jsonl": 7,
    "api_usage_logs.jsonl": 816,
}

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_HASH = re.compile(r"[0-9a-f]{64}")

_RUN_FIELDS = {
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
}
_STEP_FIELDS = {
    "step_id",
    "step_type",
    "is_target_event",
    "evaluation_enabled",
    "day",
    "period",
    "timestamp",
    "event_id",
}
_TURN_FIELDS = {"turn_id", "turn_index", "agent_id"}
_SHARED = _RUN_FIELDS | _STEP_FIELDS | _TURN_FIELDS | {"schema_version"}
_STEP_SHARED = _RUN_FIELDS | _STEP_FIELDS | {"schema_version"}
_CANDIDATE_FIELDS = {
    "memory_id",
    "category",
    "raw_recency",
    "raw_importance",
    "raw_relevance",
    "normalized_recency",
    "normalized_importance",
    "normalized_relevance",
    "final_score",
    "selected",
}
_STREAM_FIELDS = {
    "retrieval_logs.jsonl": _SHARED
    | {
        "query",
        "query_embedding",
        "weights",
        "top_k",
        "candidate_count",
        "candidates",
        "selected_memory_ids",
        "accessed_memory_ids",
    },
    "behavior_logs.jsonl": _SHARED
    | {"response", "retrieved_memory_ids", "plan_summary"},
    "memory_logs.jsonl": _SHARED
    | {
        "memory_id",
        "memory_type",
        "category",
        "text",
        "importance",
        "created_at",
        "last_accessed_at",
        "embedding",
        "metadata",
    },
    "reflection_logs.jsonl": _SHARED | {"reflection", "category", "memory_id"},
    "api_usage_logs.jsonl": _SHARED
    | {
        "call_type",
        "model",
        "prompt_template",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "status",
        "error_type",
    },
    "errors.jsonl": _SHARED
    | {"error_type", "message", "recoverable", "retry_count"},
    "event_summaries.jsonl": _STEP_SHARED
    | {
        "ordered_main_turn_ids",
        "participant_ids",
        "per_agent_main_turn_counts",
        "completed_main_turn_count",
        "reflection_count",
        "final_logical_timestamp",
        "ordered_responding_agent_ids",
        "response_count",
    },
    "natural_block_summaries.jsonl": _STEP_SHARED
    | {
        "ordered_main_turn_ids",
        "participant_ids",
        "per_agent_main_turn_counts",
        "completed_main_turn_count",
        "reflection_count",
        "final_logical_timestamp",
    },
}

_SUMMARY_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "status",
    "completed_natural_blocks",
    "completed_target_events",
    "completed_main_turns",
    "error_count",
}

_AGENT_FIELDS = {
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
_SEED_FIELDS = {"text", "category", "importance"}
_STEP_COMMON_FIELDS = {
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
_NATURAL_FIELDS = _STEP_COMMON_FIELDS | {
    "turns_per_agent",
    "total_turns",
    "teacher_opens_block",
    "turn_policy",
}
_TARGET_FIELDS = _STEP_COMMON_FIELDS | {"event_id"}
_EVENT_FIELDS = {
    "event_id",
    "title",
    "description",
    "expected_memory_themes",
    "evaluation_role",
}


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _parse_json(payload: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError, RecursionError) as exc:
        raise ReplaySchemaError(f"{filename} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ReplaySchemaError(f"{filename} must contain one JSON object")
    return value


def _require_fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ReplaySchemaError(
            f"{context} fields are invalid; missing={missing}, extra={extra}"
        )


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ReplaySchemaError(f"{field_name} must be a safe identifier")
    return value


def _require_pilot_failure_identifier(value: object, field_name: str) -> str:
    """Apply the canonical identifier grammar at the optional pilot boundary."""
    try:
        return _require_identifier(value, field_name)
    except ReplaySchemaError as exc:
        raise ReplayOverlayError(f"{field_name} must be a safe identifier") from exc


def _require_text(value: object, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReplaySchemaError(f"{field_name} must be non-empty text")
    return value


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplaySchemaError(f"{field_name} must be an integer >= {minimum}")
    return value


def _require_number(value: object, field_name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ReplaySchemaError(f"{field_name} must be a finite number")
    return value


def _require_string_list(value: object, field_name: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReplaySchemaError(f"{field_name} must be a string array")
    if not empty and not value:
        raise ReplaySchemaError(f"{field_name} must not be empty")
    return value


def _reject_symlink_components(path: Path) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                raise ReplaySecurityError("replay paths must not contain symbolic links")
        except OSError as exc:
            raise ReplayInputError("could not inspect replay path components") from exc


def resolve_input_directory(value: str | Path, *, label: str) -> Path:
    try:
        requested = Path(value)
    except (TypeError, ValueError) as exc:
        raise ReplayInputError(f"invalid {label} directory") from exc
    if ".." in requested.parts:
        raise ReplaySecurityError(f"{label} path traversal is not permitted")
    _reject_symlink_components(requested)
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReplayInputError(f"{label} directory does not exist") from exc
    if not resolved.is_dir():
        raise ReplayInputError(f"{label} path must be a directory")
    return resolved


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink():
        raise ReplaySecurityError("replay inputs must not contain symbolic links")
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReplayInputError(f"could not inspect required artifact {path.name}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ReplayInputError(f"required artifact is not a regular file: {path.name}")
    if before.st_size > MAX_INPUT_FILE_BYTES:
        raise ReplaySecurityError(f"artifact exceeds the replay size limit: {path.name}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (before.st_dev, before.st_ino):
                raise ReplaySecurityError("artifact identity changed during safe open")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_INPUT_FILE_BYTES:
                    raise ReplaySecurityError("artifact grew beyond the replay size limit")
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    except ReplaySecurityError:
        raise
    except OSError as exc:
        raise ReplayInputError(f"could not read required artifact {path.name}") from exc
    try:
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReplayInputError(f"could not recheck artifact {path.name}") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReplaySecurityError("artifact changed while it was being read")
    return payload


@dataclass(frozen=True)
class InputSnapshot:
    directory: Path
    relative_hashes: tuple[tuple[str, str], ...]
    aggregate_fingerprint: str

    @classmethod
    def from_payloads(
        cls,
        directory: Path,
        payloads: Mapping[str, bytes],
    ) -> "InputSnapshot":
        hashes = tuple(
            (name, sha256(payload).hexdigest())
            for name, payload in sorted(payloads.items())
        )
        digest = sha256()
        for name, payload in sorted(payloads.items()):
            encoded = name.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return cls(directory, hashes, digest.hexdigest())

    def verify_unchanged(self) -> None:
        expected_names = {relative for relative, _ in self.relative_hashes}
        actual_names: set[str] = set()
        try:
            for item in self.directory.rglob("*"):
                if item.is_symlink():
                    raise ReplaySecurityError("a replay input gained a symbolic link")
                if item.is_file():
                    actual_names.add(item.relative_to(self.directory).as_posix())
                    if len(actual_names) > len(expected_names):
                        raise ReplaySecurityError("a replay input inventory changed during export")
        except ReplaySecurityError:
            raise
        except OSError as exc:
            raise ReplayInputError("could not recheck replay input inventory") from exc
        if actual_names != expected_names:
            raise ReplaySecurityError("a replay input inventory changed during export")
        for relative, expected in self.relative_hashes:
            path = self.directory / Path(relative)
            actual = sha256(_read_regular_file(path)).hexdigest()
            if actual != expected:
                raise ReplaySecurityError("a replay input changed during export")

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "aggregate_sha256": self.aggregate_fingerprint,
            "artifact_hashes": [
                {"relative_path": name, "sha256": digest}
                for name, digest in self.relative_hashes
            ],
        }


@dataclass(frozen=True)
class LoadedRunArtifacts:
    snapshot: InputSnapshot
    summary: Mapping[str, Any]
    config: Mapping[str, Any]
    streams: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True)
class LoadedOverlay:
    snapshot: InputSnapshot
    value: Mapping[str, Any]


def _directory_entries(directory: Path) -> tuple[Path, ...]:
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise ReplayInputError("could not inspect replay input directory") from exc
    if any(item.is_symlink() for item in entries):
        raise ReplaySecurityError("replay input directories must not contain symlinks")
    return entries


def _read_exact_directory(
    directory: Path,
    expected_files: frozenset[str],
) -> dict[str, bytes]:
    entries = _directory_entries(directory)
    if {item.name for item in entries} != expected_files or any(
        not item.is_file() for item in entries
    ):
        raise ReplayInputError("optional overlay directory layout is invalid")
    payloads = {item.name: _read_regular_file(item) for item in entries}
    if sum(len(payload) for payload in payloads.values()) > MAX_TOTAL_INPUT_BYTES:
        raise ReplaySecurityError("overlay input exceeds the aggregate size limit")
    return payloads


def _parse_jsonl(
    payload: bytes,
    filename: str,
    *,
    record_limit: int,
    exact_fields: set[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    if not payload:
        return ()
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ReplaySchemaError(f"{filename} must be LF-terminated JSONL")
    lines = payload.splitlines(keepends=True)
    if len(lines) > record_limit:
        raise ReplaySecurityError(f"{filename} exceeds the supported record count")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line) > MAX_JSONL_LINE_BYTES:
            raise ReplaySecurityError(f"{filename} contains an oversized record")
        record = _parse_json(line, f"{filename} line {line_number}")
        if record.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise ReplaySchemaError(f"{filename} has an unsupported schema version")
        if exact_fields is not None:
            _require_fields(record, exact_fields, f"{filename} line {line_number}")
        records.append(record)
    return tuple(records)


def _validate_summary(summary: Mapping[str, Any], run_dir: Path) -> None:
    _require_fields(summary, _SUMMARY_FIELDS, "run_summary.json")
    if summary["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ReplaySchemaError("run summary schema version is unsupported")
    for name in ("experiment_id", "run_id", "retrieval_config_id"):
        _require_identifier(summary[name], name)
    repetition = _require_int(summary["repetition_index"], "repetition_index", minimum=1)
    if repetition is None or repetition > 3:
        raise ReplaySchemaError("repetition_index must be from 1 through 3")
    status_value = summary["status"]
    if status_value == "initialized":
        raise ReplayEligibilityError("initialized runs are not stable replay inputs")
    if status_value not in {"completed", "excluded", "failed", "interrupted"}:
        raise ReplaySchemaError("run status is unsupported")
    for name, maximum in (
        ("completed_natural_blocks", 7),
        ("completed_target_events", 5),
        ("completed_main_turns", 104),
    ):
        value = _require_int(summary[name], name)
        if value is None or value > maximum:
            raise ReplaySchemaError(f"{name} exceeds the fixed run maximum")
    _require_int(summary["error_count"], "error_count")
    if summary["run_id"] != run_dir.name or summary["experiment_id"] != run_dir.parent.name:
        raise ReplayIdentityError("run summary identity does not match its directory")
    if status_value == "completed" and (
        summary["completed_natural_blocks"],
        summary["completed_target_events"],
        summary["completed_main_turns"],
        summary["error_count"],
    ) != (7, 5, 104, 0):
        raise ReplaySchemaError("completed run does not satisfy exact 7/5/104/0 counts")
    if status_value == "excluded" and (
        summary["completed_natural_blocks"],
        summary["completed_target_events"],
        summary["completed_main_turns"],
        summary["error_count"],
    ) != (3, 1, 40, 0):
        raise ReplaySchemaError("excluded run does not satisfy exact Day 1-2 counts")
    if status_value in {"failed", "interrupted"} and summary["error_count"] != 1:
        raise ReplaySchemaError("failed or interrupted run must report one terminal error")


def _validate_config(config: Mapping[str, Any]) -> None:
    agents = config["agents.json"]
    _require_fields(agents, {"agents"}, "agents.json")
    if not isinstance(agents["agents"], list) or len(agents["agents"]) != 4:
        raise ReplaySchemaError("agents.json must contain exactly four agents")
    seen_agents: set[str] = set()
    for index, agent in enumerate(agents["agents"]):
        if not isinstance(agent, Mapping):
            raise ReplaySchemaError("agent rows must be objects")
        _require_fields(agent, _AGENT_FIELDS, f"agents[{index}]")
        agent_id = _require_identifier(agent["agent_id"], "agent_id")
        if agent_id in seen_agents:
            raise ReplaySchemaError("agent IDs must be unique")
        seen_agents.add(agent_id)
        for name in ("display_name", "role", "archetype", "persona_summary", "communication_style"):
            _require_text(agent[name], name)
        for name in ("traits", "goals", "motivations"):
            _require_string_list(agent[name], name)
        seeds = agent["seed_memories"]
        if not isinstance(seeds, list) or len(seeds) != 3:
            raise ReplaySchemaError("each agent must have exactly three seed memories")
        for seed in seeds:
            if not isinstance(seed, Mapping):
                raise ReplaySchemaError("seed memory must be an object")
            _require_fields(seed, _SEED_FIELDS, "seed memory")
            _require_text(seed["text"], "seed text")
            _require_text(seed["category"], "seed category")
            importance = _require_number(seed["importance"], "seed importance")
            if not 1 <= importance <= 10:
                raise ReplaySchemaError("seed importance must be in [1, 10]")
    if seen_agents != {"teacher", "active_student", "quiet_student", "academic_student"}:
        raise ReplaySchemaError("agent set differs from the architecture contract")

    steps = config["simulation_steps.json"]
    _require_fields(steps, {"steps"}, "simulation_steps.json")
    if not isinstance(steps["steps"], list) or len(steps["steps"]) != 12:
        raise ReplaySchemaError("simulation_steps.json must contain exactly 12 steps")
    seen_steps: set[str] = set()
    for index, step in enumerate(steps["steps"]):
        if not isinstance(step, Mapping):
            raise ReplaySchemaError("step rows must be objects")
        step_type = step.get("step_type")
        expected = _NATURAL_FIELDS if step_type == "natural_interaction_block" else _TARGET_FIELDS
        if step_type not in {"natural_interaction_block", "target_event"}:
            raise ReplaySchemaError("unsupported configured step type")
        _require_fields(step, expected, f"steps[{index}]")
        step_id = _require_identifier(step["step_id"], "step_id")
        if step_id in seen_steps:
            raise ReplaySchemaError("configured step IDs must be unique")
        seen_steps.add(step_id)
        _require_int(step["day"], "step day", minimum=1)
        for name in ("period", "title", "description"):
            _require_text(step[name], name)
        participants = _require_string_list(step["participants"], "participants")
        if len(participants) != 4 or set(participants) != seen_agents:
            raise ReplaySchemaError("each step must contain the four fixed agents")
        if not isinstance(step["memory_generation_enabled"], bool) or not isinstance(
            step["evaluation_enabled"], bool
        ):
            raise ReplaySchemaError("step flags must be boolean")
        if step_type == "natural_interaction_block" and (
            step["turns_per_agent"],
            step["total_turns"],
            step["teacher_opens_block"],
            step["turn_policy"],
        ) != (3, 12, True, "controlled_fixed"):
            raise ReplaySchemaError("natural block policy differs from the fixed contract")
        if step_type == "target_event":
            _require_identifier(step["event_id"], "event_id")

    events = config["target_events.json"]
    _require_fields(events, {"events"}, "target_events.json")
    if not isinstance(events["events"], list) or len(events["events"]) != 5:
        raise ReplaySchemaError("target_events.json must contain exactly five events")
    for event in events["events"]:
        if not isinstance(event, Mapping):
            raise ReplaySchemaError("target event rows must be objects")
        _require_fields(event, _EVENT_FIELDS, "target event")
        _require_identifier(event["event_id"], "event_id")
        for name in ("title", "description", "evaluation_role"):
            _require_text(event[name], name)
        _require_string_list(event["expected_memory_themes"], "expected_memory_themes")

    retrieval = config["retrieval_configs.json"]
    _require_fields(retrieval, {"configurations"}, "retrieval_configs.json")
    if not isinstance(retrieval["configurations"], list) or len(retrieval["configurations"]) != 4:
        raise ReplaySchemaError("retrieval config inventory must contain four rows")
    for item in retrieval["configurations"]:
        if not isinstance(item, Mapping):
            raise ReplaySchemaError("retrieval configuration must be an object")
        _require_fields(item, {"config_id", "weights"}, "retrieval configuration")
        _require_identifier(item["config_id"], "config_id")
        if not isinstance(item["weights"], Mapping) or set(item["weights"]) != {
            "recency",
            "importance",
            "relevance",
        }:
            raise ReplaySchemaError("retrieval weights have invalid fields")
        for name, value in item["weights"].items():
            _require_number(value, f"weights.{name}")

    settings = config["settings.json"]
    expected_settings = {
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
    _require_fields(settings, expected_settings, "settings.json")
    if (
        settings["simulation_days"],
        settings["natural_blocks_total"],
        settings["target_events_total"],
        settings["turns_per_agent_per_natural_block"],
        settings["total_turns_per_natural_block"],
        settings["teacher_opens_natural_block"],
        settings["top_k"],
        settings["repetitions"],
    ) != (6, 7, 5, 3, 12, True, 5, 3):
        raise ReplaySchemaError("settings differ from the fixed experiment contract")


def _validate_stream_shape(filename: str, records: tuple[dict[str, Any], ...]) -> None:
    for index, record in enumerate(records):
        context = f"{filename} record {index + 1}"
        for name in ("experiment_id", "run_id", "retrieval_config_id"):
            _require_identifier(record[name], f"{context}.{name}")
        _require_int(record["repetition_index"], f"{context}.repetition_index", minimum=1)
        if filename == "retrieval_logs.jsonl":
            _require_text(record["query"], "retrieval query")
            if record["query_embedding"] is not None and not isinstance(record["query_embedding"], list):
                raise ReplaySchemaError("query_embedding must be an array or null")
            if not isinstance(record["weights"], Mapping) or set(record["weights"]) != {
                "recency",
                "importance",
                "relevance",
            }:
                raise ReplaySchemaError("retrieval weights are malformed")
            _require_int(record["top_k"], "top_k", minimum=1)
            count = _require_int(record["candidate_count"], "candidate_count")
            candidates = record["candidates"]
            if not isinstance(candidates, list) or len(candidates) != count:
                raise ReplaySchemaError("candidate_count does not match candidates")
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ReplaySchemaError("retrieval candidate must be an object")
                _require_fields(candidate, _CANDIDATE_FIELDS, "retrieval candidate")
            _require_string_list(record["selected_memory_ids"], "selected_memory_ids", empty=True)
            _require_string_list(record["accessed_memory_ids"], "accessed_memory_ids", empty=True)
        elif filename == "behavior_logs.jsonl":
            _require_text(record["response"], "response")
            _require_string_list(record["retrieved_memory_ids"], "retrieved_memory_ids", empty=True)
            _require_text(record["plan_summary"], "plan_summary", optional=True)
        elif filename == "memory_logs.jsonl":
            for name in ("memory_id", "agent_id", "memory_type", "category", "text", "step_id", "step_type"):
                _require_text(record[name], name)
            _require_number(record["importance"], "importance")
            _require_int(record["created_at"], "created_at")
            _require_int(record["last_accessed_at"], "last_accessed_at")
            if record["embedding"] is not None and not isinstance(record["embedding"], list):
                raise ReplaySchemaError("memory embedding must be an array or null")
            if not isinstance(record["metadata"], Mapping):
                raise ReplaySchemaError("memory metadata must be an object")
        elif filename == "reflection_logs.jsonl":
            _require_text(record["reflection"], "reflection")
            _require_text(record["category"], "reflection category")
            _require_text(record["memory_id"], "reflection memory_id", optional=True)
        elif filename == "api_usage_logs.jsonl":
            for name in ("call_type", "model", "prompt_template", "status"):
                _require_text(record[name], name)
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                _require_int(record[name], name)
            _require_text(record["error_type"], "error_type", optional=True)
        elif filename == "errors.jsonl":
            _require_text(record["error_type"], "error_type")
            _require_text(record["message"], "message")
            if not isinstance(record["recoverable"], bool):
                raise ReplaySchemaError("recoverable must be boolean")
            _require_int(record["retry_count"], "retry_count", optional=True)


def load_run_artifacts(run_dir: str | Path) -> LoadedRunArtifacts:
    directory = resolve_input_directory(run_dir, label="source run")
    entries = _directory_entries(directory)
    if {item.name for item in entries} != RUN_ENTRIES:
        raise ReplayInputError("source run directory layout is not canonical")
    config_dir = directory / "config_snapshot"
    if config_dir.is_symlink() or not config_dir.is_dir():
        raise ReplaySecurityError("config_snapshot must be a real directory")
    config_entries = _directory_entries(config_dir)
    if {item.name for item in config_entries} != set(CONFIG_FILENAMES) or any(
        not item.is_file() for item in config_entries
    ):
        raise ReplayInputError("config_snapshot layout is not canonical")

    payloads: dict[str, bytes] = {}
    for item in entries:
        if item.name == "config_snapshot":
            continue
        if not item.is_file():
            raise ReplayInputError("source run entries must be regular files")
        payloads[item.name] = _read_regular_file(item)
    for item in config_entries:
        payloads[f"config_snapshot/{item.name}"] = _read_regular_file(item)
    if sum(len(payload) for payload in payloads.values()) > MAX_TOTAL_INPUT_BYTES:
        raise ReplaySecurityError("source run exceeds the aggregate replay size limit")

    summary = _parse_json(payloads["run_summary.json"], "run_summary.json")
    _validate_summary(summary, directory)
    config = {
        filename: _parse_json(
            payloads[f"config_snapshot/{filename}"],
            f"config_snapshot/{filename}",
        )
        for filename in CONFIG_FILENAMES
        if filename != "ground_truth.json"
    }
    _validate_config(config)

    streams: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename in STREAM_FILENAMES:
        records = _parse_jsonl(
            payloads[filename],
            filename,
            record_limit=STREAM_RECORD_LIMITS[filename],
            exact_fields=_STREAM_FIELDS[filename],
        )
        _validate_stream_shape(filename, records)
        streams[filename] = records

    expected_identity = tuple(summary[name] for name in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    ))
    for filename, records in streams.items():
        for record in records:
            actual = tuple(record[name] for name in (
                "experiment_id",
                "run_id",
                "retrieval_config_id",
                "repetition_index",
            ))
            if actual != expected_identity:
                raise ReplayIdentityError(f"{filename} contains a cross-run record")

    return LoadedRunArtifacts(
        snapshot=InputSnapshot.from_payloads(directory, payloads),
        summary=MappingProxyType(dict(summary)),
        config=MappingProxyType({key: MappingProxyType(dict(value)) for key, value in config.items()}),
        streams=MappingProxyType(
            {key: tuple(MappingProxyType(dict(item)) for item in value) for key, value in streams.items()}
        ),
    )


_MRA_FIELDS = {
    "schema_version",
    "evaluation_mode",
    "primary_eligible",
    "included_in_final_research_aggregation",
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "event_id",
    "step_id",
    "turn_id",
    "turn_index",
    "timestamp",
    "agent_id",
    "top_k",
    "retrieved_count",
    "candidate_count",
    "selected_memory_ids",
    "retrieved_categories",
    "expected_categories",
    "matched_memory_ids",
    "unmatched_memory_ids",
    "matched_categories",
    "matched_count",
    "denominator",
    "mra",
}
_BCR_EXPORT_FIELDS = {
    "schema_version",
    "evaluation_mode",
    "primary_eligible",
    "included_in_final_research_aggregation",
    "audit_metadata",
    "selected_memory_evidence",
    "prior_behavior_evidence",
    "model_metadata",
    "judge_input",
}
_EVALUATION_SUMMARY_FIELDS = {
    "schema_version",
    "source_schema_version",
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "source_run_status",
    "evaluation_mode",
    "primary_eligible",
    "included_in_final_research_aggregation",
    "mra_formula",
    "mra_aggregates",
    "bcr_export_count",
    "bcr_judgment",
    "model_metadata",
    "output_files",
}
_UNIT_IDENTITY_FIELDS = {
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "event_id",
    "step_id",
    "turn_id",
    "turn_index",
    "timestamp",
    "agent_id",
}


def load_evaluation_overlay(
    evaluation_dir: str | Path,
    *,
    run_summary: Mapping[str, Any],
) -> LoadedOverlay:
    directory = resolve_input_directory(evaluation_dir, label="evaluation")
    payloads = _read_exact_directory(directory, EVALUATION_FILENAMES)
    summary = _parse_json(payloads["evaluation_summary.json"], "evaluation_summary.json")
    _require_fields(summary, _EVALUATION_SUMMARY_FIELDS, "evaluation_summary.json")
    if summary["schema_version"] != SOURCE_SCHEMA_VERSION or summary["source_schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ReplayOverlayError("evaluation schema version is unsupported")
    identity = tuple(summary[name] for name in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    ))
    expected = tuple(run_summary[name] for name in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    ))
    if identity != expected or summary["source_run_status"] != run_summary["status"]:
        raise ReplayOverlayError("evaluation identity differs from the source run")
    expected_mode = "primary" if run_summary["status"] == "completed" else "diagnostic"
    expected_count = 20 if expected_mode == "primary" else 4
    expected_eligible = expected_mode == "primary"
    if (
        summary["evaluation_mode"] != expected_mode
        or summary["bcr_export_count"] != expected_count
        or summary["primary_eligible"] is not expected_eligible
        or summary["included_in_final_research_aggregation"] is not expected_eligible
    ):
        raise ReplayOverlayError("evaluation eligibility or unit count is invalid")

    mra = _parse_jsonl(
        payloads["mra_records.jsonl"],
        "mra_records.jsonl",
        record_limit=20,
        exact_fields=_MRA_FIELDS,
    )
    bcr = _parse_jsonl(
        payloads["bcr_export.jsonl"],
        "bcr_export.jsonl",
        record_limit=20,
        exact_fields=_BCR_EXPORT_FIELDS,
    )
    if len(mra) != expected_count or len(bcr) != expected_count:
        raise ReplayOverlayError("evaluation record counts do not match the fixed scope")
    bcr_by_turn: dict[str, tuple[dict[str, Any], str]] = {}
    for record, line in zip(bcr, payloads["bcr_export.jsonl"].splitlines(keepends=True), strict=True):
        audit = record["audit_metadata"]
        if not isinstance(audit, Mapping):
            raise ReplayOverlayError("BCR audit metadata must be an object")
        _require_fields(audit, _UNIT_IDENTITY_FIELDS, "BCR audit metadata")
        if audit["turn_id"] in bcr_by_turn:
            raise ReplayOverlayError("BCR export contains a duplicate target unit")
        if tuple(audit[name] for name in (
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "repetition_index",
        )) != expected:
            raise ReplayOverlayError("BCR export identity differs from the source run")
        bcr_by_turn[audit["turn_id"]] = (record, sha256(line).hexdigest())
    units: list[dict[str, Any]] = []
    seen_mra_turns: set[str] = set()
    for record in mra:
        if tuple(record[name] for name in (
            "experiment_id",
            "run_id",
            "retrieval_config_id",
            "repetition_index",
        )) != expected:
            raise ReplayOverlayError("MRA row identity differs from the source run")
        if (
            record["evaluation_mode"] != expected_mode
            or record["primary_eligible"] is not expected_eligible
            or record["included_in_final_research_aggregation"] is not expected_eligible
            or record["turn_id"] in seen_mra_turns
        ):
            raise ReplayOverlayError("MRA unit eligibility or identity is invalid")
        seen_mra_turns.add(record["turn_id"])
        bcr_entry = bcr_by_turn.get(record["turn_id"])
        if bcr_entry is None:
            raise ReplayOverlayError("MRA/BCR export unit identities do not match")
        audit = bcr_entry[0]["audit_metadata"]
        if any(audit[name] != record[name] for name in _UNIT_IDENTITY_FIELDS):
            raise ReplayOverlayError("MRA/BCR export unit metadata differ")
        for name in (
            "selected_memory_ids",
            "expected_categories",
            "matched_memory_ids",
            "unmatched_memory_ids",
            "matched_categories",
        ):
            _require_string_list(record[name], name, empty=True)
        _require_number(record["mra"], "persisted mra")
        units.append(
            {
                "event_id": record["event_id"],
                "step_id": record["step_id"],
                "turn_id": record["turn_id"],
                "agent_id": record["agent_id"],
                "logical_timestamp": record["timestamp"],
                "mra": record["mra"],
                "matched_count": record["matched_count"],
                "denominator": record["denominator"],
                "expected_categories": list(record["expected_categories"]),
                "matched_memory_ids": list(record["matched_memory_ids"]),
                "bcr_status": "not_judged",
                "source_bcr_record_sha256": bcr_entry[1],
            }
        )
    value = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "evaluation_mode": summary["evaluation_mode"],
        "primary_eligible": summary["primary_eligible"],
        "included_in_final_research_aggregation": summary[
            "included_in_final_research_aggregation"
        ],
        "ground_truth_visibility": "evaluation_overlay_only_not_agent_observation",
        "units": units,
        "judgment": None,
        "source_bcr_export_sha256": sha256(payloads["bcr_export.jsonl"]).hexdigest(),
    }
    return LoadedOverlay(InputSnapshot.from_payloads(directory, payloads), MappingProxyType(value))


_JUDGMENT_SUMMARY_FIELDS = {
    "schema_version",
    "source_evaluation_schema_version",
    "experiment_id",
    "run_id",
    "retrieval_config_id",
    "repetition_index",
    "evaluation_mode",
    "primary_eligible",
    "included_in_final_research_aggregation",
    "source_bcr_export_hash",
    "judged_unit_count",
    "bcr_formula",
    "run_aggregation",
    "run_mean_bcr",
    "model_metadata",
    "prompt_identity",
    "runtime_settings",
    "provider_attempt_count",
    "usage",
    "output_files",
}
_JUDGED_ROW_FIELDS = {
    "schema_version",
    "evaluation_mode",
    "primary_eligible",
    "included_in_final_research_aggregation",
    "audit_metadata",
    "source_record_hash",
    "judgment",
    "model_metadata",
    "prompt_identity",
    "provider_attempts",
}


def attach_judgment_overlay(
    evaluation: LoadedOverlay,
    judgment_dir: str | Path,
    *,
    run_summary: Mapping[str, Any],
) -> tuple[LoadedOverlay, LoadedOverlay]:
    if run_summary["status"] != "completed":
        raise ReplayOverlayError("judged BCR overlay is not permitted for an excluded pilot")
    directory = resolve_input_directory(judgment_dir, label="judgment")
    payloads = _read_exact_directory(directory, JUDGMENT_FILENAMES)
    summary = _parse_json(payloads["bcr_judgment_summary.json"], "bcr_judgment_summary.json")
    _require_fields(summary, _JUDGMENT_SUMMARY_FIELDS, "bcr_judgment_summary.json")
    if summary["schema_version"] != SOURCE_SCHEMA_VERSION or summary[
        "source_evaluation_schema_version"
    ] != SOURCE_SCHEMA_VERSION:
        raise ReplayOverlayError("judgment schema version is unsupported")
    if (
        summary["evaluation_mode"] != "primary"
        or summary["primary_eligible"] is not True
        or summary["included_in_final_research_aggregation"] is not True
    ):
        raise ReplayOverlayError("judgment overlay is not primary persisted output")
    if summary["source_bcr_export_hash"] != evaluation.value["source_bcr_export_sha256"]:
        raise ReplayOverlayError("judgment does not reference the supplied BCR export")
    expected = tuple(run_summary[name] for name in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    ))
    if tuple(summary[name] for name in (
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
    )) != expected:
        raise ReplayOverlayError("judgment identity differs from the source run")
    rows = _parse_jsonl(
        payloads["bcr_judged_records.jsonl"],
        "bcr_judged_records.jsonl",
        record_limit=20,
        exact_fields=_JUDGED_ROW_FIELDS,
    )
    if len(rows) != 20 or summary["judged_unit_count"] != 20:
        raise ReplayOverlayError("primary judgment must contain exactly 20 units")
    by_turn: dict[str, dict[str, Any]] = {}
    evaluation_units = {
        unit["turn_id"]: unit for unit in evaluation.value["units"]
    }
    for row in rows:
        audit = row["audit_metadata"]
        if not isinstance(audit, Mapping):
            raise ReplayOverlayError("judged audit metadata must be an object")
        _require_fields(audit, _UNIT_IDENTITY_FIELDS, "judged audit metadata")
        if (
            row["evaluation_mode"] != "primary"
            or row["primary_eligible"] is not True
            or row["included_in_final_research_aggregation"] is not True
            or audit["turn_id"] in by_turn
        ):
            raise ReplayOverlayError("judged BCR unit eligibility or identity is invalid")
        source_unit = evaluation_units.get(audit["turn_id"])
        if (
            source_unit is None
            or row["source_record_hash"] != source_unit["source_bcr_record_sha256"]
            or any(
                audit[name] != source_unit[alias]
                for name, alias in (
                    ("event_id", "event_id"),
                    ("step_id", "step_id"),
                    ("turn_id", "turn_id"),
                    ("agent_id", "agent_id"),
                    ("timestamp", "logical_timestamp"),
                )
            )
        ):
            raise ReplayOverlayError("judged BCR unit does not join the evaluation export")
        judgment = row["judgment"]
        if not isinstance(judgment, Mapping) or "bcr" not in judgment:
            raise ReplayOverlayError("judged row is missing its persisted BCR")
        _require_number(judgment["bcr"], "persisted bcr")
        for name in (
            "communication_style_score",
            "focus_consistency_score",
            "temporal_consistency_score",
        ):
            _require_number(judgment.get(name), f"persisted {name}")
        for name in (
            "communication_style_reason",
            "focus_consistency_reason",
            "temporal_consistency_reason",
        ):
            _require_text(judgment.get(name), f"persisted {name}")
        by_turn[audit["turn_id"]] = {
            "bcr_status": "judged",
            "bcr": judgment["bcr"],
            "communication_style_score": judgment.get("communication_style_score"),
            "focus_consistency_score": judgment.get("focus_consistency_score"),
            "temporal_consistency_score": judgment.get("temporal_consistency_score"),
            "communication_style_reason": judgment.get("communication_style_reason"),
            "focus_consistency_reason": judgment.get("focus_consistency_reason"),
            "temporal_consistency_reason": judgment.get("temporal_consistency_reason"),
        }
    value = dict(evaluation.value)
    units = []
    for unit in evaluation.value["units"]:
        merged = dict(unit)
        judged = by_turn.get(unit["turn_id"])
        if judged is None:
            raise ReplayOverlayError("judgment units do not match evaluation units")
        merged.update(judged)
        units.append(merged)
    value["units"] = units
    value["judgment"] = {
        "schema_version": summary["schema_version"],
        "judged_unit_count": summary["judged_unit_count"],
        "run_mean_bcr": summary["run_mean_bcr"],
        "model_metadata": summary["model_metadata"],
        "usage": summary["usage"],
    }
    return (
        LoadedOverlay(evaluation.snapshot, MappingProxyType(value)),
        LoadedOverlay(InputSnapshot.from_payloads(directory, payloads), MappingProxyType(dict(summary))),
    )


_ATTEMPT_FIELDS = {
    "schema_version",
    "service_role",
    "logical_call_id",
    "attempt_index",
    "call_type",
    "model",
    "prompt_template",
    "prompt_version",
    "prompt_hash",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "status",
    "error_type",
}


def _content_hash(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReplayOverlayError("pilot content is not strict JSON") from exc
    return sha256(payload).hexdigest()


def _validate_pilot_manifest(
    manifest: Mapping[str, Any],
    *,
    run: LoadedRunArtifacts,
) -> Mapping[str, Any]:
    _require_fields(manifest, _PILOT_MANIFEST_FIELDS, "pilot_manifest.json")
    base = dict(manifest)
    manifest_hash = base.pop("manifest_hash")
    if not isinstance(manifest_hash, str) or _HASH.fullmatch(manifest_hash) is None:
        raise ReplayOverlayError("pilot manifest hash is invalid")
    if _content_hash(base) != manifest_hash:
        raise ReplayOverlayError("pilot manifest content hash is invalid")
    expected = tuple(run.summary[name] for name in ("experiment_id", "run_id"))
    if (
        manifest["schema_version"] != SOURCE_SCHEMA_VERSION
        or tuple(manifest[name] for name in ("experiment_id", "run_id")) != expected
        or manifest["pilot_id"] != run.summary["run_id"]
        or manifest["pilot_only"] is not True
        or manifest["primary_eligible"] is not False
        or manifest["included_in_final_research_matrix"] is not False
        or manifest["expected_source_run_status"] != "excluded"
    ):
        raise ReplayOverlayError("pilot manifest identity or eligibility is invalid")
    plan = manifest["plan"]
    if not isinstance(plan, Mapping):
        raise ReplayOverlayError("pilot plan must be an object")
    _require_fields(plan, _PILOT_PLAN_FIELDS, "pilot plan")
    if (
        plan["mode"] not in {"dummy", "real"}
        or plan["retrieval_config_id"] != run.summary["retrieval_config_id"]
        or plan["repetition_index"] != run.summary["repetition_index"]
        or plan["days"] != [1, 2]
        or plan["step_ids"]
        != [
            "day_01_natural_01",
            "day_01_natural_02",
            "day_02_natural_03",
            "day_02_target_01",
        ]
        or (
            plan["natural_block_count"],
            plan["target_event_count"],
            plan["main_turn_count"],
            plan["post_step_reflection_count"],
            plan["target_evaluation_unit_count"],
        )
        != (3, 1, 40, 16, 4)
    ):
        raise ReplayOverlayError("pilot plan differs from the fixed Day 1-2 scope")
    call_limits = plan["call_limits"]
    if not isinstance(call_limits, Mapping) or call_limits != {
        "simulator_logical_calls": 152,
        "embedding_logical_calls": 108,
        "evaluator_logical_calls": 0,
        "total_logical_calls": 260,
        "simulator_physical_attempts": 304,
        "embedding_physical_attempts": 216,
        "evaluator_physical_attempts": 0,
        "total_physical_attempts": 520,
    }:
        raise ReplayOverlayError("pilot call limits differ from the fixed contract")
    if plan["mode"] == "real" and (
        plan["retrieval_config_id"], plan["repetition_index"]
    ) != ("balanced", 1):
        raise ReplayOverlayError("real pilot scope is not balanced repetition 1")
    maximum = manifest["max_total_tokens"]
    if plan["mode"] == "real":
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ReplayOverlayError("real pilot manifest has no positive token ceiling")
    elif maximum is not None:
        raise ReplayOverlayError("dummy pilot manifest must not claim a token ceiling")
    configs = manifest["configuration_hashes"]
    prompts = manifest["prompts"]
    if not isinstance(configs, list) or len(configs) != 6 or not isinstance(prompts, list) or len(prompts) != 6:
        raise ReplayOverlayError("pilot manifest freeze inventory is incomplete")
    for item in configs:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"filename", "sha256"}
            or not isinstance(item["sha256"], str)
            or _HASH.fullmatch(item["sha256"]) is None
        ):
            raise ReplayOverlayError("pilot configuration hash is invalid")
    for item in prompts:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "template_id",
                "filename",
                "prompt_version",
                "prompt_hash",
                "model_role",
                "output_contract",
            }
            or not isinstance(item["prompt_hash"], str)
            or _HASH.fullmatch(item["prompt_hash"]) is None
        ):
            raise ReplayOverlayError("pilot prompt identity is invalid")
    return plan


def _validate_pilot_usage(
    usage: Mapping[str, Any],
    attempts: tuple[dict[str, Any], ...],
) -> None:
    _require_fields(usage, _PILOT_USAGE_FIELDS, "pilot_usage_summary.json")
    roles = usage["role_usage"]
    if not isinstance(roles, list) or len(roles) != 3:
        raise ReplayOverlayError("pilot role usage inventory is invalid")
    for role, expected_role in zip(roles, ("simulator", "embedding", "evaluator"), strict=True):
        if not isinstance(role, Mapping):
            raise ReplayOverlayError("pilot role usage row must be an object")
        _require_fields(role, _PILOT_ROLE_USAGE_FIELDS, "pilot role usage")
        if role["service_role"] != expected_role:
            raise ReplayOverlayError("pilot role usage order is invalid")
        for name in _PILOT_ROLE_USAGE_FIELDS - {"service_role"}:
            _require_int(role[name], f"pilot role usage {name}")
        if (
            role["retry_count"]
            != role["physical_attempt_count"] - role["logical_call_count"]
            or role["success_count"] + role["failed_count"] + role["invalid_output_count"]
            != role["physical_attempt_count"]
        ):
            raise ReplayOverlayError("pilot role usage arithmetic is invalid")
        selected = [item for item in attempts if item["service_role"] == expected_role]
        logical = {(item["service_role"], item["logical_call_id"]) for item in selected}
        if (
            role["logical_call_count"] != len(logical)
            or role["physical_attempt_count"] != len(selected)
            or role["input_tokens"]
            != sum(item["input_tokens"] for item in selected if item["input_tokens"] is not None)
            or role["output_tokens"]
            != sum(item["output_tokens"] for item in selected if item["output_tokens"] is not None)
            or role["total_tokens"]
            != sum(item["total_tokens"] for item in selected if item["total_tokens"] is not None)
            or role["missing_total_usage_attempt_count"]
            != sum(item["total_tokens"] is None for item in selected)
            or role["success_count"] != sum(item["status"] == "success" for item in selected)
            or role["failed_count"] != sum(item["status"] == "failed" for item in selected)
            or role["invalid_output_count"]
            != sum(item["status"] == "invalid_output" for item in selected)
        ):
            raise ReplayOverlayError("pilot role usage differs from persisted attempts")
    for name in (
        "reported_total_tokens",
        "logical_call_count",
        "physical_attempt_count",
        "retry_count",
        "missing_total_usage_attempt_count",
    ):
        _require_int(usage[name], f"pilot usage {name}")
    if not isinstance(usage["budget_exceeded"], bool):
        raise ReplayOverlayError("pilot budget status must be boolean")
    if (
        usage["reported_total_tokens"] != sum(role["total_tokens"] for role in roles)
        or usage["logical_call_count"] != sum(role["logical_call_count"] for role in roles)
        or usage["physical_attempt_count"] != len(attempts)
        or usage["physical_attempt_count"] != sum(role["physical_attempt_count"] for role in roles)
        or usage["retry_count"] != usage["physical_attempt_count"] - usage["logical_call_count"]
        or usage["missing_total_usage_attempt_count"]
        != sum(role["missing_total_usage_attempt_count"] for role in roles)
    ):
        raise ReplayOverlayError("pilot aggregate usage arithmetic is invalid")
    maximum = usage["max_total_tokens"]
    if maximum is not None:
        _require_int(maximum, "pilot max_total_tokens", minimum=1)
    if usage["budget_exceeded"] is not (
        maximum is not None and usage["reported_total_tokens"] > maximum
    ):
        raise ReplayOverlayError("pilot budget status is inconsistent")


def _validate_attempts(attempts: tuple[dict[str, Any], ...]) -> int:
    seen: set[tuple[str, str, int]] = set()
    logical: set[tuple[str, str]] = set()
    prior_logical: tuple[str, str] | None = None
    prior_index = 0
    missing = 0
    for attempt in attempts:
        role = attempt["service_role"]
        if role not in {"simulator", "embedding"}:
            raise ReplayOverlayError("Phase 11 attempts cannot use the evaluator role")
        logical_id = _require_identifier(attempt["logical_call_id"], "logical_call_id")
        attempt_index = _require_int(attempt["attempt_index"], "attempt_index", minimum=1)
        if attempt_index not in {1, 2}:
            raise ReplayOverlayError("pilot attempt index exceeds the two-attempt policy")
        key = (role, logical_id, attempt_index)
        logical_key = (role, logical_id)
        if key in seen:
            raise ReplayOverlayError("pilot attempts contain a duplicate identity")
        if attempt_index == 1:
            if logical_key in logical:
                raise ReplayOverlayError("pilot logical operation is duplicated")
            logical.add(logical_key)
        elif prior_logical != logical_key or prior_index != 1:
            raise ReplayOverlayError("pilot retry does not immediately follow attempt one")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = attempt[name]
            if value is not None:
                _require_int(value, f"attempt {name}")
        if attempt["total_tokens"] is None:
            missing += 1
        elif attempt["input_tokens"] is not None and attempt["total_tokens"] != (
            attempt["input_tokens"] + (attempt["output_tokens"] or 0)
        ):
            raise ReplayOverlayError("pilot attempt token arithmetic is invalid")
        if attempt["status"] not in {"success", "failed", "invalid_output"}:
            raise ReplayOverlayError("pilot attempt status is invalid")
        _require_identifier(attempt["call_type"], "attempt call_type")
        if not isinstance(attempt["model"], str) or ":" not in attempt["model"]:
            raise ReplayOverlayError("pilot attempt model identity is invalid")
        prompt_values = (
            attempt["prompt_template"],
            attempt["prompt_version"],
            attempt["prompt_hash"],
        )
        if role == "embedding":
            if any(value is not None for value in prompt_values):
                raise ReplayOverlayError("embedding attempt must not claim prompt identity")
        elif (
            any(value is None for value in prompt_values)
            or not isinstance(attempt["prompt_hash"], str)
            or _HASH.fullmatch(attempt["prompt_hash"]) is None
        ):
            raise ReplayOverlayError("pilot simulator attempt prompt identity is invalid")
        if attempt["status"] == "success":
            if attempt["error_type"] is not None:
                raise ReplayOverlayError("successful pilot attempt has an error type")
        else:
            _require_identifier(attempt["error_type"], "attempt error_type")
        seen.add(key)
        prior_logical = logical_key
        prior_index = attempt_index
    return missing


def _directory_fingerprint(directory: Path) -> str:
    files: list[Path] = []
    try:
        for item in directory.rglob("*"):
            if item.is_symlink():
                raise ReplaySecurityError("fingerprinted input contains a symbolic link")
            if item.is_file():
                files.append(item)
    except OSError as exc:
        raise ReplayInputError("could not inspect fingerprinted directory") from exc
    digest = sha256()
    for item in sorted(files, key=lambda path: path.relative_to(directory).as_posix()):
        relative = item.relative_to(directory).as_posix().encode("utf-8")
        payload = _read_regular_file(item)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_pilot_overlay(
    evidence_dir: str | Path,
    *,
    run: LoadedRunArtifacts,
    review_dir: str | Path | None = None,
) -> tuple[LoadedOverlay, LoadedOverlay | None]:
    if run.summary["status"] not in {"excluded", "failed", "interrupted"}:
        raise ReplayOverlayError(
            "pilot evidence requires an excluded or terminally partial source run"
        )
    evidence = resolve_input_directory(evidence_dir, label="pilot evidence")
    payloads = _read_exact_directory(evidence, PILOT_EVIDENCE_FILENAMES)
    manifest = _parse_json(payloads["pilot_manifest.json"], "pilot_manifest.json")
    usage = _parse_json(payloads["pilot_usage_summary.json"], "pilot_usage_summary.json")
    report = _parse_json(payloads["pilot_report.json"], "pilot_report.json")
    attempts = _parse_jsonl(
        payloads["pilot_attempts.jsonl"],
        "pilot_attempts.jsonl",
        record_limit=520,
        exact_fields=_ATTEMPT_FIELDS,
    )
    for value, label in ((manifest, "manifest"), (usage, "usage"), (report, "report")):
        if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise ReplayOverlayError(f"pilot {label} schema version is unsupported")
    plan = _validate_pilot_manifest(manifest, run=run)
    missing = _validate_attempts(attempts)
    _validate_pilot_usage(usage, attempts)
    _require_fields(report, _PILOT_REPORT_FIELDS, "pilot_report.json")
    expected = tuple(run.summary[name] for name in ("experiment_id", "run_id"))
    if tuple(report.get(name) for name in ("experiment_id", "run_id")) != expected or tuple(
        manifest.get(name) for name in ("experiment_id", "run_id")
    ) != expected:
        raise ReplayOverlayError("pilot evidence identity differs from the source run")
    if (
        report.get("acceptance_state") != "pending"
        or report.get("manifest_hash") != manifest.get("manifest_hash")
        or report.get("pilot_id") != manifest.get("pilot_id")
        or report.get("mode") != plan.get("mode")
        or report.get("usage") != usage
        or report.get("source_run_directory_name") != run.summary["run_id"]
        or report.get("diagnostic_evaluation_directory_name")
        != f"{run.summary['run_id']}_diagnostic_evaluation"
        or report.get("source_run_status") != run.summary["status"]
        or report.get("primary_eligible") is not False
        or report.get("included_in_final_research_matrix") is not False
        or report.get("bcr_judgment_count") != 0
    ):
        raise ReplayOverlayError("pilot report identity or eligibility is invalid")
    if report.get("acceptance_state") != "pending":
        raise ReplayOverlayError("execution evidence must retain pending acceptance state")
    if usage.get("physical_attempt_count") != len(attempts):
        raise ReplayOverlayError("pilot attempt count differs from the usage summary")
    if usage.get("missing_total_usage_attempt_count") != missing:
        raise ReplayOverlayError("pilot missing-usage count is inconsistent")
    technical_success = report.get("technical_success") is True
    if technical_success:
        if (
            report.get("source_run_status") != "excluded"
            or report.get("mra_diagnostic_record_count") != 4
            or report.get("bcr_export_record_count") != 4
            or report.get("terminal_error_type") is not None
            or report.get("rejection_reasons") != []
            or report.get("validation") is None
            or report.get("diagnostic_evaluation_fingerprint") is None
            or report.get("source_fingerprint") is None
            or report.get("eligible_for_manual_acceptance")
            is not (plan.get("mode") == "real" and missing == 0)
        ):
            raise ReplayOverlayError("successful pilot report is internally inconsistent")
        if missing != 0:
            raise ReplayOverlayError("technical success cannot coexist with missing provider usage")
    elif (
        report.get("eligible_for_manual_acceptance") is not False
        or not isinstance(report.get("rejection_reasons"), list)
        or not report.get("rejection_reasons")
        or report.get("terminal_error_type") is None
    ):
        raise ReplayOverlayError("failed pilot report is internally inconsistent")
    if not technical_success:
        terminal_error_type = _require_pilot_failure_identifier(
            report.get("terminal_error_type"),
            "pilot terminal_error_type",
        )
        rejection_reasons = report.get("rejection_reasons")
        assert isinstance(rejection_reasons, list)  # established immediately above
        for index, reason in enumerate(rejection_reasons):
            _require_pilot_failure_identifier(
                reason,
                f"pilot rejection_reasons[{index}]",
            )
        source_errors = run.streams["errors.jsonl"]
        if run.summary["status"] in {"failed", "interrupted"} and source_errors:
            source_error_type = _require_pilot_failure_identifier(
                source_errors[0]["error_type"],
                "source terminal error_type",
            )
            if terminal_error_type != source_error_type:
                raise ReplayOverlayError(
                    "pilot terminal error type contradicts source failure evidence"
                )
    source_fingerprint = report.get("source_fingerprint")
    if source_fingerprint is not None and source_fingerprint != run.snapshot.aggregate_fingerprint:
        # Phase 11 uses the same length-prefixed directory fingerprint. Recompute
        # from the actual directory so the diagnostic remains exact.
        if source_fingerprint != _directory_fingerprint(run.snapshot.directory):
            raise ReplayOverlayError("pilot evidence references changed source artifacts")

    safe_attempts = [
        {
            "service_role": item["service_role"],
            "logical_call_id": item["logical_call_id"],
            "attempt_index": item["attempt_index"],
            "call_type": item["call_type"],
            "model": item["model"],
            "prompt_template_id": item["prompt_template"],
            "prompt_version": item["prompt_version"],
            "prompt_hash": item["prompt_hash"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "total_tokens": item["total_tokens"],
            "status": item["status"],
            "error_type": item["error_type"],
        }
        for item in attempts
    ]
    acceptance_state = "pending"
    review_value: dict[str, Any] | None = None
    review_overlay: LoadedOverlay | None = None
    if review_dir is not None:
        review = resolve_input_directory(review_dir, label="pilot review")
        review_entries = _directory_entries(review)
        names = {item.name for item in review_entries}
        if names not in (
            {"pilot_acceptance.json"},
            {"pilot_acceptance.json", "freeze_manifest.json"},
        ) or any(not item.is_file() for item in review_entries):
            raise ReplayOverlayError("pilot review directory layout is invalid")
        review_payloads = {item.name: _read_regular_file(item) for item in review_entries}
        acceptance = _parse_json(review_payloads["pilot_acceptance.json"], "pilot_acceptance.json")
        _require_fields(acceptance, _PILOT_ACCEPTANCE_FIELDS, "pilot_acceptance.json")
        acceptance_base = dict(acceptance)
        acceptance_hash = acceptance_base.pop("acceptance_hash")
        if (
            acceptance.get("schema_version") != SOURCE_SCHEMA_VERSION
            or acceptance.get("manual_review") is not True
            or acceptance.get("acceptance_state") not in {"accepted", "rejected"}
            or tuple(acceptance.get(name) for name in ("experiment_id", "run_id")) != expected
            or acceptance.get("pilot_id") != report.get("pilot_id")
            or acceptance.get("manifest_hash") != manifest.get("manifest_hash")
            or acceptance.get("source_run_status") != report.get("source_run_status")
            or acceptance.get("primary_eligible") is not False
            or acceptance.get("included_in_final_research_matrix") is not False
            or acceptance.get("evidence_hash") != _directory_fingerprint(evidence)
            or not isinstance(acceptance_hash, str)
            or _HASH.fullmatch(acceptance_hash) is None
            or _content_hash(acceptance_base) != acceptance_hash
        ):
            raise ReplayOverlayError("pilot manual review identity is invalid")
        acceptance_state = acceptance["acceptance_state"]
        review_value = {
            "acceptance_state": acceptance_state,
            "manual_review": True,
            "acceptance_hash": acceptance_hash,
            "freeze": None,
        }
        if acceptance_state == "accepted":
            if (
                report.get("technical_success") is not True
                or report.get("eligible_for_manual_acceptance") is not True
                or report.get("mode") != "real"
                or missing != 0
                or usage.get("budget_exceeded") is not False
            ):
                raise ReplayOverlayError("accepted review references ineligible pilot evidence")
            if "freeze_manifest.json" not in review_payloads:
                raise ReplayOverlayError("accepted review is missing its freeze manifest")
            freeze = _parse_json(review_payloads["freeze_manifest.json"], "freeze_manifest.json")
            _require_fields(freeze, _PILOT_FREEZE_FIELDS, "freeze_manifest.json")
            freeze_base = dict(freeze)
            freeze_hash = freeze_base.pop("freeze_hash")
            if (
                freeze.get("schema_version") != SOURCE_SCHEMA_VERSION
                or freeze.get("freeze_state") != "accepted"
                or freeze.get("pilot_id") != report.get("pilot_id")
                or freeze.get("experiment_id") != report.get("experiment_id")
                or freeze.get("run_id") != run.summary["run_id"]
                or freeze.get("acceptance_hash") != acceptance_hash
                or freeze.get("evidence_hash") != acceptance.get("evidence_hash")
                or freeze.get("manifest_hash") != manifest.get("manifest_hash")
                or freeze.get("configuration_hashes") != manifest.get("configuration_hashes")
                or freeze.get("prompts") != manifest.get("prompts")
                or freeze.get("simulator_settings") != manifest.get("simulator_settings")
                or freeze.get("evaluator_settings") != manifest.get("evaluator_settings")
                or freeze.get("embedding_settings") != manifest.get("embedding_settings")
                or freeze.get("retry_policy") != manifest.get("retry_policy")
                or freeze.get("evaluation_schema_version")
                != manifest.get("evaluation_schema_version")
                or freeze.get("research_log_schema_version")
                != manifest.get("research_log_schema_version")
                or freeze.get("bcr_rubric_version") != manifest.get("bcr_rubric_version")
                or freeze.get("primary_eligible") is not False
                or freeze.get("included_in_final_research_matrix") is not False
                or freeze.get("phase13_source_and_git_verification_required") is not True
                or not isinstance(freeze_hash, str)
                or _HASH.fullmatch(freeze_hash) is None
                or _content_hash(freeze_base) != freeze_hash
            ):
                raise ReplayOverlayError("accepted freeze identity is invalid")
            review_value["freeze"] = {
                "freeze_state": "accepted",
                "freeze_hash": freeze_hash,
                "manifest_hash": freeze.get("manifest_hash"),
                "current_drift_validation": "not_performed_by_visualizer",
            }
        elif "freeze_manifest.json" in review_payloads:
            raise ReplayOverlayError("rejected review must not contain a freeze manifest")
        review_overlay = LoadedOverlay(
            InputSnapshot.from_payloads(review, review_payloads),
            MappingProxyType(review_value),
        )

    value = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "pilot_id": report.get("pilot_id"),
        "mode": report.get("mode"),
        "technical_success": report.get("technical_success"),
        "source_run_status": report.get("source_run_status"),
        "acceptance_state": acceptance_state,
        "eligible_for_manual_acceptance": report.get("eligible_for_manual_acceptance"),
        "primary_eligible": False,
        "included_in_final_research_matrix": False,
        "terminal_error_type": report.get("terminal_error_type"),
        "rejection_reasons": report.get("rejection_reasons"),
        "usage": {
            "max_total_tokens": usage.get("max_total_tokens"),
            "reported_known_total_tokens": usage.get("reported_total_tokens"),
            "logical_call_count": usage.get("logical_call_count"),
            "physical_attempt_count": usage.get("physical_attempt_count"),
            "retry_count": usage.get("retry_count"),
            "missing_total_usage_attempt_count": usage.get(
                "missing_total_usage_attempt_count"
            ),
            "budget_exceeded": usage.get("budget_exceeded"),
            "actual_total_tokens": (
                usage.get("reported_total_tokens") if missing == 0 else None
            ),
        },
        "attempts": safe_attempts,
        "review": review_value,
    }
    return (
        LoadedOverlay(InputSnapshot.from_payloads(evidence, payloads), MappingProxyType(value)),
        review_overlay,
    )


__all__ = [
    "InputSnapshot",
    "LoadedOverlay",
    "LoadedRunArtifacts",
    "MAX_INPUT_FILE_BYTES",
    "MAX_JSONL_LINE_BYTES",
    "MAX_TOTAL_INPUT_BYTES",
    "attach_judgment_overlay",
    "load_evaluation_overlay",
    "load_pilot_overlay",
    "load_run_artifacts",
    "resolve_input_directory",
]
