"""Strict read-only validation gates for one completed pilot scope."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluation import EvaluationError, load_evaluation_run
from src.llm.settings import SIMULATOR_MODEL_IDENTITY
from src.logging_system import RunSummaryError, read_run_summary
from src.logging_system import jsonl_logger as canonical_jsonl
from src.simulation.safeguards import (
    PILOT_MAIN_TURNS,
    PILOT_NATURAL_BLOCKS,
    PILOT_POST_STEP_REFLECTIONS,
    PILOT_TARGET_EVALUATION_UNITS,
    PILOT_TARGET_EVENTS,
    PilotError,
)


_MANDATORY_STREAMS = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)
_SOURCE_ENTRIES = frozenset(
    (*_MANDATORY_STREAMS, "run_summary.json", "config_snapshot")
)
_FORBIDDEN_PERSISTED_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "credential",
        "credentials",
        "authorization",
        "authorization_header",
        "environment",
        "environment_variables",
        "rendered_prompt",
        "system_instructions",
        "user_data",
        "raw_request",
        "raw_provider_request",
        "raw_response",
        "raw_provider_response",
        "provider_response",
        "provider_exception",
        "raw_exception",
        "currency_cost",
        "monetary_cost",
        "usd_cost",
    }
)


class PilotRunValidationError(PilotError):
    """Raised when a source pilot cannot pass the post-run gates."""


class _DuplicateArtifactKeyError(ValueError):
    pass


def _strict_artifact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateArtifactKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


@dataclass(frozen=True)
class PilotRunValidation:
    source_run_status: str
    natural_block_count: int
    target_event_count: int
    main_turn_count: int
    retrieval_record_count: int
    behavior_record_count: int
    memory_record_count: int
    reflection_count: int
    error_record_count: int
    api_usage_record_count: int
    target_evaluation_unit_count: int
    simulator_model_identity: str | None
    top_k_validated: bool
    secret_scan_passed: bool
    source_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.source_run_status != "excluded"
            or (
                self.natural_block_count,
                self.target_event_count,
                self.main_turn_count,
                self.retrieval_record_count,
                self.behavior_record_count,
                self.memory_record_count,
                self.reflection_count,
                self.error_record_count,
                self.target_evaluation_unit_count,
            )
            != (3, 1, 40, 40, 40, 68, 16, 0, 4)
            or self.top_k_validated is not True
            or self.secret_scan_passed is not True
            or not isinstance(self.source_fingerprint, str)
            or len(self.source_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.source_fingerprint)
        ):
            raise PilotRunValidationError("pilot validation result is internally inconsistent")
        if self.simulator_model_identity is None:
            if self.api_usage_record_count != 0:
                raise PilotRunValidationError("dummy pilot validation claims API usage")
        elif (
            self.simulator_model_identity != SIMULATOR_MODEL_IDENTITY
            or not 152 <= self.api_usage_record_count <= 304
        ):
            raise PilotRunValidationError("real pilot validation has invalid provider usage")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_run_status": self.source_run_status,
            "natural_block_count": self.natural_block_count,
            "target_event_count": self.target_event_count,
            "main_turn_count": self.main_turn_count,
            "retrieval_record_count": self.retrieval_record_count,
            "behavior_record_count": self.behavior_record_count,
            "memory_record_count": self.memory_record_count,
            "reflection_count": self.reflection_count,
            "error_record_count": self.error_record_count,
            "api_usage_record_count": self.api_usage_record_count,
            "target_evaluation_unit_count": self.target_evaluation_unit_count,
            "simulator_model_identity": self.simulator_model_identity,
            "top_k_validated": self.top_k_validated,
            "secret_scan_passed": self.secret_scan_passed,
            "source_fingerprint": self.source_fingerprint,
        }


def fingerprint_directory(path: str | Path) -> str:
    """Hash exact relative filenames and bytes without following symlinks."""
    try:
        directory = Path(path)
    except (TypeError, ValueError) as exc:
        raise PilotRunValidationError("invalid directory for fingerprinting") from exc
    if directory.is_symlink() or not directory.is_dir():
        raise PilotRunValidationError("fingerprinted path must be a real directory")
    digest = sha256()
    files: list[Path] = []
    try:
        for item in directory.rglob("*"):
            if item.is_symlink():
                raise PilotRunValidationError(
                    "fingerprinted directory must not contain symbolic links"
                )
            if item.is_file():
                files.append(item)
    except OSError as exc:
        raise PilotRunValidationError("could not inspect fingerprinted directory") from exc
    if not files:
        raise PilotRunValidationError("fingerprinted directory must contain files")
    try:
        for item in sorted(files, key=lambda value: value.relative_to(directory).as_posix()):
            relative = item.relative_to(directory).as_posix().encode("utf-8")
            payload = item.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    except OSError as exc:
        raise PilotRunValidationError("could not read fingerprinted artifact") from exc
    return digest.hexdigest()


def _walk_json(value: object) -> tuple[tuple[str, object], ...]:
    items: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            items.append((str(key), item))
            items.extend(_walk_json(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(_walk_json(item))
    return tuple(items)


def validate_no_secret_artifacts(
    directory: str | Path,
    *,
    secret_value: str | None = None,
) -> None:
    """Reject secret-bearing fields and an exact credential value if supplied."""
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise PilotRunValidationError("secret scan requires a real directory")
    secret = (
        secret_value.encode("utf-8")
        if isinstance(secret_value, str) and secret_value
        else None
    )
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise PilotRunValidationError("secret scan refuses symbolic links")
        if not item.is_file():
            continue
        try:
            payload = item.read_bytes()
        except OSError as exc:
            raise PilotRunValidationError("secret scan could not read an artifact") from exc
        if secret is not None and secret in payload:
            raise PilotRunValidationError("credential content was found in pilot artifacts")
        if item.suffix not in {".json", ".jsonl"}:
            continue
        try:
            values = (
                [
                    json.loads(line, object_pairs_hook=_strict_artifact_object)
                    for line in payload.decode("utf-8").splitlines()
                ]
                if item.suffix == ".jsonl"
                else [
                    json.loads(
                        payload.decode("utf-8"),
                        object_pairs_hook=_strict_artifact_object,
                    )
                ]
            )
        except (UnicodeError, json.JSONDecodeError, _DuplicateArtifactKeyError) as exc:
            raise PilotRunValidationError(
                f"secret scan found malformed JSON in {item.name}"
            ) from exc
        for value in values:
            forbidden = sorted(
                key
                for key, _ in _walk_json(value)
                if key.lower() in _FORBIDDEN_PERSISTED_KEYS
            )
            if forbidden:
                raise PilotRunValidationError(
                    f"forbidden persisted security field in {item.name}: {forbidden[0]}"
                )


def _read_streams(run_dir: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    streams: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename in _MANDATORY_STREAMS:
        path = run_dir / filename
        if path.is_symlink() or not path.is_file():
            raise PilotRunValidationError(f"missing mandatory {filename}")
        try:
            streams[filename] = canonical_jsonl._read_existing_records(path)
        except Exception as exc:
            raise PilotRunValidationError(f"invalid {filename}: {type(exc).__name__}") from None
    return streams


def _validate_source_layout(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise PilotRunValidationError("source pilot must be a real run directory")
    try:
        entries = tuple(run_dir.iterdir())
    except OSError as exc:
        raise PilotRunValidationError("could not inspect source pilot layout") from exc
    actual_names = {item.name for item in entries}
    missing = sorted(_SOURCE_ENTRIES - actual_names)
    unexpected = sorted(actual_names - _SOURCE_ENTRIES)
    if missing:
        raise PilotRunValidationError(f"missing mandatory {missing[0]}")
    if unexpected:
        raise PilotRunValidationError(
            f"unexpected source pilot artifact: {unexpected[0]}"
        )
    for item in entries:
        if item.is_symlink():
            raise PilotRunValidationError("source pilot artifacts must not be symlinks")
        if item.name == "config_snapshot":
            if not item.is_dir():
                raise PilotRunValidationError("config_snapshot must be a directory")
        elif not item.is_file():
            raise PilotRunValidationError("source pilot streams must be regular files")


def _validate_real_api_usage(
    streams: Mapping[str, tuple[dict[str, Any], ...]],
) -> None:
    usage = streams["api_usage_logs.jsonl"]
    expected: dict[tuple[object, str], str] = {}
    for behavior in streams["behavior_logs.jsonl"]:
        response_call = (
            "target_event_response"
            if behavior["is_target_event"] is True
            else "natural_response"
        )
        for call_type, prompt_template in (
            ("plan", "plan"),
            (
                response_call,
                "target_event_response"
                if response_call == "target_event_response"
                else "natural_interaction",
            ),
            ("importance_score", "importance_scoring"),
        ):
            key = (behavior["turn_id"], call_type)
            if key in expected:
                raise PilotRunValidationError("duplicate expected simulator usage identity")
            expected[key] = prompt_template
    for reflection in streams["reflection_logs.jsonl"]:
        for call_type, prompt_template in (
            ("reflection", "reflect"),
            ("importance_score", "importance_scoring"),
        ):
            key = (reflection["turn_id"], call_type)
            if key in expected:
                raise PilotRunValidationError("duplicate expected reflection usage identity")
            expected[key] = prompt_template
    if len(expected) != 152 or not 152 <= len(usage) <= 304:
        raise PilotRunValidationError("real pilot API usage count is incomplete or excessive")
    grouped: dict[tuple[object, str], list[dict[str, Any]]] = {}
    for record in usage:
        key = (record["turn_id"], record["call_type"])
        if (
            key not in expected
            or record["model"] != SIMULATOR_MODEL_IDENTITY
            or record["prompt_template"] != expected[key]
            or record["status"] not in {"success", "failed", "invalid_output"}
            or (record["status"] == "success") != (record["error_type"] is None)
        ):
            raise PilotRunValidationError("real pilot API usage identity is invalid")
        grouped.setdefault(key, []).append(record)
    if set(grouped) != set(expected) or any(
        len(records) > 2
        or sum(record["status"] == "success" for record in records) != 1
        or records[-1]["status"] != "success"
        for records in grouped.values()
    ):
        raise PilotRunValidationError("real pilot API usage logical calls are incomplete")


def validate_pilot_run(
    run_dir: str | Path,
    *,
    mode: str,
    secret_value: str | None = None,
) -> PilotRunValidation:
    """Validate exact 3/1/40 excluded pilot artifacts without mutation."""
    if mode not in {"dummy", "real"}:
        raise PilotRunValidationError("pilot run mode must be dummy or real")
    source = Path(run_dir)
    _validate_source_layout(source)
    try:
        summary = read_run_summary(source / "run_summary.json")
    except RunSummaryError as exc:
        raise PilotRunValidationError("missing or invalid run_summary.json") from exc
    if summary.status != "excluded":
        raise PilotRunValidationError(
            "successful partial pilot source status must remain excluded"
        )
    actual_counts = (
        summary.completed_natural_blocks,
        summary.completed_target_events,
        summary.completed_main_turns,
        summary.error_count,
    )
    if actual_counts != (
        PILOT_NATURAL_BLOCKS,
        PILOT_TARGET_EVENTS,
        PILOT_MAIN_TURNS,
        0,
    ):
        raise PilotRunValidationError("pilot summary does not contain exact 3/1/40 counts")
    streams = _read_streams(source)
    expected_counts = {
        "retrieval_logs.jsonl": 40,
        "behavior_logs.jsonl": 40,
        "memory_logs.jsonl": 68,
        "reflection_logs.jsonl": 16,
        "errors.jsonl": 0,
        "event_summaries.jsonl": 1,
        "natural_block_summaries.jsonl": 3,
    }
    for filename, expected in expected_counts.items():
        if len(streams[filename]) != expected:
            raise PilotRunValidationError(
                f"{filename} record count must equal {expected}"
            )
    try:
        loaded = load_evaluation_run(source, diagnostic=True)
    except EvaluationError as exc:
        raise PilotRunValidationError(
            f"pilot target-event evaluation input is invalid: {type(exc).__name__}"
        ) from None
    if len(loaded.target_units) != PILOT_TARGET_EVALUATION_UNITS:
        raise PilotRunValidationError("pilot must expose exactly four target units")
    usage = streams["api_usage_logs.jsonl"]
    if mode == "dummy":
        if usage or loaded.simulator_model_identity is not None:
            raise PilotRunValidationError("dummy pilot must not claim provider usage")
    else:
        if loaded.simulator_model_identity != SIMULATOR_MODEL_IDENTITY:
            raise PilotRunValidationError("real pilot simulator identity mismatch")
        _validate_real_api_usage(streams)
    validate_no_secret_artifacts(source, secret_value=secret_value)
    return PilotRunValidation(
        source_run_status=summary.status,
        natural_block_count=summary.completed_natural_blocks,
        target_event_count=summary.completed_target_events,
        main_turn_count=summary.completed_main_turns,
        retrieval_record_count=len(streams["retrieval_logs.jsonl"]),
        behavior_record_count=len(streams["behavior_logs.jsonl"]),
        memory_record_count=len(streams["memory_logs.jsonl"]),
        reflection_count=len(streams["reflection_logs.jsonl"]),
        error_record_count=len(streams["errors.jsonl"]),
        api_usage_record_count=len(usage),
        target_evaluation_unit_count=len(loaded.target_units),
        simulator_model_identity=loaded.simulator_model_identity,
        top_k_validated=True,
        secret_scan_passed=True,
        source_fingerprint=fingerprint_directory(source),
    )


__all__ = [
    "PilotRunValidation",
    "PilotRunValidationError",
    "fingerprint_directory",
    "validate_no_secret_artifacts",
    "validate_pilot_run",
]
