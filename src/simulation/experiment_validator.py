"""Strict read-only validation for Phase 13 source/evaluation chains."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

from src.config_loader.loader import CONFIG_FILES
from src.config_loader.validators import FIXED_EVENT_IDS
from src.domain.agents import FIXED_AGENT_IDS
from src.evaluation import load_evaluation_run
from src.evaluation.judgment_reader import load_bcr_export
from src.evaluation.schemas import future_bcr_contract_dict
from src.llm.settings import SIMULATOR_MODEL_IDENTITY
from src.logging_system import read_run_summary
from src.logging_system import jsonl_logger as canonical_jsonl
from src.simulation.full_experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    MAX_MISSING_USAGE_ATTEMPTS,
    ExperimentContractError,
    ExperimentPlan,
    ExperimentRunSpec,
    FullRunScope,
)
from src.simulation.run_validator import validate_no_secret_artifacts


SOURCE_LOGS = (
    "retrieval_logs.jsonl",
    "behavior_logs.jsonl",
    "memory_logs.jsonl",
    "reflection_logs.jsonl",
    "errors.jsonl",
    "event_summaries.jsonl",
    "natural_block_summaries.jsonl",
    "api_usage_logs.jsonl",
)
SOURCE_LAYOUT = frozenset((*SOURCE_LOGS, "run_summary.json", "config_snapshot"))
EVALUATION_LAYOUT = frozenset(
    {"mra_records.jsonl", "bcr_export.jsonl", "evaluation_summary.json"}
)
_HASH_CHARS = frozenset("0123456789abcdef")


class ExperimentValidationError(ExperimentContractError):
    """Raised when a Phase 13 artifact chain is incomplete or corrupt."""


@dataclass(frozen=True)
class DirectoryFingerprint:
    directory_sha256: str
    files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            len(self.directory_sha256) != 64
            or any(character not in _HASH_CHARS for character in self.directory_sha256)
            or not self.files
        ):
            raise ExperimentValidationError("invalid directory fingerprint")
        names = tuple(item[0] for item in self.files)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ExperimentValidationError("fingerprint file order/identity is invalid")
        for name, digest in self.files:
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or len(digest) != 64
                or any(character not in _HASH_CHARS for character in digest)
            ):
                raise ExperimentValidationError("fingerprint contains an invalid file")

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory_sha256": self.directory_sha256,
            "files": [
                {"relative_path": name, "sha256": digest}
                for name, digest in self.files
            ],
        }


def fingerprint_directory(path: str | Path) -> DirectoryFingerprint:
    directory = Path(path)
    if directory.is_symlink() or not directory.is_dir():
        raise ExperimentValidationError("fingerprinted artifact must be a real directory")
    files: list[Path] = []
    try:
        for item in directory.rglob("*"):
            if item.is_symlink():
                raise ExperimentValidationError(
                    "fingerprinted artifacts must not contain symlinks"
                )
            if item.is_file():
                files.append(item)
    except OSError as exc:
        raise ExperimentValidationError("could not inspect artifact directory") from exc
    if not files:
        raise ExperimentValidationError("artifact directory contains no files")
    digest = sha256()
    inventory: list[tuple[str, str]] = []
    try:
        for item in sorted(files, key=lambda value: value.relative_to(directory).as_posix()):
            relative = item.relative_to(directory).as_posix()
            payload = item.read_bytes()
            file_digest = sha256(payload).hexdigest()
            inventory.append((relative, file_digest))
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    except OSError as exc:
        raise ExperimentValidationError("could not read artifact directory") from exc
    return DirectoryFingerprint(digest.hexdigest(), tuple(inventory))


@dataclass(frozen=True)
class SourceRunValidation:
    run_id: str
    source_status: str
    natural_block_count: int
    target_event_count: int
    step_count: int
    main_turn_count: int
    reflection_count: int
    memory_record_count: int
    target_evaluation_unit_count: int
    api_usage_record_count: int
    simulator_model_identity: str | None
    source_fingerprint: DirectoryFingerprint

    def __post_init__(self) -> None:
        if self.source_status != "completed":
            raise ExperimentValidationError("Phase 13 source status must be completed")
        if (
            self.natural_block_count,
            self.target_event_count,
            self.step_count,
            self.main_turn_count,
            self.reflection_count,
            self.memory_record_count,
            self.target_evaluation_unit_count,
        ) != (7, 5, 12, 104, 48, 164, 20):
            raise ExperimentValidationError("source validation counts are inconsistent")
        if self.simulator_model_identity is None:
            if self.api_usage_record_count != 0:
                raise ExperimentValidationError("dummy source must not claim API usage")
        elif (
            self.simulator_model_identity != SIMULATOR_MODEL_IDENTITY
            or not 408 <= self.api_usage_record_count <= 816
        ):
            raise ExperimentValidationError("real source API usage inventory is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_status": self.source_status,
            "natural_block_count": self.natural_block_count,
            "target_event_count": self.target_event_count,
            "step_count": self.step_count,
            "main_turn_count": self.main_turn_count,
            "reflection_count": self.reflection_count,
            "memory_record_count": self.memory_record_count,
            "target_evaluation_unit_count": self.target_evaluation_unit_count,
            "api_usage_record_count": self.api_usage_record_count,
            "simulator_model_identity": self.simulator_model_identity,
            "source_fingerprint": self.source_fingerprint.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationValidation:
    run_id: str
    evaluation_mode: str
    primary_eligible: bool
    mra_record_count: int
    bcr_export_record_count: int
    binary_future_bcr_contract_validated: bool
    condition_blind_judge_input_validated: bool
    evaluation_fingerprint: DirectoryFingerprint

    def __post_init__(self) -> None:
        if (
            self.evaluation_mode != "primary"
            or self.primary_eligible is not True
            or self.mra_record_count != 20
            or self.bcr_export_record_count != 20
            or self.binary_future_bcr_contract_validated is not True
            or self.condition_blind_judge_input_validated is not True
        ):
            raise ExperimentValidationError("evaluation validation result is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_mode": self.evaluation_mode,
            "primary_eligible": self.primary_eligible,
            "mra_record_count": self.mra_record_count,
            "bcr_export_record_count": self.bcr_export_record_count,
            "binary_future_bcr_contract_validated": (
                self.binary_future_bcr_contract_validated
            ),
            "condition_blind_judge_input_validated": (
                self.condition_blind_judge_input_validated
            ),
            "evaluation_fingerprint": self.evaluation_fingerprint.to_dict(),
        }


@dataclass(frozen=True)
class EligibleRunChain:
    run: ExperimentRunSpec
    source: SourceRunValidation
    evaluation: EvaluationValidation
    eligible: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run, ExperimentRunSpec)
            or self.source.run_id != self.run.run_id
            or self.evaluation.run_id != self.run.run_id
            or self.eligible is not True
        ):
            raise ExperimentValidationError("eligible run chain identity is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "source": self.source.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "eligible": True,
        }


def _source_layout(run_dir: Path) -> None:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ExperimentValidationError("source run must be a real directory")
    try:
        entries = tuple(run_dir.iterdir())
    except OSError as exc:
        raise ExperimentValidationError("could not inspect source run") from exc
    if {item.name for item in entries} != SOURCE_LAYOUT:
        raise ExperimentValidationError("source run layout is not canonical")
    for item in entries:
        if item.is_symlink():
            raise ExperimentValidationError("source run must not contain symlinks")
        if item.name == "config_snapshot":
            if not item.is_dir():
                raise ExperimentValidationError("config_snapshot must be a directory")
            names = tuple(sorted(child.name for child in item.iterdir()))
            if names != tuple(sorted(CONFIG_FILES.values())) or any(
                child.is_symlink() or not child.is_file() for child in item.iterdir()
            ):
                raise ExperimentValidationError("config_snapshot layout is invalid")
        elif not item.is_file():
            raise ExperimentValidationError("source artifacts must be regular files")


def _read_source_streams(run_dir: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename in SOURCE_LOGS:
        try:
            result[filename] = canonical_jsonl._read_existing_records(run_dir / filename)
        except Exception:
            raise ExperimentValidationError(f"invalid canonical source stream: {filename}") from None
    return result


def _validate_real_api_usage(streams: Mapping[str, tuple[dict[str, Any], ...]]) -> None:
    expected: dict[tuple[str, str], str] = {}
    for behavior in streams["behavior_logs.jsonl"]:
        action_type = (
            "target_event_response" if behavior["is_target_event"] else "natural_response"
        )
        for call_type, template in (
            ("plan", "plan"),
            (
                action_type,
                "target_event_response"
                if action_type == "target_event_response"
                else "natural_interaction",
            ),
            ("importance_score", "importance_scoring"),
        ):
            key = (behavior["turn_id"], call_type)
            if key in expected:
                raise ExperimentValidationError("duplicate expected simulator call")
            expected[key] = template
    for reflection in streams["reflection_logs.jsonl"]:
        for call_type, template in (
            ("reflection", "reflect"),
            ("importance_score", "importance_scoring"),
        ):
            key = (reflection["turn_id"], call_type)
            if key in expected:
                raise ExperimentValidationError("duplicate expected reflection call")
            expected[key] = template
    usage = streams["api_usage_logs.jsonl"]
    if len(expected) != 408 or not 408 <= len(usage) <= 816:
        raise ExperimentValidationError("simulator API usage count is incomplete")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in usage:
        key = (record["turn_id"], record["call_type"])
        if (
            key not in expected
            or record["model"] != SIMULATOR_MODEL_IDENTITY
            or record["prompt_template"] != expected[key]
            or record["status"] not in {"success", "failed", "invalid_output"}
            or (record["status"] == "success") != (record["error_type"] is None)
        ):
            raise ExperimentValidationError("simulator API usage identity is invalid")
        grouped.setdefault(key, []).append(record)
    if set(grouped) != set(expected) or any(
        len(records) > 2
        or records[-1]["status"] != "success"
        or sum(record["status"] == "success" for record in records) != 1
        for records in grouped.values()
    ):
        raise ExperimentValidationError("simulator logical-call audit is incomplete")


def validate_source_run(
    run_dir: str | Path,
    *,
    spec: ExperimentRunSpec,
    scope: FullRunScope,
    mode: str,
) -> SourceRunValidation:
    """Validate one completed full source without modifying or repairing it."""
    if mode not in {"dummy", "real"}:
        raise ExperimentValidationError("source validation mode must be dummy or real")
    source = Path(run_dir)
    _source_layout(source)
    try:
        summary = read_run_summary(source / "run_summary.json")
    except Exception:
        raise ExperimentValidationError("invalid run_summary.json") from None
    expected_identity = (
        spec.experiment_id,
        spec.run_id,
        spec.retrieval_config_id,
        spec.repetition_index,
    )
    actual_identity = (
        summary.experiment_id,
        summary.run_id,
        summary.retrieval_config_id,
        summary.repetition_index,
    )
    if actual_identity != expected_identity:
        raise ExperimentValidationError("source run identity differs from plan")
    if (
        summary.status != "completed"
        or summary.completed_natural_blocks != scope.natural_block_count
        or summary.completed_target_events != scope.target_event_count
        or summary.completed_main_turns != scope.main_turn_count
        or summary.error_count != 0
    ):
        raise ExperimentValidationError("source summary does not satisfy full-run counts")
    streams = _read_source_streams(source)
    expected_counts = {
        "retrieval_logs.jsonl": scope.main_turn_count,
        "behavior_logs.jsonl": scope.main_turn_count,
        "memory_logs.jsonl": scope.expected_memory_record_count,
        "reflection_logs.jsonl": scope.post_step_reflection_count,
        "errors.jsonl": 0,
        "event_summaries.jsonl": scope.target_event_count,
        "natural_block_summaries.jsonl": scope.natural_block_count,
    }
    for filename, expected in expected_counts.items():
        if len(streams[filename]) != expected:
            raise ExperimentValidationError(f"{filename} count differs from full-run contract")
    try:
        loaded = load_evaluation_run(source, diagnostic=False)
    except Exception:
        raise ExperimentValidationError("source is not eligible for primary Phase 09") from None
    if (
        len(loaded.target_units) != scope.target_evaluation_unit_count
        or loaded.evaluation_mode != "primary"
        or loaded.primary_eligible is not True
        or loaded.included_in_final_research_aggregation is not True
    ):
        raise ExperimentValidationError("source target-unit eligibility is incomplete")
    api_usage = streams["api_usage_logs.jsonl"]
    if mode == "dummy":
        if api_usage or loaded.simulator_model_identity is not None:
            raise ExperimentValidationError("dummy source must not contain API usage")
    else:
        if loaded.simulator_model_identity != SIMULATOR_MODEL_IDENTITY:
            raise ExperimentValidationError("real source simulator identity is invalid")
        _validate_real_api_usage(streams)
    try:
        validate_no_secret_artifacts(source)
    except Exception:
        raise ExperimentValidationError("source artifact security scan failed") from None
    return SourceRunValidation(
        run_id=spec.run_id,
        source_status=summary.status,
        natural_block_count=summary.completed_natural_blocks,
        target_event_count=summary.completed_target_events,
        step_count=(
            len(streams["natural_block_summaries.jsonl"])
            + len(streams["event_summaries.jsonl"])
        ),
        main_turn_count=summary.completed_main_turns,
        reflection_count=len(streams["reflection_logs.jsonl"]),
        memory_record_count=len(streams["memory_logs.jsonl"]),
        target_evaluation_unit_count=len(loaded.target_units),
        api_usage_record_count=len(api_usage),
        simulator_model_identity=loaded.simulator_model_identity,
        source_fingerprint=fingerprint_directory(source),
    )


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or b"\r" in payload:
            raise ValueError("invalid line ending")
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError):
        raise ExperimentValidationError(f"invalid strict JSON artifact: {path.name}") from None
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        payload = path.read_bytes()
    except OSError:
        raise ExperimentValidationError(f"could not read {path.name}") from None
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ExperimentValidationError(f"{path.name} must be non-empty LF JSONL")
    values: list[Mapping[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            )
        except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError):
            raise ExperimentValidationError(f"{path.name} contains malformed JSON") from None
        if not isinstance(value, Mapping):
            raise ExperimentValidationError(f"{path.name} rows must be objects")
        values.append(value)
    return tuple(values)


_MRA_KEYS = {
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


def _validate_mra_rows(
    rows: tuple[Mapping[str, Any], ...], spec: ExperimentRunSpec
) -> None:
    if len(rows) != 20:
        raise ExperimentValidationError("primary MRA output must contain exactly 20 rows")
    seen: set[str] = set()
    event_counts: Counter[str] = Counter()
    previous_order: tuple[int, int] | None = None
    for row in rows:
        if set(row) != _MRA_KEYS:
            raise ExperimentValidationError("MRA row does not match exact schema")
        if (
            row["schema_version"] != "1.0"
            or row["evaluation_mode"] != "primary"
            or row["primary_eligible"] is not True
            or row["included_in_final_research_aggregation"] is not True
            or (
                row["experiment_id"],
                row["run_id"],
                row["retrieval_config_id"],
                row["repetition_index"],
            )
            != (
                spec.experiment_id,
                spec.run_id,
                spec.retrieval_config_id,
                spec.repetition_index,
            )
            or row["event_id"] not in FIXED_EVENT_IDS
            or row["agent_id"] not in FIXED_AGENT_IDS
            or row["top_k"] != 5
            or row["denominator"] != 5
        ):
            raise ExperimentValidationError("MRA row identity or eligibility is invalid")
        for name in ("retrieved_count", "candidate_count", "matched_count"):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise ExperimentValidationError("MRA count field is invalid")
        for name in (
            "selected_memory_ids",
            "retrieved_categories",
            "expected_categories",
            "matched_memory_ids",
            "unmatched_memory_ids",
            "matched_categories",
        ):
            if not isinstance(row[name], list) or any(
                not isinstance(item, str) or not item for item in row[name]
            ):
                raise ExperimentValidationError("MRA evidence list is invalid")
        selected = row["selected_memory_ids"]
        if (
            len(selected) != row["retrieved_count"]
            or len(selected) != len(row["retrieved_categories"])
            or len(selected) > 5
            or row["candidate_count"] < len(selected)
            or len(selected) != len(set(selected))
            or row["matched_count"] != len(row["matched_memory_ids"])
            or set(row["matched_memory_ids"]) | set(row["unmatched_memory_ids"])
            != set(selected)
            or set(row["matched_memory_ids"]) & set(row["unmatched_memory_ids"])
            or isinstance(row["mra"], bool)
            or not isinstance(row["mra"], (int, float))
            or not isfinite(float(row["mra"]))
            or row["mra"] != row["matched_count"] / 5
        ):
            raise ExperimentValidationError("MRA evidence arithmetic is invalid")
        turn_id = row["turn_id"]
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen:
            raise ExperimentValidationError("MRA evaluation unit identity is invalid")
        seen.add(turn_id)
        order = (row["timestamp"], row["turn_index"])
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in order):
            raise ExperimentValidationError("MRA turn order is invalid")
        if previous_order is not None and order < previous_order:
            raise ExperimentValidationError("MRA rows are not chronologically ordered")
        previous_order = order
        event_counts[row["event_id"]] += 1
    if event_counts != Counter({event_id: 4 for event_id in FIXED_EVENT_IDS}):
        raise ExperimentValidationError("MRA rows must contain four agents per event")


_JUDGE_INPUT_KEYS = {
    "persona",
    "target_event",
    "response",
    "prior_target_behaviors",
    "selected_memory_texts",
}
_PERSONA_KEYS = {
    "display_name",
    "role",
    "archetype",
    "persona_summary",
    "traits",
    "goals",
    "communication_style",
    "motivations",
}


def _validate_condition_blind_judge_input(value: Mapping[str, Any]) -> None:
    if set(value) != _JUDGE_INPUT_KEYS:
        raise ExperimentValidationError("BCR judge_input schema changed")
    persona = value["persona"]
    event = value["target_event"]
    if not isinstance(persona, Mapping) or set(persona) != _PERSONA_KEYS:
        raise ExperimentValidationError("BCR judge persona payload is invalid")
    if not isinstance(event, Mapping) or set(event) != {"title", "description"}:
        raise ExperimentValidationError("BCR judge event payload is invalid")
    forbidden_keys = {
        "experiment_id",
        "run_id",
        "retrieval_config_id",
        "repetition_index",
        "weights",
        "retrieval_score",
        "normalized_retrieval_score",
        "mra",
        "ground_truth",
        "expected_categories",
        "memory_id",
        "configuration_result",
    }

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            if any(str(key).lower() in forbidden_keys for key in item):
                raise ExperimentValidationError("BCR judge_input contains blinded metadata")
            for nested in item.values():
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)

    walk(value)


def validate_phase09_evaluation(
    evaluation_dir: str | Path,
    *,
    spec: ExperimentRunSpec,
) -> EvaluationValidation:
    """Validate persisted Phase 09 output without calculating MRA or BCR."""
    directory = Path(evaluation_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ExperimentValidationError("evaluation output must be a real directory")
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise ExperimentValidationError("could not inspect evaluation output") from exc
    if {item.name for item in entries} != EVALUATION_LAYOUT or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ExperimentValidationError("evaluation output layout is invalid")
    try:
        loaded_bcr = load_bcr_export(directory)
    except Exception:
        raise ExperimentValidationError("Phase 09 BCR export bundle is invalid") from None
    if (
        loaded_bcr.experiment_id,
        loaded_bcr.run_id,
        loaded_bcr.retrieval_config_id,
        loaded_bcr.repetition_index,
        loaded_bcr.evaluation_mode,
        loaded_bcr.primary_eligible,
        loaded_bcr.included_in_final_research_aggregation,
    ) != (
        spec.experiment_id,
        spec.run_id,
        spec.retrieval_config_id,
        spec.repetition_index,
        "primary",
        True,
        True,
    ):
        raise ExperimentValidationError("evaluation identity differs from run plan")
    mra_rows = _read_jsonl(directory / "mra_records.jsonl")
    _validate_mra_rows(mra_rows, spec)
    summary = _read_json(directory / "evaluation_summary.json")
    if (
        summary.get("schema_version") != "1.0"
        or summary.get("source_run_status") != "completed"
        or summary.get("evaluation_mode") != "primary"
        or summary.get("primary_eligible") is not True
        or summary.get("included_in_final_research_aggregation") is not True
        or summary.get("bcr_export_count") != 20
        or not isinstance(summary.get("mra_aggregates"), Mapping)
        or summary["mra_aggregates"].get("unit_count") != 20
        or summary.get("bcr_judgment") != future_bcr_contract_dict()
    ):
        raise ExperimentValidationError("evaluation summary is incomplete or incompatible")
    for record in loaded_bcr.records:
        _validate_condition_blind_judge_input(record.judge_input.to_dict())
    try:
        validate_no_secret_artifacts(directory)
    except Exception:
        raise ExperimentValidationError("evaluation artifact security scan failed") from None
    return EvaluationValidation(
        run_id=spec.run_id,
        evaluation_mode="primary",
        primary_eligible=True,
        mra_record_count=len(mra_rows),
        bcr_export_record_count=len(loaded_bcr.records),
        binary_future_bcr_contract_validated=True,
        condition_blind_judge_input_validated=True,
        evaluation_fingerprint=fingerprint_directory(directory),
    )


def validate_run_chain_unchanged(chain: EligibleRunChain, *, source: Path, evaluation: Path) -> None:
    if (
        fingerprint_directory(source) != chain.source.source_fingerprint
        or fingerprint_directory(evaluation) != chain.evaluation.evaluation_fingerprint
    ):
        raise ExperimentValidationError("sealed source/evaluation artifacts changed")


def validate_complete_chains(
    plan: ExperimentPlan,
    chains: tuple[EligibleRunChain, ...],
) -> None:
    if len(chains) != 12:
        raise ExperimentValidationError("successful experiment requires exactly 12 chains")
    if tuple(chain.run for chain in chains) != plan.runs or any(
        chain.eligible is not True for chain in chains
    ):
        raise ExperimentValidationError("eligible run chains do not match plan order")
    if sum(chain.source.main_turn_count for chain in chains) != 1248:
        raise ExperimentValidationError("experiment main-turn total is incomplete")
    if sum(chain.source.reflection_count for chain in chains) != 576:
        raise ExperimentValidationError("experiment reflection total is incomplete")
    if sum(chain.evaluation.mra_record_count for chain in chains) != 240:
        raise ExperimentValidationError("experiment MRA total is incomplete")
    if sum(chain.evaluation.bcr_export_record_count for chain in chains) != 240:
        raise ExperimentValidationError("experiment BCR-export total is incomplete")


def _walk_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return tuple(keys)


def _manifest_non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _manifest_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HASH_CHARS for character in value)
    )


def _validate_manifest_fingerprint(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "directory_sha256",
        "files",
    }:
        raise ExperimentValidationError("manifest fingerprint schema is invalid")
    files = value["files"]
    if not _manifest_hash(value["directory_sha256"]) or not isinstance(files, list):
        raise ExperimentValidationError("manifest fingerprint identity is invalid")
    names: list[str] = []
    for item in files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"relative_path", "sha256"}
            or not isinstance(item["relative_path"], str)
            or not item["relative_path"]
            or Path(item["relative_path"]).is_absolute()
            or ".." in Path(item["relative_path"]).parts
            or not _manifest_hash(item["sha256"])
        ):
            raise ExperimentValidationError("manifest fingerprint file is invalid")
        names.append(item["relative_path"])
    if not names or names != sorted(names) or len(names) != len(set(names)):
        raise ExperimentValidationError("manifest fingerprint inventory is invalid")


def _validate_manifest_usage(
    value: object,
    *,
    expected_simulator_logical: int,
    expected_embedding_logical: int,
    maximum_physical_attempts: int,
    mode: str,
    require_global_budget: bool,
) -> None:
    keys = {
        "schema_version",
        "max_total_tokens",
        "logical_call_count",
        "physical_attempt_count",
        "retry_count",
        "known_input_tokens",
        "known_output_tokens",
        "known_total_tokens",
        "missing_usage_attempt_count",
        "usage_complete",
        "actual_total_tokens",
        "budget_state",
        "role_usage",
        "evaluator_logical_calls",
        "evaluator_physical_attempts",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ExperimentValidationError("manifest usage schema is invalid")
    integer_fields = (
        "logical_call_count",
        "physical_attempt_count",
        "retry_count",
        "known_input_tokens",
        "known_output_tokens",
        "known_total_tokens",
        "missing_usage_attempt_count",
        "evaluator_logical_calls",
        "evaluator_physical_attempts",
    )
    if any(not _manifest_non_negative_int(value[name]) for name in integer_fields):
        raise ExperimentValidationError("manifest usage count is invalid")
    logical = expected_simulator_logical + expected_embedding_logical
    if (
        value["schema_version"] != EXPERIMENT_SCHEMA_VERSION
        or value["logical_call_count"] != logical
        or value["physical_attempt_count"] > maximum_physical_attempts
        or value["retry_count"]
        != (
            0
            if value["physical_attempt_count"] == 0
            else value["physical_attempt_count"] - logical
        )
        or value["known_total_tokens"]
        != value["known_input_tokens"] + value["known_output_tokens"]
        or value["missing_usage_attempt_count"] > MAX_MISSING_USAGE_ATTEMPTS
        or value["usage_complete"] is not True
        or value["actual_total_tokens"] != value["known_total_tokens"]
        or value["evaluator_logical_calls"] != 0
        or value["evaluator_physical_attempts"] != 0
    ):
        raise ExperimentValidationError("manifest usage completion is invalid")
    roles = value["role_usage"]
    if not isinstance(roles, list) or len(roles) != 2:
        raise ExperimentValidationError("manifest usage role inventory is invalid")
    role_keys = {
        "service_role",
        "logical_call_count",
        "physical_attempt_count",
        "retry_count",
        "known_input_tokens",
        "known_output_tokens",
        "known_total_tokens",
        "missing_usage_attempt_count",
        "usage_complete",
        "actual_total_tokens",
        "success_count",
        "failed_count",
        "invalid_output_count",
    }
    expected_roles = (
        ("simulator", expected_simulator_logical),
        ("embedding", expected_embedding_logical),
    )
    for role, (expected_name, expected_logical) in zip(roles, expected_roles):
        if not isinstance(role, Mapping) or set(role) != role_keys:
            raise ExperimentValidationError("manifest role usage schema is invalid")
        role_integer_fields = role_keys - {
            "service_role",
            "usage_complete",
            "actual_total_tokens",
        }
        if any(
            not _manifest_non_negative_int(role[name]) for name in role_integer_fields
        ):
            raise ExperimentValidationError("manifest role usage count is invalid")
        if (
            role["service_role"] != expected_name
            or role["logical_call_count"] != expected_logical
            or role["missing_usage_attempt_count"] > MAX_MISSING_USAGE_ATTEMPTS
            or role["usage_complete"] is not True
            or role["actual_total_tokens"] != role["known_total_tokens"]
            or role["known_total_tokens"]
            != role["known_input_tokens"] + role["known_output_tokens"]
            or role["success_count"]
            + role["failed_count"]
            + role["invalid_output_count"]
            != role["physical_attempt_count"]
            or role["retry_count"]
            != (
                0
                if role["physical_attempt_count"] == 0
                else role["physical_attempt_count"] - expected_logical
            )
        ):
            raise ExperimentValidationError("manifest role usage is incomplete")
    if mode == "dummy":
        if (
            value["max_total_tokens"] is not None
            or value["physical_attempt_count"] != 0
            or value["known_total_tokens"] != 0
            or value["budget_state"] != "not_applicable"
        ):
            raise ExperimentValidationError("dummy manifest usage is invalid")
    elif require_global_budget:
        if (
            not isinstance(value["max_total_tokens"], int)
            or isinstance(value["max_total_tokens"], bool)
            or value["max_total_tokens"] < 1
            or value["physical_attempt_count"] < logical
            or value["budget_state"] not in {"admitted", "exhausted"}
        ):
            raise ExperimentValidationError("real manifest usage is invalid")
    elif (
        value["max_total_tokens"] is not None
        or value["physical_attempt_count"] < logical
        or value["budget_state"] != "not_applicable"
    ):
        raise ExperimentValidationError("real per-run manifest usage is invalid")


def _validate_manifest_run(value: object, spec: ExperimentRunSpec, *, mode: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "run",
        "source",
        "evaluation",
        "eligible",
        "usage",
    }:
        raise ExperimentValidationError("manifest run-chain schema is invalid")
    source = value["source"]
    evaluation = value["evaluation"]
    if value["run"] != spec.to_dict() or value["eligible"] is not True:
        raise ExperimentValidationError("manifest run identity is invalid")
    if not isinstance(source, Mapping) or set(source) != {
        "run_id",
        "source_status",
        "natural_block_count",
        "target_event_count",
        "step_count",
        "main_turn_count",
        "reflection_count",
        "memory_record_count",
        "target_evaluation_unit_count",
        "api_usage_record_count",
        "simulator_model_identity",
        "source_fingerprint",
    }:
        raise ExperimentValidationError("manifest source validation schema is invalid")
    if (
        source["run_id"] != spec.run_id
        or source["source_status"] != "completed"
        or tuple(
            source[name]
            for name in (
                "natural_block_count",
                "target_event_count",
                "step_count",
                "main_turn_count",
                "reflection_count",
                "memory_record_count",
                "target_evaluation_unit_count",
            )
        )
        != (7, 5, 12, 104, 48, 164, 20)
    ):
        raise ExperimentValidationError("manifest source completion is invalid")
    _validate_manifest_fingerprint(source["source_fingerprint"])
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "run_id",
        "evaluation_mode",
        "primary_eligible",
        "mra_record_count",
        "bcr_export_record_count",
        "binary_future_bcr_contract_validated",
        "condition_blind_judge_input_validated",
        "evaluation_fingerprint",
    }:
        raise ExperimentValidationError("manifest evaluation validation schema is invalid")
    if (
        evaluation["run_id"] != spec.run_id
        or evaluation["evaluation_mode"] != "primary"
        or evaluation["primary_eligible"] is not True
        or evaluation["mra_record_count"] != 20
        or evaluation["bcr_export_record_count"] != 20
        or evaluation["binary_future_bcr_contract_validated"] is not True
        or evaluation["condition_blind_judge_input_validated"] is not True
    ):
        raise ExperimentValidationError("manifest evaluation completion is invalid")
    _validate_manifest_fingerprint(evaluation["evaluation_fingerprint"])
    _validate_manifest_usage(
        value["usage"],
        expected_simulator_logical=408,
        expected_embedding_logical=268,
        maximum_physical_attempts=1352,
        mode=mode,
        require_global_budget=False,
    )


def validate_experiment_manifest(
    value: Mapping[str, Any],
    *,
    plan: ExperimentPlan,
    expected: Mapping[str, Any] | None = None,
) -> None:
    """Validate technical completion only; no research analysis is permitted."""
    required = {
        "schema_version",
        "experiment_id",
        "mode",
        "status",
        "completion_contract",
        "plan_hash",
        "ordered_matrix",
        "source_identity",
        "accepted_freeze_identity",
        "configuration_hashes",
        "prompt_identities",
        "simulator_settings",
        "embedding_settings",
        "retry_policy",
        "schema_identities",
        "expected_totals",
        "actual_totals",
        "runs",
        "usage",
        "final_validation_state",
    }
    if set(value) != required:
        raise ExperimentValidationError("experiment manifest fields do not match schema")
    if expected is not None and dict(value) != dict(expected):
        raise ExperimentValidationError("experiment manifest differs from validated content")
    totals = value.get("actual_totals")
    mode = value.get("mode")
    runs = value.get("runs")
    if (
        value.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
        or value.get("experiment_id") != plan.experiment_id
        or mode != plan.mode
        or value.get("status") != "completed"
        or value.get("completion_contract") != "source_plus_phase09_evaluation_ready"
        or value.get("plan_hash") != plan.plan_hash
        or value.get("ordered_matrix") != [run.to_dict() for run in plan.runs]
        or value.get("final_validation_state") != "valid"
        or not isinstance(totals, Mapping)
        or totals.get("run_count") != 12
        or totals.get("source_run_count") != 12
        or totals.get("evaluation_count") != 12
        or totals.get("main_turn_count") != 1248
        or totals.get("post_step_reflection_count") != 576
        or totals.get("mra_record_count") != 240
        or totals.get("bcr_export_record_count") != 240
        or totals.get("bcr_judgment_count") != 0
        or not isinstance(runs, list)
        or len(runs) != 12
        or value.get("expected_totals") != plan.to_dict()["experiment_totals"]
    ):
        raise ExperimentValidationError("experiment manifest completion is invalid")
    for item, spec in zip(runs, plan.runs):
        _validate_manifest_run(item, spec, mode=plan.mode)
    _validate_manifest_usage(
        value.get("usage"),
        expected_simulator_logical=plan.total_simulator_logical_calls,
        expected_embedding_logical=plan.total_embedding_logical_calls,
        maximum_physical_attempts=plan.maximum_physical_attempts,
        mode=plan.mode,
        require_global_budget=True,
    )
    source_identity = value.get("source_identity")
    if (
        not isinstance(source_identity, Mapping)
        or set(source_identity) != {"commit", "branch", "tags", "clean"}
        or not isinstance(source_identity["commit"], str)
        or len(source_identity["commit"]) != 40
        or any(character not in _HASH_CHARS for character in source_identity["commit"])
        or not isinstance(source_identity["branch"], str)
        or not source_identity["branch"]
        or not isinstance(source_identity["tags"], list)
        or any(not isinstance(tag, str) or not tag for tag in source_identity["tags"])
        or not isinstance(source_identity["clean"], bool)
        or (plan.mode == "real" and source_identity["clean"] is not True)
    ):
        raise ExperimentValidationError("experiment source identity is invalid")
    freeze = value.get("accepted_freeze_identity")
    if plan.mode == "dummy":
        if freeze is not None:
            raise ExperimentValidationError("dummy manifest must not claim a pilot freeze")
    elif (
        not isinstance(freeze, Mapping)
        or set(freeze)
        != {
            "pilot_id",
            "manifest_hash",
            "acceptance_hash",
            "freeze_hash",
            "prompt_version",
        }
        or any(
            not _manifest_hash(freeze[name])
            for name in ("manifest_hash", "acceptance_hash", "freeze_hash")
        )
        or not isinstance(freeze["pilot_id"], str)
        or not freeze["pilot_id"]
        or not isinstance(freeze["prompt_version"], str)
        or not freeze["prompt_version"]
    ):
        raise ExperimentValidationError("real manifest pilot freeze identity is invalid")
    forbidden = {
        "credential",
        "credentials",
        "api_key",
        "authorization",
        "environment",
        "environment_variables",
        "prompt_body",
        "raw_request",
        "raw_response",
        "raw_exception",
        "traceback",
        "currency_cost",
        "monetary_cost",
        "bcr_judgment",
        "bcr_mean",
        "configuration_ranking",
        "research_conclusion",
    }
    if any(key in forbidden for key in _walk_keys(value)):
        raise ExperimentValidationError("experiment manifest contains forbidden fields")


__all__ = [
    "DirectoryFingerprint",
    "EVALUATION_LAYOUT",
    "EligibleRunChain",
    "EvaluationValidation",
    "ExperimentValidationError",
    "SOURCE_LAYOUT",
    "SOURCE_LOGS",
    "SourceRunValidation",
    "fingerprint_directory",
    "validate_complete_chains",
    "validate_experiment_manifest",
    "validate_phase09_evaluation",
    "validate_run_chain_unchanged",
    "validate_source_run",
]
