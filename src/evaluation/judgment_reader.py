"""Strict read-only loading of Phase 09 BCR export artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.config_loader.validators import FIXED_EVENT_IDS
from src.domain.agents import FIXED_AGENT_IDS
from src.evaluation.errors import BCRJudgmentInputError
from src.evaluation.schemas import (
    BCRExportRecord,
    BCRJudgeInput,
    BCRModelMetadata,
    EvaluationUnitIdentity,
    PersonaJudgeInput,
    PriorBehaviorEvidence,
    SelectedMemoryEvidence,
    TargetEventJudgeInput,
    future_bcr_contract_dict,
)


_SUMMARY_KEYS = {
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
_ROW_KEYS = {
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
_IDENTITY_KEYS = {
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
_MODEL_KEYS = {
    "simulator_model_identity",
    "evaluator_model_identity",
    "evaluator_prompt_version",
    "evaluator_rubric_version",
}
_JUDGE_KEYS = {
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


class _DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, keys: set[str], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise BCRJudgmentInputError(f"{field_name} does not match its exact schema")
    return value


def _list(value: object, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise BCRJudgmentInputError(f"{field_name} must be a JSON array")
    return value


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    items = _list(value, field_name)
    if any(not isinstance(item, str) for item in items):
        raise BCRJudgmentInputError(f"{field_name} must contain strings")
    return tuple(items)


def _read_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise BCRJudgmentInputError(f"required source file is not regular: {path.name}")
        return path.read_bytes()
    except BCRJudgmentInputError:
        raise
    except OSError as exc:
        raise BCRJudgmentInputError(f"could not read source file {path.name}") from exc


def _decode_json(payload: bytes, field_name: str) -> Mapping[str, Any]:
    try:
        if not payload.endswith(b"\n") or b"\r" in payload:
            raise ValueError("must be LF-terminated")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise BCRJudgmentInputError(f"{field_name} is malformed strict JSON") from exc
    if not isinstance(value, Mapping):
        raise BCRJudgmentInputError(f"{field_name} must contain a JSON object")
    return value


def _parse_model(value: object) -> BCRModelMetadata:
    data = _mapping(value, _MODEL_KEYS, "model_metadata")
    try:
        return BCRModelMetadata(**data)
    except (TypeError, ValueError) as exc:
        raise BCRJudgmentInputError("model_metadata is invalid") from exc


def _parse_row(value: object) -> BCRExportRecord:
    row = _mapping(value, _ROW_KEYS, "BCR export row")
    if row["schema_version"] != "1.0":
        raise BCRJudgmentInputError("unsupported BCR export row schema version")
    identity_data = _mapping(row["audit_metadata"], _IDENTITY_KEYS, "audit_metadata")
    judge = _mapping(row["judge_input"], _JUDGE_KEYS, "judge_input")
    persona_data = _mapping(judge["persona"], _PERSONA_KEYS, "judge_input.persona")
    event_data = _mapping(
        judge["target_event"], {"title", "description"}, "judge_input.target_event"
    )
    selected_audit = _list(row["selected_memory_evidence"], "selected_memory_evidence")
    selected_texts = _text_tuple(judge["selected_memory_texts"], "selected_memory_texts")
    if len(selected_audit) != len(selected_texts):
        raise BCRJudgmentInputError("selected-memory audit/text counts differ")
    prior_audit = _list(row["prior_behavior_evidence"], "prior_behavior_evidence")
    prior_judge = _list(judge["prior_target_behaviors"], "prior_target_behaviors")
    if len(prior_audit) != len(prior_judge):
        raise BCRJudgmentInputError("prior-behavior audit/judge counts differ")
    try:
        identity = EvaluationUnitIdentity(**identity_data)
        selected = tuple(
            SelectedMemoryEvidence(
                **_mapping(audit, {"retrieval_rank", "memory_id", "category"}, "selected memory"),
                text=text,
            )
            for audit, text in zip(selected_audit, selected_texts, strict=True)
        )
        prior = tuple(
            PriorBehaviorEvidence(
                **_mapping(
                    audit,
                    {"event_id", "step_id", "turn_id", "turn_index", "timestamp"},
                    "prior behavior audit",
                ),
                **_mapping(
                    visible,
                    {"event_title", "event_description", "response"},
                    "prior behavior judge evidence",
                ),
            )
            for audit, visible in zip(prior_audit, prior_judge, strict=True)
        )
        judge_input = BCRJudgeInput(
            persona=PersonaJudgeInput(
                display_name=persona_data["display_name"],
                role=persona_data["role"],
                archetype=persona_data["archetype"],
                persona_summary=persona_data["persona_summary"],
                traits=_text_tuple(persona_data["traits"], "persona.traits"),
                goals=_text_tuple(persona_data["goals"], "persona.goals"),
                communication_style=persona_data["communication_style"],
                motivations=_text_tuple(
                    persona_data["motivations"], "persona.motivations"
                ),
            ),
            target_event=TargetEventJudgeInput(**event_data),
            response=judge["response"],
            prior_target_behaviors=prior,
            selected_memory_texts=selected_texts,
        )
        return BCRExportRecord(
            identity=identity,
            evaluation_mode=row["evaluation_mode"],
            primary_eligible=row["primary_eligible"],
            included_in_final_research_aggregation=(
                row["included_in_final_research_aggregation"]
            ),
            selected_memories=selected,
            prior_behaviors=prior,
            judge_input=judge_input,
            model_metadata=_parse_model(row["model_metadata"]),
        )
    except (TypeError, ValueError) as exc:
        raise BCRJudgmentInputError("BCR export row failed immutable schema validation") from exc


@dataclass(frozen=True)
class LoadedBCRExport:
    source_directory: Path
    records: tuple[BCRExportRecord, ...]
    source_record_hashes: tuple[str, ...]
    source_bcr_export_hash: str
    source_evaluation_schema_version: str
    experiment_id: str
    run_id: str
    retrieval_config_id: str
    repetition_index: int
    evaluation_mode: str
    primary_eligible: bool
    included_in_final_research_aggregation: bool
    simulator_model_identity: str | None
    source_fingerprint: tuple[tuple[str, str], ...]


def source_directory_fingerprint(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        files = tuple(sorted(item for item in path.iterdir() if item.is_file()))
    except OSError as exc:
        raise BCRJudgmentInputError("could not fingerprint source evaluation directory") from exc
    return tuple((item.name, sha256(_read_bytes(item)).hexdigest()) for item in files)


def load_bcr_export(evaluation_dir: str | Path) -> LoadedBCRExport:
    try:
        requested = Path(evaluation_dir)
        if requested.is_symlink():
            raise BCRJudgmentInputError("source evaluation directory must not be a symlink")
        source = requested.resolve(strict=True)
    except BCRJudgmentInputError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise BCRJudgmentInputError("source evaluation directory does not exist") from exc
    if not source.is_dir():
        raise BCRJudgmentInputError("source evaluation path must be a directory")
    expected_files = {"mra_records.jsonl", "bcr_export.jsonl", "evaluation_summary.json"}
    entries = tuple(source.iterdir())
    actual_files = {item.name for item in entries}
    if actual_files != expected_files or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise BCRJudgmentInputError("source evaluation directory layout is invalid")

    summary = _mapping(
        _decode_json(_read_bytes(source / "evaluation_summary.json"), "evaluation_summary.json"),
        _SUMMARY_KEYS,
        "evaluation_summary.json",
    )
    if summary["schema_version"] != "1.0":
        raise BCRJudgmentInputError("unsupported evaluation schema version")
    if summary["bcr_judgment"] != future_bcr_contract_dict():
        raise BCRJudgmentInputError("source summary must contain the unjudged Phase 09 contract")
    if summary["output_files"] != [
        "mra_records.jsonl",
        "bcr_export.jsonl",
        "evaluation_summary.json",
    ]:
        raise BCRJudgmentInputError("source summary output layout is invalid")
    mode = summary["evaluation_mode"]
    expected_primary = mode == "primary"
    if mode not in {"primary", "diagnostic"} or (
        summary["primary_eligible"] is not expected_primary
        or summary["included_in_final_research_aggregation"] is not expected_primary
        or summary["source_run_status"] != ("completed" if expected_primary else "excluded")
    ):
        raise BCRJudgmentInputError("source evaluation eligibility metadata is inconsistent")
    summary_model = _parse_model(summary["model_metadata"])
    if any(
        value is not None
        for value in (
            summary_model.evaluator_model_identity,
            summary_model.evaluator_prompt_version,
            summary_model.evaluator_rubric_version,
        )
    ):
        raise BCRJudgmentInputError("source Phase 09 summary is already populated as judged")

    payload = _read_bytes(source / "bcr_export.jsonl")
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise BCRJudgmentInputError("bcr_export.jsonl must be non-empty LF-terminated JSONL")
    records: list[BCRExportRecord] = []
    record_hashes: list[str] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_strict_object)
        except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
            raise BCRJudgmentInputError(
                f"bcr_export.jsonl line {line_number} is malformed"
            ) from exc
        records.append(_parse_row(value))
        record_hashes.append(sha256(line).hexdigest())
    if len(records) != summary["bcr_export_count"]:
        raise BCRJudgmentInputError("BCR export count does not match source summary")
    if expected_primary and len(records) != 20:
        raise BCRJudgmentInputError("primary BCR export must contain exactly 20 units")
    if not records:
        raise BCRJudgmentInputError("BCR export scope must not be empty")
    summary_identity = (
        summary["experiment_id"],
        summary["run_id"],
        summary["retrieval_config_id"],
        summary["repetition_index"],
        mode,
        summary["primary_eligible"],
        summary["included_in_final_research_aggregation"],
    )
    seen: set[str] = set()
    for record in records:
        actual = (
            record.identity.experiment_id,
            record.identity.run_id,
            record.identity.retrieval_config_id,
            record.identity.repetition_index,
            record.evaluation_mode,
            record.primary_eligible,
            record.included_in_final_research_aggregation,
        )
        if actual != summary_identity:
            raise BCRJudgmentInputError("BCR export row identity differs from source summary")
        if record.identity.turn_id in seen:
            raise BCRJudgmentInputError("duplicate BCR export evaluation unit")
        seen.add(record.identity.turn_id)
        if record.model_metadata != summary_model:
            raise BCRJudgmentInputError("BCR export model metadata differs from source summary")
    order = tuple((item.identity.timestamp, item.identity.turn_index) for item in records)
    if order != tuple(sorted(order)):
        raise BCRJudgmentInputError("BCR export rows are not in chronological turn order")
    if expected_primary:
        event_counts = Counter(item.identity.event_id for item in records)
        if event_counts != Counter({event_id: 4 for event_id in FIXED_EVENT_IDS}):
            raise BCRJudgmentInputError("primary BCR export must contain four units per event")
        for event_id in FIXED_EVENT_IDS:
            agents = {
                item.identity.agent_id for item in records if item.identity.event_id == event_id
            }
            if agents != set(FIXED_AGENT_IDS):
                raise BCRJudgmentInputError("primary BCR export must contain every agent per event")
    fingerprint = source_directory_fingerprint(source)
    return LoadedBCRExport(
        source_directory=source,
        records=tuple(records),
        source_record_hashes=tuple(record_hashes),
        source_bcr_export_hash=sha256(payload).hexdigest(),
        source_evaluation_schema_version=summary["schema_version"],
        experiment_id=summary["experiment_id"],
        run_id=summary["run_id"],
        retrieval_config_id=summary["retrieval_config_id"],
        repetition_index=summary["repetition_index"],
        evaluation_mode=mode,
        primary_eligible=summary["primary_eligible"],
        included_in_final_research_aggregation=(
            summary["included_in_final_research_aggregation"]
        ),
        simulator_model_identity=summary_model.simulator_model_identity,
        source_fingerprint=fingerprint,
    )


__all__ = ["LoadedBCRExport", "load_bcr_export", "source_directory_fingerprint"]
