"""Offline fixtures and provider doubles shared by Phase 10 tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from src.evaluation import (
    BCRJudgeInput,
    PersonaJudgeInput,
    PriorBehaviorEvidence,
    TargetEventJudgeInput,
    evaluate_run,
)
from src.llm import LLMContext


def natural_context(**changes: object) -> LLMContext:
    context = LLMContext(
        agent_id="teacher",
        display_name="Ms. Sunday",
        role="teacher",
        archetype="calm_classroom_guide",
        persona_summary="A calm and structured classroom teacher.",
        traits=("calm", "structured"),
        goals=("support students",),
        communication_style="Clear and concise.",
        motivations=("student understanding",),
        source_step_id="day_01_natural_01",
        source_step_type="natural_interaction_block",
        step_title="Classroom Introductions and Expectations",
        step_description="The class discusses participation.",
        day=1,
        period="morning",
        logical_timestamp=10,
        observation_text="The class is discussing participation.",
        conversation_context=("Monday asked how to participate.",),
        selected_memory_texts=("Ms. Sunday explains expectations clearly.",),
    )
    return replace(context, **changes)


def target_context(**changes: object) -> LLMContext:
    context = replace(
        natural_context(),
        source_step_id="day_02_target_01",
        source_step_type="target_event",
        step_title="Exam Announcement",
        step_description="A written test is announced.",
        event_id="event_01",
        event_title="Exam Announcement",
        event_description="A written test will take place on Day 6.",
    )
    return replace(context, **changes)


def bcr_judge_input(**changes: object) -> BCRJudgeInput:
    value = BCRJudgeInput(
        persona=PersonaJudgeInput(
            display_name="Ms. Sunday",
            role="teacher",
            archetype="calm_classroom_guide",
            persona_summary="A calm and structured classroom teacher.",
            traits=("calm", "structured"),
            goals=("support students",),
            communication_style="Clear and concise.",
            motivations=("student understanding",),
        ),
        target_event=TargetEventJudgeInput(
            title="Exam Announcement",
            description="A written test will take place on Day 6.",
        ),
        response="Please review the material and ask focused questions.",
        prior_target_behaviors=(
            PriorBehaviorEvidence(
                event_id="event_01",
                step_id="day_02_target_01",
                turn_id="day_02_target_01_turn_001",
                turn_index=1,
                timestamp=25,
                event_title="Exam Announcement",
                event_description="A written test was announced.",
                response="We will prepare with a clear review plan.",
            ),
        ),
        selected_memory_texts=("The teacher previously offered a review session.",),
    )
    return replace(value, **changes)


def prepare_phase09_evaluation(
    root: Path,
    run_dir: Path,
    *,
    diagnostic: bool = False,
    simulator_model_identity: str | None = "openai:gpt-5.6-luna",
) -> Path:
    """Create a real-model-shaped Phase 09 export without changing its schema."""
    output = root / "phase09_evaluation"
    evaluate_run(run_dir=run_dir, output_dir=output, diagnostic=diagnostic)
    bcr_path = output / "bcr_export.jsonl"
    records = [json.loads(line) for line in bcr_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["model_metadata"]["simulator_model_identity"] = simulator_model_identity
    bcr_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="",
    )
    summary_path = output / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model_metadata"]["simulator_model_identity"] = simulator_model_identity
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return output


def directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(path.iterdir())
        if item.is_file()
    }
