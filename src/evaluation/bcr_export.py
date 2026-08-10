"""Deterministic, condition-blind BCR dataset export construction."""

from __future__ import annotations

from src.evaluation.errors import EvaluationCalculationError
from src.evaluation.readers import LoadedEvaluationRun
from src.evaluation.schemas import (
    BCRExportRecord,
    BCRJudgeInput,
    BCRModelMetadata,
    PersonaJudgeInput,
    PriorBehaviorEvidence,
    TargetEventJudgeInput,
)


def build_bcr_export(
    loaded_run: LoadedEvaluationRun,
) -> tuple[BCRExportRecord, ...]:
    """Build one export row per target unit without invoking any evaluator."""
    if not isinstance(loaded_run, LoadedEvaluationRun):
        raise EvaluationCalculationError("loaded_run must be LoadedEvaluationRun")
    prior_by_agent: dict[str, list[PriorBehaviorEvidence]] = {
        agent.agent_id: [] for agent in loaded_run.config.agents
    }
    model_metadata = BCRModelMetadata(
        simulator_model_identity=loaded_run.simulator_model_identity,
    )
    records: list[BCRExportRecord] = []
    for unit in loaded_run.target_units:
        prior = tuple(prior_by_agent[unit.identity.agent_id])
        judge_input = BCRJudgeInput(
            persona=PersonaJudgeInput(
                display_name=unit.agent.display_name,
                role=unit.agent.role,
                archetype=unit.agent.archetype,
                persona_summary=unit.agent.persona_summary,
                traits=unit.agent.traits,
                goals=unit.agent.goals,
                communication_style=unit.agent.communication_style,
                motivations=unit.agent.motivations,
            ),
            target_event=TargetEventJudgeInput(
                title=unit.event.title,
                description=unit.event.description,
            ),
            response=unit.response,
            prior_target_behaviors=prior,
            selected_memory_texts=tuple(
                memory.text for memory in unit.selected_memories
            ),
        )
        records.append(
            BCRExportRecord(
                identity=unit.identity,
                evaluation_mode=loaded_run.evaluation_mode,
                primary_eligible=loaded_run.primary_eligible,
                included_in_final_research_aggregation=(
                    loaded_run.included_in_final_research_aggregation
                ),
                selected_memories=unit.selected_memories,
                prior_behaviors=prior,
                judge_input=judge_input,
                model_metadata=model_metadata,
            )
        )
        prior_by_agent[unit.identity.agent_id].append(
            PriorBehaviorEvidence(
                event_id=unit.identity.event_id,
                step_id=unit.identity.step_id,
                turn_id=unit.identity.turn_id,
                turn_index=unit.identity.turn_index,
                timestamp=unit.identity.timestamp,
                event_title=unit.event.title,
                event_description=unit.event.description,
                response=unit.response,
            )
        )
    return tuple(records)


__all__ = ["build_bcr_export"]
