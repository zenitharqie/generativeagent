BCR_PROMPT_V2 = """You are evaluating behavioral consistency in a controlled classroom simulation.

Evaluate the CURRENT RESPONSE using three separate dimensions. Judge semantic behavior, not lexical similarity.

1. COMMUNICATION STYLE
- CONSISTENT: tone, length, formality, and interaction style clearly fit the persona.
- PARTIALLY_CONSISTENT: the persona remains recognizable, but there is a noticeable minor deviation in tone, length, formality, or interaction style.
- INCONSISTENT: the response clearly contradicts a defining communication characteristic.

2. FOCUS CONSISTENCY
- CONSISTENT: the primary focus clearly aligns with the persona's role, goals, motivations, and the target event.
- PARTIALLY_CONSISTENT: the response addresses the event but is generic, mixed, or only weakly connected to the persona's characteristic focus.
- INCONSISTENT: the primary focus conflicts with the persona or is substantively unrelated to the event.

3. TEMPORAL CONSISTENCY
- CONSISTENT: the response does not contradict prior target-event responses and shows a coherent behavioral tendency.
- PARTIALLY_CONSISTENT: there is a noticeable shift or ambiguity, but no direct substantive contradiction.
- INCONSISTENT: the response directly contradicts a prior target-event response or reverses an established behavioral commitment without contextual justification.
- NOT_EVALUABLE: no prior target-event response is available.

Important rules:
- Adaptation to a different event is not automatically inconsistent.
- Do not require identical wording or identical actions across events.
- Use NOT_EVALUABLE for temporal consistency only when prior_target_behaviors is empty.
- Base every label on evidence from the supplied input.

Return exactly one JSON object:
{
  "communication_style": {"label": "CONSISTENT|PARTIALLY_CONSISTENT|INCONSISTENT", "evidence": "brief evidence"},
  "focus_consistency": {"label": "CONSISTENT|PARTIALLY_CONSISTENT|INCONSISTENT", "evidence": "brief evidence"},
  "temporal_consistency": {"label": "CONSISTENT|PARTIALLY_CONSISTENT|INCONSISTENT|NOT_EVALUABLE", "evidence": "brief evidence"},
  "reason": "brief overall rationale"
}
"""
