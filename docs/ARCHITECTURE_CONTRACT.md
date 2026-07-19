# Architecture Contract

This repository implements a controlled classroom-based generative agents experiment.

Non-negotiable design rules:

1. The system uses 4 fixed agents:
   - teacher
   - active_student
   - quiet_student
   - academic_student

2. The simulation uses a 6-day hybrid timeline:
   - Day 1: 2 natural interaction blocks
   - Day 2–6: 1 natural interaction block + 1 target event per day

3. The system uses 7 natural interaction blocks and 5 target events per run.

4. Natural interaction blocks are used only as controlled memory-building context.

5. Target events are the primary evaluation anchors for MRA and BCR.

6. Top-K retrieval must always be 5.

7. The retrieval formula must remain:
   Score(m,q) =
   w_recency * Norm(Recency(m))
   + w_importance * Norm(Importance(m))
   + w_relevance * Norm(Relevance(m,q))

8. The only independent variable is retrieval weight configuration.

9. No hidden retrieval multiplier, retrieval penalty, or additional ranking factor is allowed.

10. Prompt templates and output guardrails must not bias the system toward recency, importance, or relevance.

11. Minimal output guardrails may only protect output validity:
   - empty output
   - placeholder output
   - wrong legacy names
   - exact duplicate
   - excessive length

12. Natural interaction must not be optimized as a full human conversation simulation.

13. The visualizer is read-only log replay and must not alter simulation, memory, retrieval, or evaluation.

14. Full experiment must not start before config, retrieval, logging, evaluation, and pilot validation are complete.