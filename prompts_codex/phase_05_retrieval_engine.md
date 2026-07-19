# Codex Phase Prompts — Final Clean Repository Implementation

**Project:** Classroom-Based Generative Agents for Retrieval Weighting Experiment  
**Target repository:** `classroom-generative-agents-clean` or `classroom-memory-retrieval-agents`  
**Research focus:** Pengaruh konfigurasi bobot retrieval memori terhadap konsistensi perilaku agen generatif berbasis LLM  
**Primary documents:**

- `docs/Document_01_Final_Clean_Architecture_Blueprint.md`
- `docs/Document_02_Final_Clean_Implementation_Specification.md`
- `docs/ARCHITECTURE_CONTRACT.md`

**Purpose of this file:**  
This file contains one Codex prompt for each implementation phase. Use these prompts sequentially. Do not skip phases. Do not combine multiple phases unless the previous phases are already committed, tested, and stable.

---

# Global Rules for Every Codex Phase

Use these rules in every phase. They define the working discipline for the clean final repository.

```text
You are working on a clean final thesis implementation repository for a controlled classroom-based generative agents experiment.

Before making changes, read:
1. docs/ARCHITECTURE_CONTRACT.md
2. docs/Document_01_Final_Clean_Architecture_Blueprint.md
3. docs/Document_02_Final_Clean_Implementation_Specification.md

Do not violate the architecture contract.

Core research invariant:
The only independent variable in the final experiment is retrieval weight configuration.

Non-negotiable design constraints:
- Use exactly 4 fixed agent IDs:
  - teacher
  - active_student
  - quiet_student
  - academic_student
- Use a 6-day hybrid simulation timeline.
- Day 1 has exactly 2 natural interaction blocks.
- Days 2–6 each have exactly 1 natural interaction block followed by 1 target event.
- There are exactly 7 natural interaction blocks per run.
- There are exactly 5 target events per run.
- Each natural interaction block has exactly 12 turns.
- Each agent receives exactly 3 turns per natural interaction block.
- The teacher opens every natural interaction block.
- Natural interaction blocks are controlled memory-building context, not primary MRA/BCR evaluation points.
- Target events are the primary MRA/BCR evaluation anchors.
- Top-K retrieval must always be 5.
- Retrieval must use only normalized recency, importance, and relevance.
- Do not add hidden retrieval multipliers, penalties, filters, or additional ranking factors.
- Prompt templates and output guardrails must not bias the system toward recency, importance, or relevance.
- The visualizer, if implemented, must be read-only log replay and must not mutate simulation, memory, retrieval, or evaluation.
- Do not implement maze, spatial map, pathfinding, game server, daily life schedule, autonomous free-world simulation, or live real-time web simulation.

Development discipline:
- Implement only the current phase.
- Do not pre-implement future phases.
- Do not modify files outside the current phase scope unless required for tests or imports.
- Add or update tests for the phase.
- Run relevant tests before finishing.
- Keep dummy/offline mode working before real LLM mode.
- Do not log API keys or environment variables.
- Do not commit generated logs, API outputs, or large runtime artifacts.

At the end of the phase, provide:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions made.
5. Known limitations.
6. Recommended next phase.
```

---

---

# Phase 05 — Retrieval Engine

## Codex Prompt

```text
Implement Phase 05 — Retrieval Engine.

Context:
The repository now has validated configuration and per-agent memory streams. This phase implements the core experimental component: weighted memory retrieval using normalized recency, importance, and relevance. This phase is central to the thesis. Keep it pure, explicit, testable, and free from hidden multipliers or prompt influence.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement retrieval scoring, normalization, ranking, Top-K selection, and retrieval result structures.

Files/modules to create or update:
- src/retrieval/scoring.py
- src/retrieval/normalization.py
- src/retrieval/engine.py
- src/retrieval/result.py
- src/retrieval/__init__.py
- src/domain/retrieval.py if updates are needed
- tests/test_retrieval_scoring.py
- tests/test_retrieval_normalization.py
- tests/test_retrieval_engine.py
- tests/test_retrieval_configs_behavior.py

Retrieval formula:
Score(m, q) =
  w_recency * Norm(Recency(m))
+ w_importance * Norm(Importance(m))
+ w_relevance * Norm(Relevance(m, q))

Component requirements:

1. Recency
- Compute from current_time and memory.last_accessed_at or created_at according to the documented design.
- Use the same decay function across all configurations.
- Recency-heavy changes only the weight, not the recency formula.

2. Importance
- Use memory.importance, expected 1–10.
- Do not call LLM importance scoring in this phase.

3. Relevance
- Use cosine similarity between query embedding and memory embedding.
- If embedding is missing in final mode, the retrieval engine should expose a clear error path.
- For tests, support deterministic numeric vectors.

4. Normalization
- Normalize component values across the candidate pool to 0–1.
- If all values are equal, return 0.5 for all candidates for that component.

5. Ranking
- Compute final weighted score.
- Sort descending by final score.
- Tie-break by higher relevance, then higher importance, then newer timestamp.
- Select Top-K=5 when candidate_count >= 5.
- If candidate_count < Top-K, return all available memories and report candidate_count.

6. Last access update
- Retrieval engine should return selected IDs.
- Updating last_accessed_at should happen through the memory stream after retrieval result is produced/loggable.
- Do not update access time before ranking is recorded.

7. Retrieval result
Return a structure that includes:
- query
- agent_id
- weights
- top_k
- candidate_count
- candidates with raw and normalized component scores
- final_score
- selected flag
- selected_top_k memory IDs

Testing requirements:
Add tests for:
- recency scoring behavior
- importance scoring input use
- cosine similarity relevance
- normalization with varied values
- normalization with all equal values
- balanced ranking
- recency-heavy ranking on synthetic data
- importance-heavy ranking on synthetic data
- relevance-heavy ranking on synthetic data
- Top-K exactly 5
- fewer-than-K case
- tie-breaking order
- no hidden multiplier behavior

Acceptance criteria:
- `python main.py validate-config` still passes.
- `pytest` passes.
- Retrieval behavior is deterministic in tests.
- No LLM calls are introduced.

Forbidden:
- Do not add any retrieval component beyond recency, importance, and relevance.
- Do not add hidden score multipliers.
- Do not filter out memories by category unless explicitly requested by candidate ownership validity.
- Do not let prompts influence retrieval scoring.
- Do not implement simulation controller yet.
- Do not implement evaluation yet.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
