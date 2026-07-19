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

# Phase 09 — Evaluation Modules

## Codex Prompt

```text
Implement Phase 09 — Evaluation Modules.

Context:
The repository can now run dummy simulation and produce logs. This phase implements the primary evaluation modules: MRA calculation and BCR export dataset. Evaluation must read logs after simulation; it must not rerun or mutate the simulation.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement MRA calculation, BCR export dataset preparation, metric aggregation, and evaluation CLI commands.

Files/modules to create or update:
- src/evaluation/mra.py
- src/evaluation/bcr_export.py
- src/evaluation/aggregator.py
- src/evaluation/readers.py
- src/evaluation/schemas.py
- src/evaluation/__init__.py
- main.py
- tests/test_mra_calculation.py
- tests/test_mra_filtering.py
- tests/test_bcr_export.py
- tests/test_evaluation_aggregation.py
- tests/test_ground_truth_alignment.py

MRA requirements:
- Primary MRA uses only retrieval logs where:
  - step_type == target_event
  - is_target_event == true
  - evaluation_enabled == true if available
- Natural block retrieval logs must not enter primary MRA by default.
- MRA compares selected Top-K memory categories against ground_truth.json expected categories for event-agent pair.
- Formula:
  MRA = matched_count / K
  where K is expected Top-K=5 for the main experiment.
- If fewer than K memories exist, record actual retrieved count and make the denominator policy explicit. Prefer denominator=K for main comparability, but include retrieved_count and candidate_count for transparency. If the documents specify a different policy, follow the documents.
- Missing ground truth should produce explicit skipped or error records, not silent zero.

MRA output requirements:
Export rows containing:
- experiment_id
- run_id
- retrieval_config_id
- repetition_index
- event_id
- step_id
- agent_id
- top_k
- retrieved_count
- matched_count
- mra
- matched_memory_ids
- unmatched_memory_ids
- expected_categories
- retrieved_categories

BCR export requirements:
- Export target-event behavior logs only by default.
- Include enough context for later LLM-as-judge evaluation:
  - agent persona
  - target event context
  - response
  - retrieved memories
  - prior relevant behavior or prior memory summary if available
  - run/config metadata
- Do not run real BCR judge in this phase unless dummy judge export is required.
- The BCR export is a dataset, not final judged scores unless a dummy judge is used for tests.

Aggregation requirements:
- Aggregate MRA by:
  - run
  - configuration
  - event
  - agent
- Prepare placeholder or schema for BCR aggregation after judge results are available.

CLI requirements:
Add or update commands:
- `python main.py evaluate --run-dir <run_dir> --ground-truth config/ground_truth.json --output-dir <output_dir>`
- or separate commands:
  - `evaluate-mra`
  - `export-bcr`

Testing requirements:
Add tests for:
- MRA ignores natural blocks
- MRA uses only target events
- MRA category matching works
- missing ground truth is explicit
- BCR export includes target event responses only
- aggregation computes means correctly

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- Evaluation can run on dummy pilot logs.
- Natural blocks are excluded from primary MRA/BCR by default.

Forbidden:
- Do not change simulation logs to artificially improve MRA.
- Do not tune natural dialogue from evaluation code.
- Do not include natural blocks in primary MRA by default.
- Do not implement real LLM judge unless explicitly requested in a later phase.
- Do not modify retrieval scoring logic except for bug fixes required by tests, and document any such change.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
