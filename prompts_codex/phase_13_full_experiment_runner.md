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

# Phase 13 — Full Experiment Runner

## Codex Prompt

```text
Implement Phase 13 — Full Experiment Runner.

Context:
The repository now has core simulation, real/dummy LLM support, evaluation, pilot safeguards, and log replay visualizer. This phase implements the full 12-run experiment runner with strict confirmation, run isolation, validation, and aggregation readiness.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement full experiment execution for 4 retrieval configurations × 3 repetitions = 12 runs.

Files/modules to create or update:
- src/simulation/experiment_controller.py
- src/simulation/full_experiment.py
- src/simulation/run_validator.py if updates needed
- src/storage/experiment_directory.py
- src/evaluation/aggregator.py if integration needed
- main.py
- tests/test_full_experiment_dry_run.py
- tests/test_full_experiment_dummy.py
- tests/test_run_isolation.py
- tests/test_full_experiment_confirmation.py
- tests/test_failed_run_exclusion.py

Full experiment requirements:
- Execute exactly 4 configs:
  - balanced
  - recency_heavy
  - importance_heavy
  - relevance_heavy
- Execute exactly 3 repetitions per config.
- Total runs must be exactly 12.
- Each run must start from identical initial seed state.
- Each run must have separate run directory.
- Each run must copy config snapshots.
- Each run must write run_summary.json.
- Failed/interrupted runs must not be silently aggregated.

Execution modes:
1. Dry-run full experiment
- No LLM calls.
- Print planned runs and expected turn counts.
- Expected main turns:
  - 104 per run
  - 1,248 total main turns across 12 runs

2. Dummy full experiment
- Execute all 12 runs with dummy LLM.
- Useful for validating orchestration and logs.

3. Real full experiment
- Must require explicit confirmation token, for example:
  `--confirm-full-experiment RUN_12_CONFIGURED_RUNS`
- Must require real LLM allow flag.
- Must require token budget or confirm no budget.
- Must fail if output directory exists unless explicitly handled.

CLI requirements:
Add/update:
- `python main.py run-full-experiment --dry-run true`
- `python main.py run-full-experiment --dummy --confirm-full-experiment RUN_12_CONFIGURED_RUNS --output-root <path> --on-conflict fail`
- `python main.py run-full-experiment --real --allow-real-llm true --confirm-full-experiment RUN_12_CONFIGURED_RUNS --max-total-tokens <N> --output-root <path> --on-conflict fail`

Post-run validation:
- Confirm 12 run summaries exist.
- Confirm completed run count.
- Confirm failed run count.
- Confirm each completed run has retrieval/behavior/memory logs.
- Confirm target event logs exist.
- Confirm no failed runs are included in aggregation.

Testing requirements:
Add tests for:
- dry-run computes exactly 12 planned runs
- dry-run computes expected main turns
- dummy full experiment can complete in tmp path
- real full experiment refuses without confirmation token
- output conflict fails by default
- run isolation: memory streams do not leak between runs
- failed run excluded from aggregate plan

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- full experiment dry-run works.
- dummy full experiment works in temp output path.
- real full experiment cannot start accidentally.

Forbidden:
- Do not run real full experiment automatically.
- Do not relax confirmation requirements.
- Do not aggregate failed runs silently.
- Do not overwrite output directories silently.
- Do not change prompts/configs/retrieval/evaluation during full experiment execution.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
