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

# Phase 11 — Pilot Safeguards

## Codex Prompt

```text
Implement Phase 11 — Pilot Safeguards.

Context:
The repository now supports dummy execution, evaluation, and real LLM service integration. This phase adds strict pilot controls, cost safeguards, run validation gates, and freeze-readiness checks. The pilot is mandatory before full experiment.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement pilot execution safeguards for dummy and real LLM modes, including token budget checks, required confirmations, log completeness validation, and prompt/config freeze checks.

Files/modules to create or update:
- src/simulation/pilot.py
- src/simulation/run_validator.py
- src/simulation/safeguards.py
- src/storage/run_directory.py if needed
- src/logging_system/run_summary.py if needed
- main.py
- tests/test_pilot_safeguards.py
- tests/test_token_budget_guard.py
- tests/test_run_validation_gates.py
- tests/test_conflict_handling.py

Pilot requirements:
Default pilot scope:
- config: balanced
- repetition: 1
- days: Day 1–2 only
- includes Day 1 natural block 1
- includes Day 1 natural block 2
- includes Day 2 natural block
- includes target event 1

Pilot modes:
- dry-run: estimates planned run steps and turns without LLM calls
- dummy: executes with dummy LLM
- real: executes with real LLM only when explicitly allowed

Real mode safeguards:
- Require explicit flag such as `--real`.
- Require explicit allow flag such as `--allow-real-llm true`.
- Require max token budget argument or config value.
- Abort safely if token budget exceeded.
- Log token usage.
- Mark interrupted/failed runs explicitly.

Conflict handling:
- Default should fail if output directory exists.
- Support `--on-conflict fail` as default.
- Optional overwrite should be restricted and not used for final experiment.

Run validation gates:
After pilot, validate:
- run_summary exists
- status completed
- natural block count matches pilot scope
- target event count matches pilot scope
- retrieval logs exist
- behavior logs exist
- memory logs exist
- errors count is explicit
- target event evaluation export can be generated
- Top-K validation passes when enough memories are available
- no API secrets in logs

CLI requirements:
Update pilot command clearly:
- `python main.py run-pilot --dry-run true --config balanced --repetition 1`
- `python main.py run-pilot --dummy --config balanced --repetition 1 --output-root <path> --on-conflict fail`
- `python main.py run-pilot --real --allow-real-llm true --config balanced --repetition 1 --max-total-tokens <N> --output-root <path> --on-conflict fail`

Testing requirements:
Add tests for:
- dry-run pilot computes expected planned steps/turns
- dummy pilot completes
- real mode refuses to run without explicit allow flag
- token budget guard aborts when exceeded
- output conflict fails by default
- run validation detects missing retrieval logs
- run validation detects missing behavior logs

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- dry-run pilot works.
- dummy pilot works.
- real pilot is guarded and cannot run accidentally.

Forbidden:
- Do not run a real API call during automated tests.
- Do not run full experiment in this phase.
- Do not loosen architecture constants.
- Do not add new dialogue tuning rules after pilot; only safeguards and validation.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
