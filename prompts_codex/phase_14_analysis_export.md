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

# Phase 14 — Analysis Export

## Codex Prompt

```text
Implement Phase 14 — Analysis Export.

Context:
The repository can execute full dummy experiments and guarded real full experiments. This final phase implements analysis export tools for thesis reporting. The analysis module must read completed logs/evaluation outputs and produce structured summaries, tables, and qualitative examples. It must not modify simulation results.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement analysis export for MRA/BCR summaries, retrieval pattern examples, per-configuration comparison, per-agent breakdown, per-event breakdown, and thesis-ready tables.

Files/modules to create or update:
- src/evaluation/aggregator.py if needed
- src/evaluation/bcr_results.py if judge outputs are supported
- src/analysis/exporter.py
- src/analysis/tables.py
- src/analysis/qualitative_examples.py
- src/analysis/__init__.py
- main.py
- tests/test_analysis_export.py
- tests/test_analysis_tables.py
- tests/test_qualitative_examples.py

Analysis input requirements:
- Completed experiment directory containing 12 run directories or subset with explicit status.
- MRA records generated from evaluation phase.
- BCR judge results if available, or BCR export dataset if scores are not yet judged.
- Retrieval logs and behavior logs for qualitative examples.

Analysis output requirements:
Create outputs such as:
- mra_by_config.csv
- bcr_by_config.csv if BCR scores are available
- mra_by_event.csv
- bcr_by_agent.csv if BCR scores are available
- retrieval_examples_by_config.json
- experiment_run_status_summary.csv
- thesis_tables.md
- analysis_summary.json

MRA summary requirements:
- Mean MRA per configuration.
- Mean MRA per configuration and event.
- Mean MRA per configuration and agent.
- Repetition-level values preserved.

BCR summary requirements:
- If judged BCR scores exist, aggregate by configuration, event, and agent.
- If judged BCR scores do not exist, export a clear message that BCR judge results are pending and keep the BCR export dataset available.

Qualitative example requirements:
Provide examples for each retrieval configuration where possible:
- selected target event
- agent response
- Top-K retrieved memory categories
- retrieval component scores
- short explanation-ready notes

Data integrity requirements:
- Do not include failed/interrupted/excluded runs unless explicitly requested.
- If runs are missing, report them clearly.
- Do not recompute simulation.
- Do not mutate source logs.
- Do not silently fill missing values.

CLI requirements:
Add/update:
- `python main.py analyze-experiment --experiment-dir <path> --evaluation-dir <path> --output-dir <path>`
- Optional: `python main.py summarize-experiment --experiment-dir <path>`

Testing requirements:
Add tests for:
- analysis exports CSV/JSON/MD outputs from synthetic evaluation records
- failed runs excluded by default
- missing BCR scores handled explicitly
- per-config MRA mean correct
- per-event MRA breakdown correct
- qualitative examples include retrieval metadata

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- analysis export works on dummy full experiment outputs.
- outputs are thesis-readable and traceable.

Forbidden:
- Do not alter raw logs.
- Do not rerun simulation.
- Do not include failed runs silently.
- Do not invent missing BCR scores.
- Do not change methodology constants.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Final readiness checklist.
```

---

# Suggested Commit Messages

Use one commit after each successful phase:

```bash
git add .
git commit -m "phase-01-architecture-contract-and-repository-skeleton"
git commit -m "phase-02-agent-initialization-configuration"
git commit -m "phase-03-domain-models-and-config-validation"
git commit -m "phase-04-memory-stream"
git commit -m "phase-05-retrieval-engine"
git commit -m "phase-06-dummy-llm-and-cognitive-pipeline"
git commit -m "phase-07-logging-and-storage"
git commit -m "phase-08-simulation-controller"
git commit -m "phase-09-evaluation-modules"
git commit -m "phase-10-real-llm-service-and-prompts"
git commit -m "phase-11-pilot-safeguards"
git commit -m "phase-12-log-replay-visualizer"
git commit -m "phase-13-full-experiment-runner"
git commit -m "phase-14-analysis-export"
```

---

# Recommended Tags

Use tags at stable checkpoints:

```bash
git tag architecture-freeze-v1
git tag config-freeze-v1
git tag core-engine-freeze-v1
git tag prompt-freeze-after-pilot-v1
git tag pre-full-experiment-freeze-v1
git tag final-experiment-code-v1
```

---

# Final Pre-Experiment Checklist

Before running the real full experiment, verify:

```text
- All tests pass.
- validate-config passes.
- Dummy pilot passes.
- Dummy full experiment passes.
- Real smoke test passes.
- Real Day 1–2 pilot passes.
- Token usage is acceptable.
- Prompts are frozen.
- Config files are frozen.
- Retrieval code is frozen.
- Evaluation code is frozen.
- Architecture contract is unchanged.
- Ground truth is frozen.
- Output directories are clean.
- API key is stored only in .env and not committed.
- Full experiment command requires explicit confirmation.
```
