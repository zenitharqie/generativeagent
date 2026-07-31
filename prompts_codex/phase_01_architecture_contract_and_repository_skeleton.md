# Codex Phase Prompts — Final Clean Repository Implementation

**Project:** Classroom-Based Generative Agents for Retrieval Weighting Experiment  
**Target repository:** `classroom-generative-agents-clean` 
 
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

# Phase 01 — Architecture Contract and Repository Skeleton

## Codex Prompt

```text
Implement Phase 01 — Architecture Contract and Repository Skeleton.

Context:
This is a clean final thesis repository for a controlled classroom-based generative agents experiment. The repository must follow the final clean architecture documents and the architecture contract. This phase only creates the repository skeleton, documentation placement, initial CLI placeholder, package layout, and minimal tests.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Create a clean, testable repository scaffold that future phases can build on. Do not implement simulation, retrieval, LLM calls, evaluation, or visualizer logic yet.

Allowed work:
1. Create or verify the main folder structure:
   - config/
   - prompts/
   - src/
   - src/domain/
   - src/config_loader/
   - src/memory/
   - src/retrieval/
   - src/llm/
   - src/cognition/
   - src/simulation/
   - src/logging_system/
   - src/evaluation/
   - src/storage/
   - src/visualizer/
   - tests/
   - data/
   - logs/
   - resources/
   - visualizer/
   - docs/

2. Create or update:
   - README.md
   - .gitignore
   - .env.example
   - requirements.txt
   - main.py
   - pyproject.toml if useful for pytest/import configuration
   - empty __init__.py files where needed

3. README.md must clearly state:
   - This is a clean final thesis implementation.
   - The system is a controlled text-based classroom simulation.
   - It is not a full Smallville/Reverie reproduction.
   - Natural interaction blocks are memory-building context.
   - Target events are primary MRA/BCR evaluation anchors.
   - Only retrieval weights vary across configurations.

4. main.py should provide a minimal CLI placeholder with at least:
   - `python main.py`
   - `python main.py --help`
   - optionally `python main.py validate-config` as a placeholder command that clearly says config validation is not implemented yet.

5. Add minimal tests:
   - test_repository_structure.py
   - test_main_cli.py

Acceptance criteria:
- `python main.py` runs without crashing.
- `python main.py --help` runs without crashing.
- `pytest` passes.
- Repository structure is present.
- No simulation, retrieval, LLM, evaluation, or visualizer logic is implemented yet.

Forbidden:
- Do not implement retrieval engine.
- Do not implement memory stream.
- Do not implement simulation controller.
- Do not implement LLM API calls.
- Do not implement visualizer.
- Do not create generated logs.
- Do not alter the architecture contract.

After completing, summarize:
1. Files created or modified.
2. Tests added.
3. Commands run and their outputs.
4. Assumptions.
5. Known limitations.
6. Next recommended phase.
```

---
