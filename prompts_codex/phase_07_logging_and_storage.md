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

# Phase 07 — Logging and Storage

## Codex Prompt

```text
Implement Phase 07 — Logging and Storage.

Context:
The repository now has a dummy cognitive pipeline. This phase implements run directory management, JSONL logging, config snapshots, and structured log schemas. Logging is part of the research methodology because MRA, BCR, auditability, and reproducibility depend on it.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement storage and JSONL logging infrastructure without running full simulation yet.

Files/modules to create or update:
- src/storage/run_directory.py
- src/storage/config_snapshot.py
- src/storage/__init__.py
- src/logging_system/jsonl_logger.py
- src/logging_system/schemas.py
- src/logging_system/run_summary.py
- src/logging_system/__init__.py
- tests/test_run_directory.py
- tests/test_config_snapshot.py
- tests/test_jsonl_logger.py
- tests/test_logging_schemas.py

Run directory requirements:
- Create unique run directories for each run_id.
- Refuse to overwrite existing completed run directories by default.
- Support explicit conflict handling modes where appropriate:
  - fail
  - overwrite only if explicitly requested for temporary tests
- Create subfiles or paths for:
  - retrieval_logs.jsonl
  - behavior_logs.jsonl
  - memory_logs.jsonl
  - reflection_logs.jsonl
  - api_usage_logs.jsonl
  - event_summaries.jsonl
  - natural_block_summaries.jsonl
  - errors.jsonl
  - run_summary.json
  - config_snapshot/

Config snapshot requirements:
- Copy config JSON files into run directory before run execution.
- Do not copy .env or API keys.

JSONL logger requirements:
- Append records as valid JSON lines.
- Ensure parent directories exist.
- Flush safely.
- Do not silently swallow write errors.

Required shared metadata in log schemas:
- experiment_id
- run_id
- retrieval_config_id
- repetition_index
- step_id
- step_type
- is_target_event
- day
- period
- turn_id if relevant
- turn_index if relevant
- agent_id if relevant
- timestamp

Log record types:
1. Retrieval log
Must include candidate scores, selected flag, top_k, and weights.

2. Behavior log
Must include response, retrieved_memory_ids, plan summary if available, evaluation_enabled.

3. Memory log
Must include memory_id, memory_type, category, text, importance, source step metadata.

4. Reflection log
Must include reflection text and source step metadata.

5. API usage log
Placeholder schema for future real LLM usage.

6. Error log
Must include error_type, message, recoverable, retry_count if applicable.

Run summary requirements:
- Write status:
  - initialized
  - completed
  - failed
  - interrupted
  - excluded
- Include counts for completed natural blocks, target events, main turns, errors.

Testing requirements:
Add tests for:
- run directory creation
- no silent overwrite
- config snapshot creation excluding .env
- JSONL append and parse
- required metadata validation
- run_summary status writing

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- No full simulation execution is implemented yet.

Forbidden:
- Do not create real experiment logs in repository paths during tests unless using tmp directories.
- Do not log API keys or environment variables.
- Do not implement real API calls.
- Do not implement evaluation metrics yet.
- Do not implement visualizer.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
