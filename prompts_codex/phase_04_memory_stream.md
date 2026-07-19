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

# Phase 04 — Memory Stream

## Codex Prompt

```text
Implement Phase 04 — Memory Stream.

Context:
The repository now has validated configuration and domain models. This phase implements the per-agent memory stream. The memory stream is the core internal state that stores seed memories, observations, plans, actions, and reflections. Retrieval scoring will be implemented in the next phase, so this phase should focus on memory creation, ownership, storage, and access metadata.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement a deterministic, testable per-agent memory stream with seed initialization and memory metadata management.

Files/modules to create or update:
- src/memory/memory_entry.py
- src/memory/memory_stream.py
- src/memory/memory_factory.py
- src/memory/seed_loader.py
- src/memory/__init__.py
- src/domain/memory.py if updates are needed
- tests/test_memory_entry.py
- tests/test_memory_stream.py
- tests/test_seed_memory_initialization.py

Memory requirements:

Each memory entry must include:
- memory_id
- agent_id
- memory_type
- text
- category
- importance
- embedding or optional embedding placeholder
- created_at
- last_accessed_at
- source_step_id
- source_step_type
- metadata optional dict

Supported memory types:
- seed_persona
- observation
- plan
- action
- reflection

Memory stream requirements:
- Each agent owns a separate memory stream.
- A memory stream can add a memory.
- A memory stream can list all memories for an agent.
- A memory stream can filter by memory_type.
- A memory stream can filter by source_step_type.
- A memory stream can update last_accessed_at for selected memory IDs only.
- Memory IDs must be stable and unique within a run.
- Memory ownership must prevent cross-agent private memory leakage.

Seed initialization requirements:
- Load seed memories from agents.json through the config loader output.
- Create seed_persona memories for each agent.
- Use seed memory category labels from config.
- Use seed memory importance from config if provided, otherwise use a deterministic default.
- Seed initialization must be reproducible across runs.

Run isolation requirement:
- Provide a helper to initialize fresh memory streams for a new run.
- Running initialization twice should not share mutable memory objects across runs.

Embedding and importance scoring:
- Do not call real embedding or LLM APIs in this phase.
- Use optional placeholder fields or deterministic dummy values if tests need them.
- Real embedding/importance hooks will be added later.

Testing requirements:
Add tests for:
- memory creation with required fields
- add/list/filter operations
- update last_accessed_at only for selected memories
- separate agent memory ownership
- seed memories load for all four agents
- repeated run initialization creates independent memory stream objects

Acceptance criteria:
- `python main.py validate-config` still passes.
- `pytest` passes.
- No retrieval scoring is implemented yet.

Forbidden:
- Do not implement weighted retrieval ranking yet.
- Do not call LLM or embedding APIs.
- Do not allow an agent to retrieve another agent private memory directly.
- Do not implement simulation execution.
- Do not implement evaluation.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
