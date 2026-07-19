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

# Phase 02 — Agent Initialization Configuration

## Codex Prompt

```text
Implement Phase 02 — Agent Initialization Configuration.

Context:
This phase creates the configuration files for the first methodological stage: Agent Initialization. Agent Initialization includes environment setup, agent setup, and event setup. This phase must define configuration only. Do not implement domain models or loaders yet unless minimal syntax helpers are needed for tests.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Create stable JSON configuration files for the clean final repository:
- agents
- simulation timeline
- target events
- retrieval configurations
- settings
- placeholder ground truth

Files to create or update:
- config/agents.json
- config/simulation_steps.json
- config/target_events.json
- config/retrieval_configs.json
- config/settings.json
- config/ground_truth.json
- tests/test_config_files_exist.py
- tests/test_config_static_shape.py

Configuration requirements:

1. agents.json
Must contain exactly 4 fixed agent IDs:
- teacher
- active_student
- quiet_student
- academic_student

Each agent should include:
- agent_id
- display_name
- role
- archetype
- persona summary
- traits
- goals
- communication_style
- motivations
- seed_memories

Use the agreed clean names unless the user later changes them:
- teacher: Ms. Sunday
- active_student: Monday
- quiet_student: Tuesday
- academic_student: Wednesday

Seed memories must be useful for persona and category alignment but must not directly encode all target event answers. Use stable category labels such as:
- teacher_guidance
- classroom_norms
- social_learning
- quiet_style
- academic_focus
- exam_concern
- study_strategy
- question_asking

2. simulation_steps.json
Must define the 6-day hybrid timeline:
- Day 1: exactly 2 natural_interaction_block steps and no target event.
- Days 2–6: exactly 1 natural_interaction_block followed by exactly 1 target_event step per day.
- Total natural blocks: 7.
- Total target event steps: 5.

Each natural block step should include:
- step_id
- day
- period
- step_type = natural_interaction_block
- title
- description
- participants
- turns_per_agent = 3
- total_turns = 12
- teacher_opens_block = true
- turn_policy = controlled_fixed or controlled_semi_dynamic
- memory_generation_enabled = true
- evaluation_enabled = false

Each target event step should include:
- step_id
- day
- period
- step_type = target_event
- event_id
- title
- description
- participants
- memory_generation_enabled = true
- evaluation_enabled = true

3. target_events.json
Must contain exactly 5 target events in fixed order:
- event_01: Pengumuman Ujian
- event_02: Perubahan Aturan Kelas
- event_03: Diskusi Kelompok
- event_04: Pemilihan Ketua Kelas
- event_05: Ujian Harian

Each event should include:
- event_id
- title
- description
- expected_memory_themes
- evaluation_role

4. retrieval_configs.json
Must contain exactly 4 configurations:
- balanced: recency 0.33, importance 0.33, relevance 0.33
- recency_heavy: recency 0.60, importance 0.20, relevance 0.20
- importance_heavy: recency 0.20, importance 0.60, relevance 0.20
- relevance_heavy: recency 0.20, importance 0.20, relevance 0.60

5. settings.json
Must enforce:
- top_k = 5
- repetitions = 3
- simulation_days = 6
- natural_blocks_total = 7
- target_events_total = 5
- turns_per_agent_per_natural_block = 3
- total_turns_per_natural_block = 12
- teacher_opens_natural_block = true
- natural_block_primary_evaluation = false
- target_event_primary_evaluation = true
- response limits for natural response, target event response, and reflection
- dummy mode defaults safe

6. ground_truth.json
Create a placeholder/finalizable structure for target-event MRA. It must only apply to target events by default. Include category sets per event-agent pair, but mark that these are initial configurable values and should be frozen before final experiment.

Testing requirements:
Add static JSON tests that verify:
- all config files exist
- JSON is parseable
- exactly 4 agents exist
- exactly 7 natural blocks exist
- exactly 5 target event steps exist
- Day 1 has 2 natural blocks and no target event
- Days 2–6 each have 1 natural block and 1 target event
- top_k = 5
- repetitions = 3
- retrieval configs exist and have valid weight keys

Acceptance criteria:
- `pytest` passes.
- Config files are readable JSON.
- No code-level loader is required yet unless minimal helper is needed for tests.

Forbidden:
- Do not implement simulation execution.
- Do not implement retrieval engine.
- Do not implement LLM calls.
- Do not implement visualizer.
- Do not introduce agent IDs other than the fixed four.
- Do not make natural blocks primary evaluation targets.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
