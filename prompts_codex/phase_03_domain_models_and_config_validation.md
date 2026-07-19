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

# Phase 03 — Domain Models and Config Validation

## Codex Prompt

```text
Implement Phase 03 — Domain Models and Config Validation.

Context:
Phase 02 created the core JSON configuration files. This phase converts those configurations into typed internal structures and validates the architecture invariants. The output of this phase should make invalid experiment structures fail early before any simulation or API call is executed.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement domain models and config loading/validation for the clean final repository.

Files/modules to create or update:
- src/domain/agents.py
- src/domain/memory.py
- src/domain/simulation.py
- src/domain/retrieval.py
- src/domain/settings.py
- src/config_loader/loader.py
- src/config_loader/validators.py
- src/config_loader/__init__.py
- main.py
- tests/test_domain_models.py
- tests/test_config_loader.py
- tests/test_config_validation.py

Domain model requirements:

1. Agent models
Include structures for:
- AgentProfile
- PersonaProfile
- SeedMemoryConfig

AgentProfile must support:
- agent_id
- display_name
- role
- archetype
- persona fields
- seed memories

2. Simulation models
Include structures for:
- SimulationStep
- NaturalInteractionBlockStep
- TargetEventStep
- TargetEvent

Validation must distinguish:
- natural_interaction_block
- target_event

3. Retrieval models
Include structures for:
- RetrievalConfig
- RetrievalWeights

4. Memory model skeleton
Define a memory model that future phases can use:
- memory_id
- agent_id
- memory_type
- text
- category
- importance
- embedding placeholder or optional field
- created_at
- last_accessed_at
- source_step_id
- source_step_type

Do not implement memory stream operations yet beyond model definitions.

5. Settings model
Represent:
- top_k
- repetitions
- simulation constants
- model settings placeholders
- response limits
- evaluation flags

Config loader requirements:
- Load all config JSON files from config/.
- Return a single experiment config object or structured dict.
- Provide clear errors for missing files, invalid JSON, missing keys, or invalid constants.

Validation requirements:
Reject invalid configurations when:
- agent IDs are not exactly teacher, active_student, quiet_student, academic_student
- there are not exactly 4 agents
- simulation_days is not 6
- natural_blocks_total is not 7
- target_events_total is not 5
- top_k is not 5
- repetitions is not 3
- Day 1 does not have exactly 2 natural blocks and no target event
- Days 2–6 do not each have exactly 1 natural block and 1 target event
- a natural block does not have total_turns=12
- a natural block does not have turns_per_agent=3
- teacher_opens_block is not true
- a target event step references a missing event_id
- retrieval weights are negative
- retrieval weights do not approximately sum to 1.0
- ground truth references a non-target event unless explicitly marked optional

CLI requirement:
Update `python main.py validate-config` to load and validate the configuration. It should print a concise success message on valid config and a clear error on invalid config.

Testing requirements:
Add tests for:
- valid config passes
- invalid agent IDs fail
- invalid Day 1 structure fails
- invalid Day 2–6 structure fails
- invalid top_k fails
- invalid retrieval weights fail
- invalid target event reference fails

Acceptance criteria:
- `python main.py validate-config` passes with the current config.
- `pytest` passes.
- No simulation execution is implemented yet.

Forbidden:
- Do not implement retrieval scoring.
- Do not implement memory stream operations.
- Do not implement LLM calls.
- Do not implement simulation controller.
- Do not implement visualizer.
- Do not alter the final experiment constants to make tests easier.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
