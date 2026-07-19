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

# Phase 08 — Simulation Controller

## Codex Prompt

```text
Implement Phase 08 — Simulation Controller.

Context:
The repository now has config validation, memory streams, retrieval, dummy cognitive pipeline, and logging/storage. This phase implements the hybrid simulation controller that executes natural interaction blocks and target events in dummy mode first.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement StepRunner, NaturalBlockRunner, TargetEventRunner, TurnPolicyManager, RunController, and pilot execution in dummy mode.

Files/modules to create or update:
- src/simulation/turn_policy.py
- src/simulation/natural_block_runner.py
- src/simulation/target_event_runner.py
- src/simulation/step_runner.py
- src/simulation/run_controller.py
- src/simulation/experiment_controller.py if needed as skeleton only
- src/simulation/__init__.py
- main.py
- tests/test_turn_policy.py
- tests/test_natural_block_runner_dummy.py
- tests/test_target_event_runner_dummy.py
- tests/test_run_controller_dummy.py
- tests/test_simulation_invariants.py

Turn policy requirements:
- Teacher must open every natural interaction block.
- Each of the four agents must receive exactly 3 turns.
- Total turns must be exactly 12.
- Avoid same speaker consecutively when possible.
- The final experiment should be reproducible. Prefer a deterministic controlled turn order unless a seeded semi-dynamic option is explicitly implemented and tested.

NaturalBlockRunner requirements:
- Execute exactly 12 turns.
- Use the cognitive pipeline for each turn.
- Maintain conversation context for the current block.
- Store action memories.
- Store observation memories if the design enables witnessed observations.
- Generate bounded reflection at the end of the block: at most one insight per agent.
- Mark natural block behavior as evaluation_enabled=false and is_target_event=false.
- Write retrieval, behavior, memory, reflection, and natural block summary logs.

TargetEventRunner requirements:
- Execute each target event for the four agents.
- Use fixed participants from config.
- Use cognitive pipeline for each agent response.
- Mark logs with step_type=target_event, is_target_event=true, evaluation_enabled=true.
- Generate bounded reflection after event if enabled.
- Write event summary logs.

RunController requirements:
- Initialize one run for one retrieval configuration and one repetition.
- Create run directory.
- Copy config snapshot.
- Initialize agents and fresh memory streams from seed memories.
- Execute simulation steps in order.
- Support dummy pilot execution for Day 1–2.
- Write run_summary.json.

CLI requirements:
Add or update commands:
- `python main.py validate-config`
- `python main.py run-pilot --dummy --config balanced --repetition 1 --days 1-2 --output-root <path>`
- Optional: `python main.py run-single-repetition --dummy --config balanced --repetition 1 --output-root <path>`

Testing requirements:
Add tests for:
- valid turn order
- invalid turn order rejection
- natural block has exactly 12 turns
- each agent gets exactly 3 turns
- teacher opens natural block
- target event logs marked evaluation_enabled=true
- natural block logs marked evaluation_enabled=false
- dummy Day 1–2 pilot completes
- run summary status completed for successful dummy pilot
- memory streams are isolated per run

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- Dummy Day 1–2 pilot runs successfully to a temporary output directory.
- No real API calls are used.

Forbidden:
- Do not implement real LLM client in this phase.
- Do not run full real experiment.
- Do not implement evaluation metrics yet if not already planned for next phase.
- Do not implement visualizer.
- Do not change retrieval formula or Top-K.
- Do not make natural blocks primary evaluation points.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
