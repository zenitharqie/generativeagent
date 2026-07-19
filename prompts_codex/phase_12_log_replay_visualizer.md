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

# Phase 12 — Log Replay Visualizer

## Codex Prompt

```text
Implement Phase 12 — Log Replay Visualizer.

Context:
The core simulation, logging, evaluation, and pilot safeguards exist. This phase adds a read-only log replay visualizer. The visualizer is not a live simulation engine and must not mutate simulation state, memory, retrieval, or evaluation.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement exporter and static web visualizer for replaying existing simulation logs.

Files/modules to create or update:
- src/visualizer/exporter.py
- src/visualizer/schemas.py
- src/visualizer/__init__.py
- visualizer/index.html
- visualizer/styles.css
- visualizer/app.js
- visualizer/sample_data/.gitkeep or README only
- main.py
- tests/test_visualizer_exporter.py
- tests/test_visualizer_schema.py

Visualizer architecture:
Simulation logs → exporter → replay JSON → static website playback

The website must not:
- call LLM APIs
- run simulation
- update memory
- call retrieval engine
- alter evaluation
- write logs
- use web sockets
- require backend server beyond static file serving

Exporter requirements:
Input:
- run directory with JSONL logs
- optional evaluation directory with MRA/BCR outputs

Output:
- one replay JSON file containing:
  - run metadata
  - agents
  - timeline steps
  - natural block turns
  - target event turns
  - agent responses
  - retrieved memories per turn
  - retrieval component scores for display only
  - memory stream snapshots or memory list for display
  - reflections
  - MRA/BCR records if available

Important:
- Retrieval scores may be displayed in visualizer replay JSON, but they must never be fed back into LLM prompts.
- The visualizer uses logs as source of truth.

Website requirements:
- Static HTML/CSS/JS.
- Load a local replay JSON file selected or configured by the user.
- Provide playback controls:
  - play
  - pause
  - next
  - previous
  - speed if simple
- Display:
  - current day/step
  - current speaker
  - agent response
  - retrieved memories
  - retrieval score components
  - reflection if available
  - target event evaluation markers
- Keep the interface simple and readable.
- Do not overbuild.

CLI requirement:
Add command:
- `python main.py export-visualizer-data --run-dir <run_dir> --evaluation-dir <evaluation_dir_optional> --output <output_json>`

Testing requirements:
Add tests for:
- exporter reads dummy pilot logs
- exporter writes valid replay JSON
- replay JSON contains run metadata
- replay JSON contains turn records
- replay JSON contains retrieved memory references when available
- exporter does not mutate original logs
- exporter handles missing optional evaluation data gracefully

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- Exporter can create replay JSON from dummy pilot logs.
- Static website files exist.

Forbidden:
- Do not implement live real-time web simulation.
- Do not implement WebSocket/SSE streaming.
- Do not make visualizer required for experiment execution.
- Do not make visualizer a source of research data.
- Do not alter simulation logs to fit the visualizer.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
