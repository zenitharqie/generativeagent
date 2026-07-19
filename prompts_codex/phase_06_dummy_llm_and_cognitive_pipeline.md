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

# Phase 06 — Dummy LLM and Cognitive Pipeline

## Codex Prompt

```text
Implement Phase 06 — Dummy LLM and Cognitive Pipeline.

Context:
The repository now has validated configs, memory streams, and retrieval engine. This phase connects the cognitive loop in dummy/offline mode. The objective is to verify perceive → retrieve → plan → act → reflect → memory update without real API calls.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement a deterministic dummy LLM service and the agent cognitive pipeline for both natural interaction turns and target event turns.

Files/modules to create or update:
- src/llm/base.py
- src/llm/dummy_client.py
- src/llm/output_parsers.py
- src/llm/__init__.py
- src/cognition/perception.py
- src/cognition/query_builder.py
- src/cognition/planner.py
- src/cognition/actor.py
- src/cognition/reflector.py
- src/cognition/pipeline.py
- src/cognition/__init__.py
- tests/test_dummy_llm.py
- tests/test_output_parsers.py
- tests/test_cognitive_pipeline_dummy.py
- tests/test_retrieval_context_in_pipeline.py

Cognitive loop requirement:
For each agent turn:
1. Build perception from agent, step context, conversation context, and event context if applicable.
2. Build retrieval query.
3. Retrieve Top-K memories from the agent own memory stream.
4. Generate a short plan using dummy LLM.
5. Generate response using dummy LLM.
6. Store action memory.
7. Optionally generate reflection and store reflection memory.
8. Return a structured turn result.

Dummy LLM requirements:
- Deterministic outputs.
- No external API calls.
- Supports call types:
  - plan
  - natural_response
  - target_event_response
  - reflection
  - importance_score dummy
- Output should be valid structured data matching parser expectations.

Prompt neutrality requirement:
Even in dummy mode, do not introduce instructions that bias toward recency, importance, or relevance. Retrieved memories should be provided as context only.

Output parser requirements:
- Parse valid JSON-like outputs when applicable.
- Reject empty output.
- Reject missing required fields.
- Provide minimal cleanup for harmless wrapping when needed.
- Do not implement heavy semantic rewriting.

Minimal guardrail requirements:
Allowed guardrails only:
- empty output rejection
- placeholder output rejection
- wrong legacy name rejection
- exact duplicate rejection if previous same-agent response is provided
- excessive length trimming or rejection according to settings

Do not implement theme guard, banned phrase lists, or semantic steering.

Testing requirements:
Add tests for:
- dummy LLM returns deterministic plan
- dummy natural response valid
- dummy target event response valid
- parser rejects empty/missing response
- pipeline calls retrieval before response generation
- pipeline stores action memory
- pipeline can store reflection memory
- pipeline uses only provided retrieved memory IDs
- pipeline does not retrieve other agents private memory

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes.
- A single dummy agent turn can run without real API.

Forbidden:
- Do not implement real OpenAI/API client yet.
- Do not implement full simulation timeline controller yet.
- Do not implement logging system beyond minimal return structures unless needed for tests.
- Do not implement visualizer.
- Do not add prompt rules that over-control dialogue content.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
