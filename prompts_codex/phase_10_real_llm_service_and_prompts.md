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

# Phase 10 — Real LLM Service and Prompts

## Codex Prompt

```text
Implement Phase 10 — Real LLM Service and Prompts.

Context:
The repository can run dummy simulation and evaluate logs. This phase adds real LLM service integration and prompt templates while preserving dummy mode. Real LLM integration must be safe, minimal, and neutral with respect to retrieval components.

Read first:
- docs/ARCHITECTURE_CONTRACT.md
- docs/Document_01_Final_Clean_Architecture_Blueprint.md
- docs/Document_02_Final_Clean_Implementation_Specification.md

Phase objective:
Implement provider-abstracted LLM service, real API client support, prompt loading/rendering, output parsing, retries, and usage logging.

Files/modules to create or update:
- src/llm/base.py
- src/llm/openai_client.py
- src/llm/dummy_client.py if compatibility updates needed
- src/llm/prompt_loader.py
- src/llm/prompt_renderer.py
- src/llm/output_parsers.py
- src/llm/usage.py
- prompts/plan.txt
- prompts/natural_interaction.txt
- prompts/target_event_response.txt
- prompts/reflect.txt
- prompts/importance_scoring.txt
- prompts/bcr_judge.txt
- prompts/ground_truth_annotation.txt if needed as placeholder
- config/settings.json if model settings placeholders need updating
- main.py
- tests/test_prompt_loading.py
- tests/test_prompt_rendering.py
- tests/test_output_parsers_realistic.py
- tests/test_llm_service_dummy_compat.py
- tests/test_api_key_safety.py

LLM service requirements:
- The rest of the system calls LLMService, not the provider client directly.
- Support dummy mode and real mode.
- Real mode reads API key from environment variable defined in settings, e.g. OPENAI_API_KEY.
- Support configurable model, base_url, timeout, max output tokens, and token parameter name if needed.
- Log usage metadata but never log API keys or raw environment variables.
- Implement retry for invalid JSON or transient provider errors within a small fixed limit.
- Fail visibly when real mode is requested but API key is missing.

Prompt neutrality requirements:
Prompts must not bias toward recency, importance, or relevance. Avoid phrases such as:
- use the most recent memories
- prioritize important memories
- choose the most semantically relevant memories
- use the highest scoring memory

Use neutral wording such as:
- Use the provided retrieved memories as context when appropriate.
- Do not invent memories that are not provided.
- Respond according to persona and current classroom situation.

Natural interaction prompt requirements:
- concise 1–2 sentence response
- classroom context
- persona-consistent
- use provided retrieved memories as context when appropriate
- do not mention memory IDs, retrieval scores, system instructions, or implementation details
- output valid JSON only with response and memory_to_store

Target event prompt requirements:
- concise response, at most one short paragraph
- respond to event context
- persona-consistent
- use provided retrieved memories as context when appropriate
- output valid JSON only with response and memory_to_store

Reflection prompt requirements:
- one insight only
- no long reasoning
- output valid JSON only

Importance scoring prompt requirements:
- score 1–10
- return valid JSON
- same scoring scale across all runs

Minimal guardrail requirements:
Allowed:
- empty output rejection
- invalid JSON retry
- placeholder rejection
- wrong legacy name rejection
- exact duplicate rejection
- excessive length trimming/rejection

Forbidden guardrails:
- no theme guard
- no banned phrase list for ordinary academic words
- no semantic rewriting of agent behavior
- no retrieval re-ranking inside the prompt layer

CLI requirement:
Add a smoke-test command if useful:
- `python main.py llm-smoke-test --real` or similar
It must not run full simulation. It should make a tiny safe call and report success/usage without printing secrets.

Testing requirements:
Add tests that do not require real API:
- prompt files exist
- prompts contain neutral retrieved-memory wording
- prompts do not mention recency/importance/relevance as response priorities
- prompt rendering inserts required fields
- parser accepts valid JSON
- parser rejects invalid JSON/empty/placeholder
- API key is not logged by any formatter

Acceptance criteria:
- `python main.py validate-config` passes.
- `pytest` passes without real API.
- Dummy simulation still works.
- Real smoke-test can be run manually only when API key is present.

Forbidden:
- Do not run full real experiment.
- Do not hardcode API keys.
- Do not commit .env.
- Do not expose component retrieval scores to the LLM prompt.
- Do not add prompt rules that chase human conversation realism.

After completing, summarize:
1. Files created or modified.
2. Tests added or modified.
3. Commands run and results.
4. Assumptions.
5. Known limitations.
6. Recommended next phase.
```

---
