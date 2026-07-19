# Document 02 — Final Clean Implementation Specification

**Final Clean Implementation Specification for Classroom-Based Generative Agents**  
**Spesifikasi Implementasi Bersih untuk Agen Generatif Berbasis Kelas**

**Penelitian:** Pengaruh Konfigurasi Bobot Retrieval Memori terhadap Konsistensi Perilaku Agen Generatif Berbasis Large Language Model  
**Repository target:** `classroom-memory-retrieval-agents` / `classroom-generative-agents-clean`  
**Dokumen:** Document 02 — Final Clean Implementation Specification  
**Versi:** 1.0 — Clean Final Implementation  
**Peran dokumen:** Acuan teknis implementasi repo baru berdasarkan Document 01 — Final Clean Architecture Blueprint  

---

## Daftar Isi Ringkas

1. Executive Summary  
2. Implementation Philosophy and Scope  
3. Architecture Contract Reference  
4. Repository Structure Specification  
5. Configuration Specification  
6. Domain Model Specification  
7. Memory Stream Specification  
8. Retrieval Engine Specification  
9. Prompt Specification  
10. Minimal Output Guardrail Specification  
11. LLM Service Specification  
12. Cognitive Pipeline Specification  
13. Simulation Controller Specification  
14. Logging and Storage Specification  
15. Evaluation Specification  
16. Ground Truth and Category Alignment Specification  
17. MRA Debug Protocol  
18. Pilot and Cost-Control Protocol  
19. Log Replay Web Visualizer Specification  
20. Full Experiment Execution Specification  
21. Analysis Export Specification  
22. Testing and Validation Plan  
23. Phase-by-Phase Implementation Roadmap  
24. Freeze and Change Control Policy  
25. Final Implementation Checklist  
26. Appendices  

---

## 1. Executive Summary

### 1.1 Tujuan Dokumen

Document 02 ini adalah spesifikasi teknis final untuk membangun ulang repository penelitian agen generatif berbasis kelas secara bersih. Dokumen ini menurunkan keputusan arsitektur dari Document 01 — Final Clean Architecture Blueprint ke bentuk implementasi praktis: struktur repository, konfigurasi, data model, memory stream, retrieval engine, cognitive pipeline, prompt, logging, evaluasi, testing, roadmap, dan freeze policy.

Dokumen ini dibuat untuk mencegah masalah yang muncul pada prototype sebelumnya: implementasi menjadi terlalu banyak patch, prompt menjadi terlalu banyak aturan, output guard mulai mengatur isi perilaku agent, dan fokus pengembangan sempat bergeser dari retrieval-weight experiment ke kualitas percakapan natural. Dalam repo baru, seluruh implementasi harus dikembalikan ke tujuan utama penelitian: menguji pengaruh konfigurasi bobot retrieval memory terhadap MRA dan BCR pada controlled classroom simulation.

### 1.2 Prinsip Implementasi Final

Repo baru harus mengikuti prinsip berikut:

1. **Thesis-first, not demo-first.** Sistem dibangun untuk menjawab pertanyaan penelitian, bukan untuk membuat demo percakapan paling realistis.
2. **Core experiment must remain isolated.** Satu-satunya variabel bebas adalah konfigurasi bobot retrieval memory.
3. **Natural interaction is controlled memory-building.** Natural block digunakan untuk menghasilkan memory stream, bukan untuk mensimulasikan percakapan manusia asli secara bebas.
4. **Target events are evaluation anchors.** MRA dan BCR utama dihitung hanya pada target event.
5. **Prompt and guardrails must be minimal.** Prompt tidak boleh menjadi mekanisme tersembunyi yang mengalahkan pengaruh retrieval.
6. **Visualizer is read-only replay.** Website visualizer membaca log; ia tidak menjalankan LLM, retrieval, atau memory mutation.
7. **Freeze before full experiment.** Setelah pilot diterima, konfigurasi, prompt, retrieval code, dan evaluation code tidak boleh diubah kecuali ada bug fatal.

### 1.3 Final Design Summary

| Item | Final Decision |
|---|---|
| Simulation mode | Hybrid Controlled Natural Interaction + Target Event Evaluation |
| Environment | Text-only classroom simulation |
| Agents | 4 agents: teacher, active_student, quiet_student, academic_student |
| Simulation duration | 6 simulation days |
| Day 1 | 2 natural interaction blocks |
| Day 2-6 | 1 natural interaction block + 1 target event per day |
| Natural block | 12 turns: 4 agents x 3 turns per agent |
| Target events | 5 controlled classroom events |
| Retrieval components | Recency, importance, relevance |
| Top-K | 5 memories |
| Retrieval configurations | balanced, recency_heavy, importance_heavy, relevance_heavy |
| Repetitions | 3 repetitions per configuration |
| Total runs | 12 runs |
| Primary MRA | Target events only |
| Primary BCR | Target event responses only |
| Visualization | Read-only log replay web visualizer |

### 1.4 Perbedaan Utama dari Blueprint Lama

Document 02 versi sebelumnya sudah lengkap secara teknis, tetapi masih memberi ruang terlalu besar untuk pengembangan yang bersifat eksploratif. Document 02 baru ini menambahkan pembatasan yang lebih eksplisit:

- menambahkan architecture contract reference;
- menambahkan freeze and change control policy;
- membuat prompt lebih minimal;
- membuat output guardrail lebih teknis dan tidak mengatur tema;
- menambahkan MRA debug protocol;
- memisahkan visualizer sebagai replay-only layer;
- menambahkan allowed files, forbidden files, acceptance criteria, dan stop condition per phase;
- menegaskan bahwa dialogue realism bukan tujuan utama.

---

## 2. Implementation Philosophy and Scope

### 2.1 Filosofi Implementasi

Repo baru harus dibangun sebagai **controlled experimental software**, bukan sebagai aplikasi simulasi sosial bebas. Dalam konteks skripsi, sistem harus cukup kompleks untuk merepresentasikan mekanisme generative agents: persona, memory stream, retrieval, plan, act, reflect, dan evaluation. Namun sistem juga harus cukup sederhana agar variabel eksperimen tetap terisolasi.

Jika sistem terlalu sederhana, memory retrieval tidak benar-benar terlihat. Jika sistem terlalu bebas, retrieval-weight effect sulit dipisahkan dari variabel lain seperti topik percakapan, randomness LLM, urutan bicara, prompt bias, atau guardrail tambahan. Karena itu, desain final menggunakan pendekatan hybrid: natural block membangun memori secara terkontrol, target event menjaga evaluasi tetap stabil.

### 2.2 Scope Implementasi

Repo baru harus mencakup:

- JSON-based configuration loading;
- strict configuration validation;
- domain models untuk agent, memory, step, event, retrieval, log, dan evaluation;
- per-agent memory stream;
- importance scoring hook;
- embedding hook;
- weighted retrieval engine;
- deterministic dummy LLM mode;
- real LLM service abstraction;
- cognitive pipeline: perceive, retrieve, plan, act, reflect;
- natural block runner;
- target event runner;
- simulation controller;
- experiment controller;
- JSONL logging;
- MRA calculation;
- BCR export and judge interface;
- pilot safeguards;
- full experiment runner;
- analysis export;
- log replay visualizer exporter;
- static web visualizer.

### 2.3 Out-of-Scope Implementasi

Repo baru tidak boleh mengimplementasikan:

| Out-of-Scope Component | Alasan |
|---|---|
| Maze / tile map | Penelitian tidak menguji spatial navigation |
| Pathfinding | Tidak dibutuhkan pada classroom text simulation |
| Spatial memory | Tidak relevan untuk variabel retrieval weight penelitian |
| Full Smallville frontend | Terlalu kompleks dan tidak diperlukan untuk RQ |
| Real-time browser simulation engine | Berisiko mengubah sumber data dan menambah kompleksitas |
| Human-in-the-loop live chat | Mengganggu kontrol eksperimen |
| Open-ended social world | Ground truth MRA menjadi tidak stabil |
| Automatic weight optimization | Penelitian membandingkan konfigurasi tetap, bukan mencari optimum global |
| Theme guard / banned phrase semantic guard | Dapat menutupi efek retrieval weight |
| Hidden retrieval multiplier | Membuat bobot eksperimen tidak lagi transparan |

### 2.4 Definisi Implementasi Bersih

Implementasi bersih berarti:

1. Setiap modul memiliki tanggung jawab jelas.
2. Retrieval engine tidak memanggil LLM untuk memilih kandidat.
3. Prompt tidak memprioritaskan recency, importance, atau relevance secara eksplisit.
4. Visualizer tidak mengubah state simulasi.
5. Semua logs cukup lengkap untuk audit.
6. Setiap run dapat direproduksi dari config snapshot dan prompt snapshot.
7. Full experiment hanya dijalankan setelah semua freeze gate terpenuhi.

---

## 3. Architecture Contract Reference

### 3.1 Fungsi Architecture Contract

Architecture contract adalah daftar keputusan desain yang tidak boleh berubah selama implementasi. File ini harus dibuat pada repo baru sebagai:

```text
/docs/ARCHITECTURE_CONTRACT.md
```

Semua prompt Codex atau coding assistant harus membaca dan mengikuti file ini sebelum mengubah kode. Tujuannya adalah mencegah implementasi melebar seperti pada prototype sebelumnya.

### 3.2 Non-Negotiable Experimental Invariants

| Invariant | Nilai Final | Alasan |
|---|---|---|
| Agent IDs | teacher, active_student, quiet_student, academic_student | Menjaga konsistensi log, ground truth, dan evaluasi |
| Agent count | 4 | Scope sesuai desain classroom testbed |
| Simulation days | 6 | Menyediakan urutan memori dan event yang cukup |
| Natural blocks | 7 per run | Memory-building context terkontrol |
| Target events | 5 per run | Anchor evaluasi MRA/BCR |
| Natural block turns | 12 | 4 agents x 3 turns |
| Turns per agent | 3 | Kesempatan interaksi seimbang |
| Teacher opening | True | Menjaga konteks kelas tetap terkendali |
| Top-K | 5 | Jumlah memori konstan untuk evaluasi |
| Retrieval components | recency, importance, relevance | Variabel utama penelitian |
| Retrieval formula | Weighted sum after normalization | Transparan dan dapat diaudit |
| Retrieval configs | 4 fixed configs | Perbandingan terkontrol |
| Repetitions | 3 per config | Mengurangi efek randomness |
| Primary MRA | target_event only | Ground truth stabil |
| Primary BCR | target_event behavior only | Evaluasi perilaku pada konteks sama |
| Visualizer | read-only replay | Tidak memengaruhi data penelitian |

### 3.3 What May Change Before Freeze

Sebelum freeze, hal berikut masih boleh disesuaikan dengan hati-hati:

- wording natural block description;
- detail persona text selama agent ID dan role tetap;
- seed memory text selama category alignment jelas;
- prompt wording minimal untuk memperbaiki format output;
- JSON schema minor selama tidak merusak evaluation flow;
- file organization selama module boundary tetap jelas.

### 3.4 What Must Not Change After Freeze

Setelah config/prompt/evaluation freeze, hal berikut tidak boleh diubah:

- agent IDs;
- target event list;
- target event order;
- simulation timeline;
- retrieval weight values;
- Top-K;
- retrieval scoring formula;
- prompt templates;
- ground truth category set;
- MRA matching logic;
- BCR rubric;
- model and temperature settings;
- primary evaluation filter.

### 3.5 Architecture Contract Enforcement

Implementasi harus memiliki test yang memastikan kontrak ini tidak dilanggar. Minimal test:

```text
test_architecture_contract.py
test_agents_config.py
test_simulation_steps_config.py
test_retrieval_configs.py
test_evaluation_filters.py
```

Jika salah satu test kontrak gagal, implementasi tidak boleh dilanjutkan ke phase berikutnya.

---

## 4. Repository Structure Specification

### 4.1 Struktur Repository Final

Repository baru direkomendasikan menggunakan struktur berikut:

```text
classroom-memory-retrieval-agents/
├── README.md
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   ├── ARCHITECTURE_CONTRACT.md
│   ├── METHODOLOGY_CHECKPOINT_01_AGENT_INITIALIZATION.md
│   ├── METHODOLOGY_CHECKPOINT_02_RETRIEVAL_CONFIGURATION.md
│   ├── METHODOLOGY_CHECKPOINT_03_SIMULATION_EXECUTION.md
│   ├── METHODOLOGY_CHECKPOINT_04_EVALUATION_DESIGN.md
│   └── FREEZE_LOG.md
├── config/
│   ├── agents.json
│   ├── simulation_steps.json
│   ├── target_events.json
│   ├── retrieval_configs.json
│   ├── ground_truth.json
│   └── settings.json
├── prompts/
│   ├── natural_interaction.txt
│   ├── target_event_response.txt
│   ├── plan.txt
│   ├── reflect.txt
│   ├── importance_scoring.txt
│   ├── bcr_judge.txt
│   └── ground_truth_annotation.txt
├── src/
│   ├── domain/
│   ├── config_loader/
│   ├── memory/
│   ├── retrieval/
│   ├── cognition/
│   ├── llm/
│   ├── simulation/
│   ├── logging_system/
│   ├── evaluation/
│   ├── storage/
│   ├── analysis/
│   └── visualizer/
├── visualizer/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── sample_data/
├── scripts/
├── tests/
├── data/
│   ├── ground_truth_annotations/
│   └── evaluation_exports/
├── logs/
├── resources/
│   ├── primary/
│   ├── papers/
│   └── original_repo/
└── prompts_codex/
```

### 4.2 Module Responsibilities

| Module | Responsibility |
|---|---|
| `domain` | Data structures and typed entities |
| `config_loader` | Load, validate, and normalize config files |
| `memory` | Create, store, update, and query per-agent memory stream |
| `retrieval` | Score candidates and return Top-K memory results |
| `cognition` | Build perceive/retrieve/plan/act/reflect pipeline |
| `llm` | Abstract dummy and real LLM calls |
| `simulation` | Run timeline, natural blocks, target events, and full experiment |
| `logging_system` | Write JSONL logs and schema-compliant summaries |
| `evaluation` | Compute MRA, export BCR dataset, optionally run BCR judge |
| `storage` | Manage run directories, snapshots, and output safety |
| `analysis` | Aggregate final metrics and create thesis-ready tables |
| `visualizer` | Export replay JSON from logs |

### 4.3 Command-Line Interface Scope

The CLI should support these commands:

```bash
python3 main.py validate-config
python3 main.py run-dummy-block
python3 main.py run-pilot --dummy
python3 main.py run-pilot --real --allow-real-llm true
python3 main.py run-single-repetition --config balanced --repetition 1 --dummy
python3 main.py run-full-experiment --dummy
python3 main.py run-full-experiment --real --allow-real-llm true --confirm-full-experiment RUN_12_CONFIGURED_RUNS
python3 main.py evaluate --experiment-dir logs/<experiment_id>
python3 main.py export-visualizer-data --run-dir logs/<run_id> --evaluation-dir evaluation/<run_id> --output visualizer/sample_data/<run_id>.json
python3 main.py summarize-experiment --experiment-dir logs/<experiment_id>
```

CLI commands should fail visibly when required arguments are missing. Full real experiment must require an explicit confirmation token.

---

## 5. Configuration Specification

### 5.1 Configuration Philosophy

Configuration files define the experimental setup. They should be treated as research artifacts, not merely software settings. Once validated and approved, configuration files become part of the methodology and must be frozen before final experiment.

The configuration layer must ensure that the experiment can be repeated and audited. Every run must copy configuration snapshots into its run directory.

### 5.2 Required Configuration Files

| File | Purpose | Freeze Level |
|---|---|---|
| `agents.json` | Defines four agents, personas, and seed memories | Config Freeze |
| `simulation_steps.json` | Defines 6-day hybrid timeline | Config Freeze |
| `target_events.json` | Defines five target events | Config Freeze |
| `retrieval_configs.json` | Defines four retrieval weight configurations | Config Freeze |
| `ground_truth.json` | Defines expected categories for MRA | Evaluation Freeze |
| `settings.json` | Defines Top-K, repetitions, model settings, limits | Config/Prompt Freeze depending field |

### 5.3 `agents.json` Specification

Each agent must contain:

- `agent_id`
- `name`
- `role`
- `archetype`
- `persona`
- `communication_style`
- `goals`
- `motivations`
- `behavioral_tendencies`
- `seed_memories`

Recommended agent IDs:

| Agent ID | Display Name | Role | Purpose in Simulation |
|---|---|---|---|
| `teacher` | Ms. Sunday | Teacher / instructor | Controls class context and provides instruction |
| `active_student` | Monday | Active student | Social, curious, discussion-oriented |
| `quiet_student` | Tuesday | Quiet student | Reserved, cautious, concise |
| `academic_student` | Wednesday | Academic student | Structured, achievement-oriented, study-focused |

Names may be adjusted before config freeze, but agent IDs must not change. Agent IDs are used in logs, ground truth, and evaluation scripts.

### 5.4 `simulation_steps.json` Specification

The timeline must contain 12 steps:

| Day | Step Type | Purpose |
|---|---|---|
| Day 1 | Natural Block 1 | Initial classroom tone and participation memory |
| Day 1 | Natural Block 2 | Learning habits and classroom support memory |
| Day 2 | Natural Block 3 | Context before exam announcement |
| Day 2 | Target Event 1 | Pengumuman Ujian |
| Day 3 | Natural Block 4 | Context after exam awareness |
| Day 3 | Target Event 2 | Perubahan Aturan Kelas |
| Day 4 | Natural Block 5 | Preparation for group discussion |
| Day 4 | Target Event 3 | Diskusi Kelompok |
| Day 5 | Natural Block 6 | Context for leadership and responsibility |
| Day 5 | Target Event 4 | Pemilihan Ketua Kelas |
| Day 6 | Natural Block 7 | Final review before quiz |
| Day 6 | Target Event 5 | Ujian Harian |

Natural block descriptions should be clear but not over-scripted. They should guide the classroom topic without dictating exact agent responses.

### 5.5 `target_events.json` Specification

The target events are controlled evaluation anchors:

| Event ID | Title | Evaluation Purpose |
|---|---|---|
| `event_01` | Pengumuman Ujian | Tests exam awareness, teacher instruction, student concern, study orientation |
| `event_02` | Perubahan Aturan Kelas | Tests rule awareness, prior behavior, classroom norms |
| `event_03` | Diskusi Kelompok | Tests group coordination, quiet participation, social learning, academic contribution |
| `event_04` | Pemilihan Ketua Kelas | Tests leadership, peer trust, cautious participation, responsibility |
| `event_05` | Ujian Harian | Tests exam preparation, prior announcement recall, confidence/anxiety, study strategy |

Event descriptions must be identical across all configurations and repetitions.

### 5.6 `retrieval_configs.json` Specification

The four retrieval configurations are:

| Config ID | Recency | Importance | Relevance | Purpose |
|---|---:|---:|---:|---|
| `balanced` | 0.33 | 0.33 | 0.33 | Baseline balanced retrieval |
| `recency_heavy` | 0.60 | 0.20 | 0.20 | Tests dominance of recent memories |
| `importance_heavy` | 0.20 | 0.60 | 0.20 | Tests dominance of important memories |
| `relevance_heavy` | 0.20 | 0.20 | 0.60 | Tests dominance of semantically relevant memories |

No hidden multiplier is allowed. The retrieval engine must use these weights exactly after component normalization.

### 5.7 `settings.json` Specification

`settings.json` should contain:

```json
{
  "simulation_days": 6,
  "natural_blocks_total": 7,
  "target_events_total": 5,
  "turns_per_agent_per_natural_block": 3,
  "total_turns_per_natural_block": 12,
  "teacher_opens_natural_block": true,
  "top_k": 5,
  "repetitions": 3,
  "primary_mra_step_type": "target_event",
  "primary_bcr_step_type": "target_event",
  "natural_block_primary_evaluation": false,
  "target_event_primary_evaluation": true,
  "dummy_mode_default": true,
  "response_limits": {
    "natural_interaction_sentences": 2,
    "target_event_max_paragraphs": 1,
    "reflection_sentences": 1
  }
}
```

### 5.8 Configuration Validation Rules

The validator must enforce:

- exactly four agent IDs;
- all agent IDs are unique;
- Day 1 has exactly two natural blocks;
- Days 2-6 each have one natural block and one target event;
- there are exactly seven natural blocks;
- there are exactly five target events;
- each natural block has 12 turns;
- each natural block gives 3 turns to every agent;
- teacher opens every natural block;
- target events reference valid event IDs;
- retrieval weights are non-negative;
- retrieval weights sum to approximately 1.0;
- settings Top-K equals 5;
- settings repetitions equals 3 for final experiment;
- ground truth references target events only by default.

---

## 6. Domain Model Specification

### 6.1 Purpose of Domain Models

Domain models turn configuration and runtime state into clear program entities. They prevent the system from becoming a collection of loosely structured dictionaries. Because the experiment depends on traceability, every entity should have explicit fields and validation behavior.

### 6.2 Required Core Entities

| Entity | Description |
|---|---|
| `AgentProfile` | Static persona and identity information |
| `SeedMemoryConfig` | Seed memories loaded at run initialization |
| `MemoryEntry` | One memory in an agent memory stream |
| `SimulationStep` | One scheduled natural block or target event step |
| `TargetEvent` | Controlled evaluation event |
| `RetrievalConfig` | Retrieval weights for one condition |
| `RetrievalCandidate` | One memory candidate with component scores |
| `RetrievalResult` | Full retrieval output including Top-K |
| `BehaviorRecord` | One agent response record |
| `ReflectionRecord` | One reflection output |
| `RunMetadata` | Metadata for one run |
| `EvaluationRecord` | MRA/BCR record |

### 6.3 `AgentProfile`

Fields:

```text
agent_id: str
name: str
role: str
archetype: str
persona_summary: str
traits: list[str]
goals: list[str]
communication_style: str
motivations: list[str]
behavioral_tendencies: list[str]
seed_memories: list[SeedMemoryConfig]
```

Validation:

- `agent_id` must be one of the four fixed IDs.
- `name` must not be empty.
- `persona_summary` must not include retrieval weight information.
- `seed_memories` must include category labels.

### 6.4 `MemoryEntry`

Fields:

```text
memory_id: str
agent_id: str
run_id: str
memory_type: str
text: str
category: str
importance: int | float
embedding: list[float] | None
created_at: int
last_accessed_at: int
source_step_id: str
source_step_type: str
metadata: dict
```

Valid memory types:

```text
seed_persona
observation
plan
action
reflection
```

Important rules:

- `text` must not be empty.
- `category` must not be empty in final mode.
- `importance` must be between 1 and 10.
- `embedding` must exist in final real mode unless memory is explicitly marked dummy.
- `agent_id` indicates the owner of the memory.
- Other agents' private memory must not be accessed directly.

### 6.5 `SimulationStep`

Fields:

```text
step_id: str
day: int
period: str
step_type: str
title: str
description: str
participants: list[str]
is_target_event: bool
evaluation_enabled: bool
memory_generation_enabled: bool
turns_per_agent: int | None
total_turns: int | None
teacher_opens_block: bool | None
event_id: str | None
```

Valid `step_type` values:

```text
natural_interaction_block
target_event
```

### 6.6 `RetrievalResult`

Fields:

```text
run_id: str
step_id: str
step_type: str
turn_id: str
agent_id: str
query: str
weights: dict
candidate_count: int
top_k: int
candidates: list[RetrievalCandidate]
selected_memory_ids: list[str]
```

The `candidates` list should include component scores and selected flags. Evaluation and debugging require visibility into how Top-K was produced.

---

## 7. Memory Stream Specification

### 7.1 Purpose of Memory Stream

Memory stream is the long-term context source for each agent. It stores experiences created during the simulation and supplies candidates to retrieval. It is central to the research because retrieval weights only matter if there is a meaningful memory stream to select from.

### 7.2 Memory Ownership

Each agent owns its own memory stream. The retrieval candidate pool for an agent is that agent's memory stream only. The system may store observations about what other agents said or did, but those observations must become memories owned by the observing agent. The retrieval engine must not directly retrieve from another agent's private memory stream.

### 7.3 Memory Creation Sources

| Source | Memory Type | Description |
|---|---|---|
| Agent initialization | `seed_persona` | Stable initial persona/background memory |
| Perception | `observation` | Agent observes classroom context, conversation, or event |
| Planning | `plan` | Short internal plan summary |
| Action/response | `action` | Agent utterance or behavior text |
| Reflection | `reflection` | One higher-level insight after block/event |

### 7.4 Seed Memory Rules

Seed memories should:

- define stable persona traits and prior classroom tendencies;
- include category labels useful for later MRA;
- not encode answers to every target event directly;
- be identical across all configs and repetitions;
- be loaded at the start of every run;
- receive embeddings and importance scores consistently.

### 7.5 Memory Category Rules

Memory category labels are important for MRA. Category labels must be specific enough to distinguish expected memory types, but not so narrow that semantically equivalent memories fail to match.

Examples of useful categories:

```text
teacher_guidance
exam_announcement
exam_concern
study_strategy
academic_focus
quiet_style
quiet_participation
social_learning
peer_interaction
classroom_norms
rule_change
group_discussion
leadership
peer_trust
cautious_participation
exam_preparation
```

Avoid categories that are too broad:

```text
memory
classroom
student
response
general
```

### 7.6 Memory Insertion Flow

```text
Create memory text
→ assign memory_type
→ assign category
→ score importance
→ create embedding
→ insert into agent memory stream
→ write memory log
```

In dummy mode, deterministic fake importance and embedding may be used. In final real mode, silent fallback is not allowed.

### 7.7 Last Access Update

`last_accessed_at` should be updated only for selected Top-K memories, not for all candidates. The order must be:

```text
score all candidates
→ write retrieval log
→ select Top-K
→ update last_accessed_at for selected Top-K only
```

This order preserves auditability and makes recency effect traceable.

---

## 8. Retrieval Engine Specification

### 8.1 Retrieval Purpose

The retrieval engine is the experimental core. It determines which memories are supplied to the LLM before plan/action generation. Differences between retrieval configurations should appear through different Top-K memory sets and rankings.

### 8.2 Retrieval Formula

The retrieval score is:

```text
Score(m, q) = w_recency * Norm(Recency(m))
            + w_importance * Norm(Importance(m))
            + w_relevance * Norm(Relevance(m, q))
```

Where:

- `m` is a candidate memory;
- `q` is the retrieval query;
- `Recency(m)` is computed from timestamp distance and/or last access;
- `Importance(m)` is the memory importance score;
- `Relevance(m, q)` is cosine similarity between memory embedding and query embedding;
- all components are normalized across candidate memories before weighting.

### 8.3 Component Calculation

| Component | Calculation | Notes |
|---|---|---|
| Recency | `decay_factor ** delta_t` | Same decay formula for all configs |
| Importance | stored 1-10 score | Scored at memory creation |
| Relevance | cosine similarity | Computed from embeddings |
| Normalization | min-max normalization | If all equal, return 0.5 |
| Final Score | weighted sum | No hidden multipliers |

### 8.4 Candidate Pool

Candidate pool is the agent's memory stream. It includes:

- seed persona memories;
- observations;
- plans;
- action memories;
- reflections.

No direct cross-agent memory retrieval is allowed. Shared context must become observed memory first.

### 8.5 Query Construction

Retrieval query should be generated from:

- current agent role;
- current step title;
- current step description;
- immediate task context;
- event context for target events;
- short conversation context for natural blocks.

The query must not include retrieval config name or weight values. For example, the query must not say “use recent memories” in recency-heavy condition. The only difference between configs must be the numeric weights in the retrieval engine.

### 8.6 Top-K Selection

Rules:

- Top-K must be 5.
- If candidate count is less than 5, return all candidates and log candidate count.
- Tie-break order: higher relevance, then higher importance, then newer timestamp.
- Do not filter candidates after scoring unless invalid data exists.
- Do not apply semantic penalties or theme restrictions.

### 8.7 Retrieval Logging

Retrieval log must include:

- query;
- weights;
- candidate count;
- raw component values;
- normalized component values;
- final score;
- selected flag;
- Top-K memory IDs;
- step_type;
- is_target_event;
- agent_id;
- turn_id.

This is required because MRA and qualitative analysis depend on transparent retrieval traces.

### 8.8 Forbidden Retrieval Behaviors

The retrieval engine must not:

- call LLM to decide which memory is selected;
- use hidden multipliers;
- apply prompt-driven memory filtering;
- change decay formula per config;
- change candidate pool per config;
- expose component scores to the agent prompt;
- retrieve from other agents' private memory streams;
- silently ignore memory without embedding in final mode.

---

## 9. Prompt Specification

### 9.1 Prompt Philosophy

Prompts should provide enough context for the LLM to respond as the correct agent, but they must not become a second retrieval mechanism. The prompt should not tell the LLM to prioritize recent, important, or semantically similar memories. That priority is already handled by the retrieval engine.

### 9.2 Prompt Files

| Prompt File | Purpose |
|---|---|
| `natural_interaction.txt` | Generate natural block response |
| `target_event_response.txt` | Generate response to target event |
| `plan.txt` | Generate short internal plan |
| `reflect.txt` | Generate bounded reflection |
| `importance_scoring.txt` | Score importance of memory |
| `bcr_judge.txt` | Evaluate behavioral consistency |
| `ground_truth_annotation.txt` | Assist ground truth category construction |

### 9.3 Safe Prompt Wording for Retrieved Memories

Use this wording:

```text
Use the provided retrieved memories as context when appropriate.
Do not invent memories that are not provided.
Respond according to your persona and the current classroom situation.
```

Avoid this wording:

```text
Use the most recent memories.
Prioritize important memories.
Use the most semantically relevant memories.
Focus on the highest-scoring memory.
```

The safe wording avoids biasing the response toward recency, importance, or relevance outside the retrieval engine.

### 9.4 Natural Interaction Prompt Requirements

Inputs:

- agent persona summary;
- current day and period;
- natural block title;
- natural block description;
- conversation history in current block;
- retrieved Top-K memory texts;
- response length constraint;
- output JSON schema.

Instruction principles:

- keep response 1-2 sentences;
- stay in classroom context;
- follow persona;
- use provided memories when appropriate;
- do not mention retrieval, scoring, memory IDs, or system instructions;
- do not use placeholders;
- do not repeat the previous response exactly.

### 9.5 Target Event Prompt Requirements

Inputs:

- agent persona summary;
- target event title and description;
- recent natural block summary if available;
- retrieved Top-K memory texts;
- previous target event responses in same event if relevant;
- output JSON schema.

Instruction principles:

- respond to the event;
- remain concise;
- stay consistent with persona and prior context;
- use provided retrieved memories as context when appropriate;
- do not mention implementation details;
- return valid JSON.

### 9.6 Plan Prompt Requirements

The plan prompt should produce a short internal plan, not a full hidden chain-of-thought. It should summarize what the agent intends to do in one sentence. The plan may be stored if enabled, but it should not be long enough to dominate the memory stream.

### 9.7 Reflection Prompt Requirements

Reflection should generate at most one insight per agent per block/event. It should focus on useful behavioral or contextual information that may help future retrieval. It should not produce multiple generic reflections or long moral summaries.

### 9.8 Prompt Output Format

Natural and target event responses should return JSON:

```json
{
  "response": "agent utterance here",
  "memory_to_store": "short factual memory text here",
  "category": "category_label_here"
}
```

Plan output:

```json
{
  "plan": "short plan summary"
}
```

Reflection output:

```json
{
  "reflection": "one concise insight",
  "category": "category_label_here"
}
```

Importance output:

```json
{
  "importance": 1,
  "reason": "brief reason"
}
```

### 9.9 Prompt Freeze Rules

Prompt may be adjusted during pilot only for:

- invalid JSON;
- empty output;
- placeholder output;
- excessive length;
- wrong legacy names;
- schema mismatch.

Prompt must not be tuned after pilot merely because the conversation feels “not natural enough.” That would risk shifting the experiment from retrieval-weight analysis to dialogue optimization.

---

## 10. Minimal Output Guardrail Specification

### 10.1 Guardrail Philosophy

Output guardrail should protect technical validity, not control semantic content. Guardrail is allowed to reject malformed output, but it should not steer agent themes, preferences, or memory usage beyond persona and current context.

### 10.2 Allowed Guardrails

| Guardrail | Action |
|---|---|
| Empty output | Reject and retry |
| Invalid JSON | Retry with repair instruction |
| Placeholder text | Reject and retry |
| Wrong legacy names | Reject and retry |
| Exact duplicate same-agent response | Retry once |
| Excessive length | Trim or retry depending severity |
| Missing required field | Retry or fail visibly |

### 10.3 Forbidden Guardrails

| Forbidden Guardrail | Reason |
|---|---|
| Theme guard | Controls semantic content beyond experiment design |
| Banned academic phrase list | Can hide retrieval effects |
| Rewriting response after generation | Alters agent behavior post hoc |
| Retrieval score penalty based on output theme | Hidden intervention |
| Relevance-only post-filtering | Biases configs toward relevance |
| Prompt-specific behavioral correction per config | Breaks comparability |

### 10.4 Retry Policy

Default retry policy:

```text
1st attempt: normal prompt
2nd attempt: repair prompt for schema/format/technical issue
if still invalid: mark error and fail or skip according to mode
```

For final real experiment, failed critical LLM outputs should not be silently included in metrics. The run should be marked failed or interrupted depending severity.

---

## 11. LLM Service Specification

### 11.1 Purpose

LLMService isolates all model calls from the rest of the system. No module should call the provider API directly except the internal LLM service wrapper.

### 11.2 Required LLM Service Features

| Feature | Requirement |
|---|---|
| Provider abstraction | Support dummy and real provider through same interface |
| Prompt rendering | Fill templates with validated context |
| JSON parsing | Parse and validate structured outputs |
| Retry handling | Retry invalid JSON or technical output errors |
| Usage logging | Log token usage, model, prompt type, call type |
| API key safety | Never log key or raw environment variables |
| Timeout handling | Fail visibly when API timeout occurs |
| Cost visibility | Aggregate usage per run and experiment |
| Model config | Read from settings or environment-safe config |

### 11.3 Dummy Mode

Dummy mode must be deterministic and available for all call types:

- natural response;
- target event response;
- plan;
- reflection;
- importance scoring;
- embedding;
- BCR judge mock.

Dummy mode is not intended to produce realistic behavior. It is intended to test pipeline correctness, logs, schemas, and evaluation flow without API cost.

### 11.4 Real Mode

Real mode should be enabled only through explicit flag:

```bash
--real --allow-real-llm true
```

Full experiment real mode should also require:

```bash
--confirm-full-experiment RUN_12_CONFIGURED_RUNS
```

This prevents accidental API spending.

### 11.5 API Usage Logging

Every LLM call should log:

```text
run_id
step_id
step_type
agent_id
call_type
model
prompt_template
input_tokens
output_tokens
total_tokens
status
error_type if any
```

No API key, endpoint secret, or raw environment variable should enter logs.

---

## 12. Cognitive Pipeline Specification

### 12.1 Shared Cognitive Loop

Natural blocks and target events use the same core cognitive loop:

```text
Perceive
→ Store observation memory
→ Build retrieval query
→ Retrieve Top-K memories
→ Generate plan
→ Generate response/action
→ Store action memory
→ Generate bounded reflection when configured
→ Store reflection memory
→ Write logs
```

The difference between natural block and target event is methodological, not cognitive. Natural blocks build memory; target events provide evaluation anchors.

### 12.2 Perception

Perception converts step context into agent-observable information. For natural blocks, perception includes current classroom topic and conversation history. For target events, perception includes target event description and previous responses within the event.

Perception must not include full memory stream. Memory context must enter through retrieval Top-K.

### 12.3 Retrieval

Every main agent response should be preceded by retrieval. The Top-K memories selected by the retrieval engine must be passed to plan and act prompts as text-only context. Component scores are logged but not given to the LLM.

### 12.4 Planning

Planning generates a short response strategy. The plan is optional for output display but useful for cognitive trace. It should be concise and should not contain hidden long reasoning.

### 12.5 Acting / Responding

The act step generates the agent response. Response length depends on step type:

| Step Type | Response Limit |
|---|---|
| Natural interaction block | 1-2 sentences |
| Target event | one concise paragraph or utterance |

### 12.6 Reflection

Reflection runs after each natural block and target event. It creates at most one reflection per agent per step. Reflection should capture useful continuity, not summarize every turn.

### 12.7 Memory Update

After response and reflection, memory stream is updated. Memory generation must be logged. Each memory must include source step metadata.

---

## 13. Simulation Controller Specification

### 13.1 Simulation Controller Role

Simulation controller executes the validated timeline. It should not contain retrieval scoring logic, prompt templates, or evaluation logic directly. It coordinates modules rather than owning all behavior.

### 13.2 Required Simulation Submodules

| Submodule | Responsibility |
|---|---|
| `StepRunner` | Dispatch step by `step_type` |
| `NaturalBlockRunner` | Execute 12-turn natural block |
| `TargetEventRunner` | Execute target event responses |
| `TurnPolicyManager` | Build and validate speaker order |
| `RunController` | Execute one config/repetition run |
| `ExperimentController` | Execute all 12 runs |

### 13.3 Natural Block Runner

Natural block flow:

```text
load natural block step
→ build turn order
→ force teacher as first speaker
→ run 12 turns
→ each agent receives exactly 3 turns
→ store observations/actions
→ generate bounded reflections
→ write natural block summary
```

Natural block must not be evaluated as primary MRA/BCR.

### 13.4 Target Event Runner

Target event flow:

```text
load event details
→ build event context
→ each agent responds once
→ mark retrieval logs as is_target_event=true
→ mark behavior logs as evaluation_enabled=true
→ generate bounded reflections
→ write event summary
```

Target event order must be fixed.

### 13.5 Turn Policy

Turn policy rules:

- teacher must open each natural block;
- each agent must speak exactly three times;
- total turns must equal 12;
- avoid same speaker twice consecutively when possible;
- final experiment should use reproducible deterministic order unless semi-dynamic logic is explicitly controlled and seeded.

Recommended deterministic turn pattern:

```text
01 teacher
02 active_student
03 academic_student
04 quiet_student
05 active_student
06 teacher
07 academic_student
08 quiet_student
09 academic_student
10 active_student
11 quiet_student
12 teacher
```

### 13.6 Run Initialization

Each run should:

1. create unique `run_id`;
2. create separate run directory;
3. copy config snapshots;
4. copy prompt snapshots or record prompt version;
5. initialize agent memory streams from seed memories;
6. compute or load seed embeddings;
7. start simulation clock;
8. execute timeline.

### 13.7 Run Isolation

Run isolation is critical. Memory from one run must never affect another run. Every run starts from the same initial config and seed state. Output directories must not be overwritten silently.

---

## 14. Logging and Storage Specification

### 14.1 Logging Philosophy

Logging is not optional. It is part of the research method. Without logs, MRA cannot be reproduced, BCR context cannot be audited, and retrieval-weight effects cannot be analyzed.

### 14.2 Run Directory Structure

```text
logs/
└── experiment_<timestamp>/
    ├── experiment_summary.json
    ├── balanced_rep01/
    │   ├── config_snapshot/
    │   ├── prompt_snapshot/
    │   ├── retrieval_logs.jsonl
    │   ├── behavior_logs.jsonl
    │   ├── memory_logs.jsonl
    │   ├── reflection_logs.jsonl
    │   ├── api_usage_logs.jsonl
    │   ├── event_summaries.jsonl
    │   ├── natural_block_summaries.jsonl
    │   ├── errors.jsonl
    │   └── run_summary.json
    └── ...
```

### 14.3 Log Files

| Log File | Purpose |
|---|---|
| `retrieval_logs.jsonl` | Candidate scores and Top-K selection |
| `behavior_logs.jsonl` | Agent responses and retrieved memory IDs |
| `memory_logs.jsonl` | Created memory entries |
| `reflection_logs.jsonl` | Reflection outputs |
| `api_usage_logs.jsonl` | Token and model usage |
| `event_summaries.jsonl` | Target event summaries |
| `natural_block_summaries.jsonl` | Natural block summaries |
| `errors.jsonl` | Explicit errors and recovery status |
| `run_summary.json` | Final run status and aggregate counts |

### 14.4 Required Shared Metadata

Every relevant log should include:

```text
experiment_id
run_id
retrieval_config_id
repetition_index
step_id
step_type
is_target_event
evaluation_enabled
day
period
turn_id
turn_index
agent_id
timestamp
```

### 14.5 Storage Safety Rules

- Never overwrite a completed run directory silently.
- Failed runs must be marked as failed or interrupted.
- Partial logs must be preserved for debugging.
- Metrics aggregation must exclude failed runs unless explicitly included with reason.
- Config snapshots must be copied before run execution.
- Prompt snapshots or prompt hashes must be recorded.
- API keys must never be stored.

---

## 15. Evaluation Specification

### 15.1 Evaluation Separation

Primary metrics and optional analyses must be separated.

| Data Source | Primary MRA | Primary BCR | Optional Analysis |
|---|---:|---:|---|
| Natural block retrieval logs | No | No | Retrieval pattern, memory growth |
| Natural block behavior logs | No | No | Qualitative examples |
| Target event retrieval logs | Yes | Context only | MRA source |
| Target event behavior logs | No | Yes | BCR source |

### 15.2 MRA Definition

Memory Retrieval Accuracy measures how many retrieved Top-K memories match expected categories for a target event-agent pair.

```text
MRA(agent, event) = matched_retrieved_memories / K
K = 5
```

A memory matches when its category is included in the expected category set for that event-agent pair.

### 15.3 BCR Definition

Behaviour Consistency Rate evaluates whether the agent response is consistent with persona and prior context.

BCR dimensions:

| Dimension | Question | Score |
|---|---|---|
| Communication Style | Does the response match speaking style? | 1-5 |
| Focus Consistency | Does the response reflect goals and motivations? | 1-5 |
| Temporal Consistency | Does the response avoid contradiction with prior memory? | 1-5 |

```text
BCR_turn = (style_score + focus_score + temporal_score) / 3
```

### 15.4 Aggregation Levels

| Level | MRA | BCR |
|---|---|---|
| Turn | one agent-event retrieval result | one response score |
| Agent-event | average if multiple records | average if multiple responses |
| Event | average across agents | average across agents |
| Run | average across target events | average across target events |
| Configuration | average across 3 repetitions | average across 3 repetitions |

### 15.5 Evaluation Output Files

Recommended output:

```text
evaluation/
└── experiment_<timestamp>/
    ├── mra_records.jsonl
    ├── bcr_export.jsonl
    ├── bcr_judged_records.jsonl
    ├── mra_summary_by_config.csv
    ├── bcr_summary_by_config.csv
    ├── per_event_summary.csv
    ├── per_agent_summary.csv
    └── evaluation_summary.json
```

### 15.6 BCR Judge Rules

The BCR judge should:

- use a separate evaluator prompt;
- preferably use a stronger/different model than simulation model when feasible;
- not reward only fluency;
- not punish persona-consistent brevity;
- judge the response against persona, event, retrieved memories, and prior relevant context;
- return structured JSON.

---

## 16. Ground Truth and Category Alignment Specification

### 16.1 Purpose

Ground truth defines expected memory categories for each target event-agent pair. It is required for MRA. Since natural block memories may vary in wording, category-based matching is preferred over exact text matching.

### 16.2 Ground Truth Structure

Example:

```json
{
  "event_05": {
    "academic_student": {
      "expected_categories": [
        "exam_announcement",
        "study_strategy",
        "academic_focus",
        "prior_quiz_preparation"
      ]
    }
  }
}
```

### 16.3 Category Alignment Rules

- Use consistent category names across seed, natural block, target event, and ground truth.
- Do not use overly broad categories.
- Merge synonyms before final experiment.
- Freeze ground truth before full experiment.
- Do not change ground truth after seeing final results.

### 16.4 Recommended Ground Truth Construction

Ground truth can be constructed by:

1. defining initial expected categories based on event design;
2. using LLM-assisted annotation from multiple models if available;
3. consolidating category labels;
4. applying majority voting or manual review;
5. freezing `ground_truth.json` before final run.

### 16.5 Example Event Category Themes

| Event | Expected Themes |
|---|---|
| `event_01` | teacher_guidance, exam_announcement, exam_concern, academic_focus, question_asking |
| `event_02` | classroom_norms, rule_change, teacher_guidance, prior_behavior |
| `event_03` | group_discussion, social_learning, quiet_participation, academic_contribution |
| `event_04` | leadership, peer_trust, social_initiative, cautious_participation |
| `event_05` | exam_preparation, prior_announcement, study_strategy, anxiety_confidence |

---

## 17. MRA Debug Protocol

### 17.1 Purpose

MRA can be low for many reasons. A low MRA does not automatically mean the response is bad or the prompt needs tuning. MRA specifically measures retrieval-category match. Therefore debugging MRA must focus on memory, category, retrieval, and matching logic first.

### 17.2 MRA Debug Order

When MRA looks unexpectedly low, check in this order:

1. **Ground truth:** Are expected categories correct for the event-agent pair?
2. **Memory category assignment:** Are memory categories specific and consistent?
3. **Target event observation:** Is event observation stored with correct category?
4. **Natural block memory:** Did relevant memories exist before target event?
5. **Retrieval log:** Were relevant memories candidates but not selected?
6. **Retrieval scoring:** Did the intended config rank memories as expected?
7. **Matching logic:** Is the evaluator matching category labels correctly?
8. **Exporter/visualizer:** Is display missing memory text/category due to export bug?

### 17.3 What Not to Do

Do not immediately fix low MRA by:

- making natural dialogue more explicit;
- adding prompt instructions to mention expected categories;
- injecting ground truth into the agent prompt;
- forcing retrieval to choose event-related memories;
- adding hidden relevance filters;
- changing category labels after seeing final results.

### 17.4 Interpretation Rule

MRA and BCR measure different constructs. A response may be behaviorally consistent even when MRA is low. Conversely, high MRA does not guarantee a persona-consistent response. Both metrics must be reported together.

---

## 18. Pilot and Cost-Control Protocol

### 18.1 Pilot Is Mandatory

The real LLM pilot is mandatory before full experiment because natural blocks dominate turn count. Full experiment should not be run until token usage, output validity, and log completeness are confirmed.

### 18.2 Pilot Stages

| Stage | Mode | Purpose |
|---|---|---|
| Config validation | no LLM | Validate setup |
| Dummy Day 1-2 pilot | dummy | Validate pipeline and logs |
| Dummy full timeline | dummy | Validate all steps and evaluation export |
| Real smoke test | real | Validate model/API config |
| Real Day 1-2 pilot | real | Measure token usage and output quality |
| Freeze | n/a | Lock prompt/config/evaluation |
| Full experiment | real | Generate final data |

### 18.3 Real Pilot Scope

Recommended real pilot:

```text
configuration: balanced
repetition: 1
days: Day 1-2
natural blocks: Day 1 Block 1, Day 1 Block 2, Day 2 Natural Block
target events: event_01 only
```

### 18.4 Pilot Acceptance Criteria

Real pilot is accepted if:

- status completed;
- no placeholder output;
- no wrong legacy names;
- no empty response;
- no exact duplicate same-agent response;
- logs complete;
- retrieval Top-K valid;
- memory logs include category labels;
- MRA export works;
- BCR export works;
- token usage is acceptable;
- no core architecture invariant is violated.

### 18.5 Token Cost Controls

- Keep natural response short.
- Keep reflection one sentence.
- Do not include full memory stream in prompts.
- Do not run BCR judge on natural block responses by default.
- Batch or cache embeddings when feasible.
- Track token usage per call type.

---

## 19. Log Replay Web Visualizer Specification

### 19.1 Purpose

Visualizer helps observe simulation behavior. It is not part of the core experiment and is not the source of research data. It reads logs after simulation and displays them as playback.

### 19.2 Visualizer Architecture

```text
Simulation CLI
→ JSONL logs
→ Visualizer exporter
→ replay JSON
→ static website playback
```

### 19.3 Visualizer Must Not

Visualizer must not:

- call LLM;
- run retrieval;
- mutate memory;
- modify logs;
- change turn order;
- alter evaluation;
- require backend server for simulation;
- act as source of truth.

### 19.4 Visualizer May Show

Visualizer may show:

- classroom map-like layout;
- agent positions/icons;
- current day and step;
- speaker and response;
- retrieved memories;
- retrieval component scores;
- memory stream preview;
- reflection;
- MRA/BCR records if available;
- play/pause/next/previous/speed controls.

### 19.5 Export Command

Recommended command:

```bash
python3 main.py export-visualizer-data \
  --run-dir logs/experiment_<timestamp>/balanced_rep01 \
  --evaluation-dir evaluation/experiment_<timestamp>/balanced_rep01 \
  --output visualizer/sample_data/balanced_rep01.json
```

### 19.6 Methodology Wording

In the thesis, visualizer should be described as:

```text
Visualizer digunakan sebagai alat bantu observasi untuk menampilkan ulang proses simulasi berbasis log. Data penelitian tetap diperoleh dari log terstruktur dan hasil evaluasi MRA/BCR, bukan dari visualizer.
```

---

## 20. Full Experiment Execution Specification

### 20.1 Full Experiment Scope

Full experiment consists of:

```text
4 retrieval configurations x 3 repetitions = 12 runs
```

Each run contains:

```text
7 natural blocks x 12 turns = 84 natural turns
5 target events x 4 agents = 20 target-event turns
Total = 104 main turns per run
```

Full experiment contains:

```text
104 main turns x 12 runs = 1,248 main turns
```

### 20.2 Full Experiment Preconditions

Before full experiment:

- all tests pass;
- config freeze completed;
- prompt freeze completed;
- evaluation freeze completed;
- real pilot accepted;
- token usage reviewed;
- ground truth frozen;
- code committed;
- version tag created;
- output root is empty or conflict policy is explicit.

### 20.3 Full Experiment Command

Recommended real command:

```bash
python3 main.py run-full-experiment \
  --real \
  --allow-real-llm true \
  --confirm-full-experiment RUN_12_CONFIGURED_RUNS \
  --output-root logs/final_experiment_<timestamp> \
  --on-conflict fail
```

### 20.4 Completion Criteria

Full experiment is complete if:

- 12 run directories exist;
- every run has `run_summary.json` with status completed;
- every run has retrieval, behavior, memory, reflection, and usage logs;
- target events completed per run = 5;
- natural blocks completed per run = 7;
- MRA records = 12 x 5 x 4 = 240 if one target response per agent-event;
- BCR export records = 240 under same condition;
- no failed runs are silently included.

---

## 21. Analysis Export Specification

### 21.1 Purpose

Analysis export prepares thesis-ready tables and summaries from evaluation outputs.

### 21.2 Required Analysis Tables

| Output | Purpose |
|---|---|
| `mra_by_config.csv` | Compare MRA mean across configs |
| `bcr_by_config.csv` | Compare BCR mean across configs |
| `mra_by_config_event.csv` | Identify event-specific patterns |
| `bcr_by_config_agent.csv` | Identify persona-specific consistency patterns |
| `retrieval_component_examples.csv` | Show how configs rank memories differently |
| `run_status_summary.csv` | Audit completed/failed/excluded runs |

### 21.3 Qualitative Examples

The analysis should export selected examples:

- one example from balanced;
- one from recency-heavy;
- one from importance-heavy;
- one from relevance-heavy;
- examples where Top-K differs strongly between configurations;
- examples where MRA and BCR diverge.

### 21.4 Thesis Reporting Guidance

Report should include:

- mean and standard deviation per configuration;
- event-level breakdown;
- agent-level breakdown;
- short interpretation of why configs differ;
- limitations of controlled simulation;
- explanation that natural blocks are memory-building context.

---

## 22. Testing and Validation Plan

### 22.1 Test Categories

| Test Category | Purpose |
|---|---|
| Config tests | Validate JSON configuration and architecture constants |
| Domain tests | Validate model construction and invariants |
| Memory tests | Validate memory stream ownership and updates |
| Retrieval tests | Validate scoring, normalization, Top-K, tie-break |
| Turn policy tests | Validate teacher opening and exact quotas |
| Logging tests | Validate JSONL schemas and required metadata |
| Simulation tests | Validate dummy timeline execution |
| Evaluation tests | Validate target_event filtering and MRA calculation |
| Visualizer export tests | Validate replay JSON creation from logs |
| Freeze tests | Validate no post-freeze config/prompt drift |

### 22.2 Required Test Files

```text
tests/test_architecture_contract.py
tests/test_agents_config.py
tests/test_simulation_steps_config.py
tests/test_target_events_config.py
tests/test_retrieval_configs.py
tests/test_settings_constants.py
tests/test_ground_truth_targets_only.py
tests/test_domain_models.py
tests/test_memory_stream.py
tests/test_retrieval_scoring.py
tests/test_retrieval_normalization.py
tests/test_retrieval_top_k.py
tests/test_turn_policy.py
tests/test_logging_schema.py
tests/test_dummy_pipeline.py
tests/test_natural_block_runner.py
tests/test_target_event_runner.py
tests/test_mra_filtering.py
tests/test_bcr_export.py
tests/test_visualizer_export.py
```

### 22.3 Validation Gates

| Gate | Requirement |
|---|---|
| Config gate | All config tests pass |
| Retrieval gate | Synthetic retrieval scenarios pass |
| Logging gate | Logs contain required metadata |
| Dummy gate | Day 1-2 and full dummy timeline pass |
| Evaluation gate | MRA/BCR export works on dummy logs |
| Pilot gate | Real Day 1-2 pilot accepted |
| Freeze gate | Config, prompt, and evaluation frozen |
| Full run gate | 12 completed runs |

---

## 23. Phase-by-Phase Implementation Roadmap

### Phase 01 — Architecture Contract and Repository Skeleton

Purpose: create clean repo structure and architecture contract.

Allowed files:

```text
README.md
main.py
requirements.txt
.env.example
.gitignore
docs/ARCHITECTURE_CONTRACT.md
src/__init__.py
tests/test_repository_scaffold.py
```

Forbidden changes:

```text
Do not implement retrieval.
Do not implement LLM calls.
Do not implement visualizer.
```

Acceptance criteria:

- repo structure exists;
- placeholder CLI runs;
- architecture contract exists;
- basic pytest passes.

Commit:

```bash
git commit -m "phase-01-architecture-contract-and-repository-skeleton"
```

### Phase 02 — Agent Initialization Configuration

Purpose: create core config files for environment, agents, events, retrieval configs, and settings.

Allowed files:

```text
config/agents.json
config/simulation_steps.json
config/target_events.json
config/retrieval_configs.json
config/settings.json
config/ground_truth.json
tests/test_*config*.py
```

Forbidden changes:

```text
Do not implement simulation runner.
Do not implement prompt tuning.
```

Acceptance criteria:

- four agents defined;
- six-day timeline defined;
- seven natural blocks defined;
- five target events defined;
- four retrieval configs defined;
- Top-K=5;
- repetitions=3.

Commit:

```bash
git commit -m "phase-02-agent-initialization-configuration"
```

### Phase 03 — Domain Models and Config Validation

Purpose: implement typed models and strict validators.

Allowed files:

```text
src/domain/
src/config_loader/
tests/test_domain_models.py
tests/test_config_validation.py
```

Acceptance criteria:

- invalid timeline rejected;
- invalid agent ID rejected;
- invalid retrieval weights rejected;
- `python3 main.py validate-config` passes.

Commit:

```bash
git commit -m "phase-03-domain-models-and-config-validation"
```

### Phase 04 — Memory Stream

Purpose: implement per-agent memory stream.

Allowed files:

```text
src/memory/
tests/test_memory_stream.py
```

Acceptance criteria:

- seed memory can be loaded;
- memory can be added;
- memory ownership is enforced;
- last_accessed_at can be updated only for selected IDs.

Commit:

```bash
git commit -m "phase-04-memory-stream"
```

### Phase 05 — Retrieval Engine

Purpose: implement recency, importance, relevance, normalization, weighted sum, Top-K.

Allowed files:

```text
src/retrieval/
tests/test_retrieval_*.py
```

Forbidden changes:

```text
Do not add hidden multiplier.
Do not call LLM from retrieval scoring.
Do not add semantic post-filter.
```

Acceptance criteria:

- balanced/recency/importance/relevance synthetic scenarios pass;
- Top-K exactly 5 when enough candidates;
- fewer-than-K behavior logged;
- tie-break works.

Commit:

```bash
git commit -m "phase-05-retrieval-engine"
```

### Phase 06 — Dummy LLM and Cognitive Pipeline

Purpose: implement deterministic dummy pipeline for perceive/retrieve/plan/act/reflect.

Allowed files:

```text
src/llm/dummy.py
src/cognition/
tests/test_dummy_pipeline.py
```

Acceptance criteria:

- one agent turn can run with dummy LLM;
- retrieved Top-K is passed to plan/act;
- behavior and memory are returned in valid structures.

Commit:

```bash
git commit -m "phase-06-dummy-llm-and-cognitive-pipeline"
```

### Phase 07 — Logging and Storage

Purpose: implement run directories, config snapshots, JSONL logs.

Allowed files:

```text
src/logging_system/
src/storage/
tests/test_logging_schema.py
```

Acceptance criteria:

- logs are append-only JSONL;
- required metadata exists;
- config snapshots copied;
- output conflict policy works.

Commit:

```bash
git commit -m "phase-07-logging-and-storage"
```

### Phase 08 — Simulation Controller

Purpose: implement natural block runner, target event runner, run controller.

Allowed files:

```text
src/simulation/
tests/test_turn_policy.py
tests/test_natural_block_runner.py
tests/test_target_event_runner.py
```

Acceptance criteria:

- natural block has 12 turns;
- each agent has 3 turns;
- teacher opens;
- target events run in fixed order;
- dummy Day 1-2 completes.

Commit:

```bash
git commit -m "phase-08-simulation-controller"
```

### Phase 09 — Evaluation Modules

Purpose: implement MRA and BCR export.

Allowed files:

```text
src/evaluation/
tests/test_mra_filtering.py
tests/test_bcr_export.py
```

Acceptance criteria:

- MRA filters target_event only;
- BCR export filters target_event behavior only;
- natural block logs excluded from primary metrics.

Commit:

```bash
git commit -m "phase-09-evaluation-modules"
```

### Phase 10 — Real LLM Service and Prompts

Purpose: implement provider abstraction and minimal prompts.

Allowed files:

```text
src/llm/
prompts/
tests/test_llm_output_parsing.py
```

Forbidden changes:

```text
Do not change retrieval formula.
Do not add theme guard.
Do not add phrase ban lists.
```

Acceptance criteria:

- smoke test works when API key exists;
- invalid JSON retry works;
- API key never logged;
- dummy tests still pass.

Commit:

```bash
git commit -m "phase-10-real-llm-service-and-prompts"
```

### Phase 11 — Pilot Safeguards

Purpose: implement real pilot command, token limits, and acceptance reports.

Allowed files:

```text
src/simulation/pilot.py
src/storage/
main.py
tests/test_pilot_safeguards.py
```

Acceptance criteria:

- real mode requires explicit flag;
- token budget can abort safely;
- pilot summary generated;
- failed pilot not treated as final data.

Commit:

```bash
git commit -m "phase-11-pilot-safeguards"
```

### Phase 12 — Log Replay Visualizer

Purpose: implement replay exporter and static visualizer.

Allowed files:

```text
src/visualizer/
visualizer/
tests/test_visualizer_export.py
```

Forbidden changes:

```text
Do not modify simulation behavior for visualizer.
Do not add real-time LLM server.
```

Acceptance criteria:

- exporter reads logs and outputs replay JSON;
- website opens replay JSON;
- visualizer is read-only.

Commit:

```bash
git commit -m "phase-12-log-replay-visualizer"
```

### Phase 13 — Full Experiment Runner

Purpose: implement safe full 12-run execution.

Allowed files:

```text
src/simulation/experiment_controller.py
main.py
tests/test_full_experiment_dry_run.py
```

Acceptance criteria:

- dry-run confirms 12 runs;
- real full experiment requires confirmation token;
- output conflict policy defaults to fail;
- failed runs marked explicitly.

Commit:

```bash
git commit -m "phase-13-full-experiment-runner"
```

### Phase 14 — Analysis Export

Purpose: generate thesis-ready tables and summaries.

Allowed files:

```text
src/analysis/
scripts/
tests/test_analysis_export.py
```

Acceptance criteria:

- MRA/BCR summaries generated;
- per-config, per-event, per-agent tables generated;
- failed runs excluded with reason.

Commit:

```bash
git commit -m "phase-14-analysis-export"
```

---

## 24. Freeze and Change Control Policy

### 24.1 Freeze Types

| Freeze | When | Locked Items |
|---|---|---|
| Architecture Freeze | After Document 01 approval | Architecture contract and invariants |
| Config Freeze | After agent/environment/event approval | agents, steps, events, retrieval configs, settings |
| Prompt Freeze | After real pilot acceptance | prompt templates and output schema |
| Evaluation Freeze | After MRA/BCR validation | ground truth, matching logic, judge rubric |
| Full Experiment Freeze | Before final 12 runs | code, config, prompt, evaluation scripts |

### 24.2 Change Request Rule

After freeze, any change must be documented in `docs/FREEZE_LOG.md` with:

```text
change_id
date
changed_files
reason
bug_or_design_change
impact_on_experiment
approval_status
```

### 24.3 Allowed Post-Freeze Changes

Only critical fixes are allowed:

- API failure handling;
- invalid JSON parser bug;
- log writer bug;
- evaluation script crash;
- typo that does not change meaning;
- visualizer display bug that does not affect logs.

### 24.4 Forbidden Post-Freeze Changes

- changing prompt because output feels less natural;
- changing ground truth after seeing results;
- changing retrieval formula;
- changing target event wording;
- changing natural block turn count;
- adding new guard that affects semantic behavior;
- modifying evaluation filter.

---

## 25. Final Implementation Checklist

### 25.1 Architecture Checklist

- [ ] Architecture contract exists.
- [ ] Core experiment, dialogue guardrail, and visualizer are separated.
- [ ] Visualizer is read-only.
- [ ] Natural block is not primary evaluation.
- [ ] Target event is primary evaluation anchor.

### 25.2 Configuration Checklist

- [ ] Four agents defined.
- [ ] Seven natural blocks defined.
- [ ] Five target events defined.
- [ ] Day 1 has two natural blocks.
- [ ] Days 2-6 have natural block + target event.
- [ ] Retrieval configs are fixed.
- [ ] Top-K equals 5.
- [ ] Repetitions equal 3.

### 25.3 Retrieval Checklist

- [ ] Candidate pool is per-agent memory stream.
- [ ] Recency, importance, relevance computed.
- [ ] Normalization tested.
- [ ] Top-K tested.
- [ ] No hidden multiplier.
- [ ] No LLM inside retrieval scoring.
- [ ] All candidates logged.

### 25.4 Prompt and Guardrail Checklist

- [ ] Prompt uses “provided retrieved memories as context when appropriate.”
- [ ] Prompt does not mention recency/importance/relevance priority.
- [ ] No theme guard.
- [ ] No banned phrase list.
- [ ] Guardrail only handles technical validity.
- [ ] Prompt frozen after pilot.

### 25.5 Evaluation Checklist

- [ ] MRA filters target_event only.
- [ ] BCR export filters target_event only.
- [ ] Ground truth frozen.
- [ ] MRA debug protocol documented.
- [ ] Failed runs excluded from aggregation.

### 25.6 Full Experiment Checklist

- [ ] Dummy full timeline completed.
- [ ] Real pilot completed.
- [ ] Token usage reviewed.
- [ ] Config/prompt/evaluation frozen.
- [ ] Code committed and tagged.
- [ ] Full experiment confirmation token used.
- [ ] 12 runs completed.
- [ ] 240 MRA records expected.
- [ ] 240 BCR records expected.

---

## 26. Appendices

### Appendix A — Example `agents.json` Snippet

```json
{
  "agent_id": "quiet_student",
  "name": "Tuesday",
  "role": "student",
  "archetype": "quiet_student",
  "persona": {
    "traits": ["reserved", "careful", "observant"],
    "goals": ["understand class expectations", "avoid unnecessary attention"],
    "communication_style": "brief, cautious, and respectful",
    "motivations": ["clarity", "safety", "preparedness"]
  },
  "seed_memories": [
    {
      "text": "Tuesday usually waits for a clear invitation before speaking in class.",
      "category": "quiet_style",
      "importance": 7
    }
  ]
}
```

### Appendix B — Example Retrieval Log Candidate

```json
{
  "memory_id": "mem_academic_student_00021",
  "raw_recency": 0.981,
  "raw_importance": 8,
  "raw_relevance": 0.742,
  "norm_recency": 0.89,
  "norm_importance": 0.75,
  "norm_relevance": 0.92,
  "final_score": 0.853,
  "selected": true
}
```

### Appendix C — Example Behavior Log

```json
{
  "run_id": "balanced_rep01",
  "step_id": "day_04_event_03",
  "step_type": "target_event",
  "is_target_event": true,
  "evaluation_enabled": true,
  "day": 4,
  "turn_id": "day_04_event_03_turn_03",
  "agent_id": "quiet_student",
  "speaker": "Tuesday",
  "response": "I can help write the notes if the group agrees, but I would rather not present first.",
  "retrieved_memory_ids": ["mem_quiet_0004", "mem_quiet_0012", "mem_quiet_0020"],
  "plan_summary": "Contribute carefully while staying consistent with quiet participation."
}
```

### Appendix D — Example MRA Calculation

Expected categories for `event_05` and `academic_student`:

```text
exam_announcement
study_strategy
academic_focus
prior_preparation
```

Retrieved Top-K categories:

| Rank | Category | Match? |
|---:|---|---|
| 1 | study_strategy | Yes |
| 2 | academic_focus | Yes |
| 3 | classroom_norms | No |
| 4 | exam_announcement | Yes |
| 5 | peer_joke | No |

```text
matched_count = 3
K = 5
MRA = 3 / 5 = 0.60
```

### Appendix E — Example BCR Calculation

Response from `quiet_student`:

```text
I can help write the notes if the group agrees, but I would rather not present first.
```

Scores:

| Dimension | Score | Reason |
|---|---:|---|
| Communication Style | 5 | Brief and cautious |
| Focus Consistency | 4 | Contributes while avoiding high-visibility role |
| Temporal Consistency | 5 | Consistent with prior quiet participation |

```text
BCR = (5 + 4 + 5) / 3 = 4.67
```

### Appendix F — Common Mistakes to Avoid

| Mistake | Why It Is Problematic | Correction |
|---|---|---|
| Using natural blocks in primary MRA | Ground truth becomes unstable | Filter target_event only |
| Adding banned phrase guard | Controls behavior semantically | Keep technical guard only |
| Including full memory stream in prompt | Cost and noise increase | Use Top-K only |
| Showing scores to LLM | LLM may re-rank memories | Scores only in logs |
| Changing prompts after pilot | Results become incomparable | Freeze prompts |
| Running real API too early | Expensive debugging | Use dummy mode first |
| Visualizer modifying logs | Data integrity risk | Read-only replay only |
| Changing ground truth after results | Outcome-dependent bias | Freeze before final run |

### Appendix G — Recommended Development Workflow

```text
1. Write architecture contract.
2. Prepare config and methodology checkpoint.
3. Get supervisor feedback.
4. Implement small phase.
5. Run tests.
6. Commit.
7. Continue only after current phase passes.
8. Run dummy pilot.
9. Run real pilot.
10. Freeze.
11. Run full experiment.
12. Export analysis.
```

### Appendix H — Methodology Alignment

| Research Methodology Stage | Implementation Phases |
|---|---|
| Stage 1: Agent Initialization | Phase 01-04 |
| Stage 2: Retrieval Configuration | Phase 05 |
| Stage 3: Simulation Execution | Phase 06-08, Phase 10-11 |
| Stage 4: Evaluation | Phase 09 |
| Stage 5: Analysis and Reporting | Phase 13-14 |

### Appendix I — Final Note

This document should be treated as the technical control document for the final implementation. If a new feature does not help answer the research questions or preserve experimental validity, it should not enter the core system. The final repository should be clean, controlled, reproducible, and defensible as a thesis implementation.
