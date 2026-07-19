# Document 01 — Final Clean Architecture Blueprint

**Final Clean Architecture Blueprint for Classroom-Based Generative Agents**  
**Blueprint Arsitektur Bersih untuk Agen Generatif Berbasis Kelas**

**Penelitian:** Pengaruh Konfigurasi Bobot Retrieval Memori terhadap Konsistensi Perilaku Agen Generatif Berbasis Large Language Model  
**Repository target:** `classroom-memory-retrieval-agents` / `classroom-generative-agents-clean`  
**Dokumen:** Document 01 — Final Clean Architecture Blueprint  
**Versi:** 1.0 — Clean Final Architecture  
**Peran dokumen:** Acuan arsitektur final sebelum penyusunan Document 02 dan implementasi repo baru  

---

## Daftar Isi Ringkas

1. Pendahuluan  
2. Posisi Dokumen terhadap Repo Prototype dan Dokumen Lama  
3. Tujuan Penelitian dan Scope Sistem  
4. Lessons Learned dari Prototype Repository  
5. Ringkasan Arsitektur Generative Agents yang Diadaptasi  
6. Architecture Contract  
7. Final Clean System Architecture  
8. Layer 1 — Core Experiment Layer  
9. Layer 2 — Minimal Dialogue Quality Control Layer  
10. Layer 3 — Log Replay Visualization Layer  
11. Agent Initialization Architecture  
12. Environment Setup Architecture  
13. Agent Setup Architecture  
14. Event Setup Architecture  
15. Memory Stream Architecture  
16. Weighted Retrieval Architecture  
17. Cognitive Pipeline Architecture  
18. Hybrid Simulation Architecture  
19. Logging and Evaluation Architecture  
20. MRA and BCR Evaluation Design  
21. Methodology Alignment  
22. Design Decisions: Dipertahankan, Dimodifikasi, Dihilangkan  
23. Risks Avoided in Clean Implementation  
24. Implementation Readiness Checklist  
25. Kesimpulan  
26. Lampiran  

---

## 1. Pendahuluan

### 1.1 Latar Belakang Dokumen

Dokumen ini disusun sebagai **blueprint arsitektur final** untuk membangun ulang repository penelitian agen generatif berbasis kelas secara bersih. Dokumen ini merupakan kelanjutan dari Document 01 dan Document 02 versi sebelumnya, namun posisinya berbeda: dokumen ini tidak lagi hanya menjelaskan adaptasi dari repository Generative Agents/Reverie, melainkan menetapkan **arsitektur final yang akan digunakan sebagai dasar repo baru**.

Pada pengembangan prototype sebelumnya, sistem telah berhasil membuktikan bahwa konsep inti penelitian dapat dijalankan: agent memiliki persona, memory stream, weighted retrieval, plan/act/reflect pipeline, logging, evaluasi MRA/BCR, real LLM pilot, serta visualizer berbasis log. Namun proses pengembangan prototype juga menunjukkan risiko metodologis: ketika output agent terlihat kurang natural, pengembangan cenderung menambahkan banyak prompt rule, sanitizer, guardrail, dan theme-level restriction. Beberapa perbaikan tersebut berguna sebagai validasi teknis, tetapi jika dimasukkan ke implementasi final secara berlebihan, dapat mengaburkan variabel utama penelitian, yaitu **konfigurasi bobot retrieval memory**.

Karena itu, repo baru perlu dibangun dengan prinsip **clean final implementation**. Repo baru tidak dibuat untuk mengejar simulasi percakapan manusia yang sepenuhnya realistis, tidak dibuat sebagai game environment, dan tidak dibuat sebagai web simulation engine. Repo baru dibuat sebagai **controlled experimental testbed** untuk menguji bagaimana perbedaan bobot recency, importance, dan relevance memengaruhi memory retrieval dan konsistensi perilaku agent.

### 1.2 Tujuan Dokumen

Tujuan utama dokumen ini adalah menetapkan arsitektur final yang akan menjadi acuan sebelum implementasi repo baru. Secara spesifik, dokumen ini bertujuan untuk:

1. Menetapkan scope final sistem penelitian.
2. Menjelaskan pelajaran dari prototype repository sebelumnya.
3. Menentukan architecture contract yang tidak boleh dilanggar selama implementasi.
4. Memisahkan core experiment, dialogue quality control, dan visualization layer.
5. Menjelaskan bagaimana environment setup, agent setup, dan event setup ditempatkan dalam arsitektur.
6. Menegaskan bahwa natural interaction block adalah mekanisme pembentukan memory stream, bukan simulasi percakapan manusia asli.
7. Menegaskan bahwa target event adalah anchor evaluasi utama untuk MRA dan BCR.
8. Menjadi dasar konseptual untuk Document 02 — Final Clean Implementation Specification.

### 1.3 Ruang Lingkup Dokumen

Dokumen ini membahas arsitektur sistem, struktur komponen, batasan desain, alur data, cognitive pipeline, memory stream, retrieval engine, simulation structure, evaluation flow, dan log replay visualizer. Dokumen ini tidak membahas implementasi kode detail, daftar file secara teknis mendalam, acceptance test per phase, API retry detail, atau skema JSON lengkap. Detail tersebut akan dimasukkan ke Document 02.

Ruang lingkup sistem yang dibahas meliputi:

- controlled text-based classroom environment;
- empat agent dengan role berbeda;
- hybrid simulation structure;
- memory stream per agent;
- weighted retrieval berbasis recency, importance, dan relevance;
- Top-K memory retrieval;
- plan, act, dan reflect pipeline;
- logging berbasis JSON/JSONL;
- evaluasi MRA dan BCR pada target event;
- log replay web visualizer sebagai alat observasi opsional.

Ruang lingkup yang tidak termasuk:

- maze simulation;
- spatial map atau tile-based world;
- pathfinding;
- daily schedule 24 jam penuh;
- Smallville frontend;
- real-time web simulation engine;
- human-in-the-loop live chat;
- optimasi otomatis bobot retrieval;
- simulasi percakapan manusia yang sepenuhnya bebas.

---

## 2. Posisi Dokumen terhadap Repo Prototype dan Dokumen Lama

### 2.1 Posisi Repo Prototype

Repo yang sudah dikembangkan sebelumnya diposisikan sebagai **prototype repository**. Prototype tersebut berguna karena telah membuktikan kelayakan teknis dari banyak komponen penting. Namun prototype tidak langsung dijadikan repository final karena di dalamnya terdapat banyak eksperimen patch, khususnya pada prompt, parser, output guard, dan visualizer.

Prototype membuktikan bahwa:

- konfigurasi hybrid dapat dimuat;
- dummy simulation dapat berjalan;
- real LLM API dapat dipanggil;
- memory stream dapat dibentuk;
- retrieval Top-K dapat dihitung;
- logs dapat ditulis;
- MRA/BCR dapat diekspor;
- web visualizer berbasis log dapat dibuat;
- risiko prompt over-engineering dapat diidentifikasi.

Repo baru tidak dibuat karena prototype gagal. Repo baru dibuat karena prototype telah memenuhi peran eksplorasinya dan sekarang penelitian membutuhkan versi final yang lebih stabil, lebih minimal, dan lebih defensible secara akademik.

### 2.2 Posisi Document 01 Lama

Document 01 lama berperan sebagai **repository overview and architecture adaptation blueprint**. Fokus utamanya adalah melakukan reverse engineering terhadap repository Generative Agents/Reverie dan menjelaskan bagian mana yang dipertahankan, dimodifikasi, atau dihilangkan untuk penelitian berbasis kelas.

Document 01 lama tetap relevan sebagai dasar historis. Namun Document 01 baru ini menambahkan penekanan baru:

- architecture contract;
- pelajaran dari prototype;
- clean implementation principle;
- pemisahan layer arsitektur;
- batasan eksplisit terhadap prompt over-engineering;
- visualizer sebagai read-only log replay tool;
- natural interaction sebagai controlled memory-building mechanism.

### 2.3 Posisi Document 02 Lama

Document 02 lama berperan sebagai **full internal development blueprint**. Dokumen tersebut sudah menjelaskan module specification, data model, retrieval engine, cognitive pipeline, prompt, LLM service, logging, evaluation, testing, dan roadmap.

Document 02 baru yang akan dibuat setelah dokumen ini perlu menggunakan Document 02 lama sebagai baseline teknis, tetapi harus ditulis ulang dengan prinsip clean implementation. Perubahan utama yang perlu masuk ke Document 02 baru meliputi:

- freeze policy;
- phase-by-phase edit boundaries;
- minimal prompt design;
- minimal output guardrail;
- MRA debug protocol;
- visualizer replay-only specification;
- explicit stop condition per phase;
- prohibition against theme guard dan hidden retrieval intervention.

### 2.4 Hubungan Tiga Artefak

| Artefak | Status | Fungsi |
|---|---|---|
| Repo prototype | Referensi teknis dan lesson learned | Membuktikan feasibility dan menemukan risiko implementasi |
| Document 01 lama | Baseline adaptasi arsitektur | Menjelaskan reverse engineering dan adaptasi dari Generative Agents |
| Document 01 baru | Blueprint arsitektur final | Menetapkan clean final architecture dan architecture contract |
| Document 02 baru | Spesifikasi implementasi final | Menurunkan arsitektur ke phase, module, schema, test, dan command |
| Repo baru | Implementasi final skripsi | Menjalankan eksperimen retrieval-weight secara stabil dan reproducible |

---

## 3. Tujuan Penelitian dan Scope Sistem

### 3.1 Fokus Penelitian

Fokus penelitian adalah menganalisis pengaruh konfigurasi bobot retrieval memori terhadap konsistensi perilaku agent generatif berbasis Large Language Model. Retrieval memory dalam penelitian ini dihitung menggunakan tiga komponen: recency, importance, dan relevance. Setiap konfigurasi retrieval memberikan bobot berbeda terhadap ketiga komponen tersebut.

Penelitian tidak bertujuan mencari bobot retrieval paling optimal secara global. Penelitian membandingkan empat konfigurasi tetap untuk mengamati perbedaan pola retrieval dan konsistensi perilaku yang muncul di dalam controlled classroom simulation.

### 3.2 Pertanyaan Penelitian yang Didukung Sistem

Sistem dirancang untuk mendukung dua pertanyaan utama:

1. Bagaimana perbedaan konsistensi perilaku agent muncul pada empat konfigurasi bobot retrieval memory dalam simulasi kelas terkontrol?
2. Konfigurasi mana yang menghasilkan Behaviour Consistency Rate dan Memory Retrieval Accuracy yang lebih tinggi, serta pola apa yang terlihat dari perbandingan empat konfigurasi tersebut?

Agar pertanyaan tersebut dapat dijawab, sistem harus menjaga agar hanya bobot retrieval yang berubah antar kondisi. Jika prompt, event, turn order, jumlah memory, model, atau evaluation filter berubah secara tidak terkendali, maka hasil eksperimen tidak lagi dapat dikaitkan secara kuat dengan retrieval weight.

### 3.3 Scope Sistem Final

| Komponen | Keputusan Final |
|---|---|
| Simulation mode | Hybrid Controlled Natural Interaction + Target Event Evaluation |
| Environment | Text-based classroom environment |
| Agent count | 4 agents |
| Simulation duration | 6 simulation days |
| Natural blocks | 7 natural interaction blocks per run |
| Target events | 5 target events per run |
| Natural block size | 12 turns: 4 agents x 3 turns |
| Evaluation anchor | Target events only for primary MRA/BCR |
| Retrieval components | Recency, importance, relevance |
| Top-K | 5 retrieved memories |
| Retrieval configurations | balanced, recency-heavy, importance-heavy, relevance-heavy |
| Repetitions | 3 repetitions per configuration |
| Total runs | 12 runs |
| Visualization | Optional read-only log replay web visualizer |

### 3.4 Out-of-Scope Final

Sistem final tidak akan membangun ulang Smallville, tidak menggunakan maze, tidak menggunakan pathfinding, dan tidak mensimulasikan dunia virtual terbuka. Sistem juga tidak menjadikan percakapan manusia asli sebagai target utama. Natural interaction block hanya digunakan untuk membangun memory stream dalam konteks kelas yang terkontrol.

Komponen yang secara eksplisit berada di luar scope:

- full open-ended social simulation;
- autonomous daily life simulation;
- spatial movement;
- complex classroom map;
- real-time web simulation server;
- manual human chat intervention;
- live LLM execution dari website;
- automatic retrieval weight optimization;
- human realism benchmark;
- natural block primary MRA/BCR evaluation.

---

## 4. Lessons Learned dari Prototype Repository

### 4.1 Hal yang Berhasil Dibuktikan

Prototype sebelumnya membuktikan bahwa arsitektur penelitian ini secara teknis layak. Sistem dapat menjalankan konfigurasi, menginisialisasi agent, membangun memory stream, melakukan retrieval Top-K, menghasilkan respons agent dengan LLM, menulis log, menjalankan evaluasi, dan menampilkan replay visualizer.

Keberhasilan prototype penting karena mengurangi ketidakpastian teknis sebelum repo final dibangun. Repo baru tidak perlu mengeksplorasi dari nol. Repo baru dapat mengambil prinsip yang sudah terbukti, lalu menulis ulang implementasi dengan batasan yang lebih ketat.

Hal yang terbukti bisa dipertahankan:

- config-driven simulation;
- per-agent memory stream;
- weighted retrieval engine;
- dummy mode untuk testing;
- LLM service abstraction;
- JSONL logging;
- MRA calculation;
- BCR export;
- pilot safeguards;
- replay visualizer concept.

### 4.2 Masalah yang Muncul pada Prototype

Masalah utama prototype bukan pada retrieval formula, melainkan pada proses pengembangan yang bergerak dari core experiment menuju dialogue quality patching. Ketika output agent terlihat terlalu repetitif, terlalu formal, atau kurang natural, perbaikan dilakukan melalui penambahan prompt rule dan guardrail. Jika pola ini dibiarkan, repo final bisa terlalu banyak mengarahkan isi respons agent.

Masalah yang ditemukan:

1. Prompt menjadi terlalu panjang dan penuh larangan.
2. Output guard mulai mendekati semantic steering, bukan hanya validasi format.
3. Natural interaction mulai diperlakukan seperti simulasi percakapan manusia asli.
4. Visualizer berisiko membuat fokus bergeser ke pengalaman demo.
5. MRA rendah berisiko disalahartikan sebagai masalah dialog, padahal sering terkait category alignment.
6. Patch yang terlalu banyak membuat implementasi lebih sulit dijelaskan dalam Bab 3.

### 4.3 Prinsip Perbaikan untuk Repo Baru

Repo baru harus menghindari pola patch bertubi-tubi. Prinsipnya adalah:

```text
Design first → Architecture contract → Small implementation phases → Dummy validation → Real pilot → Freeze → Full experiment
```

Prinsip perbaikan:

- prompt harus minimal;
- guardrail harus teknis, bukan semantik berat;
- natural interaction tidak boleh menjadi objek optimasi utama;
- visualizer tidak boleh mengubah simulasi;
- MRA/BCR harus dipisahkan dari dialogue realism;
- full experiment hanya dijalankan setelah config, prompt, retrieval, dan evaluation freeze.

### 4.4 Batas antara Bug Fix dan Scope Creep

Repo baru perlu membedakan bug fix dan scope creep.

| Jenis perubahan | Contoh | Status |
|---|---|---|
| Bug fix | invalid JSON, empty response, missing log field | Boleh |
| Methodology fix | ground truth tidak sesuai target event | Boleh sebelum freeze |
| Quality polish | response terlalu panjang | Terbatas |
| Realism improvement | membuat percakapan lebih manusiawi | Hindari |
| Feature expansion | real-time web LLM simulation | Tunda / future work |
| Hidden intervention | retrieval penalty tambahan | Tidak boleh |

Kalimat prinsipnya adalah: **kalau perubahan tidak membantu menjawab RQ1/RQ2, perubahan tersebut tidak boleh masuk core experiment.**

---

## 5. Ringkasan Arsitektur Generative Agents yang Diadaptasi

### 5.1 Komponen Kognitif yang Relevan

Repository Generative Agents/Reverie memiliki banyak komponen yang berhubungan dengan simulasi dunia virtual. Namun penelitian ini hanya membutuhkan cognitive core: persona, memory stream, retrieval, planning, action generation, reflection, dan logging.

Komponen kognitif yang diadaptasi:

- persona sebagai identitas dan behavioral profile agent;
- memory stream sebagai tempat menyimpan pengalaman agent;
- retrieval sebagai mekanisme memilih memory yang akan masuk konteks LLM;
- planning sebagai tahap pembentukan niat/respons internal;
- action/response sebagai output agent;
- reflection sebagai pembentukan memory tingkat lebih tinggi;
- logging sebagai bukti audit eksperimen.

### 5.2 Komponen yang Tidak Diadaptasi

Komponen Smallville yang bersifat visual-spatial tidak digunakan karena tidak mendukung langsung variabel penelitian. Penelitian tidak memerlukan agent berjalan di tile map, berpindah lokasi, membuka objek, atau menjalankan jadwal 24 jam. Jika komponen tersebut dipertahankan, biaya dan kompleksitas akan meningkat tanpa meningkatkan validitas pengujian retrieval weight.

Komponen yang tidak diadaptasi:

- maze;
- tile map;
- pathfinding;
- spatial memory;
- visual sandbox frontend;
- game server;
- daily schedule penuh;
- legacy retrieval berbasis keyword/SPO;
- hidden multiplier dari repository asli;
- frontend replay original.

### 5.3 Prinsip Adaptasi

Prinsip adaptasi adalah **mempertahankan cognitive logic, bukan bentuk dunia virtualnya**. Sistem target tidak perlu terlihat seperti Smallville. Sistem target cukup menjaga hubungan antara pengalaman, memory, retrieval, action, dan reflection.

Alur kognitif target:

```text
Classroom context
→ Perceive
→ Store observation memory
→ Weighted Top-K retrieval
→ Plan
→ Act/respond
→ Store action memory
→ Reflect
→ Store reflection memory
→ Log for evaluation
```

Alur tersebut mempertahankan substansi Generative Agents tanpa membawa kompleksitas dunia virtual yang tidak diperlukan.

---

## 6. Architecture Contract

### 6.1 Fungsi Architecture Contract

Architecture contract adalah kontrak desain yang harus dipatuhi oleh seluruh implementasi repo baru. Kontrak ini berfungsi untuk mencegah perubahan tidak sadar yang dapat mengganggu validitas eksperimen. Setiap perubahan kode, config, prompt, atau visualizer harus diperiksa terhadap kontrak ini.

Jika terjadi konflik antara keinginan menambah fitur dan architecture contract, maka architecture contract harus diprioritaskan. Repo baru adalah sistem penelitian, bukan produk demo.

### 6.2 Non-Negotiable Experimental Invariants

| Invariant | Nilai Final | Alasan |
|---|---|---|
| Agent count | 4 | Menjaga scope kecil dan role kelas tetap jelas |
| Agent IDs | `teacher`, `active_student`, `quiet_student`, `academic_student` | Menjaga konsistensi config, log, dan evaluasi |
| Simulation days | 6 | Cukup untuk memory growth dan 5 event evaluasi |
| Natural blocks | 7 | Pembentukan memory stream secara terkontrol |
| Target events | 5 | Anchor evaluasi utama |
| Turns per natural block | 12 | 4 agents x 3 turns |
| Teacher opens block | true | Menjaga konteks natural block tetap terkendali |
| Top-K | 5 | Jumlah memory input konsisten |
| Retrieval components | recency, importance, relevance | Variabel penelitian utama |
| Retrieval formula | weighted normalized sum | Tidak boleh ada hidden multiplier |
| Retrieval configurations | 4 fixed configs | Perbandingan antar kondisi |
| Repetitions | 3 per config | Mengurangi efek randomness |
| Primary MRA/BCR | target_event only | Menjaga ground truth stabil |

### 6.3 Variabel yang Dikunci

Semua aspek berikut harus sama untuk seluruh konfigurasi dan repetition:

- persona profile;
- seed memory;
- simulation timeline;
- natural block definition;
- target event order;
- turn quota;
- Top-K;
- prompt template setelah freeze;
- model configuration setelah freeze;
- temperature;
- max token setting;
- evaluation logic;
- ground truth;
- logging schema;
- visualization input schema.

### 6.4 Satu-Satunya Variabel Bebas

Satu-satunya variabel bebas adalah konfigurasi bobot retrieval:

| Configuration | Recency | Importance | Relevance | Tujuan |
|---|---:|---:|---:|---|
| balanced | 0.33 | 0.33 | 0.33 | Baseline seimbang |
| recency_heavy | 0.60 | 0.20 | 0.20 | Menguji dominasi memori terbaru |
| importance_heavy | 0.20 | 0.60 | 0.20 | Menguji dominasi memori penting |
| relevance_heavy | 0.20 | 0.20 | 0.60 | Menguji dominasi kesesuaian semantik |

Tidak boleh ada bobot tersembunyi, multiplier tambahan, penalty tambahan, filtering tambahan, atau prompt instruction yang secara implisit memprioritaskan salah satu komponen retrieval.

### 6.5 Prohibited Architecture Changes

Perubahan berikut tidak boleh masuk repo final:

- mengubah Top-K berdasarkan config;
- memberi LLM component score retrieval dalam prompt;
- meminta LLM memprioritaskan memory terbaru;
- meminta LLM memprioritaskan memory paling penting;
- meminta LLM memprioritaskan memory paling relevan secara eksplisit;
- memasukkan full memory stream ke prompt;
- menjadikan natural block sebagai primary MRA;
- mengubah target event order setelah pilot;
- menambahkan web live simulation yang menjalankan LLM dari browser;
- menambahkan human input ke simulation run;
- mengubah prompt setelah full experiment dimulai.

### 6.6 Architecture Contract Summary

```text
Only retrieval weights vary.
Everything else is controlled.
Natural blocks create memory.
Target events evaluate retrieval and behavior.
Visualizer reads logs only.
Prompt and guardrail must not become hidden experimental variables.
```

---

## 7. Final Clean System Architecture

### 7.1 Gambaran Umum Arsitektur

Sistem final terdiri atas tiga layer besar:

1. Core Experiment Layer
2. Minimal Dialogue Quality Control Layer
3. Log Replay Visualization Layer

Ketiga layer ini harus dipisahkan dengan jelas. Core experiment menjalankan simulasi dan menghasilkan data penelitian. Dialogue quality layer hanya memastikan output valid secara teknis. Visualization layer hanya membaca log untuk membantu observasi.

Diagram arsitektur:

```text
CLI Experiment Runner
        |
        v
Simulation Controller
        |
        +--> Natural Interaction Block Runner
        |
        +--> Target Event Runner
        |
        v
Agent Cognitive Pipeline
        |
        +--> Perceive
        +--> Memory Stream
        +--> Weighted Retrieval Engine
        +--> Plan
        +--> Act / Respond
        +--> Reflect
        |
        v
Logging System
        |
        +--> Retrieval Logs
        +--> Behavior Logs
        +--> Memory Logs
        +--> Reflection Logs
        +--> Run Summary
        |
        v
Evaluation Layer
        |
        +--> MRA Calculator
        +--> BCR Export / Judge Dataset
        +--> Aggregation
        |
        v
Log Replay Exporter
        |
        v
Read-only Web Visualizer
```

### 7.2 Prinsip Data Flow

Data flow sistem harus satu arah dari simulasi menuju log, evaluasi, dan visualizer. Visualizer tidak boleh memanggil LLM, tidak boleh mengubah memory, tidak boleh menjalankan retrieval, dan tidak boleh memodifikasi hasil simulasi.

Data flow utama:

```text
Config
→ Initialize run
→ Initialize agents and seed memories
→ Execute simulation steps
→ Retrieve Top-K memory per turn
→ Generate response
→ Store new memories
→ Write logs
→ Compute/export evaluation
→ Export replay JSON
→ View replay in website
```

### 7.3 Clean Architecture Principle

Clean architecture dalam konteks penelitian ini berarti:

- retrieval engine tidak bergantung pada LLM response generation;
- evaluation tidak menjalankan ulang simulation;
- visualizer tidak menjadi sumber data penelitian;
- config menjadi sumber kebenaran untuk timeline dan agent;
- logs menjadi sumber kebenaran untuk hasil run;
- prompt tidak mengandung eksperimen tersembunyi;
- output guard tidak mengubah isi perilaku secara semantik.

---

## 8. Layer 1 — Core Experiment Layer

### 8.1 Fungsi Core Experiment Layer

Core Experiment Layer adalah bagian sistem yang menjawab pertanyaan penelitian. Layer ini memuat agent, environment, memory stream, retrieval engine, cognitive pipeline, simulation controller, logging, dan evaluation export.

Layer ini harus stabil, reproducible, dan tidak bergantung pada visualizer. Jika visualizer dihapus, eksperimen tetap harus dapat berjalan dan menghasilkan MRA/BCR.

### 8.2 Komponen Core Experiment

| Komponen | Fungsi |
|---|---|
| Config Loader | Memuat agents, simulation steps, target events, retrieval configs, settings, ground truth |
| Agent Registry | Menyimpan empat agent dan akses ke memory stream masing-masing |
| Classroom Environment | Menyimpan state simulasi berbasis teks |
| Memory Stream | Menyimpan seed, observation, action, plan, dan reflection memory |
| Retrieval Engine | Menghitung Top-K berdasarkan recency, importance, relevance |
| Cognitive Pipeline | Menjalankan perceive, retrieve, plan, act, reflect |
| Simulation Controller | Menjalankan 6-day hybrid timeline |
| Logging System | Menulis retrieval, behavior, memory, reflection, dan API usage logs |
| Evaluation Export | Menghasilkan dataset MRA/BCR dari target event logs |

### 8.3 Core Experiment Boundaries

Core Experiment Layer tidak boleh berisi:

- UI logic;
- browser-specific state;
- theme guard;
- visual animation logic;
- manual user intervention;
- hidden retrieval heuristics;
- prompt changes per config;
- evaluation logic yang membaca natural block sebagai primary metric.

---

## 9. Layer 2 — Minimal Dialogue Quality Control Layer

### 9.1 Fungsi Dialogue Quality Control

Dialogue Quality Control Layer berfungsi memastikan output LLM valid dan tidak merusak pipeline. Layer ini bukan alat untuk membuat agent berbicara seperti manusia sungguhan. Layer ini juga bukan mekanisme untuk memperbaiki MRA atau BCR secara langsung.

Fungsi utamanya:

- memastikan output tidak kosong;
- memastikan output bukan placeholder;
- memastikan output tidak memakai nama legacy yang salah;
- memastikan output tidak duplikat persis;
- membatasi panjang output;
- melakukan retry jika JSON tidak valid.

### 9.2 Guardrail yang Diizinkan

| Guardrail | Alasan |
|---|---|
| Reject empty output | Output kosong tidak bisa dilog atau dievaluasi |
| Reject placeholder | Placeholder menunjukkan LLM tidak mengikuti konteks |
| Reject wrong legacy name | Nama salah merusak persona consistency |
| Reject exact duplicate | Duplikasi persis menunjukkan output tidak valid |
| Trim excessive length | Mengontrol biaya dan noise |
| Retry invalid JSON | Menjaga pipeline tetap terstruktur |

### 9.3 Guardrail yang Tidak Diizinkan

Guardrail berikut tidak boleh digunakan karena dapat mengarahkan isi perilaku agent secara berlebihan:

- banned phrase list panjang;
- theme guard;
- semantic steering guard;
- automatic response rewriting;
- retrieval result filtering after scoring;
- hidden relevance penalty;
- hidden recency penalty;
- instruction untuk menghindari topik tertentu hanya karena terlihat repetitif;
- instruction untuk selalu bertanya, selalu klarifikasi, atau selalu menyebut memory.

### 9.4 Prompt Neutrality Rule

Prompt harus netral terhadap tiga komponen retrieval. Prompt tidak boleh mengatakan:

```text
Use the most recent memories.
Prioritize important memories.
Use only the most relevant memories.
```

Prompt yang aman:

```text
Use the provided retrieved memories as context when appropriate.
Do not invent memories that are not provided.
Respond according to your persona and the current classroom situation.
```

Kalimat tersebut aman karena tidak memberi sinyal tambahan terhadap recency, importance, atau relevance. LLM hanya menggunakan memory yang sudah dipilih oleh retrieval engine.

---

## 10. Layer 3 — Log Replay Visualization Layer

### 10.1 Fungsi Visualizer

Visualizer pada repo baru adalah **read-only log replay web visualizer**. Visualizer digunakan untuk membantu observasi, demonstrasi, dan pemahaman alur simulasi. Visualizer bukan sumber data penelitian dan bukan runtime utama simulasi.

Visualizer membaca output dari simulation logs melalui replay JSON yang dibuat oleh exporter.

Alur visualizer:

```text
Simulation logs
→ Export replay JSON
→ Website reads JSON
→ User views playback
```

### 10.2 Perbedaan dengan Live Web Simulation

| Aspek | Log Replay Visualizer | Live Web Simulation Engine |
|---|---|---|
| Menjalankan LLM | Tidak | Ya |
| Mengubah memory | Tidak | Bisa |
| Mengubah retrieval | Tidak | Bisa |
| Butuh server runtime | Tidak wajib | Ya |
| Cocok untuk skripsi ini | Ya | Tidak sebagai core |
| Risiko metodologi | Rendah | Tinggi |

Repo baru menggunakan log replay visualizer karena lebih aman, lebih sederhana, dan tidak mengganggu eksperimen.

### 10.3 Batasan Visualizer

Visualizer tidak boleh:

- memanggil LLM;
- menjalankan retrieval;
- menulis memory baru;
- mengubah behavior logs;
- mengubah evaluation output;
- memengaruhi turn order;
- memberi input manusia ke run eksperimen;
- menjadi sumber MRA/BCR.

Visualizer boleh:

- menampilkan classroom map sederhana;
- menampilkan agent positions secara statis/simbolik;
- menampilkan turn sequence;
- menampilkan response;
- menampilkan retrieved memories;
- menampilkan retrieval component scores dari logs;
- menampilkan MRA/BCR result jika sudah tersedia;
- menyediakan play, pause, next, previous, dan speed control.

---

## 11. Agent Initialization Architecture

### 11.1 Posisi Agent Initialization dalam Metodologi

Agent Initialization adalah tahap awal dalam metodologi penelitian. Tahap ini mencakup environment setup, agent setup, dan event setup. Pada tahap ini sistem belum menjalankan eksperimen penuh, melainkan menyiapkan kondisi awal yang akan dipakai konsisten di seluruh run.

Agent Initialization harus dilakukan sebelum retrieval configuration, simulation execution, evaluation, dan analysis. Kesalahan pada tahap ini dapat menyebabkan hasil eksperimen sulit dibandingkan.

### 11.2 Komponen Agent Initialization

| Subtahap | Artefak | Fungsi |
|---|---|---|
| Environment setup | classroom description, simulation timeline, rules | Menentukan konteks kelas dan urutan simulasi |
| Agent setup | agent profile, persona, seed memories | Menentukan identitas dan kondisi awal agent |
| Event setup | target event definitions | Menentukan stimulus evaluasi utama |
| Validation setup | config validator | Memastikan semua struktur sesuai architecture contract |

### 11.3 Prinsip Agent Initialization

- Semua agent harus diinisialisasi dari state yang sama untuk setiap run.
- Seed memory harus identik di semua konfigurasi.
- Target event harus identik di semua konfigurasi.
- Timeline harus identik di semua konfigurasi.
- Agent IDs tidak boleh berubah setelah config freeze.
- Agent names dapat dibuat konkret, tetapi harus stabil setelah freeze.
- Event descriptions tidak boleh berubah setelah ground truth dibuat.

### 11.4 Hubungan dengan Repo Phase

Dalam implementasi repo baru, Agent Initialization akan tercermin dalam beberapa phase awal:

| Phase Implementasi | Kaitan dengan Agent Initialization |
|---|---|
| Architecture contract setup | Menetapkan batas desain environment, agent, event |
| Configuration files | Membuat agents.json, simulation_steps.json, target_events.json |
| Domain models | Merepresentasikan agent, environment, event sebagai struktur data |
| Config validation | Memvalidasi semua kondisi awal |
| Seed memory initialization | Memuat memory awal ke memory stream agent |

---

## 12. Environment Setup Architecture

### 12.1 Definisi Environment

Environment dalam penelitian ini adalah classroom environment berbasis teks. Environment bukan peta visual, bukan maze, dan bukan dunia 3D. Environment menyediakan konteks situasional agar agent dapat menerima stimulus, berinteraksi, membentuk memory, dan merespons target event.

Environment menyimpan:

- classroom description;
- classroom rules;
- day and period;
- simulation steps;
- target event references;
- shared conversation context;
- run state;
- current step metadata.

### 12.2 Struktur Environment

```text
ClassroomEnvironment
    ├── classroom_description
    ├── classroom_rules
    ├── simulation_steps
    ├── target_events
    ├── shared_context
    ├── current_day
    ├── current_period
    ├── current_step_id
    └── run_state
```

### 12.3 Mengapa Environment Tidak Dibuat Realistik Sepenuhnya

Environment tidak dibuat seperti ruang kelas nyata sepenuhnya karena tujuan penelitian bukan mengukur realisme kelas. Environment digunakan sebagai controlled testbed. Jika environment terlalu bebas, banyak faktor baru muncul: posisi tempat duduk, gerakan, interupsi, mood, konflik, topik liar, dan relasi sosial yang berubah. Semua itu dapat mengaburkan efek retrieval weight.

Environment cukup menyediakan konteks yang stabil agar memory stream terbentuk dan target event dapat dievaluasi secara konsisten.

### 12.4 Controlled Classroom Design

Kelas dalam penelitian ini memiliki struktur sosial sederhana:

- satu guru sebagai anchor instruksional;
- satu siswa aktif sebagai anchor sosial;
- satu siswa pendiam sebagai anchor kehati-hatian;
- satu siswa akademis sebagai anchor fokus belajar.

Desain ini cukup untuk menciptakan variasi perilaku yang bisa dinilai melalui BCR tanpa membuat sistem terlalu kompleks.

---

## 13. Agent Setup Architecture

### 13.1 Agent yang Digunakan

Sistem menggunakan empat agent tetap.

| Agent ID | Role | Fungsi Eksperimental |
|---|---|---|
| teacher | Guru / instruktur | Mengarahkan kelas, memberi instruksi, menjaga aturan |
| active_student | Siswa aktif / sosial | Bertanya, mengajak diskusi, merespons secara sosial |
| quiet_student | Siswa pendiam / introvert | Berbicara singkat, berhati-hati, cenderung menunggu giliran |
| academic_student | Siswa akademis | Fokus pada materi, strategi belajar, dan detail akademik |

### 13.2 Agent Profile

Setiap agent memiliki profile yang terdiri atas:

- name;
- role;
- persona traits;
- goals;
- communication style;
- classroom motivation;
- likely behavior under uncertainty;
- likely behavior during group interaction;
- likely behavior during assessment;
- seed memories.

Agent profile harus cukup berbeda agar BCR dapat menilai konsistensi perilaku. Namun persona tidak boleh terlalu ekstrem sampai efek retrieval memory tidak terlihat.

### 13.3 Seed Memory

Seed memory adalah memory awal yang dimasukkan ke memory stream sebelum simulasi dimulai. Seed memory membentuk identitas awal agent, kebiasaan belajar, dan orientasi sosial. Seed memory bukan jawaban langsung terhadap target event.

Contoh jenis seed memory:

| Agent | Contoh Category | Fungsi |
|---|---|---|
| teacher | teacher_guidance | Menjelaskan kecenderungan memberi arahan |
| active_student | social_learning | Menjelaskan kecenderungan mengajak diskusi |
| quiet_student | quiet_style | Menjelaskan kecenderungan berbicara singkat |
| academic_student | academic_focus | Menjelaskan kecenderungan fokus pada struktur belajar |

Seed memory harus stabil di semua run. Jika seed memory berubah, maka hasil antar retrieval configuration tidak lagi murni dibandingkan berdasarkan bobot retrieval.

### 13.4 Agent Naming Policy

Nama agent boleh dibuat konkret agar output LLM tidak memakai placeholder. Namun nama harus ditetapkan sebelum prompt freeze dan tidak boleh berubah selama eksperimen.

Contoh naming final yang bisa digunakan:

| Agent ID | Display Name |
|---|---|
| teacher | Ms. Sunday |
| active_student | Monday |
| quiet_student | Tuesday |
| academic_student | Wednesday |

Nama tersebut hanya display identity. Agent ID tetap menjadi identifier utama dalam config, log, dan evaluation.

---

## 14. Event Setup Architecture

### 14.1 Fungsi Target Event

Target event adalah stimulus terkontrol yang digunakan sebagai anchor evaluasi MRA dan BCR. Target event harus sama untuk seluruh konfigurasi dan repetition. Target event memungkinkan sistem membandingkan apakah memory yang diretrieval dan respons agent berubah ketika bobot retrieval berubah.

Target event bukan sekadar skenario cerita. Target event adalah titik evaluasi metodologis.

### 14.2 Lima Target Event

| Event ID | Title | Fungsi Evaluasi |
|---|---|---|
| event_01 | Pengumuman Ujian | Menguji retrieval terkait instruksi guru, kekhawatiran ujian, dan persiapan belajar |
| event_02 | Perubahan Aturan Kelas | Menguji retrieval terkait aturan, norma kelas, dan kepatuhan sebelumnya |
| event_03 | Diskusi Kelompok | Menguji retrieval terkait partisipasi, koordinasi sosial, dan kontribusi akademik |
| event_04 | Pemilihan Ketua Kelas | Menguji retrieval terkait kepemimpinan, kepercayaan teman, dan inisiatif sosial |
| event_05 | Ujian Harian | Menguji retrieval terkait pengumuman sebelumnya, strategi belajar, dan kesiapan ujian |

### 14.3 Event Design Rules

Target event harus:

- sederhana dan bisa dijawab oleh semua agent;
- membutuhkan memory, bukan hanya reasoning dari prompt saat ini;
- terkait dengan event sebelumnya sehingga memory stream bermakna;
- memiliki expected memory category;
- stabil di seluruh run;
- tidak berubah setelah ground truth freeze.

Target event tidak boleh:

- dibuat terlalu kompleks;
- berubah secara dinamis antar konfigurasi;
- mengandung terlalu banyak sub-event;
- mengarahkan agent ke satu jawaban tertentu;
- secara eksplisit menyebut retrieval weight.

### 14.4 Event-to-Memory Dependency

| Later Event | Expected Useful Prior Memory |
|---|---|
| event_02 | teacher guidance, classroom norms, prior Day 1 interaction |
| event_03 | rule change, participation norm, social learning memory |
| event_04 | group discussion behavior, peer trust, social initiative |
| event_05 | exam announcement, prior preparation, study strategy, confidence/anxiety |

Event dependency ini membuat memory stream penting. Agent tidak hanya merespons event secara statis, tetapi dapat dipengaruhi oleh memory yang dikumpulkan dari natural block dan prior target event.

---

## 15. Memory Stream Architecture

### 15.1 Fungsi Memory Stream

Memory stream adalah pusat pengalaman agent. Semua informasi yang dapat memengaruhi retrieval dan respons agent disimpan dalam memory stream. Setiap agent memiliki memory stream sendiri. Agent tidak boleh mengambil private memory agent lain kecuali informasi tersebut sudah diamati dan disimpan sebagai memory miliknya sendiri.

### 15.2 Jenis Memory

| Memory Type | Sumber | Fungsi |
|---|---|---|
| seed_persona | Agent setup | Membentuk identitas awal agent |
| observation | Perception terhadap natural block atau target event | Menyimpan apa yang agent amati |
| plan | Cognitive pipeline | Menyimpan rencana respons jika plan logging aktif |
| action | Respons agent | Menyimpan ucapan/tindakan agent |
| reflection | Refleksi setelah block/event | Menyimpan insight yang bisa diretrieval |

### 15.3 Memory Fields

Field minimum memory:

| Field | Fungsi |
|---|---|
| memory_id | Identifier unik memory |
| agent_id | Pemilik memory |
| memory_type | Jenis memory |
| text/content | Isi naratif memory |
| category | Label kategori untuk evaluasi/analisis |
| importance | Skor kepentingan 1-10 |
| embedding | Vector embedding untuk relevance scoring |
| created_at | Timestamp memory dibuat |
| last_accessed_at | Timestamp terakhir memory masuk Top-K |
| source_step_id | Step asal memory |
| source_step_type | natural_interaction_block atau target_event |

### 15.4 Memory Ownership

Memory ownership harus tegas. Retrieval candidate pool untuk agent hanya berasal dari memory stream agent tersebut. Jika Monday berbicara dan Tuesday mendengar, maka Tuesday dapat menyimpan observation memory tentang ucapan Monday. Tetapi Tuesday tidak boleh langsung mengakses memory internal Monday.

Prinsip ini penting untuk menjaga konsistensi simulasi multi-agent dan menghindari memory leakage.

---

## 16. Weighted Retrieval Architecture

### 16.1 Fungsi Retrieval Engine

Retrieval engine adalah komponen eksperimental paling penting. Komponen ini menentukan memory mana yang diberikan kepada LLM sebelum agent membuat plan dan response. Karena variabel penelitian adalah retrieval weight, engine harus transparan, deterministik dalam scoring, dan tidak memiliki multiplier tersembunyi.

### 16.2 Formula Retrieval

Formula retrieval final:

```text
Score(m, q) = w_recency    * Norm(Recency(m))
            + w_importance * Norm(Importance(m))
            + w_relevance  * Norm(Relevance(m, q))
```

Komponen:

| Komponen | Definisi | Catatan |
|---|---|---|
| Recency | Seberapa baru memory berdasarkan timestamp/access | Formula sama di semua config |
| Importance | Skor kepentingan memory 1-10 | Dinilai saat memory dibuat |
| Relevance | Cosine similarity antara query dan memory embedding | Embedding dibuat saat memory masuk |
| Normalization | Mengubah nilai ke skala sebanding | Dilakukan per retrieval candidate set |
| Final score | Weighted sum | Dipakai untuk ranking Top-K |

### 16.3 Top-K Rule

Top-K ditetapkan 5 untuk semua retrieval. Nilai K tidak berubah antar configuration. Jika memory count kurang dari 5, sistem mengembalikan semua memory yang tersedia dan mencatat candidate_count di log.

Top-K result harus masuk ke prompt plan dan act sebagai daftar memory text. Namun component score tidak diberikan ke LLM, karena score dapat membuat LLM melakukan prioritisasi tambahan di luar retrieval engine.

### 16.4 Last Access Update Rule

Urutan update harus dijaga:

```text
Score all candidates
→ write retrieval ranking log
→ select Top-K
→ update last_accessed_at only for selected Top-K
```

Urutan ini penting agar log menunjukkan kondisi retrieval sebelum efek akses terbaru diterapkan.

### 16.5 Prompt Neutrality terhadap Retrieval

LLM hanya menerima memory yang sudah dipilih retrieval engine. Prompt tidak boleh meminta LLM memilih berdasarkan recency, importance, atau relevance. Prompt hanya boleh mengatakan bahwa memory yang diberikan dapat digunakan sebagai konteks.

Prompt safe wording:

```text
Use the provided retrieved memories as context when appropriate.
Do not invent memories that are not provided.
```

---

## 17. Cognitive Pipeline Architecture

### 17.1 Shared Cognitive Loop

Natural block dan target event menggunakan cognitive loop yang sama. Perbedaannya bukan pada core cognition, melainkan pada fungsi metodologis step tersebut. Natural block membangun memory, target event menghasilkan data evaluasi utama.

Cognitive loop:

```text
Perceive
→ Store observation
→ Build retrieval query
→ Retrieve Top-K
→ Plan
→ Act/respond
→ Store action memory
→ Reflect
→ Store reflection memory
→ Log outputs
```

### 17.2 Perceive

Perceive mengubah context saat ini menjadi observation bagi agent. Pada natural block, context berasal dari topik block dan conversation history. Pada target event, context berasal dari event description, prior context, dan respons agent sebelumnya dalam event tersebut.

Output perceive disimpan sebagai observation memory jika memory generation aktif.

### 17.3 Retrieve

Retrieve mengambil Top-K memory dari memory stream agent. Query retrieval dibangun dari agent, current context, step objective, dan event/block metadata. Retrieval menggunakan konfigurasi bobot yang sedang diuji.

### 17.4 Plan

Plan adalah ringkasan internal tentang bagaimana agent akan merespons. Plan tidak perlu panjang dan tidak perlu ditampilkan sebagai output utama. Plan dapat disimpan untuk traceability jika plan logging aktif.

### 17.5 Act / Respond

Act menghasilkan respons teks agent. Natural response dibatasi 1-2 kalimat. Target event response dapat berupa satu paragraf pendek. Respons harus sesuai persona dan context, tetapi tidak harus menyerupai percakapan manusia asli secara sempurna.

### 17.6 Reflect

Reflect menghasilkan insight singkat setelah block atau event. Reflection dibatasi maksimal satu insight per agent per step agar memory stream tidak membengkak dan biaya API tetap terkendali.

Reflection disimpan sebagai memory_type=reflection dan dapat diretrieval pada step berikutnya.

---

## 18. Hybrid Simulation Architecture

### 18.1 Alasan Hybrid Design

Desain hybrid dipilih karena event-only simulation terlalu kaku dan membuat memory stream kurang alami, sedangkan fully natural simulation terlalu bebas dan sulit dievaluasi. Hybrid design berada di tengah: natural blocks menciptakan memory context, target events menjaga evaluasi tetap stabil.

### 18.2 Simulation Timeline

| Day | Morning / Natural Context | Afternoon / Target Event | Fungsi |
|---|---|---|---|
| Day 1 | Natural Block 1 + Natural Block 2 | No target event | Initial memory formation |
| Day 2 | Natural Block 3 | Event 1: Pengumuman Ujian | Evaluation anchor 1 |
| Day 3 | Natural Block 4 | Event 2: Perubahan Aturan Kelas | Evaluation anchor 2 |
| Day 4 | Natural Block 5 | Event 3: Diskusi Kelompok | Evaluation anchor 3 |
| Day 5 | Natural Block 6 | Event 4: Pemilihan Ketua Kelas | Evaluation anchor 4 |
| Day 6 | Natural Block 7 | Event 5: Ujian Harian | Evaluation anchor 5 |

### 18.3 Natural Interaction Block

Natural interaction block adalah interaksi kelas terkontrol. Setiap block memiliki 12 turns. Setiap agent mendapat 3 turns. Teacher membuka block untuk menjaga konteks. Setelah teacher membuka, urutan dapat mengikuti controlled rotating pattern atau semi-dynamic bounded policy, tetapi quota harus tetap terpenuhi.

Natural block berfungsi untuk:

- menciptakan observation memory;
- menciptakan action memory;
- memberi kesempatan agent merespons agent lain;
- membentuk reflection memory;
- memperkaya memory stream sebelum target event.

Natural block tidak berfungsi sebagai:

- primary MRA evaluation;
- primary BCR evaluation;
- benchmark human-like conversation realism;
- open-ended social simulation.

### 18.4 Target Event

Target event adalah stimulus evaluasi utama. Setiap target event dijalankan untuk semua agent. Retrieval log dari target event digunakan untuk MRA. Behavior log dari target event digunakan untuk BCR.

Target event harus tetap sama di seluruh configuration dan repetition.

### 18.5 Turn Count

```text
Natural blocks per run = 7
Turns per natural block = 12
Natural turns per run = 7 x 12 = 84
Target events per run = 5
Target event turns per run = 5 x 4 agents = 20
Main turns per run = 104
Full experiment = 104 x 12 = 1248 main turns
```

Angka ini penting untuk estimasi biaya API, logging, dan planning eksperimen.

---

## 19. Logging and Evaluation Architecture

### 19.1 Fungsi Logging

Logging adalah bagian inti metodologi. Tanpa log, retrieval tidak dapat diaudit dan MRA/BCR tidak dapat dihitung secara transparan. Setiap run harus menghasilkan log lengkap yang dapat digunakan untuk merekonstruksi keputusan retrieval dan respons agent.

### 19.2 Log yang Dibutuhkan

| Log File | Fungsi |
|---|---|
| retrieval_logs.jsonl | Menyimpan candidate, component score, final score, dan Top-K |
| behavior_logs.jsonl | Menyimpan respons agent, retrieved memory IDs, dan metadata step |
| memory_logs.jsonl | Menyimpan memory yang dibuat selama run |
| reflection_logs.jsonl | Menyimpan reflection output |
| api_usage_logs.jsonl | Menyimpan token usage dan metadata API call |
| natural_block_summaries.jsonl | Menyimpan ringkasan natural block |
| event_summaries.jsonl | Menyimpan ringkasan target event |
| errors.jsonl | Menyimpan error yang terjadi |
| run_summary.json | Menyimpan status run |

### 19.3 Required Metadata

Setiap log harus menyimpan metadata minimum:

- experiment_id;
- run_id;
- retrieval_config_id;
- repetition_index;
- step_id;
- step_type;
- is_target_event;
- day;
- period;
- turn_id;
- turn_index;
- agent_id;
- timestamp.

### 19.4 Source of Truth

Source of truth penelitian adalah logs, bukan visualizer. MRA dan BCR dihitung dari log. Visualizer hanya membaca log yang sudah ada.

---

## 20. MRA and BCR Evaluation Design

### 20.1 Evaluation Separation

Evaluasi utama hanya dilakukan pada target event. Natural block dapat dianalisis secara kualitatif atau sebagai secondary analysis, tetapi tidak masuk primary MRA/BCR.

| Data Source | Primary MRA | Primary BCR | Optional Analysis |
|---|---|---|---|
| Natural block retrieval logs | No | No | Retrieval pattern, memory growth |
| Natural block behavior logs | No | No | Qualitative examples |
| Target event retrieval logs | Yes | Indirect context | Main MRA source |
| Target event behavior logs | No | Yes | Main BCR source |

### 20.2 Memory Retrieval Accuracy

MRA mengukur proporsi retrieved Top-K memory yang category-nya sesuai expected category untuk event-agent pair.

```text
MRA(agent, event) = matched retrieved memories / K
K = 5
```

Jika retrieved memory category masuk expected_categories pada ground_truth.json, memory tersebut dihitung match.

MRA bukan ukuran apakah respons final bagus. MRA mengukur kualitas retrieval terhadap ground truth category.

### 20.3 Behaviour Consistency Rate

BCR menilai apakah respons agent konsisten dengan persona dan prior context. BCR tidak menilai apakah memory yang diretrieval tepat secara kategori; itu tugas MRA. BCR menilai perilaku yang muncul dari respons.

Dimensi BCR:

| Dimension | Pertanyaan Evaluasi | Score |
|---|---|---|
| Communication Style | Apakah gaya bicara sesuai persona? | 1-5 |
| Focus Consistency | Apakah respons sesuai role, motivasi, dan fokus agent? | 1-5 |
| Temporal Consistency | Apakah respons tidak bertentangan dengan memory/aksi sebelumnya? | 1-5 |

```text
BCR_turn = (style_score + focus_score + temporal_score) / 3
```

### 20.4 MRA Debug Protocol

Jika MRA rendah, jangan langsung mengubah prompt natural interaction. Lakukan debug berurutan:

1. Cek apakah ground_truth.json memiliki expected category yang benar.
2. Cek apakah memory category assignment konsisten.
3. Cek apakah target event observation memory tersimpan.
4. Cek apakah retrieval log memuat candidate dan Top-K dengan benar.
5. Cek apakah MRA matching logic memakai category yang sama.
6. Cek apakah visualizer/exporter tidak menampilkan memory kosong.

MRA rendah dapat menjadi hasil eksperimen yang valid. MRA rendah tidak selalu berarti sistem gagal.

### 20.5 BCR Interpretation Rule

BCR tinggi tidak selalu berarti MRA tinggi. Agent bisa menghasilkan respons persona-consistent dari memory yang hanya sebagian relevan. Sebaliknya, MRA tinggi tidak selalu menjamin BCR tinggi jika agent tidak memakai memory tersebut secara konsisten dengan persona.

Karena itu, MRA dan BCR harus dilaporkan bersama.

---

## 21. Methodology Alignment

### 21.1 Lima Tahap Metodologi

Arsitektur ini mendukung lima tahap metodologi penelitian:

| Tahap | Nama | Fungsi |
|---|---|---|
| Tahap 1 | Agent Initialization | Environment setup, agent setup, event setup |
| Tahap 2 | Retrieval Configuration | Menetapkan bobot retrieval dan Top-K |
| Tahap 3 | Simulation Execution | Menjalankan hybrid simulation |
| Tahap 4 | Evaluation | Menghitung MRA dan BCR |
| Tahap 5 | Analysis and Reporting | Membandingkan konfigurasi dan menyusun hasil |

### 21.2 Tahap 1 — Agent Initialization

Tahap ini mencakup:

- environment setup;
- agent setup;
- event setup;
- seed memory initialization;
- config validation.

Output metodologis:

- environment kelas terkontrol;
- empat agent tetap;
- lima target event tetap;
- timeline enam hari;
- initial memory stream.

### 21.3 Tahap 2 — Retrieval Configuration

Tahap ini menetapkan:

- retrieval formula;
- recency component;
- importance component;
- relevance component;
- Top-K=5;
- empat konfigurasi bobot.

Output metodologis:

- retrieval_configs.json;
- scoring rules;
- normalization rules;
- Top-K selection policy.

### 21.4 Tahap 3 — Simulation Execution

Tahap ini menjalankan:

- natural interaction block;
- target event;
- agent cognitive pipeline;
- logging.

Output metodologis:

- retrieval logs;
- behavior logs;
- memory logs;
- reflection logs;
- run summaries.

### 21.5 Tahap 4 — Evaluation

Tahap ini menghitung:

- MRA dari target_event retrieval logs;
- BCR dari target_event behavior logs;
- aggregation by event, agent, run, configuration.

Output metodologis:

- mra_records;
- bcr_records;
- evaluation summaries.

### 21.6 Tahap 5 — Analysis and Reporting

Tahap ini membandingkan hasil antar konfigurasi:

- MRA mean per configuration;
- BCR mean per configuration;
- per-event breakdown;
- per-agent breakdown;
- qualitative retrieval examples;
- interpretation and limitation.

---

## 22. Design Decisions: Dipertahankan, Dimodifikasi, Dihilangkan

### 22.1 Komponen yang Dipertahankan

| Komponen | Alasan Dipertahankan |
|---|---|
| Persona | Dibutuhkan untuk BCR dan perilaku agent |
| Memory stream | Pusat pengalaman agent |
| Weighted retrieval | Variabel utama penelitian |
| Plan/Act/Reflect | Cognitive core Generative Agents |
| Importance scoring | Salah satu komponen retrieval |
| Embedding relevance | Salah satu komponen retrieval |
| JSONL logging | Auditability dan reproducibility |
| MRA/BCR | Metric utama penelitian |

### 22.2 Komponen yang Dimodifikasi

| Komponen Asal | Modifikasi Final |
|---|---|
| Persona original | Diubah menjadi Agent sederhana tanpa spatial state |
| Conversation system | Diubah menjadi controlled natural interaction block |
| Event-only flow | Diubah menjadi hybrid simulation step flow |
| Daily schedule | Diubah menjadi compressed 6-day classroom timeline |
| Visual replay | Diganti log replay visualizer opsional |

### 22.3 Komponen yang Dihilangkan

| Komponen | Alasan Dihilangkan |
|---|---|
| Maze | Tidak diperlukan untuk retrieval-weight experiment |
| Spatial memory | Tidak ada spatial task |
| Pathfinding | Interaksi berbasis teks |
| Full frontend | Bukan source of truth penelitian |
| Game server | Menambah kompleksitas di luar scope |
| Legacy retrieval | Mengganggu isolasi weighted retrieval |
| Hidden multiplier | Merusak interpretasi bobot eksperimen |
| Full daily life simulation | Terlalu luas untuk penelitian S1 |

### 22.4 Komponen Baru yang Ditambahkan

| Komponen Baru | Fungsi |
|---|---|
| Architecture contract | Menjaga eksperimen tidak melebar |
| Minimal output guardrail | Menjaga validitas output tanpa semantic steering |
| Freeze policy | Mencegah perubahan setelah pilot |
| MRA debug protocol | Mencegah prompt tuning yang salah arah |
| Log replay visualizer | Membantu observasi tanpa mengubah simulasi |

---

## 23. Risks Avoided in Clean Implementation

### 23.1 Risiko Prompt Over-Engineering

Prompt yang terlalu panjang dapat membuat perilaku agent lebih ditentukan oleh instruksi prompt daripada memory retrieval. Repo baru menghindari hal ini dengan prompt minimal, neutral wording, dan freeze policy.

### 23.2 Risiko Hidden Experimental Variable

Jika guardrail atau prompt berbeda antar configuration, maka retrieval weight bukan satu-satunya variabel. Repo baru mengunci prompt, model, temperature, event, timeline, agent, dan evaluation logic.

### 23.3 Risiko Natural Interaction Terlalu Bebas

Natural interaction yang terlalu bebas dapat membuat memory stream antar run sangat berbeda. Repo baru menjaga natural interaction melalui fixed timeline, turn quota, teacher opener, dan response limit.

### 23.4 Risiko Visualizer Mengubah Fokus

Visualizer yang terlalu interaktif dapat membuat sistem bergeser menjadi demo. Repo baru membatasi visualizer sebagai read-only log replay.

### 23.5 Risiko MRA Dituning lewat Dialog

Jika MRA rendah lalu prompt natural diubah agar memory tertentu muncul, maka hasil menjadi bias. Repo baru menetapkan MRA debug protocol dan ground truth freeze.

### 23.6 Risiko Run Leakage

Jika memory antar run tidak terisolasi, hasil antar configuration tidak valid. Repo baru mewajibkan setiap run dimulai dari seed state identik dan output directory terpisah.

---

## 24. Implementation Readiness Checklist

Sebelum repo baru dibuat penuh, checklist berikut harus terpenuhi:

### 24.1 Architecture Checklist

- [ ] Architecture contract selesai.
- [ ] Scope core experiment disetujui.
- [ ] Out-of-scope components jelas.
- [ ] Natural block didefinisikan sebagai memory-building context.
- [ ] Target event didefinisikan sebagai evaluation anchor.
- [ ] Visualizer didefinisikan sebagai read-only log replay.

### 24.2 Methodology Checklist

- [ ] Environment setup disetujui.
- [ ] Agent setup disetujui.
- [ ] Event setup disetujui.
- [ ] Retrieval configurations disetujui.
- [ ] MRA/BCR definitions disetujui.
- [ ] Ground truth strategy disetujui.

### 24.3 Implementation Control Checklist

- [ ] Prompt minimal disepakati.
- [ ] Output guardrail minimal disepakati.
- [ ] No theme guard.
- [ ] No hidden retrieval filter.
- [ ] No score shown to LLM.
- [ ] No full memory stream in prompt.
- [ ] Freeze policy disepakati.

### 24.4 Pre-Full Experiment Checklist

- [ ] Config validation passed.
- [ ] Retrieval unit tests passed.
- [ ] Dummy Day 1-2 pilot passed.
- [ ] Dummy full timeline passed.
- [ ] Real smoke test passed.
- [ ] Real pilot Day 1-2 passed.
- [ ] Token usage reviewed.
- [ ] Prompt frozen.
- [ ] Config frozen.
- [ ] Evaluation frozen.
- [ ] Code committed and tagged.

---

## 25. Kesimpulan

Document 01 — Final Clean Architecture Blueprint menetapkan arsitektur final untuk repo baru penelitian agen generatif berbasis kelas. Arsitektur ini tetap mempertahankan inti desain sebelumnya: hybrid controlled natural interaction + target event evaluation, memory stream, weighted retrieval, Top-K=5, empat konfigurasi retrieval, dan evaluasi MRA/BCR pada target event.

Perubahan utama bukan pada topik penelitian, melainkan pada disiplin arsitektur. Repo baru harus menghindari prompt over-engineering, theme guard, hidden retrieval intervention, dan visualizer yang terlalu masuk ke core simulation. Natural interaction tetap digunakan, tetapi hanya sebagai controlled memory-building mechanism. Target event tetap menjadi anchor evaluasi utama. Visualizer tetap boleh ada, tetapi hanya sebagai read-only log replay.

Dengan architecture contract, layer separation, minimal guardrail, dan freeze policy, repo baru dapat menjadi final thesis implementation yang lebih mudah dijelaskan di Bab 3, lebih aman untuk full experiment, dan lebih kuat secara metodologis.

Kalimat ringkas arsitektur final:

```text
The system is a controlled text-based generative agent testbed that isolates retrieval-weight configurations as the main experimental variable. Natural interaction blocks build memory context, target events provide stable evaluation anchors, and all outputs are logged for MRA/BCR analysis. The visualizer is read-only and does not affect the experiment.
```

---

# Lampiran

## Lampiran A — Ringkasan Arsitektur Final

```text
Config
  ↓
Agent Initialization
  ↓
Hybrid Simulation
  ├── Natural Interaction Block
  │     ├── Perceive
  │     ├── Retrieve Top-K
  │     ├── Plan
  │     ├── Act
  │     ├── Reflect
  │     └── Store Memory
  │
  └── Target Event
        ├── Perceive
        ├── Retrieve Top-K
        ├── Plan
        ├── Act
        ├── Reflect
        ├── Store Memory
        └── Mark for MRA/BCR
  ↓
Logs
  ↓
Evaluation
  ├── MRA
  └── BCR
  ↓
Analysis and Reporting
  ↓
Optional Log Replay Visualizer
```

## Lampiran B — Ringkasan Konfigurasi Eksperimen

| Item | Nilai |
|---|---|
| Agent count | 4 |
| Simulation days | 6 |
| Natural blocks per run | 7 |
| Target events per run | 5 |
| Turns per natural block | 12 |
| Turns per agent per natural block | 3 |
| Top-K | 5 |
| Retrieval configurations | 4 |
| Repetitions per configuration | 3 |
| Total runs | 12 |
| Main turns per run | 104 |
| Full experiment main turns | 1248 |

## Lampiran C — Retrieval Formula

```text
Recency(m)    = decay_factor ^ delta_t
Importance(m) = importance score of memory
Relevance(m,q)= cosine_similarity(embedding_m, embedding_q)

Score(m,q) = w_recency    * Norm(Recency(m))
           + w_importance * Norm(Importance(m))
           + w_relevance  * Norm(Relevance(m,q))
```

No hidden multiplier is allowed.

## Lampiran D — Safe Natural Prompt Skeleton

```text
You are {agent_name}.
Follow your persona.
Current classroom situation:
{current_context}

Conversation so far:
{conversation_history}

Retrieved memories provided by the system:
{retrieved_memories}

Instruction:
Use the provided retrieved memories as context when appropriate.
Do not invent memories that are not provided.
Respond according to your persona and the current classroom situation.
Keep the response concise.
Do not use placeholders.
Do not repeat your previous response exactly.
Output only valid JSON.
```

## Lampiran E — Unsafe Prompt Patterns to Avoid

```text
Use the most recent memories.
Prioritize the most important memories.
Use only the most relevant memories.
Avoid phrases such as ...
Make the dialogue sound like a real human conversation.
Always clarify before answering.
Always mention evidence from memory.
Rewrite the response if it does not sound natural.
```

Prompt di atas tidak digunakan karena dapat menjadi intervensi tambahan yang mengaburkan efek retrieval weight.

## Lampiran F — Glossary

| Istilah | Definisi |
|---|---|
| Agent | Entitas simulasi dengan persona, memory stream, dan cognitive pipeline |
| Memory Stream | Kumpulan memory kronologis milik agent |
| Natural Interaction Block | Interaksi kelas terkontrol untuk membentuk memory stream |
| Target Event | Stimulus evaluasi utama untuk MRA/BCR |
| MRA | Memory Retrieval Accuracy, proporsi retrieved memory yang sesuai ground truth category |
| BCR | Behaviour Consistency Rate, skor konsistensi perilaku terhadap persona dan prior context |
| Top-K | Jumlah memory terpilih yang diberikan ke LLM |
| Architecture Contract | Batas desain yang tidak boleh dilanggar selama implementasi |
| Log Replay Visualizer | Website read-only untuk menampilkan ulang log simulasi |

## Lampiran G — Final Architecture Statement untuk Bab 3

```text
Sistem penelitian dibangun sebagai controlled text-based classroom simulation untuk menguji pengaruh konfigurasi bobot retrieval memory terhadap konsistensi perilaku agent generatif. Sistem mengadaptasi cognitive core dari Generative Agents berupa persona, memory stream, retrieval, planning, action generation, dan reflection, tetapi tidak mengimplementasikan maze, spatial map, pathfinding, frontend sandbox, atau daily schedule penuh. Natural interaction block digunakan sebagai mekanisme pembentukan memory stream secara terkontrol, sedangkan target event digunakan sebagai anchor evaluasi utama untuk Memory Retrieval Accuracy dan Behaviour Consistency Rate. Dengan mengunci persona, timeline, target event, prompt, model setting, Top-K, dan evaluation procedure, penelitian menjaga agar perbedaan hasil antar kondisi terutama dapat dikaitkan dengan konfigurasi bobot retrieval memory.
```
