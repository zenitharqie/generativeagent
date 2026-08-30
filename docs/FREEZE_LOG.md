# FREEZE LOG

Setiap perubahan setelah titik freeze wajib dicatat di sini (Doc 02 §24.2).

---

## change_001 — v1.8.0: Memory-stock apparatus revision (MRA Debug Protocol §17)

- **date**: 2026-08-27
- **changed_files**:
  - `config/simulation_steps.json` (pengayaan deskripsi 6 blok natural)
  - `prompts/natural_interaction.txt`, `prompts/target_event_response.txt`, `prompts/reflect.txt` (klausa tendency-first + refleksi forward-looking)
  - `src/simulation/safeguards.py` (3 prompt hash baru)
  - `scripts/sbe_report.py` (instrumen baru: Stock-Before-Event)
- **reason**: Diagnosis full experiment v3 menunjukkan 3/5 event (01, 04, 05) memiliki stok memori berlabel-expected = 0 sebelum event, sehingga MRA=0 struktural. Perbaikan dilakukan di hulu produksi memori (MRA Debug Protocol §17 langkah 1–3), bukan pada GT/retrieval/evaluasi.
- **bug_or_design_change**: design_change (apparatus calibration)
- **impact_on_experiment**: GT beku (fingerprint c6e310cc tidak berubah); bobot retrieval, Top-K, formula, evaluasi, dan judge tidak disentuh. Semua perubahan berlaku identik untuk 4 konfigurasi → variabel bebas tetap tunggal. Full experiment sebelumnya (v1–v3) dinyatakan superseded oleh run v4.
- **approval_status**: disetujui peneliti (sesi analisis v1.7.5)
- **evidence**: `docs/METHODOLOGY_REVISION_PLAN_v1.8.md`; diagnosis stok: 0 unit expected pra-event_04 di 12/12 run

---

## change_002 — v1.8.2: Operational tolerance for missing-usage attempts (Phase 13)

- **date**: 2026-08-27
- **changed_files**:
  - `src/simulation/full_experiment.py` (konstanta `MAX_MISSING_USAGE_ATTEMPTS = 5`; `usage_complete` memakai cap)
  - `src/simulation/experiment_controller.py` (mid-run authorize + manifest publish memakai cap)
  - `src/simulation/experiment_validator.py` (validasi manifest/role memakai cap)
  - `tests/test_full_experiment_usage.py`, `tests/test_full_experiment_dummy.py` (uji batas cap)
- **reason**: Dua full attempt (v2 rep02, v4 relevance_rep01) gugur oleh SATU attempt tanpa usage akibat blip server OpenAI (5xx/timeout) — padahal retry berhasil dan data eksperimen utuh. Dengan 8.112 panggilan, blip tunggal statistiknya hampir pasti terjadi; aturan `missing_usage == 0` mutlak tidak realistis secara operasional.
- **bug_or_design_change**: design_change (kalibrasi toleransi audit, bukan perubahan semantik evaluasi)
- **impact_on_experiment**: MRA/BCR, bobot, Top-K, GT tidak tersentuh. Attempt tanpa usage tetap tercatat eksplisit di manifest (`missing_usage_attempt_count`) — transparan dan dibatasi cap 5 (≈0,06% dari 8.112 call). Melebihi cap tetap ditolak.
- **approval_status**: disetujui peneliti (sesi analisis v1.8.1)
- **evidence**: `logs/full_experiment_v2` (rep02), `logs/full_experiment_v4` (relevance_rep01) — masing-masing gugur oleh missing_usage=1
