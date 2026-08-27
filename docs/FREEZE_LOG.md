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
