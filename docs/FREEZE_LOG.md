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

---

## change_003 — v1.9.0: Proportional theme calibration (theme-crowding hypothesis test)

- **date**: 2026-08-27
- **changed_files**:
  - `config/simulation_steps.json` (hanya 3 deskripsi blok natural: day_02_natural_03 muatan ujian dikembalikan netral; day_03_natural_04 fokus norma/raise-hand + persiapan sekolah; day_06_natural_07 diarahkan ke eksekusi ujian)
- **hypothesis (H2-theme crowding)**: Pemanjatan tema ujian pada blok day-2/3 di v1.8.0 menggeser distribusi label stream sehingga event_02 kehilangan stok partisipasi/norma (MRA 0.625→0.217). Kalibrasi proporsional diprediksi memulihkan event_02 tanpa mengorbankan event_01.
- **pre-registered predictions (dideklarasikan sebelum run)**:
  - event_02: 0.217 → 0.40–0.55
  - event_05: 0.138 → 0.20–0.30
  - event_01: boleh turun ke 0.6–0.7 (trade-off diakui)
  - event_04: bertahan ≥0.40
  - recency_heavy di event_02 diprediksi pulih paling besar (indikator mekanisme crowding)
- **bug_or_design_change**: design_change (konteks blok natural; bukan GT/bobot/evaluasi)
- **impact_on_experiment**: GT beku (c6e310cc), bobot, Top-K, formula, judge tidak disentuh; perubahan identik untuk 4 konfigurasi.
- **approval_status**: disetujui peneliti (framing: eksperimen hipotesis-tes untuk perbandingan v5 vs v6)
- **evidence**: analisis theme-crowding — komposisi 240 slot Top-5 event_02 di v5: TEST/PREP 50%, PARTICIPATION 36%; recency_heavy event_02 = 0.017

---

## change_004 — v1.9.1: Lower operational disk-space threshold (5 GiB → 3.5 GiB)

- **date**: 2026-08-27
- **changed_files**: `src/simulation/experiment_controller.py` (konstanta `OPERATIONAL_MINIMUM_FREE_BYTES`)
- **reason**: Ambang 5 GiB terbukti terlalu konservatif — 12 run aktual hanya menulis ±150 MB artefak, sementara disk peneliti kronis sempit sehingga gerbang berulang kali memblokir run yang valid. 3,5 GiB tetap memberi margin aman bagi artefak + ruang napas sistem.
- **bug_or_design_change**: design_change (operasional infrastruktur; bukan semantik evaluasi)
- **impact_on_experiment**: Tidak menyentuh GT/bobot/evaluasi. Gerbang tetap melindungi dari disk habis di tengah run (toleransi turun dari 5 ke 3,5 GiB — peneliti wajib tetap memantau ruang).
- **approval_status**: diminta peneliti, disetujui sesi analisis
