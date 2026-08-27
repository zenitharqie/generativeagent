# METHODOLOGY REVISION PLAN v1.8 — Targeted Memory-Stock Repair

**Status:** Draf rancangan (menunggu persetujuan peneliti)
**Basis:** Diagnosis lengkap atas full experiment `logs/full_experiment_v3` (12/12 completed)
**Prinsip:** Ground truth TIDAK diubah. Hanya sisi instrumen simulasi diperbaiki. Semua ubahan seragam untuk 4 konfigurasi → netralitas terjaga.

---

## 1. Diagnosis Terkonsolidasi (bukti)

| Event | MRA (12 run) | Stok memori expected sebelum event | Penyebab dominan |
|---|---|---|---|
| event_01 | 0.046 | 0–1 unit | Tema GT mencampur kata "test" padahal pengumuman baru terjadi SAAT event |
| event_02 | 0.625 | 7–12 unit ✔ | Sehat — label partisipasi/norma bersifat harian |
| event_03 | 0.471 | 4–4–0–0 | Sebagian sehat (active/academic gap) |
| event_04 | **0.000** | **0 unit di 12/12 run** | Label kepemimpinan tak pernah dipilih walau blok pagi harinya bertema responsibility |
| event_05 | 0.008 | 0–1 unit | Meski pengumuman ujian sudah sejak hari-2, percakapan hari 3–6 jarang berlabel persiapan ujian |

**Kesimpulan kausal:** retrieval bekerja normal (bukti: event_02 = 0.62, relevance_heavy unggul 0.78 di sana). Kegagalan terjadi di **hulu — produksi memori berlabel-expected**. Kategori frasa GT banyak yang *event-anchored* (memuat kata "test"/"leader"), sehingga tidak natural terpilih sebelum temanya muncul di timeline.

---

## 2. Perubahan A — Kalibrasi tema blok natural (`config/simulation_steps.json`)

Blok tetap netral (deskripsi topik kelas, bukan instruksi ke agen):

| Blok | Tambahan kalimat pada `description` | Memupuk stok untuk |
|---|---|---|
| day_01_natural_01 | Each student also shares how they prefer to learn and prepare for lessons. | event_01 (kebiasaan belajar) |
| day_01_natural_02 | …including what makes each student confident or nervous about schoolwork in general. | event_01 (kecemasan/kepercayaan diri) |
| day_02_natural_03 | …and how reviewing the material will help them prepare for the announced end-of-week written test. | event_05 + penguatan pasca-pengumuman |
| day_04_natural_05 | …including how groups decide who guides the explanation and how classmates evaluate each contribution fairly. | event_04 + event_03 (hari yang sama) |
| day_05_natural_06 | …and what makes a classmate dependable when taking responsibility as class leader. | event_04 (pagi sebelum sorenya) |
| day_06_natural_07 | …as students compare their preparation strategies right before the scheduled written test begins. | event_05 (langsung pra-ujian) |

---

## 3. Perubahan B — Panduan pelabelan lebih mendalam (2 prompt)

Tambah satu kalimat di `prompts/natural_interaction.txt`, `target_event_response.txt`, dan `reflect.txt`:

> When several categories fit, choose the one naming the lasting behavioural tendency or concern behind the action, even if its wording refers to situations beyond today's topic.

Klausa refleksi (`reflect.txt`) ditambah:

> If appropriate, frame the insight as an ongoing tendency that may matter in a future situation.

---

## 4. Instrumen ukur baru — SBE (Stock-Before-Event)

Skrip `scripts/sbe_report.py`: menghitung, per run × event × agent, apakah minimal ada 1 memori non-seed berlabel-expected yang dibuat SEBELUM step event.

| Gerbang | Kriteria |
|---|---|
| Pilot v8 (Day 1–2 saja) | event_01: ≥3 dari 4 agent punya stok ≥1 |
| Full experiment v4 | event_04 & event_05: tiap agent punya stok ≥1 di ≥10/12 run; event_01 dinikhi sebagai ceiling struktural (dilaporkan, bukan digarap paksa) |

---

## 5. Tata urutan eksekusi

```
0. Simpan analisis ini + catat di docs/FREEZE_LOG.md (revisi aparatur, bukan GT)
1. Edit simulation_steps.json (tabel §2)
2. Edit 3 prompt (§3) → hitung ulang hash → update safeguards
3. pytest penuh (harus 995 hijau) → dummy pilot
4. Commit v1.8.0 + tag
5. PILOT real v8 → jalankan SBE report → cek gerbang pilot
6. review-pilot accepted
7. FULL EXPERIMENT v4
8. Analisis akhir: MRA matrix + SBE + BCR judge
```

## 6. Estimasi biaya & waktu

| Item | Nilai |
|---|---|
| Editing & testing (langkah 1–4) | ±30 menit kerja, Rp0 |
| Pilot v8 | ~160 rb token (~$0,15) |
| Full experiment v4 | ~5,5 jt token (~$2–3), 4–6 jam |
| Ulangi jika gerbang gagal | hanya pilot (murah) |

## 7. Batasan yang diakui jujur

- event_01 punya **ceiling struktural**: sebagian kategorinya menyebut "test" sebelum ujian diumumkan. Naikkan semaksimal mungkin lewat temuan umum Day 1, sisanya dilaporkan sebagai limitasi desain (dan itu temuan metodologis yang valid).
- Revisi ini dilakukan SETELAH melihat hasil — justifikasinya dicatat eksplisit: mengikuti **MRA Debug Protocol §17** (perbaikan hulu: assign kategori & stok memori), bukan tuning menuju hasil; GT beku, bobot beku, Top-K beku.

---
*Dibuat otomatis oleh sesi analisis v1.7.5 — revisi menunggu persetujuan peneliti.*
