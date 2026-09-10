#!/usr/bin/env python3
"""Tahap 2 — Union frasa per unit + embedding + klasterisasi cosine ≥ 0,70.

Input : annotationv3/{gpt5_6_sol,claude_sonnet_5,qwen3_8}.json
Output: canonicalization_result.json  (klaster, varian, dukungan per unit)

Jalankan:  python3 scripts/canonicalize_freeform.py
Butuh     : OPENAI_API_KEY di environment (untuk embedding frasa unik)
"""
import json, os, sys, itertools
from collections import defaultdict
from pathlib import Path

REPO = Path("/Users/zenitharq/Documents/classroom-generative-agents-clean")
ROOT = Path("/Users/zenitharq/ground_truth_research")
SRC = ROOT / "annotationv3"
AGENTS = ["gpt5_6_sol", "claude_sonnet_5", "qwen3_8"]
THRESHOLD = 0.70
OUT = ROOT / "canonicalization_result.json"

# ── 1. Validasi & union ──────────────────────────────────────────────
data = {}
for aid in AGENTS:
    d = json.load(open(SRC / f"{aid}.json", encoding="utf-8"))
    assert len(d) == 20, f"{aid}: unit != 20"
    data[aid] = d

unit_order = [r["annotation_unit_id"] for r in data["gpt5_6_sol"]]
for aid in AGENTS[1:]:
    ids = [r["annotation_unit_id"] for r in data[aid]]
    assert ids == unit_order, f"{aid}: urutan unit tidak kanonik"

# cek input_sha256 konsisten
for uid in unit_order:
    shas = set()
    for aid in AGENTS:
        r = next(x for x in data[aid] if x["annotation_unit_id"] == uid)
        shas.add(r["input_sha256"])
    assert len(shas) == 1, f"{uid}: input_sha256 berbeda antar anotator"

# union frasa unik per unit (normalisasi: strip + casefold utk dedup persis)
union = {}       # uid -> list of {"text": original, "norm": normalized, "who": [agent...]}
for aid in AGENTS:
    for r in data[aid]:
        uid = r["annotation_unit_id"]
        for c in r["categories"]:
            txt = c["category"].strip()
            norm = " ".join(txt.casefold().split())
            found = False
            for p in union.setdefault(uid, []):
                if p["norm"] == norm:
                    if aid not in p["who"]:
                        p["who"].append(aid)
                    found = True
                    break
            if not found:
                union.setdefault(uid, []).append({"text": txt, "norm": norm, "who": [aid]})

total_phrases = sum(len(v) for v in union.values())
print(f"[1] union selesai: {total_phrases} frasa unik di 20 unit")

# ── 2. Embedding ─────────────────────────────────────────────────────
from openai import OpenAI
client = OpenAI()  # butuh OPENAI_API_KEY

all_phrases = []
for uid in unit_order:
    for p in union[uid]:
        all_phrases.append(p["norm"])
# dedup lintas unit tidak perlu — klaster per unit
phrase_vec = {}
BATCH = 100
uniq = list(dict.fromkeys(all_phrases))
for i in range(0, len(uniq), BATCH):
    batch = uniq[i:i+BATCH]
    resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
    for j, d in enumerate(resp.data):
        phrase_vec[batch[j]] = d.embedding
print(f"[2] embedding selesai: {len(uniq)} frasa")

# ── 3. Klasterisasi per unit (greedy, ambang 0,70) ───────────────────
import math
def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

clusters = {}   # uid -> list of clusters; cluster = {"members":[norm...], "who":set}
for uid in unit_order:
    phrases = union[uid]
    cl = []
    for p in phrases:
        vec = phrase_vec[p["norm"]]
        placed = False
        for c in cl:
            # cek terhadap representatif (frasa pertama di klaster)
            rep = phrase_vec[c["members"][0]]
            if cos(vec, rep) >= THRESHOLD:
                c["members"].append(p["norm"]); c["who"] |= set(p["who"])
                placed = True; break
        if not placed:
            cl.append({"members": [p["norm"]], "who": set(p["who"])})
    # representatif = frasa dengan dukungan anotator terbanyak, lalu leksikal
    out = []
    for c in cl:
        best = max(c["members"], key=lambda m: (-len(next(u for u in union[uid] if u["norm"]==m)["who"]), m))
        out.append({"canonical": best, "variants": sorted(c["members"]),
                    "who": sorted(c["who"])})
    out.sort(key=lambda c: c["canonical"])
    clusters[uid] = out

print("[3] klasterisasi selesai")

# ── 4. Simpan hasil ─────────────────────────────────────────────────
result = {
    "method": "free-form union + cosine clustering (text-embedding-3-small, threshold 0.70)",
    "threshold": THRESHOLD,
    "annotators": AGENTS,
    "units": clusters,
}
json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"[4] tersimpan: {OUT}")

# ringkasan
for uid in unit_order:
    n_cl = len(clusters[uid])
    maxsup = max(len(c["who"]) for c in clusters[uid])
    print(f"  {uid:<32} klaster:{n_cl:>3}  dukungan max: {maxsup}/3")
