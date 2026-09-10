#!/usr/bin/env python3
"""Evaluate MRA of v9 logs against the NEW free-form GT (annotationv3/final_ground_truth.json).

Three matching methods:
  1. OpenAI cosine similarity, threshold 0.70
  2. OpenAI cosine similarity, threshold 0.85
  3. Semantic review by LLM judge (gpt-5.6-luna)

Usage:
  export OPENAI_API_KEY=sk-...
  python3 scripts/evaluate_mra_gtbaru.py

Outputs:
  ground_truth_research/mra_v10_cosine007.json
  ground_truth_research/mra_v10_cosine085.json
  ground_truth_research/mra_v10_llm_judge.json
  ground_truth_research/mra_v10_summary.json
"""

import json
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
GT_FILE = Path.home() / "ground_truth_research/annotationv3/final_ground_truth.json"
LOG_GLOB = str(REPO / "logs/full_experiment_v10/phase13_full_experiment/*_rep*/retrieval_logs.jsonl")
OUT_DIR = Path.home() / "ground_truth_research"

MODEL_EMBED = "text-embedding-3-small"
MODEL_JUDGE = "gpt-5.6-luna"
TOP_K = 5
SIM_TOLERANCE = 1e-6

JUDGE_PROMPT = """You are judging whether two behavior category labels describe the same underlying student/teacher behavior.

Category A (from ground truth): "{a}"
Category B (retrieved from agent memory): "{b}"

Consider semantic meaning, not wording. Answer with JSON exactly:
{{"match": true/false, "confidence": 0.0-1.0}}
"""

client = OpenAI()


def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL_EMBED, input=texts)
    vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def judge_pair(a: str, b: str, cache: dict) -> bool:
    key = f"{a}|||{b}"
    if key in cache:
        return cache[key]
    resp = client.chat.completions.create(
        model=MODEL_JUDGE,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(a=a, b=b)}],
        response_format={"type": "json_object"},
    )
    for _ in range(4):
        try:
            out = json.loads(resp.choices[0].message.content)
            match = bool(out["match"])
            cache[key] = match
            return match
        except (json.JSONDecodeError, KeyError, AttributeError):
            time.sleep(2)
    raise RuntimeError(f"LLM judge failed for pair: {a!r} / {b!r}")


def load_units() -> dict[str, dict]:
    """Collect 240 target-event retrieval entries from v9 logs."""
    gt = json.load(open(GT_FILE))
    gt_map = {}
    for uid, u in gt["ground_truth"].items():
        gt_map[f"{u['event_id']}_{u['agent_id']}"] = u["categories"]

    units = []
    for rf in sorted(glob.glob(LOG_GLOB)):
        run = rf.split("/")[-2]
        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if not d.get("is_target_event"):
                    continue
                key = f"{d['event_id']}_{d['agent_id']}"
                selected = [c for c in d["candidates"] if c.get("selected")]
                units.append({
                    "run_id": run,
                    "config": d["retrieval_config_id"],
                    "event_id": d["event_id"],
                    "agent_id": d["agent_id"],
                    "expected": gt_map.get(key, []),
                    "retrieved": [c["category"] for c in selected],
                })
    if len(units) != 240:
        print(f"WARNING: expected 240 target entries, found {len(units)}", file=sys.stderr)
    return units


def cosine_mra(units: list[dict], threshold: float, cache: dict) -> tuple[list[dict], dict]:
    unique = sorted({t for u in units for t in u["expected"] + u["retrieved"]})
    vecs = {}
    B = 512
    for i in range(0, len(unique), B):
        batch = unique[i:i + B]
        emb = embed_texts(batch)
        for t, v in zip(batch, emb):
            vecs[t] = v

    results = []
    for u in units:
        matched = []
        scores = []
        for ret in u["retrieved"]:
            best, best_score = None, -1.0
            for exp in u["expected"]:
                score = float(np.dot(vecs[ret], vecs[exp]))
                if score > best_score:
                    best, best_score = exp, score
            if best is not None and best_score >= threshold - SIM_TOLERANCE:
                matched.append(best)
                scores.append(round(best_score, 4))
        results.append({
            "run_id": u["run_id"], "config": u["config"],
            "event_id": u["event_id"], "agent_id": u["agent_id"],
            "expected_categories": u["expected"],
            "retrieved_categories": u["retrieved"],
            "matched_categories": matched, "similarity_scores": scores,
            "matched_count": len(matched), "denominator": TOP_K,
            "mra": round(len(matched) / TOP_K, 4),
        })
    summary = aggregate(results)
    return results, summary


def llm_mra(units: list[dict], cache: dict) -> tuple[list[dict], dict]:
    results = []
    for idx, u in enumerate(units):
        matched = []
        for ret in u["retrieved"]:
            for exp in u["expected"]:
                if judge_pair(exp, ret, cache):
                    matched.append(exp)
                    break
        results.append({
            "run_id": u["run_id"], "config": u["config"],
            "event_id": u["event_id"], "agent_id": u["agent_id"],
            "expected_categories": u["expected"],
            "retrieved_categories": u["retrieved"],
            "matched_categories": matched,
            "matched_count": len(matched), "denominator": TOP_K,
            "mra": round(len(matched) / TOP_K, 4),
        })
        if (idx + 1) % 20 == 0:
            print(f"  LLM judge: {idx + 1}/{len(units)} units done")
    summary = aggregate(results)
    return results, summary


def aggregate(results: list[dict]) -> dict:
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r["mra"])
    cfg = {c: round(sum(v) / len(v), 4) for c, v in sorted(by_config.items())}
    overall = round(sum(r["mra"] for r in results) / len(results), 4) if results else 0.0
    return {"by_config": cfg, "overall": overall, "n_units": len(results)}


def save(name: str, method: str, extra: dict, results: list[dict], summary: dict):
    out = {
        "metadata": {
            "method": method,
            "gt_source": str(GT_FILE),
            "gt_categories": 46,
            "log_source": "logs/full_experiment_v10/phase13_full_experiment/",
            "embedding_model": MODEL_EMBED if "cosine" in method else None,
            "judge_model": MODEL_JUDGE if "LLM" in method else None,
            "top_k": TOP_K,
            **extra,
        },
        "summary": summary,
        "results": results,
    }
    path = OUT_DIR / name
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"Saved: {path}")


def main():
    units = load_units()
    print(f"Loaded {len(units)} target-event entries")

    print("\n[1/3] Cosine similarity, threshold 0.70 ...")
    res, summ = cosine_mra(units, 0.70, {})
    save("mra_v10_cosine007.json", "OpenAI cosine similarity", {"threshold": 0.70}, res, summ)
    print("   ", summ)

    print("\n[2/3] Cosine similarity, threshold 0.85 ...")
    res, summ = cosine_mra(units, 0.85, {})
    save("mra_v10_cosine085.json", "OpenAI cosine similarity", {"threshold": 0.85}, res, summ)
    print("   ", summ)

    print("\n[3/3] LLM judge (gpt-5.6-luna) ...")
    cache: dict = {}
    res, summ = llm_mra(units, cache)
    save("mra_v10_llm_judge.json", "LLM semantic review", {"judge_pairs_cached": len(cache)}, res, summ)
    print("   ", summ)

    combined = {
        "metadata": {
            "description": "MRA with NEW free-form GT (46 categories) vs v9 logs, 3 matching methods",
            "gt_source": str(GT_FILE),
            "top_k": TOP_K,
        },
        "methods": {},
    }
    for label, fname in [
        ("cosine_0.70", "mra_v10_cosine007.json"),
        ("cosine_0.85", "mra_v10_cosine085.json"),
        ("llm_judge", "mra_v10_llm_judge.json"),
    ]:
        d = json.load(open(OUT_DIR / fname))
        combined["methods"][label] = d["summary"]
    json.dump(combined, open(OUT_DIR / "mra_v10_summary.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nSaved: {OUT_DIR / 'mra_v10_summary.json'}")

    print("\n=== SUMMARY (MRA, new GT) ===")
    print(f"{'config':20s} {'cos0.70':>9s} {'cos0.85':>9s} {'llm':>9s}")
    cfgs = sorted(next(iter(combined["methods"].values()))["by_config"])
    for c in cfgs:
        v = [combined["methods"][m]["by_config"][c] for m in combined["methods"]]
        print(f"{c:20s} {v[0]:9.3f} {v[1]:9.3f} {v[2]:9.3f}")
    o = [combined["methods"][m]["overall"] for m in combined["methods"]]
    print(f"{'OVERALL':20s} {o[0]:9.3f} {o[1]:9.3f} {o[2]:9.3f}")


if __name__ == "__main__":
    main()