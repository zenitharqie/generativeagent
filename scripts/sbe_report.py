#!/usr/bin/env python3
"""SBE (Stock-Before-Event) report.

Menghitung, per run x event x agent, apakah ada minimal satu memori non-seed
berlabel-expected yang diciptakan SEBELUM step event terjadi.

Usage:
    python3 scripts/sbe_report.py <run_dir> [<run_dir> ...]
Contoh:
    python3 scripts/sbe_report.py logs/pilot_real_v8/phase08_dummy_pilot/balanced_rep01_pilot_day01_02
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

EVENT_DAY = {"event_01": 2, "event_02": 3, "event_03": 4, "event_04": 5, "event_05": 6}


def sbe_for_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    gt_targets = json.load(open(run_dir / "config_snapshot" / "ground_truth.json"))["targets"]
    expected = defaultdict(set)
    for t in gt_targets:
        expected[(t["event_id"], t["agent_id"])] |= set(t["expected_categories"])

    memories = [json.loads(l) for l in open(run_dir / "memory_logs.jsonl")]
    report = {}
    for (event_id, agent_id), cats in sorted(expected.items()):
        day_limit = EVENT_DAY[event_id]
        pre = [
            m
            for m in memories
            if m["agent_id"] == agent_id
            and m.get("memory_type") != "seed_persona"
            and m.get("day", 99) < day_limit
        ]
        stock = sum(1 for m in pre if m.get("category") in cats)
        report[f"{event_id}|{agent_id}"] = {
            "stock_before": stock,
            "pre_memories": len(pre),
        }
    return report


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    fail = False
    for run_dir in sys.argv[1:]:
        report = sbe_for_run(Path(run_dir))
        run_name = Path(run_dir).name
        print(f"\n=== SBE: {run_name} ===")
        for key, value in report.items():
            mark = "OK " if value["stock_before"] >= 1 else "GAP"
            print(f"  [{mark}] {key:<36} stock={value['stock_before']}  (pre-mem {value['pre_memories']})")
            if value["stock_before"] == 0:
                fail = True
    print("\nKesimpulan:", "ADA GAP STOK" if fail else "SEMUA UNIT PUNYA STOK")


if __name__ == "__main__":
    main()
