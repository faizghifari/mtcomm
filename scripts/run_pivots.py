"""Indonesian pivots (NLLB, Google) on su<->ko, which the main runs played only on su<->jv.

`dialogue` rewrites records.jsonl with the episodes it was asked for, and some parents predate later rules, so the parent runs
are left alone: each task gets its own run, main3p_<task>, then scripts/merge_runs.py combines them. Same flags as the parent,
and a copy of its cache, so the opening messages are the ones the parent's conditions used and the comparison stays paired.

Usage: python scripts/run_pivots.py [--only sequence ...] [--batch-only]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT

HERE = Path(__file__).resolve().parent
COMMON = ["--player", "gemini-3.8-flash", "--monolingual", "--pairs", "su-ko", "ko-su", "--conds", "nllb-id", "google-id", "--pivots-all"]
# task: (parent runs, flags of the parent)
TASKS = {
    "sequence": (["main2x_sequence"], ["--n", "30", "--seq-words", "20", "--seq-questions", "0", "--seq-manifest", str(DATA_ROOT / "derived" / "bloom_vist" / "manifest_seq_v3.json")]),
    "mcq": (["main3_mcq"], ["--n", "30", "--mcq-lid-gate"]),
    "twentyq": (["main3_twentyq"], ["--n", "30", "--twentyq-pool", "paper140", "--twentyq-questions", "7"]),
    "route": (["main2x_route", "main3_route_e16"], ["--n", "30", "--route-level", "easy"]),
    "image": (["main2xf_image"], ["--n", "30"]),
    "culture": (["main3_culture"], ["--n", "20", "--culture-style", "targeted", "--culture-screen", str(DATA_ROOT / "derived" / "blend" / "culture_screen_38-flash.json")]),
}


def run_name(task):
    return f"main3p_{task}"


def seed(task):
    """Copy the parents' caches once. seed.json keeps their spend, because `report` counts every record in the cache."""
    d = OUTPUT_ROOT / run_name(task)
    if (d / "cache.jsonl").exists():
        return
    d.mkdir(exist_ok=True)
    parents = TASKS[task][0]
    with open(d / "cache.jsonl", "w") as out:
        for run in parents:
            out.write(open(OUTPUT_ROOT / run / "cache.jsonl").read())
    json.dump({"parents": parents, "spend_usd": sum(json.load(open(OUTPUT_ROOT / r / "summary.json"))["spend_total_usd"] for r in parents)},
              open(d / "seed.json", "w"), indent=1)


def run(task, batch_only):
    seed(task)
    args = ["--run", run_name(task), "--tasks", task, *COMMON, *TASKS[task][1]]
    py = [sys.executable, str(HERE / "run_main.py")]
    steps = [["batch", "--loop", "--interval", "60"]] + ([] if batch_only else [["dialogue", "--workers", "8"]])
    for cmd in steps:
        rc = subprocess.run(py + cmd + args).returncode
        if rc:
            return f"{task}: {cmd[0]} failed with exit code {rc}"
    return f"{task}: done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--batch-only", action="store_true", help="check that the openings come from the parents' cache")
    args = ap.parse_args()
    for task in args.only:
        print(run(task, args.batch_only), flush=True)


if __name__ == "__main__":
    main()
