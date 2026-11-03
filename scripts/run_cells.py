"""Cells added to the finished PoC runs. `dialogue` rewrites records.jsonl with the episodes it was asked for, so a parent run is left
alone: each addition gets its own run directory, and scripts/merge_runs.py combines them.

  pivots   NLLB and Google through Indonesian on su<->ko -> main3p_<task>
  qwen     Qwen on the tasks that ran while taco was down -> main3q_<task>
  repeat   a whole task again with fresh sampling -> main4_<task>_r<k>
  nllb     NLLB and NLLB via Indonesian again, after the input fix (NLLBChannel) -> main3n_<task>, and main4n_<task>_r<k> with --k

pivots, qwen and nllb copy the parents' cache, so the opening messages are the ones the parents' conditions used and the comparison stays
paired. A repeat has its own cache, so nothing is replayed. It runs every condition but passthrough. Flags come from the parent's
config.json, so a cell cannot drift from the run it extends.

Usage: python scripts/run_cells.py qwen [--only mcq ...]
       python scripts/run_cells.py repeat --only sequence --k 1 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mtcomm.paths import OUTPUT_ROOT

HERE = Path(__file__).resolve().parent
PLAYER = ["--player", "gemini-3.8-flash", "--monolingual"]
# task: parent runs, the first one giving the flags
PIVOT_PARENTS = {"sequence": ["main2x_sequence"], "mcq": ["main3_mcq"], "twentyq": ["main3_twentyq"],
                 "route": ["main2x_route", "main3_route_e16"], "image": ["main2xf_image"], "culture": ["main3_culture"]}
QWEN_PARENTS = {"mcq": "main3_mcq", "twentyq": "main3_twentyq", "route": "main3_route_e16", "culture": "main3_culture"}
REPEAT_CONDS = ["sol", "flash-lite", "nllb", "google", "qwen", "nllb-id", "google-id", "ceiling", "prior"]


def flags_from(config):
    """The command-line flags that reproduce a run's items and rules, from its config.json."""
    out = ["--n", str(config["n"])]
    tasks = config["tasks"]
    if "route" in tasks:
        out += ["--route-level", config["route_level"], "--route-start", str(config.get("route_start", 0))]
    if "sequence" in tasks:
        seq = config["sequence"]
        out += ["--seq-words", str(seq["max_words"]), "--seq-questions", str(seq["max_questions"])]
        out += ["--seq-manifest", seq["manifest"]] if seq.get("manifest") else []
    items = config.get("items") or {}
    if "twentyq" in tasks:
        out += ["--twentyq-pool", items["twentyq_pool"], "--twentyq-questions", str(items["twentyq_questions"])]
    if "mcq" in tasks and items.get("mcq_lid_gate"):
        out += ["--mcq-lid-gate"]
    if "culture" in tasks:
        out += ["--culture-style", items["culture_style"], "--culture-screen", items["culture_screen"]]
    return out


def parent_flags(run):
    return flags_from(json.load(open(OUTPUT_ROOT / run / "config.json")))


def seed(name, parents):
    """Copy the parents' caches once. seed.json keeps their spend, because `report` counts every record in the cache."""
    d = OUTPUT_ROOT / name
    if (d / "cache.jsonl").exists():
        return
    d.mkdir(exist_ok=True)
    with open(d / "cache.jsonl", "w") as out:
        for run in parents:
            out.write(open(OUTPUT_ROOT / run / "cache.jsonl").read())
    json.dump({"parents": parents, "spend_usd": sum(json.load(open(OUTPUT_ROOT / r / "summary.json"))["spend_total_usd"] for r in parents)},
              open(d / "seed.json", "w"), indent=1)


def play(name, args, workers):
    py = [sys.executable, str(HERE / "run_main.py")]
    for cmd in (["batch", "--loop", "--interval", "60"], ["dialogue", "--workers", str(workers)]):
        rc = subprocess.run(py + cmd + args).returncode
        if rc:
            return f"{name}: {cmd[0]} failed with exit code {rc}"
    return f"{name}: done"


def pivots(task, workers):
    name, parents = f"main3p_{task}", PIVOT_PARENTS[task]
    seed(name, parents)
    flags = ["--n", "30", "--route-level", "easy"] if task == "route" else parent_flags(parents[0])
    return play(name, ["--run", name, "--tasks", task, *PLAYER, "--pairs", "su-ko", "ko-su", "--conds", "nllb-id", "google-id", "--pivots-all", *flags], workers)


def qwen(task, workers):
    name, parent = f"main3q_{task}", QWEN_PARENTS[task]
    seed(name, [parent])
    return play(name, ["--run", name, "--tasks", task, *PLAYER, "--conds", "qwen", *parent_flags(parent)], workers)


def repeat(task, k, workers):
    parent = PIVOT_PARENTS[task][0]
    flags = ["--n", "30", "--route-level", "easy"] if task == "route" else parent_flags(parent)
    return play(f"main4_{task}_r{k}", ["--run", f"main4_{task}", "--repeat", str(k), "--tasks", task, *PLAYER, "--pivots-all", "--conds", *REPEAT_CONDS, *flags], workers)


def nllb(task, workers):
    name, parents = f"main3n_{task}", PIVOT_PARENTS[task]
    seed(name, parents)
    flags = ["--n", "30", "--route-level", "easy"] if task == "route" else parent_flags(parents[0])
    return play(name, ["--run", name, "--tasks", task, *PLAYER, "--conds", "nllb", "nllb-id", "--pivots-all", *flags], workers)


def nllb_repeat(task, k, workers):
    """A repeat played before the fix: same cache, so its opening messages and the other conditions stay paired."""
    name, parent = f"main4n_{task}_r{k}", f"main4_{task}_r{k}"
    seed(name, [parent])
    flags = ["--n", "30", "--route-level", "easy"] if task == "route" else parent_flags(PIVOT_PARENTS[task][0])
    return play(name, ["--run", name, "--tasks", task, *PLAYER, "--conds", "nllb", "nllb-id", "--pivots-all", *flags], workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pivots", "qwen", "repeat", "nllb"])
    ap.add_argument("--only", nargs="*", default=None, choices=list(PIVOT_PARENTS))
    ap.add_argument("--k", nargs="*", type=int, default=None, help="repeat numbers (repeat: default 1 2; nllb: none means the main run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tasks = args.only or list(QWEN_PARENTS if args.cmd == "qwen" else PIVOT_PARENTS)
    for task in tasks:
        if args.cmd == "pivots":
            print(pivots(task, args.workers), flush=True)
        elif args.cmd == "qwen":
            print(qwen(task, args.workers), flush=True)
        elif args.cmd == "nllb":
            for k in args.k or [None]:
                print(nllb_repeat(task, k, args.workers) if k else nllb(task, args.workers), flush=True)
        else:
            for k in args.k or [1, 2]:
                print(repeat(task, k, args.workers), flush=True)


if __name__ == "__main__":
    main()
