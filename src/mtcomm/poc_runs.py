"""The runs behind the PoC results, with one player (3.8 Flash) on every task. Analysis scripts read this list.

Each poc3_<task> is the task's main run plus cells played later, merged by scripts/merge_runs.py: the Indonesian pivots on su<->ko
(scripts/run_cells.py), Qwen on the tasks that ran while taco was down, and for route the 30 easy maps (main2x_route e01-e15, main3_route_e16). Sequence and
image come from main2, which already used 3.8 Flash. A condition goes in only when it has all the items.
"""

import json

from .paths import OUTPUT_ROOT

ANALYSIS = OUTPUT_ROOT / "analysis_poc3"
# title, run, task
FINAL = [("MCQ conversation", "poc3_mcq", "mcq"), ("Twenty Questions", "poc3_twentyq", "twentyq"),
         ("Sequence ordering", "poc3_sequence", "sequence"), ("Image near-twin", "poc3_image", "image"),
         ("Route, easy maps", "poc3_route", "route"), ("Culture (BLEnD)", "poc3_culture", "culture")]
RUNS = [run for _, run, _ in FINAL]


def available():
    """FINAL without runs that have not finished."""
    return [t for t in FINAL if (OUTPUT_ROOT / t[1] / "records.jsonl").exists()]


def runs_of(run):
    """A merged run and its repeats: poc3_<task>, rep_<task>_r1, ..."""
    reps = OUTPUT_ROOT.glob(f"rep_{run.removeprefix('poc3_')}_r[0-9]*/records.jsonl")
    return [run, *sorted(p.parent.name for p in reps)]


def all_games(run):
    """Finished games of a task over the run and its repeats, as dicts: run, cond, key, item, ok.

    The key is the pair, or the scored language for ceiling and prior, whose games are not tied to one pair."""
    out = []
    for name in runs_of(run):
        for line in open(OUTPUT_ROOT / name / "records.jsonl"):
            r = json.loads(line)
            if r["status"] == "done":
                out.append({"run": name, "cond": r["cond"], "key": r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"],
                            "item": r["item"]["item_id"], "ok": int(bool(r["score"].get("correct")))})
    return out
