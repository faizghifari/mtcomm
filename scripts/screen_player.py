"""Screen one player model on every task: ceiling and prior (plus passthrough on MCQ and sequence, to see how much
of an untranslated partner the player reads), with main2's exact task settings, so the games pair item by item with
the existing 3.1 Flash-Lite and 3.8 Flash ceilings.

Item lists are the first n of main2's lists. Route uses only the easy maps (15 exist; hard maps are dropped).

Decision rule, fixed before the results (docs/poc_next_steps.md, Phase 1):
  - ceiling below 50%: the player cannot play this task (a player failure);
  - ceiling above 80%: the task needs calibrating for this player (not a reason to reject it);
  - ceiling at least 30 points above the prior, otherwise the task does not need the conversation;
  - culture is left out of the decision: its prior is high whichever player runs it (docs/poc_diagnostics.md);
  - 60 games per task is about +/-12 points, so borderline results are read against the paired ceilings.

Usage: python scripts/screen_player.py run --player gemini-3.5-flash-lite [--tasks mcq ...] [--n 20]
       python scripts/screen_player.py report --player gemini-3.5-flash-lite
"""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import is_not_understood

HERE = Path(__file__).resolve().parent
TASKS = ["mcq", "twentyq", "sequence", "image", "route", "culture"]
SETTINGS = {"sequence": ["--seq-manifest", str(DATA_ROOT / "derived" / "bloom_vist" / "manifest_seq_v3.json"),
                         "--seq-words", "20", "--seq-questions", "0"],
            "route": ["--route-level", "easy"]}
PASSTHROUGH = {"mcq", "sequence"}
# Existing runs on the same items, for the paired comparison.
REFERENCE = {"3.1 Flash-Lite": {"mcq": "main2x_mcq", "twentyq": "main2x_twentyq", "image": "main2x_image", "culture": "main2x_culture"},
             "3.8 Flash": {"mcq": "main2r_flash_mcq", "sequence": "main2x_sequence", "image": "main2xf_image", "route": "main2x_route"}}


TAG = ""  # --tag v2: a separate run of the same screen


def run_name(player, task):
    return f"screen_{player.replace('gemini-', '').replace('.', '')}{TAG}_{task}"


def run_task(player, task, n):
    # Route: the first 15 easy maps, the ones main2 played.
    conds = ["ceiling", "prior"] + (["passthrough"] if task in PASSTHROUGH else [])
    args = ["--run", run_name(player, task), "--tasks", task, "--player", player, "--monolingual",
            "--n", "15" if task == "route" else str(n), "--conds", *conds, *SETTINGS.get(task, [])]
    py = [sys.executable, str(HERE / "run_main.py")]
    log = open(OUTPUT_ROOT / f"{run_name(player, task)}.log", "a")
    for cmd in (["batch", "--loop", "--interval", "60"], ["dialogue", "--workers", "8"], ["report"]):
        rc = subprocess.run(py + cmd + args, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            return f"{task}: {cmd[0]} failed with exit code {rc}"
    return f"{task}: done"


def outcomes(run, conds=None):
    """{(scope, cond, item_id): 0/1}; ceiling and prior are keyed by receiving language, passthrough by pair."""
    p = OUTPUT_ROOT / run / "records.jsonl"
    d = {}
    if not p.exists():
        return d
    for line in open(p):
        r = json.loads(line)
        if r["status"] == "done" and (conds is None or r["cond"] in conds):
            scope = r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"]
            d[(scope, r["cond"], r["item"]["item_id"])] = int(bool(r["score"].get("correct")))
    return d


def rate(values):
    return (100 * sum(values) / len(values), len(values)) if values else (None, 0)


def fmt(rn):
    return "–" if rn[0] is None else f"{rn[0]:.0f}% ({rn[1]})"


def mcnemar(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def verdict(task, ceil, prior):
    if ceil is None:
        return "no data"
    if task == "culture":
        return "not used for the decision"
    if ceil < 50:
        return "too low: cannot play"
    if prior is not None and ceil - prior < 30:
        return "headroom under 30 points"
    return "ok, calibrate down" if ceil > 80 else "ok"


def report(args):
    md = [f"# Player screen: {args.player}\n",
          "Ceiling and prior per task, main2's settings, n per receiving language. Paired = same items in an existing run: "
          "difference in points, games better / worse, exact McNemar p. Opening refusals: games where the first message is the "
          "refusal sentence (with the rewrite rule they are rejected first, and remain only if the rewrites fail too).\n",
          "| task | ceiling | per language | prior | verdict | vs 3.1 Flash-Lite | vs 3.8 Flash | passthrough su↔ko | passthrough su↔jv | opening refusals | $/game |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        run = run_name(args.player, task)
        d = outcomes(run)
        ce = {k: v for k, v in d.items() if k[1] == "ceiling"}
        pr = [v for k, v in d.items() if k[1] == "prior"]
        ceil, prior = rate(list(ce.values())), rate(pr)
        langs = defaultdict(list)
        for k, v in ce.items():
            langs[k[0]].append(v)
        per = " ".join(f"{L} {100 * sum(v) / len(v):.0f}" for L, v in sorted(langs.items()))
        refs = []
        for name in REFERENCE:
            ref = REFERENCE[name].get(task)
            o = outcomes(ref, {"ceiling"}) if ref else {}
            pairs = [(v, o[k]) for k, v in ce.items() if k in o]
            if not pairs:
                refs.append("–")
                continue
            b, c = sum(a > x for a, x in pairs), sum(a < x for a, x in pairs)
            refs.append(f"{100 * (b - c) / len(pairs):+.0f} ({b}/{c}, p={mcnemar(b, c):.2f}, n={len(pairs)})")
        pt = defaultdict(list)
        for k, v in d.items():
            if k[1] == "passthrough":
                pt["ko" if "ko" in k[0] else "jv"].append(v)
        opening, ceiling_games = 0, 0
        rp = OUTPUT_ROOT / run / "records.jsonl"
        if rp.exists():
            for line in open(rp):
                r = json.loads(line)
                if r["status"] == "done" and r["cond"] == "ceiling" and r["turns"]:
                    ceiling_games += 1
                    opening += is_not_understood(r["turns"][0]["source"], r["eval_lang"])
        spend = "–"
        sp = OUTPUT_ROOT / run / "summary.json"
        if sp.exists() and d:
            spend = f"{json.load(open(sp))['spend_total_usd'] / len(d):.4f}"
        md.append(f"| {task} | {fmt(ceil)} | {per} | {fmt(prior)} | {verdict(task, ceil[0], prior[0])} | {refs[0]} | {refs[1]} | "
                  f"{fmt(rate(pt['ko']))} | {fmt(rate(pt['jv']))} | {opening}/{ceiling_games} | {spend} |")
    out = OUTPUT_ROOT / f"screen_{args.player.replace('gemini-', '').replace('.', '')}{TAG}.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for run names, e.g. v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--player", required=True)
    r.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    r.add_argument("--n", type=int, default=20)
    q = sub.add_parser("report")
    q.add_argument("--player", required=True)
    args = ap.parse_args()
    TAG = f"_{args.tag}" if args.tag else ""
    if args.cmd == "report":
        return report(args)
    with ThreadPoolExecutor(len(args.tasks)) as ex:
        for msg in ex.map(lambda t: run_task(args.player, t, args.n), args.tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
