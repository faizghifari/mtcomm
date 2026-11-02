"""Phase 2: calibrate the tasks for the single player, 3.8 Flash (docs/poc_next_steps.md). Ceiling and prior only, so no
translator is called. Each variant has its own run directory; `run` runs them in parallel.

Rule, fixed before the results: a variant is usable when its ceiling is 50-80% (target 60-75%) and at least 30 points above
the prior. n=60 games is about +/-12 points, so a variant within that of the band's edge is read as borderline.

Usage: python scripts/calibrate.py run [--only mcq_q8 ...]
       python scripts/calibrate.py report   -> outputs/mtcomm/calibrate_38flash.md
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import is_not_understood

HERE = Path(__file__).resolve().parent
PLAYER = "gemini-3.8-flash"
BV = DATA_ROOT / "derived" / "bloom_vist"
MCQ = DATA_ROOT / "derived" / "mcq"
SEQ = ["--seq-questions", "0"]
BOTH = ["ceiling", "prior"]
# name: (task, n per scored language, conditions, options, what it changes)
VARIANTS = {
    "mcq_q8": ("mcq", 20, ["ceiling"], ["--mcq-questions", "8"], "v2 items, 8 questions"),
    "mcq_w2": ("mcq", 20, BOTH, ["--mcq-screen", str(MCQ / "mcq_screen_38_w2.json")], "prior wrong in 2 of 3 languages, 4 questions"),
    "mcq_v2m": ("mcq", 20, ["ceiling", "passthrough"], ["--mcq-maybe-rule"], "v2 items, answerer told to say maybe rather than refuse"),
    "twentyq_q10": ("twentyq", 20, ["ceiling"], ["--twentyq-pool", "paper140", "--twentyq-questions", "10"], "paper's 140 words, 10 questions"),
    "mcq_gate": ("mcq", 20, ["ceiling", "passthrough"], ["--mcq-lid-gate"], "v2 items, language check as the answerer's gate"),
    "mcq_gatep": ("mcq", 20, ["passthrough"], ["--mcq-lid-gate"], "language-check gate, untranslated partner"),
    "twentyq_q7": ("twentyq", 20, ["ceiling"], ["--twentyq-pool", "paper140", "--twentyq-questions", "7"], "paper's 140 words, 7 questions"),
    "twentyq": ("twentyq", 20, BOTH, ["--twentyq-pool", "paper140"], "paper's 140 words"),
    "seq_w12": ("sequence", 20, ["ceiling"], ["--seq-manifest", str(BV / "manifest_seq_v3.json"), "--seq-words", "12", *SEQ], "v3, 12 words"),
    "seq_k4": ("sequence", 20, BOTH, ["--seq-manifest", str(BV / "manifest_seq_v3k4.json"), "--seq-words", "20", *SEQ], "4 distractors, 20 words"),
    "seq_k4w12": ("sequence", 20, ["ceiling"], ["--seq-manifest", str(BV / "manifest_seq_v3k4.json"), "--seq-words", "12", *SEQ], "4 distractors, 12 words"),
    "seq_p7": ("sequence", 20, BOTH, ["--seq-manifest", str(BV / "manifest_seq7_v1.json"), "--seq-words", "20", *SEQ], "7 pages, 20 words"),
    "wayfind_easy": ("wayfind", 10, BOTH, ["--wayfind-level", "easy"], "directions game, 2 turns"),
    "wayfind_mid": ("wayfind", 10, BOTH, ["--wayfind-level", "mid"], "directions game, 3 turns"),
    "wayfind_hard": ("wayfind", 10, BOTH, ["--wayfind-level", "hard"], "directions game, 4 turns"),
    "culture_t": ("culture", 20, BOTH, ["--culture-style", "targeted", "--culture-screen", str(DATA_ROOT / "derived" / "blend" / "culture_screen_38-flash.json")], "targeted prompt, screened items"),
}


def run_name(name):
    return f"cal38_{name}"


def run_variant(name):
    task, n, conds, opts, _ = VARIANTS[name]
    args = ["--run", run_name(name), "--tasks", task, "--player", PLAYER, "--monolingual", "--n", str(n), "--conds", *conds, *opts]
    py = [sys.executable, str(HERE / "run_main.py")]
    with open(OUTPUT_ROOT / f"{run_name(name)}.log", "a") as log:
        for cmd in (["batch", "--loop", "--interval", "60"], ["dialogue", "--workers", "8"], ["report"]):
            rc = subprocess.run(py + cmd + args, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                return f"{name}: {cmd[0]} failed with exit code {rc}"
    return f"{name}: done"


def verdict(ceil, prior):
    if ceil is None:
        return "no data"
    if ceil < 50:
        return "too low"
    if prior is not None and ceil - prior < 30:
        return "headroom under 30"
    if ceil > 80:
        return "too easy"
    return "target" if 60 <= ceil <= 75 else "in band"


def report():
    md = ["# Phase 2 calibration, Gemini 3.8 Flash\n",
          "Ceiling and prior only. Band: ceiling 50-80% (target 60-75%), at least 30 points above the prior. "
          "Refusals = ceiling games with a refusal sentence anywhere in the dialogue.\n",
          "| variant | task | what changes | ceiling (n) | ko / su / jv | prior | verdict | graded | refusals | $/game |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (task, _, _, _, what) in VARIANTS.items():
        p = OUTPUT_ROOT / run_name(name) / "records.jsonl"
        if not p.exists():
            md.append(f"| {name} | {task} | {what} | – | – | – | no data | – | – | – |")
            continue
        ce, pr, langs, graded, refused, games = [], [], defaultdict(list), [], 0, 0
        for line in open(p):
            r = json.loads(line)
            if r["status"] != "done":
                continue
            games += 1
            ok = int(bool(r["score"].get("correct")))
            if r["cond"] == "prior":
                pr.append(ok)
                continue
            if r["cond"] != "ceiling":  # e.g. a passthrough check of the same variant
                continue
            ce.append(ok)
            langs[r["eval_lang"]].append(ok)
            refused += any(is_not_understood(t["source"], t.get("lang", r["eval_lang"])) for t in r["turns"] if t.get("source"))
            for key in ("kendall_tau", "step_accuracy"):
                if r["score"].get(key) is not None:
                    graded.append(r["score"][key])
        rate = lambda v: 100 * sum(v) / len(v) if v else None
        c, q = rate(ce), rate(pr)
        per = " / ".join(f"{rate(langs[L]):.0f}" if langs[L] else "–" for L in ("ko", "su", "jv"))
        spend = json.load(open(p.parent / "summary.json"))["spend_total_usd"] / games if (p.parent / "summary.json").exists() and games else None
        md.append(f"| {name} | {task} | {what} | {f'{c:.0f}% ({len(ce)})' if ce else '–'} | {per} | "
                  f"{f'{q:.0f}% ({len(pr)})' if pr else '–'} | {verdict(c, q)} | "
                  f"{f'{sum(graded) / len(graded):.2f}' if graded else '–'} | {refused}/{len(ce)} | "
                  f"{f'{spend:.4f}' if spend is not None else '–'} |")
    out = OUTPUT_ROOT / "calibrate_38flash.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--only", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()
    if args.cmd == "report":
        return report()
    with ThreadPoolExecutor(len(args.only)) as ex:
        for msg in ex.map(run_variant, args.only):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
