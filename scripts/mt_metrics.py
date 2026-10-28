"""Standard MT metrics on every translated message, following mt-eval (research/mt_compare).

  export   collect (source, translation) for every message a translator delivered in the given runs;
           identical (source, translation, direction) triples are scored once
  comet    CometKiwi and XCOMET-XL, reference-free, as in mt-eval's compute_comet.py (scripts/comet_score.py
           in the separate .venv-metrics env; local GPU, run when no NLLB process holds the GPU)
  report   join scores back to messages and games: mean per (task, pair, condition), and how well the
           episode's mean score separates successful from failed games (AUC)

Usage: python scripts/mt_metrics.py export --runs main_image main_mcq
       python scripts/mt_metrics.py comet
       python scripts/mt_metrics.py report
"""

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from mtcomm.paths import OUTPUT_ROOT

OUT = OUTPUT_ROOT / "metrics"
TRANSLATED = {"sol", "flash-lite", "nllb", "google", "nllb-id", "google-id"}
MODELS = ["cometkiwi", "xcomet"]


def _seg_id(src, mt, a, b):
    return hashlib.sha256(json.dumps([src, mt, a, b], ensure_ascii=False).encode()).hexdigest()[:16]


def export(runs):
    segs, msgs = {}, []
    for run in runs:
        for line in open(OUTPUT_ROOT / run / "records.jsonl"):
            r = json.loads(line)
            if r["status"] != "done" or r["cond"] not in TRANSLATED:
                continue
            for k, t in enumerate(r["turns"]):
                if t["kind"] != "message" or not t.get("delivered"):
                    continue
                a, b = r["roles"][t["role"]], next(v for role, v in r["roles"].items() if role != t["role"])
                sid = _seg_id(t["source"], t["delivered"], a, b)
                segs[sid] = {"record_id": sid, "source_text": t["source"], "post_edit": t["delivered"],
                             "src_lang": a, "tgt_lang": b}
                msgs.append({"seg": sid, "run": run, "episode": r["episode"], "turn": k, "task": r["task"],
                             "pair": r["pair"], "cond": r["cond"], "role": t["role"],
                             "correct": bool(r["score"].get("correct"))})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "segments.jsonl", "w") as f:
        for s in segs.values():
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(OUT / "messages.jsonl", "w") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{len(msgs)} translated messages, {len(segs)} distinct segments -> {OUT}")


def comet(batch_size, models):
    """Score in the separate .venv-metrics env (see comet_score.py for why not mt-eval's env)."""
    here = Path(__file__).resolve().parents[1]
    cmd = [str(here / ".venv-metrics" / "bin" / "python"), str(here / "scripts" / "comet_score.py"),
           str(OUT / "segments.jsonl"), str(OUT / "comet_scores.jsonl"), "--models", *models,
           "--batch-size", str(batch_size)]
    subprocess.run(cmd, check=True)


def _auc(pos, neg):
    """P(score of a successful game > score of a failed one); ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def report():
    scores = {}
    for line in open(OUT / "comet_scores.jsonl"):
        r = json.loads(line)
        scores[r["record_id"]] = {m: r.get(f"{m}_score") for m in MODELS}
    cells = defaultdict(lambda: defaultdict(list))
    episodes = defaultdict(lambda: defaultdict(list))
    outcome = {}
    for line in open(OUT / "messages.jsonl"):
        m = json.loads(line)
        sc = scores.get(m["seg"])
        if not sc:
            continue
        for metric, v in sc.items():
            if v is not None:
                cells[(m["task"], m["pair"], m["cond"])][metric].append(v)
                episodes[(m["task"], m["episode"])][metric].append(v)
        outcome[(m["task"], m["episode"])] = m["correct"]
    rows = [{"task": t, "pair": p, "cond": c, "n_messages": len(next(iter(v.values()), [])),
             **{f"{metric}_mean": round(sum(x) / len(x), 4) for metric, x in v.items()}}
            for (t, p, c), v in sorted(cells.items())]
    auc = {}
    for task in sorted({t for t, _ in episodes}):
        for metric in MODELS:
            ep = [(sum(v[metric]) / len(v[metric]), outcome[k]) for k, v in episodes.items()
                  if k[0] == task and v[metric]]
            auc[f"{task}/{metric}"] = _auc([s for s, ok in ep if ok], [s for s, ok in ep if not ok])
    json.dump({"cells": rows, "auc_success_vs_episode_mean": auc}, open(OUT / "summary.json", "w"), indent=1)
    for r in rows:
        print(r)
    print("AUC (episode mean metric separates success from failure; 0.5 = no signal):", auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "comet", "report"])
    ap.add_argument("--runs", nargs="*", default=["main_image", "main_mcq"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.runs)
    elif args.cmd == "comet":
        comet(args.batch_size, args.models)
    else:
        report()


if __name__ == "__main__":
    main()
