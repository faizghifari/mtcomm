"""Standard MT metrics on every translated message, following mt-eval (research/mt_compare).

  export   collect (source, translation) for every message a translator delivered in the given runs;
           identical (source, translation, direction) triples are scored once
  comet    CometKiwi and XCOMET-XL, reference-free, as in mt-eval's compute_comet.py (scripts/comet_score.py
           in the separate .venv-metrics env; local GPU, run when no NLLB process holds the GPU)
  gemba    GEMBA-MQM (mt-eval's prompt, parser and severity weights, imported from its compute_gemba.py)
           on a stratified sample, up to --per-cell messages per (task, condition), as one Gemini batch;
           judge Gemini 3.8 Flash (mt-eval's gemini-3-flash-preview judge may be retired). Re-run to poll.
  report   join scores back to messages and games: mean per (task, pair, condition), and how well the
           episode's mean score separates successful from failed games (AUC)

Usage: python scripts/mt_metrics.py export --runs main_image main_mcq
       python scripts/mt_metrics.py comet
       python scripts/mt_metrics.py gemba [--per-cell 20] [--loop]
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
TRANSLATED = {"sol", "flash-lite", "nllb", "google", "qwen", "nllb-id", "google-id"}
MODELS = ["cometkiwi", "xcomet"]  # xcomet only with comet_score.py --low-mem on this machine


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


GEMBA_DIR = Path("/mnt/d/dev/mt-eval/interpreter-agent-eval/research/mt_compare")
JUDGE = "gemini-3.8-flash"


def _mt_eval_gemba():
    import importlib.util
    import sys

    sys.path.insert(0, str(GEMBA_DIR))
    spec = importlib.util.spec_from_file_location("compute_gemba", GEMBA_DIR / "compute_gemba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gemba(per_cell, loop):
    import random
    import time

    from mtcomm.backends import make_backend
    from mtcomm.languages import get as lang

    g = _mt_eval_gemba()
    segs = {json.loads(line)["record_id"]: json.loads(line) for line in open(OUT / "segments.jsonl")}
    cells = defaultdict(set)
    for line in open(OUT / "messages.jsonl"):
        m = json.loads(line)
        cells[(m["task"], m["cond"])].add(m["seg"])
    rng = random.Random(0)
    sample = sorted({sid for ids in cells.values() for sid in rng.sample(sorted(ids), min(per_cell, len(ids)))})
    backend = make_backend(JUDGE)
    state_p, out_p = OUT / "gemba_state.json", OUT / "gemba_scores.jsonl"
    done = {json.loads(line)["record_id"] for line in open(out_p)} if out_p.exists() else set()
    while True:
        state = json.load(open(state_p)) if state_p.exists() else {"job": None, "ids": [], "rounds": 0}
        if state["job"]:
            st = backend.status(state["job"])
            if st == "running":
                print(f"GEMBA batch {state['job']} running", flush=True)
                if not loop:
                    return
                time.sleep(120)
                continue
            res = backend.results(state["job"]) if st == "succeeded" else {}
            with open(out_p, "a") as f:
                for sid in state["ids"]:
                    r = res.get(sid, {})
                    if "text" not in r:
                        continue
                    errors = g.parse_errors(r["text"])
                    raw, extra = g.score_from_errors(errors)
                    f.write(json.dumps({"record_id": sid, "gemba_score": raw, **extra, "errors": errors,
                                        "judge": JUDGE, "usage": r.get("usage")}, ensure_ascii=False) + "\n")
                    done.add(sid)
            state = {"job": None, "ids": [], "rounds": state.get("rounds", 0)}
            json.dump(state, open(state_p, "w"))
        todo = [sid for sid in sample if sid not in done]
        if not todo or state.get("rounds", 0) >= 2:  # unparseable after two batches: leave unscored
            print(f"GEMBA: {len(done & set(sample))}/{len(sample)} sampled segments scored -> {out_p}")
            return
        reqs = []
        for sid in todo:
            sg = segs[sid]
            prompt = g.PROMPT_TEMPLATE.format(src_lang=lang(sg["src_lang"]).name, tgt_lang=lang(sg["tgt_lang"]).name,
                                              source=sg["source_text"], hypothesis=sg["post_edit"])
            reqs.append((sid, backend.build(prompt, g.SYSTEM_PROMPT, [])))
        state = {"job": backend.submit(reqs, OUT), "ids": todo, "rounds": state.get("rounds", 0) + 1}
        json.dump(state, open(state_p, "w"))
        print(f"GEMBA: submitted {len(todo)} segments -> {state['job']}", flush=True)
        if not loop:
            return


def _auc(pos, neg):
    """P(score of a successful game > score of a failed one); ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def _spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def report():
    scores = {}
    for line in open(OUT / "comet_scores.jsonl"):
        r = json.loads(line)
        scores[r["record_id"]] = {m: r.get(f"{m}_score") for m in MODELS}
    for xp in sorted(OUT.glob("xcomet_v*.jsonl")):  # XCOMET-XL: v1 on taco CPU (--gpus 0), v2 here (--low-mem)
        for line in open(xp):
            r = json.loads(line)
            scores.setdefault(r["record_id"], {})["xcomet"] = r.get("xcomet_score")
    gp = OUT / "gemba_scores.jsonl"
    if gp.exists():
        for line in open(gp):
            r = json.loads(line)
            scores.setdefault(r["record_id"], {})["gemba"] = r["gemba_norm"]
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
        for metric in MODELS + ["gemba"]:
            ep = [(sum(v[metric]) / len(v[metric]), outcome[k]) for k, v in episodes.items()
                  if k[0] == task and v[metric]]
            auc[f"{task}/{metric}"] = _auc([s for s, ok in ep if ok], [s for s, ok in ep if not ok])
    # cell level: does a cell's mean metric rank cells like their task success does?
    cell_ok = defaultdict(list)
    for (task, ep), ok in outcome.items():
        cell_ok[(task,) + tuple(ep.split("|")[1:3][::-1])].append(ok)  # episode id: task|cond|pair|item
    spear = {}
    for task in sorted({t for t, _ in episodes}):
        for metric in MODELS + ["gemba"]:
            xs, ys = [], []
            for (t, p, c), v in cells.items():
                if t == task and v.get(metric) and cell_ok.get((t, p, c)):
                    xs.append(sum(v[metric]) / len(v[metric]))
                    ys.append(sum(cell_ok[(t, p, c)]) / len(cell_ok[(t, p, c)]))
            spear[f"{task}/{metric}"] = (_spearman(xs, ys), len(xs)) if len(xs) > 2 else None
    json.dump({"cells": rows, "auc_success_vs_episode_mean": auc, "cell_spearman_vs_success": spear},
              open(OUT / "summary.json", "w"), indent=1)
    print("cell Spearman (metric mean vs success rate, n cells):", spear)
    for r in rows:
        print(r)
    print("AUC (episode mean metric separates success from failure; 0.5 = no signal):", auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "comet", "gemba", "report"])
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--runs", nargs="*", default=["main_image", "main_mcq"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--models", nargs="*", default=["cometkiwi"])
    ap.add_argument("--out", default="metrics", help="output dir under outputs/mtcomm (e.g. metrics2 for the main2 runs)")
    args = ap.parse_args()
    global OUT
    OUT = OUTPUT_ROOT / args.out
    if args.cmd == "export":
        export(args.runs)
    elif args.cmd == "comet":
        comet(args.batch_size, args.models)
    elif args.cmd == "gemba":
        gemba(args.per_cell, args.loop)
    else:
        report()


if __name__ == "__main__":
    main()
