"""Can a model play MuG-Eval's tasks (MCQ Conversation, Twenty Questions) at all, in its own language and
without translation?

The same model is both questioner and answerer, everything in one language L, with MuG-Eval's
released prompts verbatim (MCQConversationGame(original=True): the answerer reads the passage in L
and replies "Yes."/"No."/"Maybe." in English, the questioner asks in L). This is the task ceiling a
player must clear before its passthrough or translated results mean anything; the no-talk prior
(the questioner answers alone) is the floor. MCQ items are a random Belebele sample, not the set
screened for Gemini in the main runs. Twenty Questions uses the 100-candidate easy setting, with the
entity given to the answerer in L, as in MuG-Eval (su/jv words: native check pending).

API models (gemini-*, gpt-*) run the same screen as an anchor: the opening calls and the single-call
priors go through the batch API first (half price), then the dependent turns run synchronously.

Usage: python scripts/mugeval_ceiling.py run --model exaone-3.5-7.8b-instruct --lang ko [--task twentyq] [--n 50]
       [--ctx 16384 --retry-errors]  (re-asks calls that failed, e.g. on context overflow; answered calls replay)
       python scripts/mugeval_ceiling.py report
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mtcomm import lid
from mtcomm.backends import LocalBackend, make_backend
from mtcomm.channels import IdentityChannel
from mtcomm.data_mugeval import load_mcq_items, load_twentyq_items
from mtcomm.game import run_episode
from mtcomm.games.mugeval import MCQConversationGame, TwentyQuestionsGame
from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import BatchGenerate, CallFailed, Pending, ReplayCache, SyncGenerate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_main import PRICES, _tokens  # noqa: E402

OUT = OUTPUT_ROOT / "mugeval_ceiling"
LMS = "/mnt/c/Users/user/.lmstudio/bin/lms.exe"
TASKS = {"mcq": (MCQConversationGame, load_mcq_items), "twentyq": (TwentyQuestionsGame, load_twentyq_items)}
API = ("gemini", "gpt")


def records_path(d, task):
    return d / ("records.jsonl" if task == "mcq" else f"records_{task}.jsonl")


def batch_openings(be, eps, play, cache, workdir):
    """Opening calls and single-call priors through the batch API; later turns stay synchronous."""
    pending = {}
    for ep in eps:
        try:
            play(ep, lambda eid, role: BatchGenerate(be, cache, eid, role, pending))
        except (Pending, CallFailed):
            pass
    if not pending:
        return
    job = be.submit([(k, v["request"]) for k, v in pending.items()], workdir)
    print(f"submitted {len(pending)} opening calls -> {job}", flush=True)
    while (st := be.status(job)) == "running":
        time.sleep(60)
    res = be.results(job) if st == "succeeded" else {}
    for k, v in pending.items():
        r = res.get(k, {"error": f"batch {st}"})
        rec = {"key": k, "request_hash": v["request_hash"], "model": be.model, "mode": "batch", "usage": r.get("usage")}
        if "text" in r:
            rec["text"] = r["text"]
        else:
            rec.update(error=r["error"], attempts=v["attempts"] + 1)
        cache.put(rec)


def spend(cache):
    """USD from the cache's usage records (batch calls at half price); local models are free."""
    total = 0.0
    for r in cache.records.values():
        pi, po = PRICES.get(r.get("model"), (0, 0))
        i, o = _tokens(r.get("usage"))
        total += (0.5 if r.get("mode") == "batch" else 1.0) * (i * pi + o * po) / 1e6
    return total


def run(args):
    d = OUT / args.model.replace(":", "_")  # taco:<model>; no colons on the Windows drive
    d.mkdir(parents=True, exist_ok=True)
    api = args.model.startswith(API)
    if api or args.model.startswith("taco:"):
        be = make_backend(args.model, **({"workers": args.workers} if not api else {}))
    else:
        subprocess.run([LMS, "unload", "--all"], capture_output=True)
        subprocess.run([LMS, "load", args.model, "--gpu", "max", "-c", str(args.ctx), "--parallel", str(args.workers),
                        "-y"], check=True, capture_output=True)
        be = LocalBackend(f"local:{args.model}", workers=args.workers)
    cache = ReplayCache(d / "cache.jsonl")
    if args.retry_errors:
        cache.records = {k: r for k, r in cache.records.items() if "text" in r}
    Game, load = TASKS[args.task]
    eps = [(L, cond, it) for L in args.lang for it in load(L, L, n=args.n, seed=args.seed)
           for cond in ("ceiling", "prior")]

    def play(ep, make_gen):
        L, cond, it = ep
        eid = f"{args.task}|{cond}|{L}|{it['item_id']}"
        gm = Game(it, L, L, allow_message=cond == "ceiling", original=True)
        players = {r: LLMPlayer(make_gen(eid, r), be.model) for r in ("questioner", "answerer")}
        return run_episode(gm, players, IdentityChannel(), lid=lid.check, max_moves=50)

    if api:
        (d / f"batch_{args.task}").mkdir(exist_ok=True)
        batch_openings(be, eps, play, cache, d / f"batch_{args.task}")

    def one(ep):
        L, cond, it = ep
        try:
            rec = play(ep, lambda eid, role: SyncGenerate(be, cache, eid, role))
            rec.update(status="done")
        except Exception as e:
            rec = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        rec.update(episode=f"{args.task}|{cond}|{L}|{it['item_id']}", model=args.model, task=args.task, lang=L,
                   cond=cond)
        return rec

    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, eps))
    with open(records_path(d, args.task), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"{args.model} {args.task}: {len(recs)} episodes, {sum(r['status'] != 'done' for r in recs)} errors, "
          f"spend so far ${spend(cache):.2f}")


def report(args):
    summary = {}
    for p in sorted(OUT.glob("*/records*.jsonl")):
        cells = defaultdict(list)
        for r in map(json.loads, open(p)):
            cells[(r.get("task", "mcq"), r["lang"], r["cond"])].append(r)
        for (task, L, cond), rs in sorted(cells.items()):
            done = [r for r in rs if r["status"] == "done"]
            ans = Counter()
            q_lid = []
            for r in done:
                for t in r["turns"]:
                    if t["role"] == "answerer":
                        ans[t["source"].strip().strip(".").lower()[:10]] += 1
                    elif t["kind"] == "message" and t.get("lid") and len(t["source"].split()) >= 3:
                        q_lid.append(t["lid"]["ok"])
            n_ans = sum(ans.values())
            row = {"n": len(rs), "errors": len(rs) - len(done),
                   "success": round(sum(r["score"]["correct"] for r in done) / len(done), 3) if done else None,
                   "valid_format": round(sum(r["score"]["valid_format"] for r in done) / len(done), 3) if done else None,
                   "questions": round(sum(r["score"]["questions_asked"] for r in done) / len(done), 2) if done else None,
                   "answers": {k: round(ans[k] / n_ans, 2) for k in ("yes", "no", "maybe")} if n_ans else None,
                   "other_answers": round(1 - sum(ans[k] for k in ("yes", "no", "maybe")) / n_ans, 2) if n_ans else None,
                   "questioner_lid": round(sum(q_lid) / len(q_lid), 3) if q_lid else None}
            summary[f"{p.parent.name}|{task}|{L}|{cond}"] = row
            print(f"{p.parent.name:38s} {task:7s} {L} {cond:8s} success={row['success']} valid={row['valid_format']} "
                  f"q={row['questions']} answers={row['answers']} other={row['other_answers']} lid={row['questioner_lid']}")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--lang", nargs="+", required=True, choices=["su", "jv", "ko", "id", "en"])
    r.add_argument("--task", default="mcq", choices=sorted(TASKS))
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--ctx", type=int, default=8192)
    r.add_argument("--retry-errors", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
