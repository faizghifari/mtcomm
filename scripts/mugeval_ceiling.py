"""Can a model play MuG-Eval's MCQ Conversation at all, in its own language and without translation?

The same model is both questioner and answerer, everything in one language L, with MuG-Eval's
released prompts verbatim (MCQConversationGame(original=True): the answerer reads the passage in L
and replies "Yes."/"No."/"Maybe." in English, the questioner asks in L). This is the task ceiling a
player must clear before its passthrough or translated results mean anything; the no-talk prior
(the questioner answers alone) is the floor. Items are a random Belebele sample, not the set
screened for Gemini in the main runs.

Usage: python scripts/mugeval_ceiling.py run --model exaone-3.5-7.8b-instruct --lang ko [--n 50]
       python scripts/mugeval_ceiling.py report
"""

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from mtcomm import lid
from mtcomm.backends import LocalBackend
from mtcomm.channels import IdentityChannel
from mtcomm.data_mugeval import load_mcq_items
from mtcomm.game import run_episode
from mtcomm.games.mugeval import MCQConversationGame
from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import ReplayCache, SyncGenerate

OUT = OUTPUT_ROOT / "mugeval_ceiling"
LMS = "/mnt/c/Users/user/.lmstudio/bin/lms.exe"


def run(args):
    d = OUT / args.model
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([LMS, "unload", "--all"], capture_output=True)
    subprocess.run([LMS, "load", args.model, "--gpu", "max", "-c", "8192", "--parallel", str(args.workers), "-y"],
                   check=True, capture_output=True)
    be = LocalBackend(f"local:{args.model}", workers=args.workers)
    cache = ReplayCache(d / "cache.jsonl")
    eps = [(L, cond, it) for L in args.lang for it in load_mcq_items(L, L, n=args.n, seed=args.seed)
           for cond in ("ceiling", "prior")]

    def one(ep):
        L, cond, it = ep
        eid = f"mcq|{cond}|{L}|{it['item_id']}"
        gm = MCQConversationGame(it, L, L, allow_message=cond == "ceiling", original=True)
        players = {r: LLMPlayer(SyncGenerate(be, cache, eid, r), be.model) for r in ("questioner", "answerer")}
        try:
            rec = run_episode(gm, players, IdentityChannel(), lid=lid.check, max_moves=50)
            rec.update(status="done")
        except Exception as e:
            rec = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        rec.update(episode=eid, model=args.model, lang=L, cond=cond)
        return rec

    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, eps))
    with open(d / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"{args.model}: {len(recs)} episodes, {sum(r['status'] != 'done' for r in recs)} errors")


def report(args):
    summary = {}
    for p in sorted(OUT.glob("*/records.jsonl")):
        cells = defaultdict(list)
        for r in map(json.loads, open(p)):
            cells[(r["lang"], r["cond"])].append(r)
        for (L, cond), rs in sorted(cells.items()):
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
            summary[f"{p.parent.name}|{L}|{cond}"] = row
            print(f"{p.parent.name:38s} {L} {cond:8s} success={row['success']} valid={row['valid_format']} "
                  f"q={row['questions']} answers={row['answers']} other={row['other_answers']} lid={row['questioner_lid']}")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--lang", nargs="+", required=True, choices=["su", "jv", "ko", "id", "en"])
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--workers", type=int, default=4)
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
