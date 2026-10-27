"""Screen Belebele items with the no-talk prior, so the MCQ game needs the conversation.

The questioner sees the question and options but never the passage. Items it answers correctly
alone (Gemini often guesses Belebele from the options) say nothing about communication, so the
main runs keep items the prior gets wrong in every questioner language (su, jv, ko).

Usage: python scripts/mcq_prior_screen.py [--pool 300] [--keep 30] [--loop]
  Re-run until it prints the selection (submit, poll, ingest are resumable).
"""

import argparse
import json
import time

from mtcomm.backends import make_backend
from mtcomm.data_mugeval import load_mcq_items
from mtcomm.games.mugeval import MCQConversationGame
from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import BatchGenerate, Pending, ReplayCache

PLAYER = "gemini-3.8-flash"
LANGS = ["su", "jv", "ko"]
RUN_DIR = OUTPUT_ROOT / "mcq_screen"
OUT = DATA_ROOT / "derived" / "mcq" / "mcq_screen_v1.json"


def prior_answers(pool, backend, cache, pending):
    """{item_id: {lang: correct or None (pending)}} from one replay over the pool."""
    out = {}
    for la in LANGS:
        for it in load_mcq_items(la, la, n=pool):
            gm = MCQConversationGame(it, la, la, allow_message=False)
            player = LLMPlayer(BatchGenerate(backend, cache, f"screen|{la}|{it['item_id']}", "questioner", pending),
                               PLAYER)
            try:
                gm.apply("questioner", player.act(gm.observe("questioner")))
                out.setdefault(it["item_id"], {})[la] = gm.score()["correct"]
            except Pending:
                out.setdefault(it["item_id"], {})[la] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backend = make_backend(PLAYER)
    cache = ReplayCache(RUN_DIR / "cache.jsonl")
    state_p = RUN_DIR / "state.json"
    while True:
        state = json.load(open(state_p)) if state_p.exists() else {"job": None, "sent": {}}
        if state["job"]:
            st = backend.status(state["job"])
            if st == "running":
                print(f"batch {state['job']} running", flush=True)
                if not args.loop:
                    return
                time.sleep(120)
                continue
            res = backend.results(state["job"]) if st == "succeeded" else {}
            for key, info in state["sent"].items():
                r = res.get(key, {"error": f"batch {st}"})
                rec = {"key": key, "request_hash": info["request_hash"], "model": PLAYER, "mode": "batch",
                       "usage": r.get("usage")}
                rec.update({"text": r["text"]} if "text" in r else {"error": r["error"], "attempts": info["attempts"] + 1})
                cache.put(rec)
            state = {"job": None, "sent": {}}
            json.dump(state, open(state_p, "w"))
        pending = {}
        answers = prior_answers(args.pool, backend, cache, pending)
        if pending:
            state["job"] = backend.submit([(k, v["request"]) for k, v in pending.items()], RUN_DIR)
            state["sent"] = {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]} for k, v in pending.items()}
            json.dump(state, open(state_p, "w"))
            print(f"submitted {len(pending)} prior calls -> {state['job']}", flush=True)
            if not args.loop:
                return
            continue
        break
    acc = {la: sum(a[la] for a in answers.values()) / len(answers) for la in LANGS}
    # Pool order is the seeded Belebele sample order; prefer items wrong in all three languages.
    ranked = sorted(answers, key=lambda i: sum(answers[i][la] for la in LANGS))
    keep = [i for i in ranked if sum(answers[i][la] for la in LANGS) == 0][:args.keep]
    if len(keep) < args.keep:  # top up with items wrong in two of the three languages
        keep += [i for i in ranked if sum(answers[i][la] for la in LANGS) == 1][:args.keep - len(keep)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": 1, "player": PLAYER, "pool": args.pool, "prior_accuracy_on_pool": acc,
               "n_wrong_all_langs": sum(1 for a in answers.values() if not any(a.values())),
               "item_ids": keep, "prior": {i: answers[i] for i in keep}}, open(OUT, "w"), indent=1)
    print(f"prior accuracy on the pool: {acc}; kept {len(keep)} -> {OUT}")


if __name__ == "__main__":
    main()
