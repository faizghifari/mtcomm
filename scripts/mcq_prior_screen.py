"""Screen Belebele items with the no-talk prior, so the MCQ game needs the conversation.

The questioner sees the question and options but never the passage. Items it answers correctly
alone (Gemini often guesses Belebele from the options) say nothing about communication, so the
main runs keep items the prior gets wrong in every questioner language (su, jv, ko).

v2 adds a passage check: v1's items gave a same-language ceiling of only 40-50%, because picking
what the prior gets wrong also picked items that are tricky even with the passage (negations,
ambiguous options). v2 keeps items the prior gets wrong AND a reader given the passage gets right,
in every language, so the passage really fills the gap.

Usage: python scripts/mcq_prior_screen.py [--pool 600] [--keep 30] [--loop]
  Re-run until it prints the selection (submit, poll, ingest are resumable).
"""

import argparse
import json
import re
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
OUT = DATA_ROOT / "derived" / "mcq" / "mcq_screen_v2.json"
DIGIT = re.compile(r"\[\[\s*([1-4])\s*\]\]|\b([1-4])\b")


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


def passage_answers(item_ids, pool, backend, cache, pending):
    """{item_id: {lang: correct or None}}: a reader answers with the passage in front of it."""
    out = {}
    for la in LANGS:
        for it in load_mcq_items(la, la, n=pool):
            if it["item_id"] not in item_ids:
                continue
            opts = "\n".join(f"({i}) {o}" for i, o in enumerate(it["options_q"], 1))
            prompt = (f"Read the passage and answer the multiple-choice question.\n\nPassage:\n{it['passage_a']}\n\n"
                      f"Question: {it['question_q']}\n{opts}\n\nAnswer with only the number in double brackets, "
                      f"e.g. [[2]].")
            gen = BatchGenerate(backend, cache, f"passage|{la}|{it['item_id']}", "reader", pending)
            try:
                m = DIGIT.search(gen(prompt, None, []))
                out.setdefault(it["item_id"], {})[la] = bool(m) and int(m.group(1) or m.group(2)) == it["gold"]
            except Pending:
                out.setdefault(it["item_id"], {})[la] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=600)
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
        wrong = {i for i, a in answers.items() if all(v is False for v in a.values())}
        readable = passage_answers(wrong, args.pool, backend, cache, pending) if not pending else {}
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
    ok = [i for i in answers if i in wrong and all(readable.get(i, {}).get(la) for la in LANGS)]
    keep = ok[:args.keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": 2, "player": PLAYER, "pool": args.pool, "prior_accuracy_on_pool": acc,
               "n_wrong_all_langs": len(wrong), "n_wrong_and_passage_right": len(ok),
               "passage_accuracy_on_wrong": {la: sum(bool(readable[i][la]) for i in wrong) / max(1, len(wrong))
                                             for la in LANGS},
               "item_ids": keep, "prior": {i: answers[i] for i in keep}}, open(OUT, "w"), indent=1)
    print(f"prior accuracy {acc}; wrong in all languages {len(wrong)}; of those right with the passage in all "
          f"languages {len(ok)}; kept {len(keep)} -> {OUT}")


if __name__ == "__main__":
    main()
