"""Screen BLEnD local-visitor items so the culture game needs the local (docs/poc_diagnostics.md, section 2).

Keeps an item when, for the given player and in the visitor's language:
  - the no-talk prior is wrong in every sample (--samples, default 3), so the model cannot answer from what it knows;
  - an oracle reader told only the local's most common answer picks the gold option, so the local's knowledge is enough.
Pool: the 33-40 items of each (local culture, visitor language) pair in CULTURE_PAIRS, so the kept set is small.

Usage: python scripts/culture_screen.py --player gemini-3.5-flash-lite [--samples 3] [--loop]
  Re-run until it prints the selection (submit, poll, ingest are resumable).
  Writes data/derived/blend/culture_screen_<player>.json; pass it to run_main.py with --culture-screen.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from mtcomm.backends import make_backend
from mtcomm.data import load_blend_items
from mtcomm.games.blend_lv import BlendLocalVisitorGame
from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import BatchGenerate, Pending, ReplayCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_main import CULTURE_PAIRS  # noqa: E402

LETTER = re.compile(r"\b([A-Z])\b")


def pool():
    items = load_blend_items()
    return [it for it in items if (it["local_culture"], it["visitor_lang"]) in CULTURE_PAIRS
            and len({o.strip().lower() for o in it["options"].values()}) == len(it["options"])]


def prior_answers(items, samples, backend, cache, pending):
    out = {}
    for it in items:
        for s in range(samples):
            gm = BlendLocalVisitorGame(it, it["visitor_lang"], it["visitor_lang"], max_questions=0, prompt_style="neutral")
            player = LLMPlayer(BatchGenerate(backend, cache, f"prior|{s}|{it['item_id']}", "visitor", pending), backend.model)
            try:
                gm.apply("visitor", player.act(gm.observe("visitor")))
                out.setdefault(it["item_id"], []).append(bool(gm.score()["correct"]))
            except Pending:
                out.setdefault(it["item_id"], []).append(None)
    return out


def oracle_answers(items, backend, cache, pending):
    """The local's top answer is in the local language, as the game gives it; the options are in the visitor's."""
    out = {}
    for it in items:
        top = ", ".join(next(k for k in it["local_knowledge"] if k["target"])["answers"][0]["answers"])
        opts = "\n".join(f"{a}. {t}" for a, t in sorted(it["options"].items()))
        prompt = (f"People in {it['local_place']} were asked about this topic and most commonly answered: {top}\n\n"
                  f"Question: {it['question']}\n{opts}\n\nWhich option matches their answer? Reply with only the letter.")
        gen = BatchGenerate(backend, cache, f"oracle|{it['item_id']}", "reader", pending)
        try:
            m = LETTER.search(gen(prompt, None, []).upper())
            out[it["item_id"]] = bool(m) and m.group(1) == it["answer_idx"]
        except Pending:
            out[it["item_id"]] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    tag = args.player.replace("gemini-", "").replace(".", "")
    run_dir = OUTPUT_ROOT / f"culture_screen_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = DATA_ROOT / "derived" / "blend" / f"culture_screen_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args.player)
    cache = ReplayCache(run_dir / "cache.jsonl")
    state_p = run_dir / "state.json"
    items = pool()
    while True:
        state = json.load(open(state_p)) if state_p.exists() else {"job": None, "sent": {}}
        if state["job"]:
            st = backend.status(state["job"])
            if st == "running":
                print(f"batch {state['job']} running", flush=True)
                if not args.loop:
                    return
                time.sleep(60)
                continue
            res = backend.results(state["job"]) if st == "succeeded" else {}
            for key, info in state["sent"].items():
                r = res.get(key, {"error": f"batch {st}"})
                rec = {"key": key, "request_hash": info["request_hash"], "model": args.player, "mode": "batch", "usage": r.get("usage")}
                rec.update({"text": r["text"]} if "text" in r else {"error": r["error"], "attempts": info["attempts"] + 1})
                cache.put(rec)
            state = {"job": None, "sent": {}}
            json.dump(state, open(state_p, "w"))
        pending = {}
        prior = prior_answers(items, args.samples, backend, cache, pending)
        oracle = oracle_answers(items, backend, cache, pending)
        if not pending:
            break
        state["job"] = backend.submit([(k, v["request"]) for k, v in pending.items()], run_dir)
        state["sent"] = {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]} for k, v in pending.items()}
        json.dump(state, open(state_p, "w"))
        print(f"submitted {len(pending)} calls -> {state['job']}", flush=True)
        if not args.loop:
            return
    keep = [it for it in items if not any(prior[it["item_id"]]) and oracle[it["item_id"]]]
    pair = lambda it: f"{it['local_culture']}>{it['visitor_lang']}"  # noqa: E731
    summary = {"player": args.player, "samples": args.samples, "pool": dict(Counter(map(pair, items))), "kept": dict(Counter(map(pair, keep))),
               "prior_accuracy_on_pool": round(sum(sum(v) for v in prior.values()) / (args.samples * len(items)), 3),
               "oracle_accuracy_on_pool": round(sum(oracle.values()) / len(items), 3), "item_ids": [it["item_id"] for it in keep]}
    json.dump(summary, open(out_path, "w"), indent=1)
    print({k: v for k, v in summary.items() if k != "item_ids"}, "->", out_path)


if __name__ == "__main__":
    main()
