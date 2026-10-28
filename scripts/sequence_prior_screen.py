"""Screen Bloom-VIST stories with the no-talk prior, so the sequence game needs the narration.

The arranger sees the shuffled pictures and nobody tells the story. A story whose order can be
read off the pictures (a fable with recurring characters) tests the model's story sense, not
communication. Each story gets `--samples` prior attempts (the request is language-independent);
the kept stories are never put exactly in order and have the lowest mean Kendall's tau.

Usage: python scripts/sequence_prior_screen.py [--pool manifest_seq_pool120.json] [--keep 30] [--loop]
  Writes data/derived/bloom_vist/manifest_seq_v2.json (the selected stories, in rank order).
"""

import argparse
import json
import time

from mtcomm.backends import make_backend
from mtcomm.data import load_sequence_items
from mtcomm.games.sequence import SequenceGame
from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import BatchGenerate, Pending, ReplayCache

PLAYER = "gemini-3.8-flash"
RUN_DIR = OUTPUT_ROOT / "sequence_screen"
SEQ_DIR = DATA_ROOT / "derived" / "bloom_vist"


def prior_scores(items, samples, backend, cache, pending):
    out = {}
    for it in items:
        for k in range(samples):
            gm = SequenceGame(it, "en", "en", allow_message=False)
            gen = BatchGenerate(backend, cache, f"seqscreen|{it['story_id']}|s{k}", "arranger", pending)
            try:
                gm.apply("arranger", LLMPlayer(gen, PLAYER).act(gm.observe("arranger")))
                out.setdefault(it["story_id"], []).append(gm.score())
            except Pending:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="manifest_seq_pool120.json")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    backend = make_backend(PLAYER)
    cache = ReplayCache(RUN_DIR / "cache.jsonl")
    items = load_sequence_items(manifest_path=SEQ_DIR / args.pool)
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
        scores = prior_scores(items, args.samples, backend, cache, pending)
        if pending:
            state["job"] = backend.submit([(k, v["request"]) for k, v in pending.items()], RUN_DIR)
            state["sent"] = {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]} for k, v in pending.items()}
            json.dump(state, open(state_p, "w"))
            print(f"submitted {len(pending)} prior calls -> {state['job']}", flush=True)
            if not args.loop:
                return
            continue
        break

    def mean_tau(sid):
        taus = [s["kendall_tau"] if s["kendall_tau"] is not None else 0.0 for s in scores[sid]]
        return sum(taus) / len(taus)

    exact = {sid for sid, ss in scores.items() if any(s["correct"] for s in ss)}
    ranked = sorted((sid for sid in scores if sid not in exact), key=mean_tau)
    keep = ranked[:args.keep]
    pool = json.load(open(SEQ_DIR / args.pool, encoding="utf-8"))
    by_id = {s["story_id"]: s for s in pool["stories"]}
    out = dict(pool, version=2, screen={"player": PLAYER, "samples": args.samples, "pool": args.pool,
                                        "pool_size": len(scores), "exact_by_prior": len(exact),
                                        "mean_tau_pool": round(sum(map(mean_tau, scores)) / len(scores), 3),
                                        "mean_tau_kept": round(sum(map(mean_tau, keep)) / max(1, len(keep)), 3)},
               stories=[dict(by_id[sid], prior_mean_tau=round(mean_tau(sid), 3)) for sid in keep])
    json.dump(out, open(SEQ_DIR / "manifest_seq_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"pool {len(scores)}: {len(exact)} put exactly in order by the prior at least once; mean tau "
          f"{out['screen']['mean_tau_pool']} (kept: {out['screen']['mean_tau_kept']}); kept {len(keep)}")


if __name__ == "__main__":
    main()
