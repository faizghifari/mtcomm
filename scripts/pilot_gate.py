"""Language-gate pilot: can each player model play the tasks in each language, with no MT?

Blocks (identity channel, 20 items each):
  A  image game, LLM describer + LLM guesser, same language (en, ko, id, su, jv)
  B  image game, native XM3600 caption + LLM guesser (en, ko, id)
  C  blind guess, no message (once per model)
  D  BLEnD local-visitor ceiling: local and visitor share the visitor's language (su, jv, id, ko),
     at most 3 questions; plus the no-talk prior (max_questions=0). Renderings are the
     unreviewed Haiku drafts.

Runs in batch waves (see mtcomm.replay). Each invocation of `step` ingests finished batches,
replays every episode, and submits the next wave; `--loop` repeats until done.

Usage:
  python scripts/pilot_gate.py plan              # count wave-1 requests per model, no calls
  python scripts/pilot_gate.py step [--loop]     # submit / poll / ingest
  python scripts/pilot_gate.py report
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from mtcomm import lid
from mtcomm.backends import make_backend
from mtcomm.channels import IdentityChannel
from mtcomm.data import load_blend_items, load_image_items
from mtcomm.game import Move, run_episode
from mtcomm.games.blend_lv import BlendLocalVisitorGame
from mtcomm.games.image_ref import ImageRefGame
from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import LLMPlayer, ScriptedPlayer
from mtcomm.replay import BatchGenerate, CachedChannel, CallFailed, Pending, ReplayCache

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
          "gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol"]
N_ITEMS = 20
MAX_WAVES = 10
ACTIVE = set(MODELS)  # models whose episodes are replayed and advanced
RUN_DIR = OUTPUT_ROOT / "pilot_gate"  # overridden by --run
OPTS = {"blend_items": None, "manifest": None, "max_words": None, "blocks": None,
        "prompt_style": "explicit", "min_questions": 0}
# BLEnD ceiling: visitor language -> local culture whose items it uses
BLEND_PAIRS = {"su": "South_Korea", "jv": "West_Java", "id": "South_Korea", "ko": "West_Java"}


def episodes():
    """Yield (episode_id, block, lang, build) where build(gen) -> (game, players)."""
    images = load_image_items("hard", manifest_path=OPTS["manifest"])[:N_ITEMS]
    for lang in ["en", "ko", "id", "su", "jv"]:
        for it in images:
            yield (f"A|{lang}|{it['item_id']}", "A", lang,
                   lambda gen, it=it, lang=lang: (ImageRefGame(it, lang, lang, max_words=OPTS["max_words"]),
                                                  {"describer": LLMPlayer(gen("describer"), ""),
                                                   "guesser": LLMPlayer(gen("guesser"), "")}))
    for lang in ["en", "ko", "id"]:
        for it in images:
            yield (f"B|{lang}|{it['item_id']}", "B", lang,
                   lambda gen, it=it, lang=lang: (ImageRefGame(it, lang, lang, max_words=OPTS["max_words"]),
                                                  {"describer": ScriptedPlayer([Move("message", it["captions"][lang])]),
                                                   "guesser": LLMPlayer(gen("guesser"), "")}))
    for it in images:
        yield (f"C|en|{it['item_id']}", "C", "en",
               lambda gen, it=it: (ImageRefGame(it, "en", "en", allow_message=False, max_words=OPTS["max_words"]),
                                   {"describer": None, "guesser": LLMPlayer(gen("guesser"), "")}))
    blend = load_blend_items(items_path=OPTS["blend_items"])
    for vlang, local in BLEND_PAIRS.items():
        # Items whose rendered options collide (e.g. football/soccer -> one word) are unanswerable.
        items = [b for b in blend if b["visitor_lang"] == vlang and b["local_culture"] == local
                 and len({v.strip().lower() for v in b["options"].values()}) == len(b["options"])][:N_ITEMS]
        for it in items:
            for block, maxq in [("D", 3), ("Dprior", 0)]:
                yield (f"{block}|{vlang}|{it['item_id']}", block, vlang,
                       lambda gen, it=it, vlang=vlang, maxq=maxq: (
                           BlendLocalVisitorGame(it, vlang, vlang, max_questions=maxq,
                                                 min_questions=OPTS["min_questions"],
                                                 prompt_style=OPTS["prompt_style"]),
                           {"visitor": LLMPlayer(gen("visitor"), ""), "local": LLMPlayer(gen("local"), "")}))


def replay(backends, cache):
    """Replay all episodes once. Returns finished records and pending requests per model."""
    pending = {m: {} for m in backends}
    records, n_pending, n_failed = [], 0, 0
    eps = [e for e in episodes() if not OPTS["blocks"] or e[1] in OPTS["blocks"]]
    for model, be in backends.items():
        for eid, block, lang, build in eps:
            gen = lambda role, eid=eid: BatchGenerate(be, cache, eid, role, pending[model])  # noqa: E731
            gm, players = build(gen)
            channel = CachedChannel(IdentityChannel(), cache, f"{model}|{eid}")
            try:
                rec = run_episode(gm, players, channel, lid=lid.check)
                rec.update(model=model, episode=eid, block=block, lang=lang, status="done")
                records.append(rec)
            except Pending:
                n_pending += 1
            except CallFailed as e:
                n_failed += 1
                records.append({"model": model, "episode": eid, "block": block, "lang": lang,
                                "status": "error", "error": str(e)[:300]})
    return records, pending, n_pending, n_failed


def load_state():
    p = RUN_DIR / "state.json"
    return json.load(open(p)) if p.exists() else {"wave": {}, "jobs": {}, "sent": {}, "history": []}


def save_state(state):
    with open(RUN_DIR / "state.json", "w") as f:
        json.dump(state, f, indent=1)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def step(backends, cache):
    """One step: ingest finished batches, replay, submit the next wave for idle active models.
    Models advance independently, so a slow batch for one model never blocks another.
    Returns True when nothing is running and nothing is left to send."""
    state = load_state()
    if isinstance(state["wave"], int):  # older state format: one wave counter for all models
        state["wave"] = {m: state["wave"] for m in state["jobs"]}
    for model, job in list(state["jobs"].items()):
        st = backends[model].status(job)
        if st == "running":
            continue
        if st == "succeeded":
            res = backends[model].results(job)
            sent = state["sent"][model]
            for key, info in sent.items():
                r = res.get(key, {"error": "missing from batch output"})
                rec = {"key": key, "request_hash": info["request_hash"], "model": model,
                       "wave": state["wave"][model], "usage": r.get("usage")}
                if "text" in r:
                    rec["text"] = r["text"]
                else:
                    rec.update(error=r["error"], attempts=info["attempts"] + 1)
                cache.put(rec)
            log(f"{model} wave {state['wave'][model]}: ingested {len(sent)} results "
                f"({sum('text' not in res.get(k, {}) for k in sent)} errors)")
        else:
            log(f"{model}: batch FAILED ({job}); its requests will be resubmitted")
        state["history"].append({"model": model, "wave": state["wave"][model], "job": job, "status": st,
                                 "finished_at": time.time()})
        del state["jobs"][model]
        del state["sent"][model]
        save_state(state)

    # Replay every model so finished episodes of inactive models are still recorded,
    # but only send requests for active models that have no batch running.
    records, pending, n_pending, n_failed = replay(backends, cache)
    with open(RUN_DIR / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    submitted = 0
    for model in sorted(ACTIVE):
        reqs = pending.get(model) or {}
        if not reqs or model in state["jobs"]:
            continue
        w = state["wave"].get(model, 0)
        if w >= MAX_WAVES:
            log(f"{model}: wave cap {MAX_WAVES} reached with {len(reqs)} requests pending")
            continue
        state["wave"][model] = w + 1
        wave_dir = RUN_DIR / f"wave{w + 1:02d}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        job = backends[model].submit([(k, v["request"]) for k, v in reqs.items()], wave_dir)
        state["jobs"][model] = job
        state["sent"][model] = {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]}
                                for k, v in reqs.items()}
        save_state(state)  # after every submit, so a crash never resubmits a sent model
        submitted += 1
        log(f"{model} wave {w + 1}: submitted {len(reqs)} requests -> {job}")
    running = sorted(state["jobs"])
    log(f"replay: {len(records)} episodes done/errored ({n_failed} errored); running: {', '.join(running) or 'none'}")
    if not running and not submitted:
        log("pilot finished")
        return True
    return False


def plan(backends):
    cache = ReplayCache(Path("/dev/null"))
    _, pending, n_pending, _ = replay(backends, cache)
    for model, reqs in pending.items():
        n_img = sum(sum(1 for c in json.dumps(v["request"]).split('"image/jpeg"')[1:]) for v in reqs.values())
        size = sum(len(json.dumps(v["request"])) for v in reqs.values())
        log(f"{model}: wave-1 requests {len(reqs)}, images {n_img}, payload {size / 1e6:.1f} MB")


def report():
    recs = [json.loads(line) for line in open(RUN_DIR / "records.jsonl")]
    table = defaultdict(lambda: {"n": 0, "correct": 0, "error": 0, "lid_ok": 0, "lid_n": 0, "own": 0})
    for r in recs:
        t = table[(r["model"], r["block"], r["lang"])]
        t["n"] += 1
        if r["status"] != "done":
            t["error"] += 1
            continue
        t["correct"] += bool(r["score"].get("correct"))
        t["own"] += bool(r["score"].get("own_culture"))
        for turn in r["turns"]:
            if turn.get("lid") is not None:
                t["lid_n"] += 1
                t["lid_ok"] += bool(turn["lid"]["ok"])
    rows = []
    for (model, block, lang), t in sorted(table.items()):
        acc = t["correct"] / t["n"] if t["n"] else 0
        lid_rate = t["lid_ok"] / t["lid_n"] if t["lid_n"] else None
        rows.append({"model": model, "block": block, "lang": lang, "n": t["n"], "acc": round(acc, 3),
                     "errors": t["error"], "own_culture": t["own"],
                     "lid_pass": None if lid_rate is None else round(lid_rate, 3)})
    with open(RUN_DIR / "summary.json", "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "step", "report"])
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--run", default="pilot_gate", help="run directory name under outputs/mtcomm")
    ap.add_argument("--blend-items", default=None, help="BLEnD items jsonl (default: current build)")
    ap.add_argument("--manifest", default=None, help="XM3600 manifest (default: v2, text-free)")
    ap.add_argument("--max-words", type=int, default=None, help="describer word limit (default: none)")
    ap.add_argument("--blocks", nargs="*", default=None, help="only these blocks (A B C D Dprior)")
    ap.add_argument("--prompt-style", default="explicit", choices=["explicit", "neutral"],
                    help="culture-game prompts (explicit = first pilots)")
    ap.add_argument("--min-questions", type=int, default=0, help="culture game: messages required before answering")
    args = ap.parse_args()
    global RUN_DIR
    RUN_DIR = OUTPUT_ROOT / args.run
    OPTS.update(blend_items=args.blend_items, manifest=args.manifest, max_words=args.max_words, blocks=args.blocks,
                prompt_style=args.prompt_style, min_questions=args.min_questions)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    global ACTIVE
    ACTIVE = set(args.models)
    # Replay the active models plus any model already in this run's cache or state (so earlier
    # results stay in records); only ACTIVE models are advanced.
    cache_models = set()
    if (RUN_DIR / "cache.jsonl").exists():
        cache_models = {json.loads(line).get("model") for line in open(RUN_DIR / "cache.jsonl") if line.strip()}
    if (RUN_DIR / "state.json").exists():
        cache_models |= set(json.load(open(RUN_DIR / "state.json"))["jobs"])
    backends = {m: make_backend(m) for m in MODELS if m in ACTIVE or m in cache_models}
    if args.cmd == "plan":
        return plan(backends)
    if args.cmd == "report":
        return report()
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump({"models": {m: b.config() for m, b in backends.items()}, "active": sorted(ACTIVE), **OPTS},
                  f, indent=1, default=str)
    cache = ReplayCache(RUN_DIR / "cache.jsonl")
    while True:
        done = step(backends, cache)
        if done or not args.loop:
            break
        time.sleep(args.interval)
    if done:
        report()


if __name__ == "__main__":
    sys.exit(main())
