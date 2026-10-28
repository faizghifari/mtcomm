"""Main PoC runs: every task x language pair x condition, with Gemini 3.8 Flash as both players.

Conditions (one channel each):
  sol, flash-lite       LLM translators (mt-eval's InterpreterAgent brief)
  nllb, google          NLLB-200 3.3B (local GPU) and the Google web endpoint
  nllb-id, google-id    the same, pivoting through Indonesian (su<->jv pairs only)
  ceiling               both players in the scored player's language, no translator
  prior                 no message reaches the scored player (it answers alone)
Ceiling and prior depend only on the scored player's language, so they run once per language
and are shared by every pair with that language.

Two phases, both resumable (deterministic replay, see mtcomm.replay):
  batch     The opening call of every episode does not depend on the channel, so it is sent once
            per distinct request through the batch API (half price), with the single-call priors.
  dialogue  The rest of each episode runs with synchronous calls, episodes in parallel threads.

Usage:
  python scripts/run_main.py plan     [--run R] [--tasks ...] [--pairs ...] [--n N]
  python scripts/run_main.py batch    [...] [--loop]      # submit / poll / ingest the opening calls
  python scripts/run_main.py dialogue [...] [--workers 12]
  python scripts/run_main.py report   [...]
"""

import argparse
import json
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from mtcomm import lid
from mtcomm.backends import make_backend
from mtcomm.channels import GoogleWebChannel, IdentityChannel, NLLBChannel, NoChannel, PivotChannel, llm_translator
from mtcomm.data import load_blend_items, load_image_items, load_route_items, load_sequence_items
from mtcomm.data_mugeval import load_mcq_items, load_twentyq_items
from mtcomm.game import run_episode
from mtcomm.games.blend_lv import BlendLocalVisitorGame
from mtcomm.games.image_ref import ImageRefGame
from mtcomm.games.mugeval import MCQConversationGame, TwentyQuestionsGame
from mtcomm.games.route import RouteGame
from mtcomm.games.sequence import SequenceGame
from mtcomm.paths import DATA_ROOT, OUTPUT_ROOT
from mtcomm.players import LLMPlayer
from mtcomm.replay import BatchGenerate, CachedChannel, CallFailed, Pending, ReplayCache, SyncGenerate

PLAYER = "gemini-3.8-flash"
PERTURB_MANIFEST = DATA_ROOT / "derived" / "perturb" / "full60" / "manifest_perturb_full60.json"
TASKS = ["image", "mcq", "twentyq", "culture", "sequence", "sequence_step", "route"]
# (a, b): image describer=a, guesser=b; sequence narrator=a, arranger=b; route giver=a, follower=b;
# MuG-Eval questioner=a, answerer=b.
PAIRS = [("su", "jv"), ("jv", "su"), ("su", "ko"), ("ko", "su")]
# Culture: (local culture, visitor language); the local speaks the culture's language.
CULTURE_PAIRS = [("West_Java", "jv"), ("West_Java", "ko"), ("South_Korea", "su")]
TRANSLATORS = ["sol", "flash-lite", "nllb", "google"]
PIVOTS = ["nllb-id", "google-id"]  # su<->jv only
# USD per million tokens (input, output); batch is half. Gemini 3.8 Flash doubles from 2027-01-01.
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.1-flash-lite": (0.25, 1.50), "gpt-6.1-sol": (2.0, 10.0)}
TRANSLATOR_MODELS = {"sol": "gpt-6.1-sol", "flash-lite": "gemini-3.1-flash-lite"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Channels:
    """Builds each translator once (NLLB loads 3.3B weights on first use) and shares it across threads."""

    def __init__(self):
        self._made = {}
        self._lock = threading.RLock()  # RLock: a pivot builds its inner channel while holding it

    def get(self, cond):
        with self._lock:
            return self._get(cond)

    def _get(self, cond):
        if cond not in self._made:
            if cond in TRANSLATOR_MODELS:
                ch = llm_translator(TRANSLATOR_MODELS[cond])
            elif cond == "nllb":
                ch = NLLBChannel("facebook/nllb-200-3.3B")
            elif cond == "google":
                ch = GoogleWebChannel()
            elif cond in PIVOTS:
                ch = PivotChannel(self.get(cond.split("-")[0]), "id")
            elif cond == "ceiling":
                ch = IdentityChannel()
            elif cond == "prior":
                ch = NoChannel()
            else:
                raise ValueError(cond)
            self._made[cond] = ch
        return self._made[cond]


def conditions(a, b):
    conds = list(TRANSLATORS)
    if {a, b} == {"su", "jv"}:
        conds += PIVOTS
    return conds


# v2 (prior wrong and passage right, in su/jv/ko) once built; v1 (prior wrong only) was used by run main_mcq.
MCQ_SCREEN = next((p for p in (DATA_ROOT / "derived" / "mcq" / f"mcq_screen_v{v}.json" for v in (2, 1)) if p.exists()),
                  DATA_ROOT / "derived" / "mcq" / "mcq_screen_v1.json")


def task_items(task, load, a, b, n):
    """MCQ uses the items the no-talk prior got wrong (scripts/mcq_prior_screen.py) when screened."""
    if task != "mcq" or not MCQ_SCREEN.exists():
        return load(a, b, n=n)
    screen = json.load(open(MCQ_SCREEN))
    by_id = {it["item_id"]: it for it in load(a, b, n=screen["pool"])}
    return [by_id[i] for i in screen["item_ids"]][:n]


def episodes(opts):
    """Yield dicts: id, task, pair, cond, eval_lang, build() -> (game, roles).

    Ceiling/prior episodes are keyed by the scored language instead of the pair, so pairs sharing
    that language share them."""
    n, tasks, pairs = opts["n"], opts["tasks"], opts["pairs"]
    seen = set()

    def emit(task, pair, cond, eval_lang, item_id, build):
        key = f"{task}|{cond}|{eval_lang if cond in ('ceiling', 'prior') else pair}|{item_id}"
        if key not in seen:
            seen.add(key)
            return [{"id": key, "task": task, "pair": pair, "cond": cond, "eval_lang": eval_lang, "build": build}]
        return []

    if "image" in tasks:
        items = load_image_items("hard", manifest_path=PERTURB_MANIFEST)[:n]
        for a, b in pairs:
            for it in items:
                for cond in conditions(a, b) + ["ceiling", "prior"]:
                    da = b if cond == "ceiling" else a
                    yield from emit("image", f"{a}-{b}", cond, b, it["item_id"],
                                    lambda it=it, da=da, b=b, cond=cond: (
                                        ImageRefGame(it, da, b, allow_message=cond != "prior", max_words=14,
                                                     max_questions=3 if cond != "prior" else 0),
                                        ["describer", "guesser"]))
    for task in [t for t in ("sequence", "sequence_step") if t in tasks]:
        items = load_sequence_items()[:n]
        for a, b in pairs:
            for it in items:
                for cond in conditions(a, b) + ["ceiling", "prior"]:
                    na = b if cond == "ceiling" else a
                    yield from emit(task, f"{a}-{b}", cond, b, it["item_id"],
                                    lambda it=it, na=na, b=b, cond=cond, task=task: (
                                        SequenceGame(it, na, b, allow_message=cond != "prior",
                                                     stepwise=task == "sequence_step"),
                                        ["narrator", "arranger"]))
    if "route" in tasks:
        items = load_route_items(langs=sorted({x for p in pairs for x in p}))
        easy = [it for it in items if it["map"]["level"] == "easy"]
        hard = [it for it in items if it["map"]["level"] == "hard"]
        level = opts.get("route_level", "both")  # half easy, half hard; "hard" re-runs only the hard half
        items = (easy[:(n + 1) // 2] if level != "hard" else []) + (hard[:n // 2] if level != "easy" else [])
        for a, b in pairs:
            for it in items:
                for cond in conditions(a, b) + ["ceiling", "prior"]:
                    ga = b if cond == "ceiling" else a
                    yield from emit("route", f"{a}-{b}", cond, b, it["item_id"],
                                    lambda it=it, ga=ga, b=b, cond=cond: (
                                        RouteGame(it, ga, b, allow_message=cond != "prior"),
                                        ["giver", "follower"]))
    for task, Game, load in [("mcq", MCQConversationGame, load_mcq_items),
                             ("twentyq", TwentyQuestionsGame, load_twentyq_items)]:
        if task not in tasks:
            continue
        for a, b in pairs:
            for cond in conditions(a, b) + ["ceiling", "prior"]:
                lb = a if cond == "ceiling" else b
                for it in task_items(task, load, a, lb, n):
                    yield from emit(task, f"{a}-{b}", cond, a, it["item_id"],
                                    lambda it=it, a=a, lb=lb, cond=cond, Game=Game: (
                                        Game(it, a, lb, allow_message=cond != "prior"),
                                        ["questioner", "answerer"]))
    if "culture" in tasks:
        blend = load_blend_items()
        for culture, v in CULTURE_PAIRS:
            local_lang = {"West_Java": "su", "South_Korea": "ko"}[culture]
            if pairs != PAIRS and (local_lang, v) not in pairs and (v, local_lang) not in pairs:
                continue
            # Items whose rendered options collide (e.g. football/soccer -> one word) are unanswerable.
            items = [it for it in blend if it["visitor_lang"] == v and it["local_culture"] == culture
                     and len({o.strip().lower() for o in it["options"].values()}) == len(it["options"])][:n]
            for it in items:
                for cond in conditions(local_lang, v) + ["ceiling", "prior"]:
                    ll = v if cond == "ceiling" else local_lang
                    yield from emit("culture", f"{culture}>{v}", cond, v, it["item_id"],
                                    lambda it=it, ll=ll, v=v, cond=cond: (
                                        BlendLocalVisitorGame(it, ll, v, max_questions=0 if cond == "prior" else 3,
                                                              min_questions=0 if cond == "prior" else 1,
                                                              prompt_style="neutral"),
                                        ["visitor", "local"]))


def build(ep, make_gen):
    """Build an episode. Only the role that speaks first gets a shared (request-keyed) first call: its
    opening request is the same in every condition. Later roles' first calls depend on the channel."""
    gm, roles = ep["build"]()
    opener = gm.next_role()
    return gm, {role: LLMPlayer(make_gen(role, share_first=(role == opener)), PLAYER) for role in roles}


def run_one(ep, backend, cache, chans, mode, pending=None):
    if mode == "batch":
        make_gen = lambda role, share_first=False: BatchGenerate(  # noqa: E731
            backend, cache, ep["id"], role, pending, share_first=share_first)
    else:
        make_gen = lambda role, share_first=False: SyncGenerate(  # noqa: E731
            backend, cache, ep["id"], role, share_first=share_first)
    gm, players = build(ep, make_gen)
    # In the batch phase an uncached translation is left for `dialogue` (translators are synchronous).
    channel = CachedChannel(chans.get(ep["cond"]), cache, ep["id"], offline=(mode == "batch"))
    rec = run_episode(gm, players, channel, lid=lid.check, max_moves=50)
    rec.update(episode=ep["id"], task=ep["task"], pair=ep["pair"], cond=ep["cond"], eval_lang=ep["eval_lang"],
               model=PLAYER, status="done")
    return rec


def cmd_plan(opts, backend):
    eps = list(episodes(opts))
    by = defaultdict(int)
    for e in eps:
        by[(e["task"], e["cond"])] += 1
    for k, v in sorted(by.items()):
        print(k, v)
    pending = {}
    cache = ReplayCache(opts["run_dir"] / "plan_null.jsonl")
    for ep in eps:
        try:
            run_one(ep, backend, cache, Channels(), "batch", pending)
        except Pending:
            pass
    first = {k for k in pending if "|first|" in k}
    log(f"{len(eps)} episodes; {len(first)} distinct opening calls go to batch")
    (opts["run_dir"] / "plan_null.jsonl").unlink(missing_ok=True)


def cmd_batch(opts, backend, cache, loop, interval):
    """One batch wave of opening calls (and single-call priors); later turns are left to `dialogue`."""
    state_p = opts["run_dir"] / "batch_state.json"
    while True:
        state = json.load(open(state_p)) if state_p.exists() else {"job": None, "sent": {}, "history": []}
        if state["job"]:
            st = backend.status(state["job"])
            if st == "running":
                log(f"batch {state['job']} running")
                if not loop:
                    return
                time.sleep(interval)
                continue
            if st == "succeeded":
                res = backend.results(state["job"])
                for key, info in state["sent"].items():
                    r = res.get(key, {"error": "missing from batch output"})
                    rec = {"key": key, "request_hash": info["request_hash"], "model": PLAYER, "mode": "batch",
                           "usage": r.get("usage")}
                    if "text" in r:
                        rec["text"] = r["text"]
                    else:
                        rec.update(error=r["error"], attempts=info["attempts"] + 1)
                    cache.put(rec)
                log(f"ingested {len(state['sent'])} batch results "
                    f"({sum('text' not in res.get(k, {}) for k in state['sent'])} errors)")
            else:
                log(f"batch {state['job']} failed; unanswered requests will be resubmitted")
            state["history"].append({"job": state["job"], "status": st, "finished_at": time.time()})
            state["job"], state["sent"] = None, {}
            json.dump(state, open(state_p, "w"), indent=1)
        pending = {}
        chans = Channels()
        for ep in episodes(opts):
            try:
                run_one(ep, backend, cache, chans, "batch", pending)
            except (Pending, CallFailed):
                pass
            except Exception:  # a translator failing on an opening message is retried in `dialogue`
                pass
        first = {k: v for k, v in pending.items() if "|first|" in k}
        if not first:
            log("batch phase complete: every opening call is answered")
            return
        if len(state["history"]) >= 3:
            log(f"{len(first)} opening calls still unanswered after 3 batches; `dialogue` will call them directly")
            return
        wave_dir = opts["run_dir"] / f"batch{len(state['history']) + 1:02d}"
        wave_dir.mkdir(parents=True, exist_ok=True)
        state["job"] = backend.submit([(k, v["request"]) for k, v in first.items()], wave_dir)
        state["sent"] = {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]} for k, v in first.items()}
        json.dump(state, open(state_p, "w"), indent=1)
        log(f"submitted {len(first)} opening calls -> {state['job']}")
        if not loop:
            return


def cmd_dialogue(opts, backend, cache, workers):
    eps = list(episodes(opts))
    chans = Channels()
    records, errors = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, ep, backend, cache, chans, "sync"): ep for ep in eps}
        for i, fut in enumerate(as_completed(futs), 1):
            ep = futs[fut]
            try:
                records.append(fut.result())
            except Exception as e:
                errors += 1
                records.append({"episode": ep["id"], "task": ep["task"], "pair": ep["pair"], "cond": ep["cond"],
                                "eval_lang": ep["eval_lang"], "model": PLAYER, "status": "error",
                                "error": f"{type(e).__name__}: {e}"[:400],
                                "trace": traceback.format_exc()[-1500:]})
            if i % 25 == 0 or i == len(eps):
                log(f"{i}/{len(eps)} episodes ({errors} errors)")
    with open(opts["run_dir"] / "records.jsonl", "w") as f:
        for r in sorted(records, key=lambda r: r["episode"]):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    cmd_report(opts)


def _tokens(usage):
    """(input, output) tokens from a Gemini usageMetadata or OpenAI usage dict; thinking counts as output."""
    if not usage:
        return 0, 0
    if "promptTokenCount" in usage:
        return usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def cost(opts):
    """Spend so far from the run cache: player calls (batch or sync) and LLM-translator calls."""
    spend, toks = defaultdict(float), defaultdict(lambda: [0, 0, 0])
    p = opts["run_dir"] / "cache.jsonl"
    if not p.exists():
        return spend, toks
    latest = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            latest.setdefault(r["key"], []).append(r)
    for key, recs in latest.items():
        for r in recs:  # every attempt was billed, so count all records, not only the latest
            if key.startswith("channel|"):
                model = key.split("|")[1].removeprefix("llm:")  # LLM channels are named "llm:<model>"
                for u in (r.get("meta") or {}).get("usage") or []:
                    i, o = _tokens(u)
                    pi, po = PRICES.get(model, (0, 0))
                    spend[f"translator {model}"] += (i * pi + o * po) / 1e6
                    t = toks[f"translator {model}"]
                    t[0], t[1], t[2] = t[0] + i, t[1] + o, t[2] + 1
            elif r.get("usage"):
                i, o = _tokens(r["usage"])
                pi, po = PRICES[r["model"]]
                half = 0.5 if r.get("mode") == "batch" else 1.0
                lab = f"player {r['model']} ({r.get('mode', 'batch')})"
                spend[lab] += half * (i * pi + o * po) / 1e6
                t = toks[lab]
                t[0], t[1], t[2] = t[0] + i, t[1] + o, t[2] + 1
    return spend, toks


def cmd_report(opts):
    p = opts["run_dir"] / "records.jsonl"
    recs = [json.loads(line) for line in open(p)] if p.exists() else []
    cell = defaultdict(lambda: {"n": 0, "ok": 0, "err": 0, "turns": 0, "lid_n": 0, "lid_ok": 0})
    for r in recs:
        c = cell[(r["task"], r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"], r["cond"])]
        c["n"] += 1
        if r["status"] != "done":
            c["err"] += 1
            continue
        c["ok"] += bool(r["score"].get("correct"))
        msgs = [t for t in r["turns"] if t["kind"] == "message"]
        c["turns"] += len(msgs)
        for t in msgs:
            if t.get("lid") and len(t["source"].split()) >= 3:  # GlotLID is unreliable on "Henteu." etc.
                c["lid_n"] += 1
                c["lid_ok"] += bool(t["lid"]["ok"])
    rows = []
    for (task, pair, cond), c in sorted(cell.items()):
        done = c["n"] - c["err"]
        rows.append({"task": task, "pair": pair, "cond": cond, "n": c["n"], "errors": c["err"],
                     "success": round(c["ok"] / done, 3) if done else None,
                     "msgs_per_game": round(c["turns"] / done, 1) if done else None,
                     "lid_pass": round(c["lid_ok"] / c["lid_n"], 3) if c["lid_n"] else None})
    spend, toks = cost(opts)
    out = {"cells": rows, "spend_usd": {k: round(v, 4) for k, v in spend.items()},
           "spend_total_usd": round(sum(spend.values()), 4),
           "tokens": {k: {"in": v[0], "out": v[1], "calls": v[2]} for k, v in toks.items()}}
    json.dump(out, open(opts["run_dir"] / "summary.json", "w"), indent=1)
    for r in rows:
        print(f"{r['task']:8s} {r['pair']:16s} {r['cond']:10s} n={r['n']:3d} err={r['errors']:2d} "
              f"success={r['success']} msgs={r['msgs_per_game']} lid={r['lid_pass']}")
    for k, v in sorted(out["spend_usd"].items()):
        print(f"spend {k}: ${v} ({out['tokens'][k]})")
    print(f"spend total: ${out['spend_total_usd']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "batch", "dialogue", "report"])
    ap.add_argument("--run", default="main")
    ap.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    ap.add_argument("--pairs", nargs="*", default=None, help="e.g. su-jv ko-su (default: all four)")
    ap.add_argument("--n", type=int, default=30, help="items per task and pair")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--route-level", default="both", choices=["both", "easy", "hard"])
    args = ap.parse_args()
    pairs = [tuple(p.split("-")) for p in args.pairs] if args.pairs else PAIRS
    opts = {"run_dir": OUTPUT_ROOT / args.run, "tasks": args.tasks, "pairs": pairs, "n": args.n,
            "route_level": args.route_level}
    opts["run_dir"].mkdir(parents=True, exist_ok=True)
    backend = make_backend(PLAYER)
    if args.cmd == "plan":
        return cmd_plan(opts, backend)
    if args.cmd == "report":
        return cmd_report(opts)
    with open(opts["run_dir"] / "config.json", "w") as f:
        json.dump({"player": backend.config(), "tasks": args.tasks, "pairs": pairs, "n": args.n,
                   "route_level": args.route_level, "translators": TRANSLATORS, "pivots": PIVOTS}, f, indent=1, default=str)
    cache = ReplayCache(opts["run_dir"] / "cache.jsonl")
    if args.cmd == "batch":
        return cmd_batch(opts, backend, cache, args.loop, args.interval)
    return cmd_dialogue(opts, backend, cache, args.workers)


if __name__ == "__main__":
    sys.exit(main())
