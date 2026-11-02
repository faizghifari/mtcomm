"""Why does 3.8 Flash say "I don't understand" to MCQ questions in its own language? (docs/poc_diagnostics.md)

Replays the first question of each MCQ ceiling game of cal38_mcq_v2m, so every call has the same context, under variants of
the answerer's setup. All calls go through the batch API.
  same     the system prompt and thinking level of the run (how often does it repeat?)
  nosys    no monolingual system prompt
  medium   the run's system prompt, thinking level medium
  v3       a system prompt that makes the language judgement separate from whether the question can be answered
  v3_pass  v3 on untranslated questions from another language: the gate must still refuse these

Usage: python scripts/mcq_refusal_check.py [--loop]
"""

import argparse
import json
import time
from collections import defaultdict

from mtcomm.backends import make_backend
from mtcomm.data_mugeval import load_mcq_items
from mtcomm.games.mugeval import MCQConversationGame
from mtcomm.languages import get as lang
from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import LLMPlayer, is_not_understood, monolingual_system, monolingual_system_v3
from mtcomm.replay import BatchGenerate, Pending, ReplayCache

PLAYER = "gemini-3.8-flash"
RUN_DIR = OUTPUT_ROOT / "mcq_refusal_check"
SOURCE = OUTPUT_ROOT / "cal38_mcq_v2m" / "records.jsonl"
PASS_TO = {"su": ["ko", "jv"], "jv": ["su"], "ko": ["su"]}


def first_questions():
    out = []
    for line in open(SOURCE):
        r = json.loads(line)
        if r["status"] == "done" and r["cond"] == "ceiling" and r["turns"] and r["turns"][0]["kind"] == "message":
            out.append((r["eval_lang"], r["item"]["item_id"], r["turns"][0]["source"]))
    return out


def run_group(name, cases, backend, cache, pending, system):
    """cases: (questioner lang, answerer lang, item_id, question). Returns [(answerer lang, reply or None)]."""
    results = []
    for q_lang, a_lang, item_id, question in cases:
        items = {it["item_id"]: it for it in load_mcq_items(q_lang, a_lang, n=600)}
        gm = MCQConversationGame(items[item_id], q_lang, a_lang)
        gm.inbox["answerer"].append(question)
        obs = gm.observe("answerer")
        player = LLMPlayer(BatchGenerate(backend, cache, f"{name}|{q_lang}|{a_lang}|{item_id}", "answerer", pending), PLAYER,
                           system(a_lang))
        try:
            results.append((a_lang, player.generate(player.prompt(obs), player.system, obs.images)))
        except Pending:
            results.append((a_lang, None))
    return results


def step(backend, pending, state_p, loop):
    if not pending:
        return True
    state = {"job": backend.submit([(k, v["request"]) for k, v in pending.items()], RUN_DIR),
             "sent": {k: {"request_hash": v["request_hash"], "attempts": v["attempts"]} for k, v in pending.items()}}
    json.dump(state, open(state_p, "w"))
    print(f"submitted {len(pending)} calls -> {state['job']}", flush=True)
    return False


def collect(backend, cache, state_p, loop):
    """Waits for this backend's batch; returns False when it should be polled again later."""
    if not state_p.exists():
        return True
    state = json.load(open(state_p))
    if not state["job"]:
        return True
    st = backend.status(state["job"])
    if st == "running":
        return False
    res = backend.results(state["job"]) if st == "succeeded" else {}
    for key, info in state["sent"].items():
        r = res.get(key, {"error": f"batch {st}"})
        rec = {"key": key, "request_hash": info["request_hash"], "model": PLAYER, "mode": "batch", "usage": r.get("usage")}
        rec.update({"text": r["text"]} if "text" in r else {"error": r["error"], "attempts": info["attempts"] + 1})
        cache.put(rec)
    json.dump({"job": None, "sent": {}}, open(state_p, "w"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    mono = lambda code: monolingual_system(lang(code).name, code)  # noqa: E731
    v3 = lambda code: monolingual_system_v3(lang(code).name, code)  # noqa: E731
    qs = first_questions()
    same = [(la, la, i, q) for la, i, q in qs]
    passing = [(la, to, i, q) for la, i, q in qs for to in PASS_TO[la]]
    backends = {"low": make_backend(PLAYER), "medium": make_backend(PLAYER, thinking_level="medium")}
    groups = {"same": ("low", same, mono), "nosys": ("low", same, lambda c: None), "medium": ("medium", same, mono),
              "v3": ("low", same, v3), "v3_pass": ("low", passing, v3)}
    while True:
        done, out = True, {}
        for tag, be in backends.items():
            cache = ReplayCache(RUN_DIR / f"cache_{tag}.jsonl")
            state_p = RUN_DIR / f"state_{tag}.json"
            if not collect(be, cache, state_p, args.loop):
                done = False
                continue
            pending = {}
            for name, (t, cases, system) in groups.items():
                if t == tag:
                    out[name] = run_group(name, cases, be, cache, pending, system)
            if not step(be, pending, state_p, args.loop):
                done = False
        if done:
            break
        if not args.loop:
            return
        time.sleep(60)
    lines = ["| variant | calls | refused % | by language |", "|---|---|---|---|"]
    for name, res in out.items():
        by = defaultdict(list)
        for la, text in res:
            by[la].append(is_not_understood(text, la))
        all_ = [x for v in by.values() for x in v]
        lines.append(f"| {name} | {len(all_)} | {100 * sum(all_) / len(all_):.0f} | "
                     + " ".join(f"{k} {100 * sum(v) / len(v):.0f}" for k, v in sorted(by.items())) + " |")
    (RUN_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
