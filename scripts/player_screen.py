"""Player screen: is a local model strong in its own language and weak in the others it may meet?

A good simulated player for language L reads L well and gets little from text in its partner's
language, a neighbour language, or a widely known language (English, Chinese). Otherwise it can
read an untranslated or half-translated message directly, and the game measures the player, not
the translator (the passthrough control in the main runs reached the same-language ceiling).

Each cell is (model, player language L, reading language X, prompt). The instructions are in
English as in the games, the Belebele question and options are in L, and only the passage is in X,
so a cell mimics a passthrough message. X = "none" is the no-passage prior. Two prompts: "plain",
and "restricted" (told it only understands L, answer only in L). The reply's explanation is
language-checked with GlotLID.

Selection rule: own-language accuracy high; partner and neighbour accuracy close to the prior.

Usage: python scripts/player_screen.py run --model qwen-sea-lion-v4-8b-vl --player su jv ko [--n 100]
       python scripts/player_screen.py report
"""

import argparse
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mtcomm import lid
from mtcomm.backends import LocalBackend
from mtcomm.data_mugeval import _belebele
from mtcomm.languages import get as lang
from mtcomm.paths import OUTPUT_ROOT

OUT = OUTPUT_ROOT / "player_screen"
LMS = "/mnt/c/Users/user/.lmstudio/bin/lms.exe"
READ = {  # player language -> reading languages: own, partners, neighbours, widely known
    "su": ["su", "jv", "ko", "id", "ms", "en", "zh"],
    "jv": ["jv", "su", "ko", "id", "ms", "en", "zh"],
    "ko": ["ko", "su", "jv", "ja", "zh", "en"],
}
PROMPTS = ("plain", "restricted")


def restricted_system(name):
    return (f"You are a native speaker of {name}, and {name} is the only language you understand. Apart from "
            f"these instructions, you cannot read any other language: not English, not Chinese, and not the "
            f"languages of neighbouring regions. If a text is not in {name}, treat it as unreadable and do not "
            f"use it. Always write only in {name}.")


def prompt(item, x, name):
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(item["options"], 1))
    if x == "none":
        head = (f"No passage is available. Choose the most likely answer to the question anyway. Reply in {name}: "
                f"first a line 'Answer: <number 1-4>', then one short sentence in {name} explaining your choice.")
        return f"{head}\n\n<|Question|>\n{item['question']}\n\n<|Options|>\n{opts}\n\n<|Language|>\n{name}"
    head = (f"Answer the question based only on the passage. Reply in {name}: first a line 'Answer: <number 1-4>', "
            f"then one short sentence in {name} explaining your choice. If you cannot read the passage, say so in "
            f"{name} and give your best guess.")
    return (f"{head}\n\n<|Passage|>\n{item['passages'][x]}\n\n<|Question|>\n{item['question']}\n\n"
            f"<|Options|>\n{opts}\n\n<|Language|>\n{name}")


def items_for(player, n, seed):
    tables = {x: _belebele(x) for x in READ[player]}
    keys = sorted(set.intersection(*(set(t) for t in tables.values())))
    out = []
    for key in random.Random(seed).sample(keys, n):
        q = tables[player][key]
        out.append({"key": f"{key[0]}#{key[1]}", "question": q["question"],
                    "options": [q[f"mc_answer{i}"] for i in range(1, 5)], "gold": int(q["correct_answer_num"]),
                    "passages": {x: tables[x][key]["flores_passage"] for x in READ[player]}})
    return out


def parse(text):
    m = re.search(r"Answer\s*[:：]\s*[*<\s]*([1-4])", text, re.I) or re.search(r"\b([1-4])\b", text)
    return int(m.group(1)) if m else None


def explanation(text):
    lines = [l for l in text.splitlines() if l.strip() and not re.match(r"\s*\**Answer", l, re.I)]
    return " ".join(lines).strip()


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}.jsonl"
    done = set()
    if path.exists():
        done = {(r["player"], r["read"], r["prompt"], r["key"]) for r in map(json.loads, open(path)) if not r["error"]}
    subprocess.run([LMS, "unload", "--all"], capture_output=True)
    subprocess.run([LMS, "load", args.model, "--gpu", "max", "-c", "8192", "--parallel", str(args.workers), "-y"],
                   check=True, capture_output=True)
    be = LocalBackend(f"local:{args.model}", max_tokens=256, workers=args.workers)
    jobs = []
    for player in args.player:
        name = lang(player).name
        for item in items_for(player, args.n, args.seed):
            for x in READ[player] + ["none"]:
                for p in PROMPTS:
                    if (player, x, p, item["key"]) not in done:
                        jobs.append((player, x, p, item, be.build(prompt(item, x, name),
                                                                  restricted_system(name) if p == "restricted" else None, [])))
    print(f"{args.model}: {len(jobs)} requests ({len(done)} cached)", flush=True)

    def one(job):
        player, x, p, item, req = job
        out = be.call(req)
        text = out.get("text", "")
        expl = explanation(text)
        return {"model": args.model, "player": player, "read": x, "prompt": p, "key": item["key"],
                "gold": item["gold"], "pred": parse(text), "text": text, "error": out.get("error"),
                "lid": lid.check(expl, player) if expl else None}

    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"  {i}/{len(jobs)}", flush=True)


def report(args):
    rows = [r for p in sorted(OUT.glob("*.jsonl")) for r in map(json.loads, open(p)) if not r["error"]]  # errors are retried
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["player"], r["prompt"], r["read"])].append(r)
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        for player in READ:
            for p in PROMPTS:
                line = {}
                for x in READ[player] + ["none"]:
                    c = cells.get((model, player, p, x))
                    if not c:
                        continue
                    acc = sum(r["pred"] == r["gold"] for r in c) / len(c)
                    lid_ok = [r["lid"]["ok"] for r in c if r["lid"]]
                    line[x] = {"n": len(c), "acc": round(acc, 3),
                               "no_answer": round(sum(r["pred"] is None for r in c) / len(c), 3),
                               "lid_ok": round(sum(lid_ok) / len(lid_ok), 3) if lid_ok else None}
                if line:
                    summary[f"{model}|{player}|{p}"] = line
                    print(f"{model:30s} {player} {p:10s} " +
                          "  ".join(f"{x}:{v['acc']:.2f}" + (f"(lid {v['lid_ok']:.2f})" if x == player and v['lid_ok'] is not None else "")
                                    for x, v in line.items()))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--player", nargs="+", choices=sorted(READ), required=True)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--workers", type=int, default=4)
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
