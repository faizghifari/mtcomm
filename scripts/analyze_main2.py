"""Tables for the main2 runs (monolingual players): success per task x pair x condition with Wilson
intervals, success relative to the ceiling, gain over the prior, refusal rates, the pivot effect, and how
stable the translator ranking is across player models.

Reads outputs/mtcomm/<run>/records.jsonl (not summary.json, so cells can be recomputed with intervals).
Writes outputs/mtcomm/analysis_main2/{tables.md,cells.json}.

Usage: python scripts/analyze_main2.py [--runs main2_mcq main2_image ...]
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations

from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import NOT_UNDERSTOOD, is_not_understood

OUT = OUTPUT_ROOT / "analysis_main2"
RUNS = ["main2_mcq", "main2_image", "main2f_image", "main2_culture", "main2_twentyq", "main2_route", "main2_sequence"]
ROBUST = {"mcq": ["main2_mcq", "main2r_flash_mcq", "main2r_qwen_mcq"], "image": ["main2f_image", "main2_image", "main2r_qwen_image"]}
TRANSLATORS = ["sol", "flash-lite", "nllb", "google", "qwen"]
ORDER = TRANSLATORS + ["nllb-id", "google-id", "passthrough", "ceiling", "prior"]


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return round(num / den, 3) if den else None


def graded(r):
    """Secondary score: route step accuracy, sequence positions correct (fraction); None elsewhere."""
    s = r.get("score") or {}
    if r["task"] == "sequence" and s.get("gold"):
        return s.get("positions_correct", 0) / len(s["gold"])
    for k in ("step_accuracy", "steps_correct_frac"):
        if k in s:
            return s[k]
    return None


def cells_of(run):
    p = OUTPUT_ROOT / run / "records.jsonl"
    if not p.exists():
        return {}
    cell = defaultdict(lambda: {"n": 0, "ok": 0, "err": 0, "msgs": 0, "refused": 0, "graded": []})
    for line in open(p):
        r = json.loads(line)
        key = (r["task"], r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"], r["cond"])
        c = cell[key]
        if r["status"] != "done":
            c["err"] += 1
            continue
        c["n"] += 1
        c["ok"] += bool(r["score"].get("correct"))
        g = graded(r)
        if g is not None:
            c["graded"].append(g)
        for t in r["turns"]:
            if t["kind"] == "message":
                c["msgs"] += 1
                c["refused"] += any(is_not_understood(t["source"], k) for k in NOT_UNDERSTOOD)
    out = {}
    for k, c in cell.items():
        out[k] = {"n": c["n"], "errors": c["err"], "success": round(c["ok"] / c["n"], 3) if c["n"] else None,
                  "ci": wilson(c["ok"], c["n"]), "refusal": round(c["refused"] / c["msgs"], 3) if c["msgs"] else None,
                  "graded": round(sum(c["graded"]) / len(c["graded"]), 3) if c["graded"] else None}
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    md, dump = ["# Main2 results (monolingual players)\n"], {}
    for run in args.runs:
        cells = cells_of(run)
        if not cells:
            continue
        dump[run] = {"|".join(k): v for k, v in cells.items()}
        task = next(iter(cells))[0]
        pairs = sorted({p for (_, p, c) in cells if c not in ("ceiling", "prior")})
        conds = [c for c in ORDER if any(k[2] == c for k in cells)]
        md.append(f"\n## {run} ({task})\n")
        md.append("Success % (n); refusal % of messages in brackets. Ceiling/prior are per receiving language.\n")
        md.append("| pair | " + " | ".join(conds) + " |")
        md.append("|---" * (len(conds) + 1) + "|")
        for p in pairs:
            recv = p.split(">")[-1].split("-")[-1]  # culture pairs are "<culture>><visitor>"
            row = []
            for c in conds:
                k = (task, recv if c in ("ceiling", "prior") else p, c)
                v = cells.get(k)
                if v is None:
                    row.append("")
                    continue
                extra = f" [{pct(v['refusal'])}]" if v["refusal"] and c not in ("ceiling", "prior") else ""
                g = f" g{pct(v['graded'])}" if v["graded"] is not None else ""
                row.append(f"{pct(v['success'])} ({v['n']}){extra}{g}")
            md.append(f"| {p} | " + " | ".join(row) + " |")
        # relative to ceiling and gain over prior, translators only
        md.append("\nTranslator success relative to the ceiling (translated / ceiling) and share of the ceiling-over-prior gain kept:\n")
        md.append("| pair | " + " | ".join(t for t in TRANSLATORS if t in conds) + " |")
        md.append("|---" * (1 + sum(t in conds for t in TRANSLATORS)) + "|")
        for p in pairs:
            recv = p.split(">")[-1].split("-")[-1]
            ce, pr = cells.get((task, recv, "ceiling")), cells.get((task, recv, "prior"))
            row = []
            for t in TRANSLATORS:
                if t not in conds:
                    continue
                v = cells.get((task, p, t))
                if not (v and ce and pr and ce["success"]):
                    row.append("")
                    continue
                rel = v["success"] / ce["success"]
                gain = ce["success"] - pr["success"]
                kept = (v["success"] - pr["success"]) / gain if gain > 0 else None
                row.append(f"{rel:.2f} / {'–' if kept is None else f'{kept:.2f}'}")
            md.append(f"| {p} | " + " | ".join(row) + " |")
        # pivot effect
        piv = [(p, d, cells.get((task, p, d)), cells.get((task, p, d + "-id"))) for p in pairs for d in ("nllb", "google")]
        piv = [(p, d, a, b) for p, d, a, b in piv if a and b]
        if piv:
            md.append("\nPivot through Indonesian vs direct (success %):\n")
            md.append("| pair | system | direct | via id |\n|---|---|---|---|")
            for p, d, a, b in piv:
                md.append(f"| {p} | {d} | {pct(a['success'])} | {pct(b['success'])} |")
    # ranking stability across players
    md.append("\n## Translator ranking across player models\n")
    for task, runs in ROBUST.items():
        per = {}
        for run in runs:
            c = cells_of(run)
            if c:
                per[run] = {(p, t): v["success"] for (tk, p, t), v in c.items() if t in TRANSLATORS}
        if len(per) < 2:
            continue
        md.append(f"\n{task}: Spearman over (pair, translator) cells\n")
        for a, b in combinations(per, 2):
            keys = sorted(set(per[a]) & set(per[b]))
            rho = spearman([per[a][k] for k in keys], [per[b][k] for k in keys]) if len(keys) > 2 else None
            md.append(f"- {a} vs {b}: rho = {rho} over {len(keys)} cells")
            dump.setdefault("ranking", {})[f"{task}|{a}|{b}"] = {"rho": rho, "cells": len(keys)}
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    json.dump(dump, open(OUT / "cells.json", "w"), indent=1)
    print(f"wrote {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
