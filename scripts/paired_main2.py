"""Paired (same-item) comparisons for the main2 runs.

Every condition in a run plays the same 30 items per pair, so two conditions can be compared item by
item: an exact McNemar (two-sided binomial on the discordant games). This removes item difficulty, which
the unpaired Wilson intervals and the cell-reliability in analyze_main2.py leave in, and is the right test
for "does translator A beat translator B / the ceiling", pooled over pairs (n=120 matched games).

Also reports the item effect: the intraclass correlation of success across the 5 translators on one item.

p-values are NOT corrected: 10 translator contrasts per task (60 across the six tasks); Bonferroni within a
task is p < 0.005. Games are single samples (Gemini default sampling, one run per cell), so item-sampling
noise is removed by pairing but sampling noise of the players is not separated from translation effects.
The ceiling of a receiving language is re-used by every pair that shares it (su twice), so the vs-ceiling
tests are slightly anti-conservative.

Usage: python scripts/paired_main2.py   -> outputs/mtcomm/analysis_main2/paired.md
"""

import json
import math
from collections import defaultdict
from itertools import combinations

from mtcomm.paths import OUTPUT_ROOT

OUT = OUTPUT_ROOT / "analysis_main2" / "paired.md"
RUNS = {"MCQ (Flash-Lite)": "main2x_mcq", "Twenty Questions (Flash-Lite)": "main2x_twentyq",
        "Sequence (3.8 Flash)": "main2x_sequence", "Image (3.8 Flash)": "main2xf_image",
        "Route (3.8 Flash)": "main2x_route", "Culture (Flash-Lite)": "main2x_culture"}
TRANSLATORS = ["sol", "flash-lite", "nllb", "google", "qwen"]


def mcnemar_p(b, c):
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(run):
    """{(pair or receiving language, cond, item_id): 0/1}; ceiling/prior are keyed by receiving language."""
    d = {}
    for line in open(OUTPUT_ROOT / run / "records.jsonl"):
        r = json.loads(line)
        if r["status"] != "done":
            continue
        scope = r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"]
        d[(scope, r["cond"], r["item"]["item_id"])] = int(bool(r["score"].get("correct")))
    return d


def receiver(pair):
    return pair.split(">")[-1].split("-")[-1]  # culture pairs are "<culture>><visitor>"


def item_icc(d, pairs):
    rows = [[d.get((p, t, it)) for t in TRANSLATORS] for p in pairs for it in {k[2] for k in d if k[0] == p}]
    rows = [r for r in rows if None not in r]
    k, n = len(TRANSLATORS), len(rows)
    if n < 2:
        return None
    gm = sum(map(sum, rows)) / (n * k)
    msb = k * sum((sum(r) / k - gm) ** 2 for r in rows) / (n - 1)
    msw = sum(sum((x - sum(r) / k) ** 2 for x in r) for r in rows) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), n, sum(all(r) for r in rows) / n, sum(not any(r) for r in rows) / n


def main():
    md = ["# Paired (same-item) comparisons, main2\n",
          "Exact McNemar on discordant games, pooled over pairs; p uncorrected (10 translator contrasts per task, "
          "Bonferroni within task p < 0.005). Diff in points; `a>b` counts games a solved and b did not.\n"]
    for name, run in RUNS.items():
        d = load(run)
        pairs = sorted({k[0] for k in d if k[1] == "sol"})
        md.append(f"\n## {name} — {run}\n")
        md.append("| translator vs ceiling | n | diff | games better / worse | p |\n|---|---|---|---|---|")
        for t in TRANSLATORS:
            b = c = n = 0
            for (p, cond, it), v in d.items():
                if cond == t and p in pairs and (ce := d.get((receiver(p), "ceiling", it))) is not None:
                    n += 1
                    b += v == 1 and ce == 0
                    c += v == 0 and ce == 1
            if n:
                md.append(f"| {t} | {n} | {100 * (b - c) / n:+.1f} | {b} / {c} | {mcnemar_p(b, c):.3f} |")
        md.append("\n| translator contrast | n | diff | a>b / b>a | p |\n|---|---|---|---|---|")
        rows = []
        for a, bb in combinations(TRANSLATORS, 2):
            x = y = n = 0
            for (p, cond, it), v in d.items():
                if cond == a and p in pairs and (w := d.get((p, bb, it))) is not None:
                    n += 1
                    x += v == 1 and w == 0
                    y += v == 0 and w == 1
            if n:
                rows.append((mcnemar_p(x, y), a, bb, n, x, y))
        for p, a, bb, n, x, y in sorted(rows):
            md.append(f"| {a} vs {bb} | {n} | {100 * (x - y) / n:+.1f} | {x} / {y} | {p:.3f} |")
        if (icc := item_icc(d, pairs)):
            md.append(f"\nItem effect: ICC across the 5 translators on one item = {icc[0]:.2f} ({icc[1]} pair-items); "
                      f"solved by all 5: {100 * icc[2]:.0f}%, by none: {100 * icc[3]:.0f}%.")
    OUT.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
