"""How much do results move when a whole task is played again? -> analysis_poc3/consistency.md

Compares the merged run (poc3_<task>, "r0") with fresh repeats (main4_<task>_r<k>, scripts/run_cells.py), which have their own cache and
so their own opening messages. On NLLB and Google only the players vary between runs, since the translators are deterministic; on Sol
and Flash-Lite the translator varies too. Qwen runs at temperature 0.

Reported per task:
  - success per condition in each run, the SD across runs, and the SD that sampling n games alone would give
  - per-game agreement and Cohen's kappa between runs, per condition
  - each fixed contrast in each run, and whether it survives Holm in that run
  - Kendall's tau between the runs' orderings of the translators
  - variance of one game's outcome split into item and residual (sampling), and the standard error of a cell for a few designs

Sequence and image r0 predate the rule that rejects a refused opening, so r0 is not exchangeable with their repeats. For those tasks the
r0 comparisons are also shown without the items whose r0 game opened with a refusal.

Usage: python scripts/repeat_consistency.py [--tasks sequence twentyq]
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations

from scipy.stats import kendalltau

from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import NOT_UNDERSTOOD, is_not_understood
from mtcomm.poc_runs import ANALYSIS
import mixed_effects

TITLES = {"mcq": "MCQ conversation", "twentyq": "Twenty Questions", "sequence": "Sequence ordering", "image": "Image near-twin", "route": "Route, easy maps", "culture": "Culture (BLEnD)"}
ORDER = ["ceiling", "sol", "flash-lite", "google", "nllb", "qwen", "nllb-id", "google-id"]
TRANSLATORS = ["sol", "flash-lite", "google", "nllb", "qwen"]
OLD_RULE = {"sequence", "image"}  # r0 played before refused openings were rejected
NAMES = {"ceiling": "Ceiling", "sol": "Sol", "flash-lite": "Flash-Lite", "google": "Google", "nllb": "NLLB", "qwen": "Qwen", "nllb-id": "NLLB via id", "google-id": "Google via id"}


def refused_opening(r):
    msgs = [t for t in r["turns"] if t.get("kind") == "message" and t.get("source")]
    return bool(msgs) and any(is_not_understood(msgs[0]["source"], k) for k in NOT_UNDERSTOOD)


def load(run):
    """{(cond, pair or scored language, item): 0/1} and the keys whose game opened with a refusal. Ceiling and prior games are keyed by language."""
    games, bad = {}, set()
    for line in open(OUTPUT_ROOT / run / "records.jsonl"):
        r = json.loads(line)
        if r["status"] != "done":
            continue
        key = (r["cond"], r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"], r["item"]["item_id"])
        games[key] = int(bool(r["score"].get("correct")))
        if refused_opening(r):
            bad.add(key)
    return games, bad


def kappa(xs, ys):
    """Cohen's kappa for two binary lists. None when chance agreement is 1 (all games the same in both)."""
    n = len(xs)
    agree = sum(x == y for x, y in zip(xs, ys)) / n
    px, py = sum(xs) / n, sum(ys) / n
    chance = px * py + (1 - px) * (1 - py)
    return None if chance >= 1 else (agree - chance) / (1 - chance)


def components(by_item):
    """Variance of one game's outcome: item (between items, sampling removed) and residual (between repeats of one item).

    `by_item` maps an item to its outcomes over the repeats, at least 2 each. One-way random-effects ANOVA estimators."""
    groups = [v for v in by_item.values() if len(v) >= 2]
    if not groups:
        return None
    r = sum(len(v) for v in groups) / len(groups)
    resid = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1) for v in groups) / len(groups)
    means = [sum(v) / len(v) for v in groups]
    grand = sum(means) / len(means)
    between = sum((m - grand) ** 2 for m in means) / (len(means) - 1)
    return {"item": max(0.0, between - resid / r), "resid": resid}


def se_design(item, resid, n_items, repeats):
    """SE of a cell's success rate, in points, with n_items items each played `repeats` times."""
    return 100 * math.sqrt(item / n_items + resid / (n_items * repeats))


def section(task):
    runs = [(f"r0", f"poc3_{task}")]
    k = 1
    while (OUTPUT_ROOT / f"main4_{task}_r{k}" / "records.jsonl").exists():
        runs.append((f"r{k}", f"main4_{task}_r{k}"))
        k += 1
    if len(runs) < 2:
        return []
    data = {name: load(run) for name, run in runs}
    names = [n for n, _ in runs]
    conds = [c for c in ORDER if all(any(key[0] == c for key in data[n][0]) for n in names)]
    md = [f"\n## {TITLES[task]} ({', '.join(f'{n} = {r}' for n, r in runs)})\n"]
    if task in OLD_RULE:
        md.append("r0 predates the rule that rejects a refused opening, so compare r0 with care; r1 and r2 follow the same rules.\n")

    # success per run
    md += ["| condition | n | " + " | ".join(names) + " | SD across runs | SD from n alone |", "|---|---|" + "---|" * (len(names) + 2)]
    rates = {}
    for c in conds:
        rs, ns = [], []
        for name in names:
            v = [x for key, x in data[name][0].items() if key[0] == c]
            rs.append(100 * sum(v) / len(v))
            ns.append(len(v))
        rates[c] = dict(zip(names, rs))
        mean = sum(rs) / len(rs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in rs) / (len(rs) - 1))
        expected = 100 * math.sqrt(mean / 100 * (1 - mean / 100) / ns[0])
        md.append(f"| {NAMES[c]} | {ns[0]} | " + " | ".join(f"{x:.0f}" for x in rs) + f" | {sd:.1f} | {expected:.1f} |")

    # agreement per game
    pairs = list(combinations(names, 2))
    labels = [f"{a} vs {b}" for a, b in pairs]
    md += ["\nPer-game agreement (% of games with the same outcome) and Cohen's kappa:\n", "| condition | " + " | ".join(labels) + " |", "|---|" + "---|" * len(pairs)]
    for c in conds:
        cells = []
        for a, b in pairs:
            keys = [key for key in data[a][0] if key[0] == c and key in data[b][0]]
            xs, ys = [data[a][0][key] for key in keys], [data[b][0][key] for key in keys]
            ka = kappa(xs, ys)
            cells.append(f"{100 * sum(x == y for x, y in zip(xs, ys)) / len(keys):.0f}%, " + (f"κ {ka:.2f}" if ka is not None else "κ n/a"))
        md.append(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    if task in OLD_RULE:
        bad = data["r0"][1]
        md += ["\nWithout items whose r0 game opened with a refusal:\n", "| condition | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for c in conds:
            keys = [key for key in data["r0"][0] if key[0] == c and key not in bad and all(key in data[n][0] for n in names)]
            md.append(f"| {NAMES[c]} | {len(keys)} | " + " | ".join(f"{100 * sum(data[n][0][key] for key in keys) / len(keys):.0f}" for n in names) + " |")

    # contrasts per run
    per_run = {n: {(r["a"], r["b"]): r for r in mixed_effects.contrasts(mixed_effects.load([run]))} for n, run in runs}
    md += ["\nFixed contrasts: difference in points per run; * = Holm p < 0.05 within that run.\n",
           "| contrast | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    shown = [key for key in per_run[names[0]] if all(key in per_run[n] for n in names)]
    for key in shown:
        md.append(f"| {NAMES[key[0]]} vs {NAMES[key[1]]} | " + " | ".join(f"{per_run[n][key]['diff']:+.0f}{'*' if per_run[n][key]['holm'] < 0.05 else ''}" for n in names) + " |")
    pool = mixed_effects.contrasts(mixed_effects.load([r for _, r in runs]))
    md += [f"\nAll {len(runs)} runs together (paired within pair × item × run):\n", "| contrast | n | diff | a wins / b wins | Holm p |", "|---|---|---|---|---|"]
    for r in pool:
        md.append(f"| {NAMES[r['a']]} vs {NAMES[r['b']]} | {r['n']} | {r['diff']:+.1f} | {r['x']} / {r['y']} | {r['holm']:.3f} |")

    # translator order
    present = [c for c in TRANSLATORS if c in conds]
    if len(present) >= 3:
        taus = []
        for a, b in pairs:
            tau = kendalltau([rates[c][a] for c in present], [rates[c][b] for c in present]).statistic
            taus.append(f"{a} vs {b} {tau:.2f}")
        md.append("\nKendall's tau of the translator order (" + ", ".join(NAMES[c] for c in present) + "): " + "; ".join(taus) + ".")

    # variance split
    use = names if task not in OLD_RULE or len(names) < 3 else names[1:]
    md += [f"\nVariance of one game's outcome, from runs {', '.join(use)}, and the standard error of one pooled cell (points) for other designs. "
           "Item and residual variance are the one-way ANOVA estimates. `30×1` is 30 items per pair × 4 pairs, played once.\n",
           "| condition | item | residual | item share | SE 120×1 | SE 120×3 | SE 360×1 |", "|---|---|---|---|---|---|---|"]
    for c in conds:
        by_item = defaultdict(list)
        for n in use:
            for key, x in data[n][0].items():
                if key[0] == c:
                    by_item[key[1:]].append(x)
        v = components(by_item)
        if not v:
            continue
        share = v["item"] / (v["item"] + v["resid"]) if v["item"] + v["resid"] else 0
        m = len(by_item)
        md.append(f"| {NAMES[c]} | {v['item']:.3f} | {v['resid']:.3f} | {100 * share:.0f}% | {se_design(v['item'], v['resid'], m, 1):.1f} | {se_design(v['item'], v['resid'], m, 3):.1f} | {se_design(v['item'], v['resid'], 3 * m, 1):.1f} |")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TITLES))
    args = ap.parse_args()
    md = ["# Consistency across repeated runs\n",
          "r0 is the merged run behind the headline results. Each repeat has its own cache, hence its own opening messages. Items, rules, translators and player are the same, apart from the opening rule noted on sequence and image.\n"]
    for task in args.tasks:
        md += section(task)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "consistency.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
