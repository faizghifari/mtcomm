"""Fixed contrasts and variance components for a task's runs (repeats pooled).

Games in different conditions share items, so comparisons are paired within (pair, item, repeat):
  - fixed contrast list, decided in advance: exact McNemar with Holm correction over the list, plus the odds ratio from a
    conditional logistic model (success ~ condition, strata = pair x item), left blank when the data are separated;
  - variance components: a Bayesian mixed logistic model (success ~ translator + (1|item) + (1|pair) + (1|repeat)),
    posterior mean SD on the logit scale, showing how much item, pair and repeat move the outcome.

Usage: python scripts/mixed_effects.py main2x_sequence [main2x_sequence_r1 ...]   (one task per call; appends to --out)
       python scripts/mixed_effects.py --all    (one section per PoC task, mtcomm.poc_runs)
Writes mixed_effects.md in the analysis directory by default.
"""

import argparse
import json
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from mtcomm.paths import OUTPUT_ROOT
from mtcomm.poc_runs import ANALYSIS, available

TRANSLATORS = ["sol", "flash-lite", "google", "nllb", "qwen"]
CONTRASTS = [("sol", "nllb"), ("sol", "qwen"), ("sol", "google"), ("flash-lite", "nllb"), ("flash-lite", "qwen"),
             ("sol", "ceiling"), ("flash-lite", "ceiling"), ("google", "ceiling"), ("nllb", "ceiling"), ("qwen", "ceiling")]


def load(runs):
    """One row per game: repeat, pair, item, cond, success. A ceiling game is copied to every pair sharing its language."""
    rows, ceiling = [], []
    for rep, run in enumerate(runs):
        pairs = set()
        for line in open(OUTPUT_ROOT / run / "records.jsonl"):
            r = json.loads(line)
            if r["status"] != "done":
                continue
            row = {"rep": rep, "pair": r["pair"], "item": r["item"]["item_id"], "cond": r["cond"], "ok": int(bool(r["score"].get("correct"))), "recv": r["eval_lang"]}
            if r["cond"] == "ceiling":
                ceiling.append(row)
            elif r["cond"] in TRANSLATORS:
                rows.append(row)
                pairs.add((rep, r["pair"], r["eval_lang"]))
        for (rp, pair, recv) in pairs:
            rows += [dict(c, pair=pair) for c in ceiling if c["rep"] == rp and c["recv"] == recv]
    return pd.DataFrame(rows)


def mcnemar(b, c):
    n = b + c
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = np.argsort(ps)
    adj, top = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        top = max(top, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = top
    return adj


def contrasts(df):
    wide = df.pivot_table(index=["rep", "pair", "item"], columns="cond", values="ok", aggfunc="first")
    rows = []
    for a, b in CONTRASTS:
        if a not in wide or b not in wide:
            continue
        w = wide[[a, b]].dropna()
        x, y = int(((w[a] == 1) & (w[b] == 0)).sum()), int(((w[a] == 0) & (w[b] == 1)).sum())
        rows.append({"a": a, "b": b, "n": len(w), "diff": 100 * (x - y) / max(1, len(w)), "x": x, "y": y, "p": mcnemar(x, y)})
    for r, h in zip(rows, holm([r["p"] for r in rows])):
        r["holm"] = h
    return rows


def conditional_or(df):
    """Odds ratio of each condition relative to the ceiling, conditional on pair x item x repeat."""
    d = df[df["cond"].isin(TRANSLATORS + ["ceiling"])].copy()
    d["stratum"] = d["rep"].astype(str) + "|" + d["pair"] + "|" + d["item"]
    X = pd.get_dummies(d["cond"]).drop(columns="ceiling").astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConditionalLogit(d["ok"], X, groups=d["stratum"]).fit(disp=0, maxiter=200)
        return m.params, m.cov_params()
    except Exception:
        return None, None


def logit_or(params, cov, a, b):
    if params is None:
        return ""
    va = params.get(a, 0.0)
    vb = params.get(b, 0.0)
    v = va - vb
    var = sum(cov.loc[i, j] * si * sj for i, si in ((a, 1), (b, -1)) if i in cov for j, sj in ((a, 1), (b, -1)) if j in cov)
    if abs(v) > 6 or not np.isfinite(var) or var <= 0 or var > 25:  # separated: the estimate runs away
        return "separated"
    return f"{math.exp(v):.2f} ({math.exp(v - 1.96 * math.sqrt(var)):.2f}–{math.exp(v + 1.96 * math.sqrt(var)):.2f})"


def variance_components(df):
    d = df[df["cond"].isin(TRANSLATORS)].copy()
    d["pair_item"] = d["pair"] + "|" + d["item"]
    vc = {"item": "0 + C(pair_item)", "pair": "0 + C(pair)"}
    if d["rep"].nunique() > 1:
        d["rep_s"] = d["rep"].astype(str)
        vc["repeat"] = "0 + C(rep_s)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = BinomialBayesMixedGLM.from_formula("ok ~ C(cond)", vc, d).fit_vb()
    return {name: float(np.exp(fit.vcp_mean[i])) for i, name in enumerate(fit.model.vcp_names)}


def section(name, runs):
    df = load(runs)
    params, cov = conditional_or(df)
    md = [f"\n## {name} ({', '.join(runs)}; {df['rep'].nunique()} run(s))\n",
          "| contrast | n | diff (points) | a wins / b wins | p | Holm p | odds ratio (95% CI) |", "|---|---|---|---|---|---|---|"]
    for r in contrasts(df):
        md.append(f"| {r['a']} vs {r['b']} | {r['n']} | {r['diff']:+.1f} | {r['x']} / {r['y']} | {r['p']:.3f} | {r['holm']:.3f} | {logit_or(params, cov, r['a'], r['b'])} |")
    vc = variance_components(df)
    md.append("\nVariance components, SD on the logit scale: " + ", ".join(f"{k} {v:.2f}" for k, v in vc.items()) + ".")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ANALYSIS / "mixed_effects.md"))
    args = ap.parse_args()
    md = ["# Fixed contrasts and variance components\n",
          "Paired within pair x item x repeat. Holm is over the task's fixed contrast list. The ceiling game of a language is shared by the pairs that end in it, so vs-ceiling tests are slightly anti-conservative. Variance components come from a variational Bayes fit and are rough.\n"]
    for runs in ([[t[1]] for t in available()] if args.all else [args.runs]):
        if load(runs)["cond"].isin(TRANSLATORS).any():  # a control with no translators has nothing to contrast
            md += section(runs[0], runs)
    open(args.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
