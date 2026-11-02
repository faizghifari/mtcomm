"""Figures for the short PoC summary (docs/poc_summary_short.md).

fig1_pooled.png          success per condition, pooled over the 4 pairs, with Wilson 95% intervals (one panel per task)
fig2_per_pair.png        the same per language pair (rows = tasks, columns = pairs); dashed line = ceiling
fig3_metrics.png         MT metrics vs task success: per-game AUC (bootstrap interval) and per-condition Spearman
fig4_sequence_scatter.png  condition success vs mean metric on sequence, the one task whose conditions can be ranked

Reads cells.json in the analysis directory (run scripts/analyze_main2.py first) and outputs/mtcomm/metrics3/*. Writes outputs/mtcomm/figures/*.png. Needs matplotlib, numpy and scipy.

Colours: the first three slots of the validated categorical palette (blue, orange, aqua; all-pairs CVD safe) and
neutral greys. Orange is used only for the Indonesian pivots in figures 1-2 and for the second metric in figures 3-4.
"""

import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mtcomm.languages import scored_lang
from mtcomm.paths import OUTPUT_ROOT
from mtcomm.poc_runs import ANALYSIS as D
from mtcomm.poc_runs import available

M = OUTPUT_ROOT / "metrics3"
FIG = OUTPUT_ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, REF, REF_LIGHT = "#0b0b0b", "#52514e", "#e3e2de", "#52514e", "#a9a8a3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK, "text.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

ROWS = [("ceiling", "Ceiling"), ("sol", "Sol"), ("flash-lite", "Flash-Lite"), ("google", "Google"), ("nllb", "NLLB"),
        ("qwen", "Qwen"), ("nllb-id", "NLLB via id"), ("google-id", "Google via id"), ("passthrough", "Passthrough"),
        ("prior", "Prior (no message)")]
TRANSLATORS = {"sol", "flash-lite", "google", "nllb", "qwen"}
TASKS = available()  # title, run, task


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def colour(cond):
    return BLUE if cond in TRANSLATORS else ORANGE if cond.endswith("-id") else REF


def style(ax, xlim=(0, 100)):
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def fig1(cells):
    panels = [(t[0], {c: (v["k"], v["n"]) for c, v in cells["pooled"][t[1]].items()}) for t in TASKS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for ax in list(axes.flat)[len(panels):-1]:
        ax.axis("off")
    for ax, (title, data) in zip(axes.flat, panels):
        style(ax)
        ce = data["ceiling"]
        lo, hi = wilson(*ce)
        ax.axvspan(100 * lo, 100 * hi, color=REF_LIGHT, alpha=0.18, lw=0)
        ax.axvline(100 * ce[0] / ce[1], color=REF, lw=1, ls=(0, (4, 3)))
        for i, (key, _) in enumerate(ROWS):
            if key not in data:
                continue
            k, n = data[key]
            lo, hi = wilson(k, n)
            col = colour(key)
            ax.plot([100 * lo, 100 * hi], [i, i], color=col, lw=1.6, solid_capstyle="round", alpha=0.55 if key != "ceiling" else 0.9)
            ax.plot(100 * k / n, i, "o", ms=6.5, mfc=col, mec="white", mew=1.2, zorder=3)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("success %")
    ax0 = axes.flat[0]
    ax0.set_yticks(range(len(ROWS)), [r[1] for r in ROWS])
    ax0.invert_yaxis()
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    ax = axes.flat[-1]
    ax.axis("off")
    ax.text(0, 0.98, "How to read", fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 0.90, "Dot: success rate pooled over the four pairs\n(n=120 per translator and pivot; culture 42).\nBar: Wilson 95% interval.\n"
            "Dashed line and band: the ceiling and its interval,\nwhich is another n=30-per-language condition,\nnot an upper bound.\n\n"
            "Blue: translators.  Orange: NLLB and Google\nvia Indonesian.  Grey: references.\n\n"
            "Image prior is invalid\n(the player asks instead of guessing; chance is 17%).",
            va="top", fontsize=8.5, color=INK2, linespacing=1.5, transform=ax.transAxes)
    fig.suptitle("Task success by condition, player Gemini 3.8 Flash, pooled over the four pairs", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig1_pooled.png", dpi=180)
    plt.close(fig)


def fig2(cells):
    rows_t = [(t[0], t[2], {k.split("|", 1)[1]: (round(v["success"] * v["n"]), v["n"]) for k, v in cells[t[1]].items()
                            if v["success"] is not None}) for t in TASKS]
    rows_t = [r for r in rows_t if any(k.split("|")[1] == "sol" for k in r[2])]  # a control with no translators has no pairs
    conds = [r for r in ROWS if r[0] not in ("ceiling", "prior")]
    fig, axes = plt.subplots(len(rows_t), 4, figsize=(13, 2.75 * len(rows_t)), sharex=True, sharey=True)
    for ri, (title, task, data) in enumerate(rows_t):
        pairs = sorted({k.split("|")[0] for k in data if k.split("|")[1] == "sol"})
        for ci in range(4):
            ax = axes[ri, ci]
            if ci >= len(pairs):
                ax.axis("off")
                continue
            p = pairs[ci]
            recv = scored_lang(task, p)
            style(ax)
            ce = data.get(f"{recv}|ceiling")
            if ce:
                ax.axvline(100 * ce[0] / ce[1], color=REF, lw=1, ls=(0, (4, 3)))
            for i, (key, _) in enumerate(conds):
                v = data.get(f"{p}|{key}")
                if not v:
                    continue
                ax.plot(100 * v[0] / v[1], i, "o", ms=6.5, mfc=colour(key), mec="white", mew=1.1, zorder=3)
            ax.set_xticks([0, 50, 100])
            label = (" asks ".join(p.split("-")) if task in ("mcq", "twentyq") else p.replace("-", " → ").replace(">", " → "))
            ax.set_title(label, fontsize=9, loc="left", pad=3)
            if ci == 0:
                ax.text(-0.02, 1.32, title, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left")
    axes[0, 0].set_yticks(range(len(conds)), [c[1] for c in conds])
    axes[0, 0].invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("success %")
    fig.suptitle("Task success per language pair. Dashed line = ceiling of the scored language.",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.998)
    fig.text(0.01, 0.976, "n=30 games per dot, 20–22 on culture (95% interval about ±17 points). a → b: a sends, b is scored. a asks b: a is scored, b holds the information. Orange: NLLB and Google via Indonesian.",
             fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.968), h_pad=2.2)
    fig.savefig(FIG / "fig2_per_pair.png", dpi=160)
    plt.close(fig)


def load_metric_games():
    scores = defaultdict(dict)
    for line in open(M / "comet_scores.jsonl"):
        r = json.loads(line)
        scores[r["record_id"]]["cometkiwi"] = r.get("cometkiwi_score")
    for p in sorted(M.glob("xcomet_v*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            scores[r["record_id"]]["xcomet"] = r.get("xcomet_score")
    for line in open(M / "gemba_scores.jsonl"):
        r = json.loads(line)
        scores[r["record_id"]]["gemba"] = r["gemba_norm"]
    ep = defaultdict(lambda: defaultdict(list))
    outcome = {}
    for line in open(M / "messages.jsonl"):
        m = json.loads(line)
        for metric, v in scores.get(m["seg"], {}).items():
            if v is not None:
                ep[(m["task"], m["episode"])][metric].append(v)
        outcome[(m["task"], m["episode"])] = bool(m["correct"])
    return ep, outcome


METRICS = [("cometkiwi", "CometKiwi", BLUE), ("xcomet", "XCOMET-XL", ORANGE), ("gemba", "GEMBA-MQM (LLM judge)", AQUA)]
TASK_ORDER = [("sequence", "Sequence"), ("route", "Route"), ("mcq", "MCQ"), ("twentyq", "Twenty Questions"), ("image", "Image"), ("culture", "Culture")]


def gap_ci(x, y, reps=400, seed=0):
    """Mean metric of successful minus failed games, with a bootstrap 95% interval over games."""
    x, y = np.asarray(x), np.asarray(y, bool)
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(reps):
        i = rng.integers(0, len(x), len(x))
        if 0 < y[i].sum() < len(i):
            gaps.append(x[i][y[i]].mean() - x[i][~y[i]].mean())
    return x[y].mean(), x[~y].mean(), np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)


def fig3(cells):
    """Per game: the mean metric of successful vs failed games, one dumbbell per task and metric."""
    ep, outcome = load_metric_games()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
    for ax, (m, name, col) in zip(axes, METRICS):
        style(ax, (0, 1))
        lo_all, hi_all = 1, 0
        for ti, (task, label) in enumerate(TASK_ORDER):
            ks = [k for k in ep if k[0] == task and ep[k].get(m)]
            x = [np.mean(ep[k][m]) for k in ks]
            y = [outcome[k] for k in ks]
            ok, bad, lo, hi = gap_ci(x, y)
            ax.plot([bad, ok], [ti, ti], color=GRID if False else REF_LIGHT, lw=2.2, solid_capstyle="round", zorder=1)
            ax.plot(bad, ti, "o", ms=8.5, mfc="white", mec=REF, mew=1.6, zorder=3)
            ax.plot(ok, ti, "o", ms=8.5, mfc=col, mec="white", mew=1.2, zorder=3)
            ax.text(max(ok, bad) + 0.02, ti, f"{ok - bad:+.2f}", va="center", fontsize=8.5, color=INK2)
            lo_all, hi_all = min(lo_all, ok, bad), max(hi_all, ok, bad)
        ax.set_xlim(max(0, lo_all - 0.08), min(1, hi_all + 0.14))
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(name, loc="left")
        ax.set_xlabel("mean metric score of a game's messages")
    axes[0].set_yticks(range(len(TASK_ORDER)), [t[1] for t in TASK_ORDER])
    axes[0].set_ylim(len(TASK_ORDER) - 0.5, -0.5)
    fig.suptitle("CometKiwi and XCOMET-XL score successful and failed games alike; the LLM judge separates them somewhat", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.9, "Open dot: average score in failed games.  Filled dot: average score in successful games.  If a metric predicted success, the filled dot "
             "would sit clearly to the right.\nNumber: the gap (successful minus failed). Scores run 0–1, higher = better; about 800–1,000 games per task (GEMBA: fewer, it scored a sample of messages).",
             fontsize=8.8, color=INK2, va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(FIG / "fig3_metric_gap.png", dpi=180)
    plt.close(fig)


def fig4(cells):
    """Sequence: rank the five translators by task success and by each metric; one line per translator."""
    summ = json.load(open(M / "summary.json"))
    names = {"sol": "Sol", "flash-lite": "Flash-Lite", "google": "Google", "nllb": "NLLB", "qwen": "Qwen"}
    pooled = cells["pooled"]["poc3_sequence"]
    success = {c: pooled[c]["k"] / pooled[c]["n"] for c in names}
    agg = {c: {m: [0, 0] for m, _, _ in METRICS} for c in names}
    for r in summ["cells"]:
        if r["task"] == "sequence" and r["cond"] in names:
            for m, _, _ in METRICS:
                if r.get(f"{m}_mean") is not None:
                    agg[r["cond"]][m][0] += r[f"{m}_mean"] * r["n_messages"]
                    agg[r["cond"]][m][1] += r["n_messages"]
    metric = {m: {c: agg[c][m][0] / agg[c][m][1] for c in names} for m, _, _ in METRICS}

    def ranks(d):
        order = sorted(d, key=lambda c: -d[c])
        return {c: i + 1 for i, c in enumerate(order)}
    rs = ranks(success)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, (m, name, col) in zip(axes, METRICS):
        rm = ranks(metric[m])
        for c in names:
            hl = c == "qwen"
            ax.plot([0, 1], [rs[c], rm[c]], color=ORANGE if hl else REF_LIGHT, lw=3.2 if hl else 1.8, alpha=1 if hl else 0.75, zorder=3 if hl else 2, solid_capstyle="round")
            for xx, r in ((0, rs[c]), (1, rm[c])):
                ax.plot(xx, r, "o", ms=8, mfc=ORANGE if hl else REF, mec="white", mew=1.1, zorder=4)
            ax.text(-0.06, rs[c], f"{names[c]}  {100 * success[c]:.0f}%", ha="right", va="center", fontsize=9, color=ORANGE if hl else INK, fontweight="bold" if hl else None)
            ax.text(1.06, rm[c], f"{names[c]}  {metric[m][c]:.2f}", ha="left", va="center", fontsize=9, color=ORANGE if hl else INK, fontweight="bold" if hl else None)
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(5.6, 0.4)
        ax.axis("off")
        ax.text(0, 0.0, "ranked by\ntask success", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(1, 0.0, f"ranked by\n{name}", ha="center", va="bottom", fontsize=9, color=INK2, fontweight="bold")
        ax.text(0.5, 5.75, "best at top", ha="center", va="top", fontsize=8, color=INK2) if m == "xcomet" else None
    fig.suptitle("Sequence ordering: does the metric order the translators like task success does?", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.9, "Rank 1 is best. Left of each panel: translators by pooled success rate. Right: the same translators by the metric's mean score. "
             "Parallel lines mean the metric agrees.\nGEMBA (an LLM judge) reproduces the success order exactly; CometKiwi and XCOMET-XL put Qwen, "
             "the translator with the lowest success, at or near the top.", fontsize=8.8, color=INK2, va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(FIG / "fig4_sequence_ranks.png", dpi=180)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells = json.load(open(D / "cells.json"))
    for f in (fig1, fig2, *((fig3, fig4) if (M / "summary.json").exists() else ())):
        f(cells)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
