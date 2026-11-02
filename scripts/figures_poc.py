"""Figures for the short PoC summary (docs/poc_summary_short.md).

fig1_pooled.png          success per condition, pooled over the 4 pairs, with Wilson 95% intervals (one panel per task)
fig2_per_pair.png        the same per language pair (rows = tasks, columns = pairs); dashed line = ceiling
fig3_metrics.png         MT metrics vs task success: per-game AUC (bootstrap interval) and per-condition Spearman
fig4_sequence_scatter.png  condition success vs mean metric on sequence, the one task whose conditions can be ranked

Reads outputs/mtcomm/analysis_main2/cells.json (run scripts/analyze_main2.py first), main2x_route records and
outputs/mtcomm/metrics2/*. Writes outputs/mtcomm/figures/*.png. Needs matplotlib, numpy and scipy.

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
from scipy.stats import rankdata

from mtcomm.paths import OUTPUT_ROOT

D = OUTPUT_ROOT / "analysis_main2"
M = OUTPUT_ROOT / "metrics2"
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
TASKS = [("MCQ conversation", "main2x_mcq", "mcq", "Flash-Lite"), ("Twenty Questions", "main2x_twentyq", "twentyq", "Flash-Lite"),
         ("Sequence ordering", "main2x_sequence", "sequence", "3.8 Flash"), ("Image near-twin", "main2xf_image", "image", "3.8 Flash"),
         ("Route, easy + hard maps", "main2x_route", "route", "3.8 Flash"), ("Culture (BLEnD)", "main2x_culture", "culture", "Flash-Lite")]


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


def route_easy():
    """Pooled and per-pair counts (k, n) for the easy route maps; ceiling/prior weighted by pairs that share them."""
    cell = defaultdict(lambda: [0, 0])
    for line in open(OUTPUT_ROOT / "main2x_route" / "records.jsonl"):
        r = json.loads(line)
        if r["status"] != "done" or r["score"].get("level") != "easy":
            continue
        scope = r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"]
        c = cell[(scope, r["cond"])]
        c[0] += bool(r["score"].get("correct"))
        c[1] += 1
    pairs = sorted({s for s, c in cell if c == "sol"})
    pooled = defaultdict(lambda: [0, 0])
    for (s, c), (k, n) in cell.items():
        w = sum(p.split("-")[-1] == s for p in pairs) if c in ("ceiling", "prior") else 1
        pooled[c][0] += w * k
        pooled[c][1] += w * n
    return cell, pooled


def fig1(cells):
    pooled = {t[1]: {c: tuple(v.values()) for c, v in cells["pooled"][t[1]].items()} for t in TASKS}
    easy_cell, easy_pooled = route_easy()
    panels = [(t[0], f"player: {t[3]}", {c: (v["k"], v["n"]) for c, v in cells["pooled"][t[1]].items()}) for t in TASKS]
    panels.insert(5, ("Route, easy maps only", "player: 3.8 Flash", {c: tuple(v) for c, v in easy_pooled.items()}))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for ax, (title, sub, data) in zip(axes.flat, panels):
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
        ax.set_title(title, loc="left", pad=17)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", ha="left")
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
    ax.text(0, 0.90, "Dot: success rate pooled over the four pairs\n(n=120 per translator, 60 per pivot).\nBar: Wilson 95% interval.\n"
            "Dashed line and band: the ceiling and its interval,\nwhich is another n=30-per-language condition,\nnot an upper bound.\n\n"
            "Blue: translators.  Orange: Indonesian pivots\n(su↔jv only).  Grey: references.\n\n"
            "Image prior is invalid for 3.8 Flash\n(it asks instead of guessing; chance is 17%).",
            va="top", fontsize=8.5, color=INK2, linespacing=1.5, transform=ax.transAxes)
    fig.suptitle("Task success by condition, pooled over su→jv, jv→su, su→ko, ko→su", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig1_pooled.png", dpi=180)
    plt.close(fig)


def fig2(cells):
    easy_cell, _ = route_easy()
    rows_t = [(t[0], t[3], {k.split("|", 1)[1]: (round(v["success"] * v["n"]), v["n"]) for k, v in cells[t[1]].items()
                           if v["success"] is not None}) for t in TASKS]
    easy = ("Route, easy maps only (n=15)", "3.8 Flash", {f"{s}|{c}": tuple(v) for (s, c), v in easy_cell.items()})
    rows_t.insert(5, (easy[0], easy[1], {k: (v[0], v[1]) for k, v in easy[2].items()}))
    conds = [r for r in ROWS if r[0] not in ("ceiling", "prior")]
    fig, axes = plt.subplots(len(rows_t), 4, figsize=(13, 16.5), sharex=True, sharey=True)
    for ri, (title, player, data) in enumerate(rows_t):
        pairs = sorted({k.split("|")[0] for k in data if k.split("|")[1] == "sol"})
        for ci in range(4):
            ax = axes[ri, ci]
            if ci >= len(pairs):
                ax.axis("off")
                continue
            p = pairs[ci]
            recv = p.split(">")[-1].split("-")[-1]
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
            ax.set_title(p.replace("-", " → ").replace(">", " → "), fontsize=9, loc="left", pad=3)
            if ci == 0:
                ax.text(-0.02, 1.32, f"{title}  ({player})", transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left")
    axes[0, 0].set_yticks(range(len(conds)), [c[1] for c in conds])
    axes[0, 0].invert_yaxis()
    for ax in axes[-1]:
        ax.set_xlabel("success %")
    fig.suptitle("Task success per language pair (sender → receiver; the receiver is scored). Dashed line = ceiling of the receiving language.",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.998)
    fig.text(0.01, 0.976, "n=30 games per dot (95% interval about ±17 points; ±25 for the easy-route rows). Orange: Indonesian pivots (su↔jv only).",
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


def auc_boot(x, y, reps=400, seed=0):
    x, y = np.asarray(x), np.asarray(y, bool)

    def auc(xs, ys):
        r = rankdata(xs)
        n1, n0 = ys.sum(), (~ys).sum()
        return (r[ys].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(x), len(x))
        if 0 < y[i].sum() < len(i):
            bs.append(auc(x[i], y[i]))
    return auc(x, y), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


METRICS = [("cometkiwi", "CometKiwi", BLUE), ("xcomet", "XCOMET-XL", ORANGE), ("gemba", "GEMBA-MQM (LLM judge)", AQUA)]


def fig3(cells):
    ep, outcome = load_metric_games()
    summ = json.load(open(M / "summary.json"))
    order = ["sequence", "route", "mcq", "twentyq", "image", "culture"]
    label = {"sequence": "Sequence", "route": "Route", "mcq": "MCQ", "twentyq": "Twenty Questions", "image": "Image", "culture": "Culture"}
    rel = {t[2]: cells["reliability"][t[1]]["reliability"] for t in TASKS}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    off = {0: -0.24, 1: 0, 2: 0.24}
    for ti, task in enumerate(order):
        keys = [k for k in ep if k[0] == task]
        for mi, (m, name, col) in enumerate(METRICS):
            ks = [k for k in keys if ep[k].get(m)]
            x = [np.mean(ep[k][m]) for k in ks]
            y = [outcome[k] for k in ks]
            a, lo, hi = auc_boot(x, y)
            assert abs(a - summ["auc_success_vs_episode_mean"][f"{task}/{m}"]) < 0.002, (task, m, a)
            yy = ti + off[mi]
            a1.plot([lo, hi], [yy, yy], color=col, lw=1.6, alpha=0.6, solid_capstyle="round")
            a1.plot(a, yy, "o", ms=6.5, mfc=col, mec="white", mew=1.1, zorder=3, label=name if ti == 0 else None)
            rho, n = summ["cell_spearman_vs_success"][f"{task}/{m}"]
            z, se = math.atanh(rho), math.sqrt((1 + rho * rho / 2) / (n - 3))
            a2.plot([math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)], [yy, yy], color=col, lw=1.6, alpha=0.6, solid_capstyle="round")
            a2.plot(rho, yy, "o", ms=6.5, mfc=col, mec="white", mew=1.1, zorder=3)
    for ax, xl, title, chance in ((a1, (0.2, 0.8), "Per game: does a game's mean metric separate\nsuccessful from failed games?", 0.5),
                                  (a2, (-1, 1), "Per condition: does a condition's mean metric\nrank conditions like their success rate?", 0)):
        style(ax, xl)
        ax.axvline(chance, color=REF, lw=1, ls=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_ylim(len(order) - 0.45, -0.55)
    a1.set_xlabel("AUC (0.5 = no signal)")
    a2.set_xlabel("Spearman ρ (24 conditions; 17 for culture)")
    a1.set_yticks(range(len(order)), [label[t] for t in order])
    for ti, task in enumerate(order):
        if rel[task] < 0.3:
            a2.axhspan(ti - 0.5, ti + 0.5, color=REF_LIGHT, alpha=0.16, lw=0)
        a2.text(1.02, ti, f"reliability {rel[task]:.2f}" + ("" if rel[task] >= 0.3 else "\nconditions not rankable"), transform=a2.get_yaxis_transform(),
                va="center", fontsize=8, color=INK2)
    fig.legend(*a1.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=3, frameon=False, fontsize=9)
    fig.text(0.99, 0.905, "Bars: 95% intervals (bootstrap over games; Fisher approximation over conditions).",
             ha="right", fontsize=8.5, color=INK2)
    fig.suptitle("Standard MT metrics do not predict task success; only the LLM judge tracks it, and only where conditions can be ranked",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.88))
    fig.savefig(FIG / "fig3_metrics.png", dpi=180)
    plt.close(fig)


def fig4(cells):
    summ = json.load(open(M / "summary.json"))
    seq = cells["main2x_sequence"]
    pts = [(r, seq.get(f"sequence|{r['pair']}|{r['cond']}")) for r in summ["cells"] if r["task"] == "sequence"]
    pts = [(r, c["success"] * 100) for r, c in pts if c and c["success"] is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, (m, name, col) in zip(axes, METRICS):
        xs = np.array([r[f"{m}_mean"] for r, _ in pts])
        ys = np.array([y for _, y in pts])
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        qwen = np.array([r["cond"] == "qwen" for r, _ in pts])
        ax.plot(xs[~qwen], ys[~qwen], "o", ms=7, mfc=col, mec="white", mew=1.1, alpha=0.9)
        ax.plot(xs[qwen], ys[qwen], "o", ms=9, mfc="none", mec=INK, mew=1.4)
        ax.plot(xs[qwen], ys[qwen], "o", ms=7, mfc=col, mec="white", mew=1.1)
        rho = summ["cell_spearman_vs_success"][f"sequence/{m}"][0]
        ax.set_title(f"{name}:  ρ = {rho:+.2f}", loc="left")
        ax.set_xlabel("mean metric over the condition's messages")
    axes[0].set_ylabel("sequence success % (n=30)")
    fig.suptitle("Sequence ordering, the one task whose conditions can be ranked (reliability 0.73): one dot per translator × pair",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.90, "Ringed dots: Qwen as translator (57–63% success). Their metric scores sit inside the range of the best translators, which is why the neural metrics miss them.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig4_sequence_scatter.png", dpi=180)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells = json.load(open(D / "cells.json"))
    for f in (fig1, fig2, fig3, fig4):
        f(cells)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
