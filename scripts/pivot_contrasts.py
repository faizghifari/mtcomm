"""Does routing through Indonesian help? Pivot vs direct, same system, same item, per task -> analysis_poc3/pivot.md

Exact McNemar, Holm over the 12 pooled contrasts (6 tasks x NLLB, Google). This list is its own family, separate from the translator
contrasts in mixed_effects.py. su<->jv and su<->ko are shown separately, because Indonesian is close to Sundanese and Javanese and not
to Korean.

Sequence and image parents predate the rule that re-asks a refused opening, and the su<->ko pivot games were played under it. So a
game whose direct version opened with a refusal favours the pivot. The last column repeats the contrast without those items.
"""

import json
from collections import defaultdict

from scipy.stats import binomtest

from mtcomm.paths import OUTPUT_ROOT
from mtcomm.players import NOT_UNDERSTOOD, is_not_understood
from mtcomm.poc_runs import ANALYSIS, available

SYSTEMS = [("nllb", "NLLB"), ("google", "Google")]


def family(pair):
    """su<->jv or su<->ko; culture pairs are named culture>visitor."""
    if ">" in pair:
        return "su↔jv" if pair.endswith(">jv") else "su↔ko"
    return "su↔jv" if "jv" in pair else "su↔ko"


def refused_opening(r):
    msgs = [t for t in r["turns"] if t.get("kind") == "message" and t.get("source")]
    return bool(msgs) and any(is_not_understood(msgs[0]["source"], k) for k in NOT_UNDERSTOOD)


def load(run):
    games, opened_badly = {}, set()
    for line in open(OUTPUT_ROOT / run / "records.jsonl"):
        r = json.loads(line)
        if r["cond"] in ("ceiling", "prior"):
            continue
        key = (r["pair"], r["item"]["item_id"])
        games[(r["cond"], *key)] = int(bool(r["score"].get("correct")))
        if r["cond"] in ("nllb", "google") and refused_opening(r):
            opened_badly.add((r["cond"], *key))
    return games, opened_badly


def contrast(games, sys_, keep):
    """(n, direct wins, pivot wins, direct %, pivot %) over the items present in both."""
    a = b = n = d_ok = p_ok = 0
    for (cond, pair, item), v in games.items():
        if cond != sys_ or not keep(pair, item):
            continue
        w = games.get((f"{sys_}-id", pair, item))
        if w is None:
            continue
        n += 1
        d_ok += v
        p_ok += w
        a += v > w
        b += w > v
    return n, a, b, 100 * d_ok / n if n else None, 100 * p_ok / n if n else None


def mcnemar(a, b):
    return binomtest(a, a + b, 0.5).pvalue if a + b else 1.0


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, top = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        top = max(top, min(1.0, ps[i] * (len(ps) - rank)))
        out[i] = top
    return out


def main():
    rows, ps = [], []
    for title, run, task in available():
        games, bad = load(run)
        for sys_, name in SYSTEMS:
            n, a, b, d, p = contrast(games, sys_, lambda pair, item: True)
            if not n:
                continue
            ps.append(mcnemar(a, b))
            rows.append((title, name, sys_, games, bad, n, a, b, d, p))
    adj = holm(ps)
    fmt = lambda d, p: f"{d:.0f} → {p:.0f}" if d is not None else "–"  # noqa: E731
    md = ["# Pivot through Indonesian vs direct\n",
          "Same system, same item. a → b: success % direct → via Indonesian. Direct / pivot wins = games only that version solved. "
          "Holm over the 12 pooled contrasts. Without opening refusals: su↔ko, leaving out items whose direct game opened with a refusal.\n",
          "| task | system | pooled (n) | direct / pivot wins | p | Holm p | su↔jv | su↔ko | su↔ko, without opening refusals (n) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (title, name, sys_, games, bad, n, a, b, d, p), raw, h in zip(rows, ps, adj):
        _, _, _, jd, jp = contrast(games, sys_, lambda pair, item: family(pair) == "su↔jv")
        _, _, _, kd, kp = contrast(games, sys_, lambda pair, item: family(pair) == "su↔ko")
        cn, _, _, cd, cp = contrast(games, sys_, lambda pair, item: family(pair) == "su↔ko" and (sys_, pair, item) not in bad)
        md.append(f"| {title} | {name} | {fmt(d, p)} ({n}) | {a} / {b} | {raw:.3f} | {h:.3f} | {fmt(jd, jp)} | {fmt(kd, kp)} | {fmt(cd, cp)} ({cn}) |")
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / "pivot.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
