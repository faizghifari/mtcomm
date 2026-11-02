"""Per-language-pair success tables for the PoC runs (appendix of the short PoC summary).

One compact table per task: rows = pair (scored language: mtcomm.languages.scored_lang), columns =
conditions, cells = success % at n=30 games (ceiling and prior: n=30 per scored language, shared by the pairs
that score it). Reads cells.json in the analysis directory (run scripts/analyze_main2.py first) and writes
pair_breakdown.md there.
"""

import json

from mtcomm.languages import scored_lang
from mtcomm.poc_runs import ANALYSIS as D
from mtcomm.poc_runs import available

CONDS = ["ceiling", "sol", "flash-lite", "google", "nllb", "qwen", "nllb-id", "google-id", "passthrough", "prior"]
HEAD = {"ceiling": "ceiling", "sol": "Sol", "flash-lite": "Flash-Lite", "google": "Google", "nllb": "NLLB", "qwen": "Qwen",
        "nllb-id": "NLLB via id", "google-id": "Google via id", "passthrough": "passthru", "prior": "prior"}


def main():
    cells = json.load(open(D / "cells.json"))
    md = ["# Per-pair results (player: 3.8 Flash)\n",
          "Success %, n=30 games per cell (95% interval about ±17 points), culture 11–20 (up to ±28). Sequence, image and route: `a-b` means a sends and b, "
          "the receiver, is scored. MCQ and Twenty Questions: a asks and is scored, b holds the information, and both directions "
          "are translated. Ceiling and prior belong to the scored language.\n"]
    for title, run, _ in available():
        c = cells.get(run)
        if not c:
            continue
        keys = [k.split("|") for k in c]
        task = keys[0][0]
        pairs = sorted({p for _, p, cd in keys if cd not in ("ceiling", "prior")})
        conds = [x for x in CONDS if any(k[2] == x for k in keys)]
        md.append(f"\n## {title}\n")
        md.append("| pair | " + " | ".join(HEAD[x] for x in conds) + " |")
        md.append("|---" * (len(conds) + 1) + "|")
        for p in pairs:
            recv = scored_lang(task, p)
            row = []
            for x in conds:
                v = c.get(f"{task}|{recv if x in ('ceiling', 'prior') else p}|{x}")
                row.append("" if v is None or v["success"] is None else f"{100 * v['success']:.0f}")
            md.append(f"| {p} | " + " | ".join(row) + " |")
    (D / "pair_breakdown.md").write_text("\n".join(md) + "\n")
    print(f"wrote {D / 'pair_breakdown.md'}")


if __name__ == "__main__":
    main()
