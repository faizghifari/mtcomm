"""Per-language-pair success tables for the main2 runs (appendix of the short PoC summary).

One compact table per task: rows = pair (scored language: mtcomm.languages.scored_lang), columns =
conditions, cells = success % at n=30 games (ceiling and prior: n=30 per scored language, shared by the pairs
that score it). Route also gets an easy-maps-only table (n=15 per cell). Reads
outputs/mtcomm/analysis_main2/cells.json (run scripts/analyze_main2.py first) and the route records.
Writes outputs/mtcomm/analysis_main2/pair_breakdown.md.
"""

import json

from mtcomm.languages import scored_lang
from mtcomm.paths import OUTPUT_ROOT

D = OUTPUT_ROOT / "analysis_main2"
TASKS = [("MCQ conversation", "main2x_mcq", "Flash-Lite"), ("Twenty Questions", "main2x_twentyq", "Flash-Lite"),
         ("Sequence ordering", "main2x_sequence", "3.8 Flash"), ("Image near-twin", "main2xf_image", "3.8 Flash"),
         ("Route (easy+hard maps)", "main2x_route", "3.8 Flash"), ("Culture (BLEnD)", "main2x_culture", "Flash-Lite")]
CONDS = ["ceiling", "sol", "flash-lite", "google", "nllb", "qwen", "nllb-id", "google-id", "passthrough", "prior"]
HEAD = {"ceiling": "ceiling", "sol": "Sol", "flash-lite": "Flash-Lite", "google": "Google", "nllb": "NLLB", "qwen": "Qwen",
        "nllb-id": "NLLB via id", "google-id": "Google via id", "passthrough": "passthru", "prior": "prior"}


def route_easy():
    """Route success on the easy maps only (the design going forward): 15 items per cell."""
    cell = {}
    for line in open(OUTPUT_ROOT / "main2x_route" / "records.jsonl"):
        r = json.loads(line)
        if r["status"] != "done" or r["score"].get("level") != "easy":
            continue
        scope = r["eval_lang"] if r["cond"] in ("ceiling", "prior") else r["pair"]
        k = cell.setdefault((scope, r["cond"]), [0, 0])
        k[0] += 1
        k[1] += bool(r["score"].get("correct"))
    conds = [x for x in CONDS if any(k[1] == x for k in cell)]
    out = ["\n## Route, easy maps only (player: 3.8 Flash; n=15 per cell, interval about ±25 points)\n",
           "| pair | " + " | ".join(HEAD[x] for x in conds) + " |", "|---" * (len(conds) + 1) + "|"]
    for p in sorted({k[0] for k in cell if k[1] == "sol"}):
        row = []
        for x in conds:
            v = cell.get((p.split("-")[-1] if x in ("ceiling", "prior") else p, x))
            row.append("" if not v else f"{100 * v[1] / v[0]:.0f}")
        out.append(f"| {p} | " + " | ".join(row) + " |")
    return out


def main():
    cells = json.load(open(D / "cells.json"))
    md = ["# Per-pair results (main2)\n",
          "Success %, n=30 games per cell (95% interval about ±17 points). Sequence, image and route: `a-b` means a sends and b, "
          "the receiver, is scored. MCQ and Twenty Questions: a asks and is scored, b holds the information, and both directions "
          "are translated. Ceiling and prior belong to the scored language. Pivots exist for su↔jv only.\n"]
    for title, run, player in TASKS:
        c = cells.get(run)
        if not c:
            continue
        keys = [k.split("|") for k in c]
        task = keys[0][0]
        pairs = sorted({p for _, p, cd in keys if cd not in ("ceiling", "prior")})
        conds = [x for x in CONDS if any(k[2] == x for k in keys)]
        md.append(f"\n## {title} (player: {player})\n")
        md.append("| pair | " + " | ".join(HEAD[x] for x in conds) + " |")
        md.append("|---" * (len(conds) + 1) + "|")
        for p in pairs:
            recv = scored_lang(task, p)
            row = []
            for x in conds:
                v = c.get(f"{task}|{recv if x in ('ceiling', 'prior') else p}|{x}")
                row.append("" if v is None or v["success"] is None else f"{100 * v['success']:.0f}")
            md.append(f"| {p} | " + " | ".join(row) + " |")
    md += route_easy()
    (D / "pair_breakdown.md").write_text("\n".join(md) + "\n")
    print(f"wrote {D / 'pair_breakdown.md'}")


if __name__ == "__main__":
    main()
