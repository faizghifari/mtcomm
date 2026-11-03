"""Per-language-pair success tables for the PoC runs (appendix of the short PoC summary), over the first run and its repeats.

One compact table per task: rows = pair (scored language: mtcomm.languages.scored_lang), columns = conditions. Ceiling and prior are per
scored language, shared by the pairs that score it. Writes pair_breakdown.md in the analysis directory.
"""

from collections import defaultdict

from mtcomm.languages import scored_lang
from mtcomm.poc_runs import ANALYSIS as D
from mtcomm.poc_runs import all_games, available

CONDS = ["ceiling", "sol", "flash-lite", "google", "nllb", "qwen", "nllb-id", "google-id", "passthrough", "prior"]
HEAD = {"ceiling": "ceiling", "sol": "Sol", "flash-lite": "Flash-Lite", "google": "Google", "nllb": "NLLB", "qwen": "Qwen",
        "nllb-id": "NLLB via id", "google-id": "Google via id", "passthrough": "passthru", "prior": "prior"}


def main():
    md = ["# Per-pair results (player: 3.8 Flash)\n",
          "Success %, pooled over the first run and its two repeats: n=90 games per cell (30 items, 3 runs; 95% interval ±5–17 points, resampling items), culture 33–60 (up to ±24), passthrough 30 (first run only). "
          "Sequence, image and route: `a-b` means a sends and b, the receiver, is scored. MCQ and Twenty Questions: a asks and is scored, b holds the information, and both directions "
          "are translated. Ceiling and prior belong to the scored language.\n"]
    for title, run, task in available():
        cells = defaultdict(lambda: [0, 0])
        for g in all_games(run):
            cells[g["key"], g["cond"]][0] += g["ok"]
            cells[g["key"], g["cond"]][1] += 1
        pairs = sorted({key for key, cond in cells if cond not in ("ceiling", "prior")})
        conds = [x for x in CONDS if any(c == x for _, c in cells)]
        md.append(f"\n## {title}\n")
        md.append("| pair | " + " | ".join(HEAD[x] for x in conds) + " |")
        md.append("|---" * (len(conds) + 1) + "|")
        for p in pairs:
            recv = scored_lang(task, p)
            row = []
            for x in conds:
                k, n = cells.get((recv if x in ("ceiling", "prior") else p, x), (0, 0))
                row.append(f"{100 * k / n:.0f}" if n else "")
            md.append(f"| {p} | " + " | ".join(row) + " |")
    (D / "pair_breakdown.md").write_text("\n".join(md) + "\n")
    print(f"wrote {D / 'pair_breakdown.md'}")


if __name__ == "__main__":
    main()
