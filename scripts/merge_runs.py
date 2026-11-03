"""Merge the runs behind each PoC task into one, outputs/mtcomm/poc3_<task>: the main run plus the extra cells played later. Also each
repeat, into rep_<task>_r<k>.

A condition goes in only when it has games on all the items of the task, so a pooled rate never mixes item sets (Qwen stays out of
route until its e16-e30 games exist). A game already present is not taken twice. Re-run after adding conditions.
"""

import json
from collections import defaultdict

from mtcomm.paths import OUTPUT_ROOT

NLLB = ("nllb", "nllb-id")
old = lambda r: r["cond"] not in NLLB  # noqa: E731  # played before the input fix in NLLBChannel; main3n_<task> replaces them
easy_old = lambda r: old(r) and r["item"]["item_id"].startswith("e")  # noqa: E731
# task: (run, filter); runs that do not exist yet are skipped
MERGES = {
    "poc3_mcq": [("main3_mcq", old), ("main3p_mcq", old), ("main3q_mcq", None), ("main3n_mcq", None)],
    "poc3_twentyq": [("main3_twentyq", old), ("main3p_twentyq", old), ("main3q_twentyq", None), ("main3n_twentyq", None)],
    "poc3_sequence": [("main2x_sequence", old), ("main3p_sequence", old), ("main3n_sequence", None)],
    "poc3_image": [("main2xf_image", old), ("main3p_image", old), ("main3n_image", None)],
    "poc3_route": [("main2x_route", easy_old), ("main3_route_e16", old), ("main3p_route", old), ("main3q_route", None), ("main3n_route", None)],
    "poc3_culture": [("main3_culture", old), ("main3p_culture", old), ("main3q_culture", None), ("main3n_culture", None)],
}


def repeats():
    """rep_<task>_r<k>: a repeat, with NLLB from main4n_ if the repeat was played before the fix."""
    out = {}
    for d in sorted(OUTPUT_ROOT.glob("main4_*_r[0-9]*")):
        fixed = OUTPUT_ROOT / d.name.replace("main4_", "main4n_")
        out[d.name.replace("main4_", "rep_")] = [(d.name, old), (fixed.name, None)] if (fixed / "records.jsonl").exists() else [(d.name, None)]
    return out


def main():
    for name, sources in {**MERGES, **repeats()}.items():
        rows, items, seen, used = [], defaultdict(set), set(), []
        for run, keep in sources:
            p = OUTPUT_ROOT / run / "records.jsonl"
            if not p.exists():
                continue
            used.append(run)
            for line in open(p):
                r = json.loads(line)
                if r["status"] == "done" and (keep is None or keep(r)) and r["episode"] not in seen:
                    seen.add(r["episode"])
                    rows.append(r)
                    items[r["cond"]].add(r["item"]["item_id"])
        if not rows:
            continue
        all_items = set().union(*items.values())
        dropped = {c: len(m) for c, m in items.items() if m != all_items}
        rows = [r for r in rows if r["cond"] not in dropped]
        out = OUTPUT_ROOT / name
        out.mkdir(exist_ok=True)
        with open(out / "records.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"sources": used, "items": len(all_items), "dropped_conditions": dropped}, open(out / "merge.json", "w"), indent=1)
        print(f"{name}: {len(rows)} games on {len(all_items)} items from {used}; dropped for missing items: {dropped}")


if __name__ == "__main__":
    main()
