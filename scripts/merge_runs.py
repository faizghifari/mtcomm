"""Merge the runs behind each PoC task into one, outputs/mtcomm/poc3_<task>: the main run plus the extra cells played later.

A condition goes in only when it has games on all the items of the task, so a pooled rate never mixes item sets (Qwen stays out of
route until its e16-e30 games exist). A game already present is not taken twice. Re-run after adding conditions.
"""

import json
from collections import defaultdict

from mtcomm.paths import OUTPUT_ROOT

easy = lambda r: r["item"]["item_id"].startswith("e")  # noqa: E731
# task: (run, filter); runs that do not exist yet are skipped
MERGES = {
    "mcq": [("main3_mcq", None), ("main3p_mcq", None), ("main3q_mcq", None)],
    "twentyq": [("main3_twentyq", None), ("main3p_twentyq", None), ("main3q_twentyq", None)],
    "sequence": [("main2x_sequence", None), ("main3p_sequence", None)],
    "image": [("main2xf_image", None), ("main3p_image", None)],
    "route": [("main2x_route", easy), ("main3_route_e16", None), ("main3p_route", None), ("main3q_route", None)],
    "culture": [("main3_culture", None), ("main3p_culture", None), ("main3q_culture", None)],
}


def main():
    for task, sources in MERGES.items():
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
        out = OUTPUT_ROOT / f"poc3_{task}"
        out.mkdir(exist_ok=True)
        with open(out / "records.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"sources": used, "items": len(all_items), "dropped_conditions": dropped}, open(out / "merge.json", "w"), indent=1)
        print(f"{task}: {len(rows)} games on {len(all_items)} items from {used}; dropped for missing items: {dropped}")


if __name__ == "__main__":
    main()
