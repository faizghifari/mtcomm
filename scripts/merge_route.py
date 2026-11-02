"""Merge the easy route games into one run, outputs/mtcomm/poc3_route: main2x_route's easy maps plus main3_route_e16.

Keeps a condition only if it has games on all the maps, so a pooled rate never mixes map sets (Qwen stays out until its
e16-e30 games exist). Re-run after adding conditions.
"""

import json
from collections import defaultdict

from mtcomm.paths import OUTPUT_ROOT

SOURCES = [("main2x_route", lambda r: r["item"]["item_id"].startswith("e")), ("main3_route_e16", lambda r: True)]
OUT = OUTPUT_ROOT / "poc3_route"


def main():
    rows, maps = [], defaultdict(set)
    for run, keep in SOURCES:
        for line in open(OUTPUT_ROOT / run / "records.jsonl"):
            r = json.loads(line)
            if r["status"] == "done" and keep(r):
                rows.append(r)
                maps[r["cond"]].add(r["item"]["item_id"])
    all_maps = set().union(*maps.values())
    dropped = {c: len(m) for c, m in maps.items() if m != all_maps}
    rows = [r for r in rows if r["cond"] not in dropped]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "records.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    spend = sum(json.load(open(OUTPUT_ROOT / run / "summary.json"))["spend_total_usd"] for run, _ in SOURCES)
    json.dump({"sources": [s[0] for s in SOURCES], "maps": len(all_maps), "dropped_conditions": dropped}, open(OUT / "merge.json", "w"), indent=1)
    print(f"{len(rows)} games on {len(all_maps)} maps; dropped for missing maps: {dropped}; sources spent ${spend:.2f} in all")


if __name__ == "__main__":
    main()
