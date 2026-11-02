"""Generate the street maps for the directions game (mtcomm.games.wayfind).

Usage: python scripts/wayfind_build_maps.py [--easy 30] [--mid 30] [--hard 30] [--seed 0]
Writes data/derived/wayfind/maps_v1.json; per-language images are drawn by data.load_wayfind_items.
"""

import argparse
import json
from collections import Counter

from mtcomm.paths import DATA_ROOT
from mtcomm.streetmap import build_maps

OUT = DATA_ROOT / "derived" / "wayfind" / "maps_v1.json"


def main():
    ap = argparse.ArgumentParser()
    for level in ("easy", "mid", "hard"):
        ap.add_argument(f"--{level}", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    maps = build_maps(args.seed, {"easy": args.easy, "mid": args.mid, "hard": args.hard})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": 1, "seed": args.seed, "maps": maps}, open(OUT, "w"), indent=1)
    kinds = Counter(c["kind"] for m in maps for c in m["candidates"].values())
    print(f"{len(maps)} maps -> {OUT}; candidate kinds {dict(kinds)}; landmarks per map "
          f"{min(len(m['landmarks']) for m in maps)}-{max(len(m['landmarks']) for m in maps)}")


if __name__ == "__main__":
    main()
