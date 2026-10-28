"""Generate the route-game maps (Map-Task style; see mtcomm.routemap) and save them for reuse.

Usage: python scripts/route_build_maps.py [--easy 15] [--hard 15] [--seed 0]
Writes data/derived/route/maps_v1.json; per-language map images are drawn by data.load_route_items.
"""

import argparse
import json
import random

from mtcomm.paths import DATA_ROOT
from mtcomm.routemap import make_picture_map

OUT = DATA_ROOT / "derived" / "route" / "maps_v1.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy", type=int, default=15)
    ap.add_argument("--hard", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    maps = []
    for level, n in [("easy", args.easy), ("hard", args.hard)]:
        for k in range(n):
            m = make_picture_map(rng, level)
            maps.append(dict(m, map_id=f"{level[0]}{k + 1:02d}"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": 1, "seed": args.seed, "maps": maps}, open(OUT, "w"), indent=1)
    print(f"{len(maps)} maps -> {OUT}")


if __name__ == "__main__":
    main()
