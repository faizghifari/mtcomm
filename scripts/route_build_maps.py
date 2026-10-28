"""Generate the route-game maps (Map-Task style; see mtcomm.routemap) and save them for reuse.

Usage: python scripts/route_build_maps.py [--easy 15] [--hard 15] [--seed 0]
       python scripts/route_build_maps.py --v2   (keep v1's easy maps, draw new hard maps hb01.. -> maps_v2.json)
Writes data/derived/route/maps_v1.json (or v2); per-language map images are drawn by data.load_route_items.
New hard maps get new ids because map images are cached by id and labels.
"""

import argparse
import json
import random

from mtcomm.paths import DATA_ROOT
from mtcomm.routemap import make_picture_map

OUT = DATA_ROOT / "derived" / "route" / "maps_v1.json"
OUT_V2 = DATA_ROOT / "derived" / "route" / "maps_v2.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy", type=int, default=15)
    ap.add_argument("--hard", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.v2:
        maps = [m for m in json.load(open(OUT))["maps"] if m["level"] == "easy"]
        maps += [dict(make_picture_map(rng, "hard"), map_id=f"hb{k + 1:02d}") for k in range(args.hard)]
        json.dump({"version": 2, "seed": args.seed, "maps": maps}, open(OUT_V2, "w"), indent=1)
        print(f"{len(maps)} maps -> {OUT_V2}")
        return
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
