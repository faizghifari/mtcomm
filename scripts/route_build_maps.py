"""Generate the route-game maps (Map-Task style; see mtcomm.routemap) and save them for reuse.

Usage: python scripts/route_build_maps.py [--easy 15] [--hard 15] [--seed 0]
       python scripts/route_build_maps.py --v2   (keep v1's easy maps, draw new hard maps hb01.. -> maps_v2.json)
       python scripts/route_build_maps.py --v3 --easy 45 --seed 3  (keep every v2 map, append new easy maps e16.. -> maps_v3.json)
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
OUT_V3 = DATA_ROOT / "derived" / "route" / "maps_v3.json"


def route_signature(m):
    return tuple(m["giver_landmarks"][r["landmark"]]["name"] for r in m["route"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy", type=int, default=15)
    ap.add_argument("--hard", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v3", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.v3:
        old = json.load(open(OUT_V2))["maps"]
        seen = {route_signature(m) for m in old}
        first = sum(m["level"] == "easy" for m in old) + 1
        new = []
        while len(new) < args.easy:
            m = make_picture_map(rng, "easy")
            if route_signature(m) not in seen:  # no two maps share a route
                seen.add(route_signature(m))
                new.append(dict(m, map_id=f"e{first + len(new):02d}"))
        json.dump({"version": 3, "seed": args.seed, "maps": old + new}, open(OUT_V3, "w"), indent=1)
        print(f"{len(old)} old + {len(new)} new easy maps -> {OUT_V3}")
        return
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
