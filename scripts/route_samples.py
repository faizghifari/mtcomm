"""Prototype synthetic route-redrawing maps, rendered for design review (not yet a game).

Two representations of the same idea (a Map-Task-style route between two players whose maps differ):
  picture  a 2-D map with labelled landmarks; the giver's copy shows the route; the follower answers
           with the ordered list of landmarks the route passes (and on which side).
  grid     a text grid without coordinates; the giver's copy marks the route; the follower answers
           with the path as moves from the start cell S.
Difficulty knobs: number of landmarks, repeated landmark types told apart by an attribute, route
length (waypoints), and map mismatches (landmarks missing from or added to the follower's map).

Usage: python scripts/route_samples.py [--out DIR]
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LANDMARKS = ["well", "big tree", "small tree", "red house", "blue house", "bridge", "mosque", "school", "pond",
             "rock", "statue", "market", "rice field", "gate", "tower", "bench", "warung", "bus stop", "temple",
             "football field", "fence", "boat", "cave", "waterfall"]
LEVELS = {
    "easy": {"n_landmarks": 9, "n_waypoints": 5, "n_missing": 0, "n_extra": 0, "dup_types": 0},
    "hard": {"n_landmarks": 14, "n_waypoints": 9, "n_missing": 2, "n_extra": 1, "dup_types": 2},
}


def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _crosses(pts):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    segs = list(zip(pts, pts[1:]))
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            (a, b), (c, d) = segs[i], segs[j]
            if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
                return True
    return False


def make_picture_map(rng, level, tries=200):
    """Retry until the drawn route does not cross itself (Map Task routes never do)."""
    for _ in range(tries):
        m = _make_picture_map(rng, level)
        if not _crosses(route_points(m)):
            return m
    return m


def _make_picture_map(rng, level):
    cfg = LEVELS[level]
    names = rng.sample(LANDMARKS, cfg["n_landmarks"] - cfg["dup_types"])
    # Repeated types: the same kind twice, told apart only by an attribute in the label.
    for base in rng.sample(["well", "rock", "bench", "gate"], cfg["dup_types"]):
        names += [f"{base} (with bucket)" if base == "well" else f"{base} (tall)", f"{base} (broken)"]
    pts = []
    while len(pts) < len(names):
        p = (rng.uniform(60, 540), rng.uniform(60, 540))
        if all(math.dist(p, q) > 85 for q in pts):
            pts.append(p)
    lms = [{"id": i, "name": n, "x": x, "y": y} for i, (n, (x, y)) in enumerate(zip(names, pts))]
    # Route: start bottom-left, greedy nearest-unvisited walk through n_waypoints landmarks.
    start = (30, 570)
    cur, route, left = start, [], list(range(len(lms)))
    for _ in range(cfg["n_waypoints"]):
        j = min(left, key=lambda k: math.dist(cur, (lms[k]["x"], lms[k]["y"])) + rng.uniform(0, 60))
        left.remove(j)
        side = rng.choice(["left", "right"])
        route.append({"landmark": j, "side": side})
        cur = (lms[j]["x"], lms[j]["y"])
    on_route = [r["landmark"] for r in route]
    missing = rng.sample(on_route[1:-1], cfg["n_missing"]) if cfg["n_missing"] else []
    extra = []
    for _ in range(cfg["n_extra"]):
        while True:
            p = (rng.uniform(60, 540), rng.uniform(60, 540))
            if all(math.dist(p, (l["x"], l["y"])) > 85 for l in lms):
                break
        extra.append({"id": 100 + len(extra), "name": rng.choice([n for n in LANDMARKS if n not in names]),
                      "x": p[0], "y": p[1]})
    follower = [l for l in lms if l["id"] not in missing] + extra
    gold = [{"landmark": lms[r["landmark"]]["name"], "side": r["side"]} for r in route
            if r["landmark"] not in missing]
    return {"level": level, "start": start, "giver_landmarks": lms, "follower_landmarks": follower,
            "route": route, "missing_on_follower": [lms[m]["name"] for m in missing],
            "extra_on_follower": [e["name"] for e in extra], "gold_answer": gold}


def route_points(m):
    lms = {l["id"]: l for l in m["giver_landmarks"]}
    pts, prev = [m["start"]], m["start"]
    for r in m["route"]:
        l = lms[r["landmark"]]
        dx, dy = l["x"] - prev[0], l["y"] - prev[1]
        n = math.hypot(dx, dy) or 1
        # Pass 30px to the traveller's left or right of the landmark.
        ox, oy = (-dy / n, dx / n) if r["side"] == "left" else (dy / n, -dx / n)
        pts.append((l["x"] + 30 * ox, l["y"] + 30 * oy))
        prev = (l["x"], l["y"])
    return pts


def draw_picture(m, who, path):
    im = Image.new("RGB", (600, 640), (250, 247, 235))
    d = ImageDraw.Draw(im)
    f, fb = font(13), font(16)
    d.text((10, 608), f"{m['level'].upper()} — {'GIVER (sees route)' if who == 'giver' else 'FOLLOWER (no route; map differs)'}",
           fill="black", font=fb)
    for l in (m["giver_landmarks"] if who == "giver" else m["follower_landmarks"]):
        extra = l["id"] >= 100
        d.ellipse([l["x"] - 9, l["y"] - 9, l["x"] + 9, l["y"] + 9], fill=(200, 120, 60) if not extra else (90, 140, 200))
        d.text((l["x"] + 11, l["y"] - 8), l["name"], fill="black", font=f)
    d.rectangle([m["start"][0] - 8, m["start"][1] - 8, m["start"][0] + 8, m["start"][1] + 8], fill="green")
    d.text((m["start"][0] + 10, m["start"][1] - 22), "START", fill="green", font=f)
    if who == "giver":
        d.line(route_points(m), fill=(200, 30, 30), width=3, joint="curve")
        end = route_points(m)[-1]
        d.text((end[0] + 6, end[1] + 6), "FINISH", fill=(200, 30, 30), font=f)
    im.save(path)


def make_grid_map(rng, level):
    """Text grid without coordinates; route as moves from S; follower map drops/adds landmarks."""
    cfg = LEVELS[level]
    size = 7 if level == "easy" else 9
    grid = [["." for _ in range(size)] for _ in range(size)]
    symbols = {}
    names = rng.sample(LANDMARKS, min(len(LANDMARKS), cfg["n_landmarks"]))
    cells = rng.sample([(r, c) for r in range(size) for c in range(size) if (r, c) != (size - 1, 0)], len(names))
    for i, ((r, c), n) in enumerate(zip(cells, names)):
        sym = "ABCDEFGHJKLMNPQRSTUVWXYZ"[i]
        grid[r][c] = sym
        symbols[sym] = n
    # Random self-avoiding walk from S over empty cells.
    steps = 8 if level == "easy" else 14
    pos, path, moves = (size - 1, 0), [(size - 1, 0)], []
    dirs = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    while len(moves) < steps:
        opts = [(k, (pos[0] + v[0], pos[1] + v[1])) for k, v in dirs.items()
                if 0 <= pos[0] + v[0] < size and 0 <= pos[1] + v[1] < size
                and (pos[0] + v[0], pos[1] + v[1]) not in path and grid[pos[0] + v[0]][pos[1] + v[1]] == "."]
        if not opts:
            break
        k, nxt = rng.choice(opts)
        moves.append(k)
        path.append(nxt)
        pos = nxt
    giver = [row[:] for row in grid]
    for (r, c) in path[1:]:
        giver[r][c] = "*"
    giver[size - 1][0] = "S"
    giver[path[-1][0]][path[-1][1]] = "#"
    follower = [row[:] for row in grid]
    follower[size - 1][0] = "S"
    near = [s for s, n in symbols.items() if any(abs(r - pr) + abs(c - pc) == 1
            for (pr, pc) in path for r, c in [next((rr, cc) for rr in range(size) for cc in range(size) if grid[rr][cc] == s)])]
    dropped = rng.sample(near, min(cfg["n_missing"], len(near)))
    for s in dropped:
        for r in range(size):
            for c in range(size):
                if follower[r][c] == s:
                    follower[r][c] = "."
    legend = {s: n for s, n in symbols.items()}
    return {"level": level, "giver": ["".join(r) for r in giver], "follower": ["".join(r) for r in follower],
            "legend": legend, "dropped_on_follower": {s: symbols[s] for s in dropped}, "gold_moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "derived" / "route_samples"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    samples = {}
    for level in ["easy", "hard"]:
        m = make_picture_map(rng, level)
        draw_picture(m, "giver", out / f"picture_{level}_giver.png")
        draw_picture(m, "follower", out / f"picture_{level}_follower.png")
        a, b = Image.open(out / f"picture_{level}_giver.png"), Image.open(out / f"picture_{level}_follower.png")
        both = Image.new("RGB", (1220, 640), "white")
        both.paste(a, (0, 0))
        both.paste(b, (620, 0))
        both.save(out / f"picture_{level}.png")
        g = make_grid_map(rng, level)
        samples[level] = {"picture": m, "grid": g}
    json.dump(samples, open(out / "samples.json", "w"), indent=1)
    with open(out / "grid_samples.txt", "w") as f:
        for level, s in samples.items():
            g = s["grid"]
            f.write(f"===== GRID {level.upper()} =====\nGIVER sees (S start, # finish, * route):\n")
            f.write("\n".join(" ".join(r) for r in g["giver"]) + "\n\nFOLLOWER sees (no route; some landmarks missing):\n")
            f.write("\n".join(" ".join(r) for r in g["follower"]) + "\n\nLegend: " +
                    ", ".join(f"{k}={v}" for k, v in g["legend"].items()) + "\n")
            f.write(f"Missing on follower's map: {g['dropped_on_follower']}\nGold answer (moves from S): {g['gold_moves']}\n\n")
            m = s["picture"]
            f.write(f"PICTURE {level}: follower answers the ordered landmarks passed and the side: {m['gold_answer']}\n"
                    f"  missing on follower's map: {m['missing_on_follower']}; extra on follower's map: {m['extra_on_follower']}\n\n")
    print(out)


if __name__ == "__main__":
    main()
