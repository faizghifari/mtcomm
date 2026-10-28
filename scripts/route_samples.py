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

from PIL import Image

from mtcomm.routemap import LANDMARKS, LEVELS, draw_picture, example_answer, make_picture_map, no_talk_baseline, score_picture

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
        m["example_answer"] = example_answer(m)
        m["example_score"] = score_picture(m, m["example_answer"])
        draw_picture(m, "scored", out / f"picture_{level}_scored.png", m["example_answer"])
        both = Image.new("RGB", (1840, 640), "white")
        for k, who in enumerate(["giver", "follower", "scored"]):
            both.paste(Image.open(out / f"picture_{level}_{who}.png"), (k * 620, 0))
        both.save(out / f"picture_{level}.png")
        g = make_grid_map(rng, level)
        samples[level] = {"picture": m, "grid": g}
    for level in ["easy", "hard"]:
        samples[level]["no_talk_baseline"] = no_talk_baseline(random.Random(1), level)
        print(level, "no-talk baseline (nearest-landmark walk):", samples[level]["no_talk_baseline"])
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
            f.write(f"PICTURE {level}: gold steps (landmark + side, or grid cell for a landmark the follower lacks): "
                    f"{m['gold_answer']}\n  example answer: {m['example_answer']}\n  example score: {m['example_score']}\n"
                    f"  missing on follower's map: {m['missing_on_follower']}; extra on follower's map: {m['extra_on_follower']}\n\n")
    print(out)


if __name__ == "__main__":
    main()
