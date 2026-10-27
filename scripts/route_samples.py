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
    """Retry until the drawn route does not cross itself (Map Task routes never do) and hidden landmarks are recoverable."""
    for _ in range(tries):
        m = _make_picture_map(rng, level)
        if m is not None and not _crosses(route_points(m)):
            return m
    raise RuntimeError(f"no valid {level} map in {tries} tries")


def _make_picture_map(rng, level):
    cfg = LEVELS[level]
    # Repeated types: the same kind twice, told apart only by an attribute in the label (never also a plain one).
    bases = rng.sample(["well", "rock", "bench", "gate"], cfg["dup_types"])
    names = rng.sample([n for n in LANDMARKS if n not in bases], cfg["n_landmarks"] - 2 * cfg["dup_types"])
    for base in bases:
        names += [f"{base} (with bucket)" if base == "well" else f"{base} (tall)", f"{base} (broken)"]
    pts = []
    while len(pts) < len(names):
        p = (rng.uniform(60, 540), rng.uniform(60, 540))
        if all(math.dist(p, q) > 85 for q in pts):
            pts.append(p)
    lms = [{"id": i, "name": n, "x": x, "y": y} for i, (n, (x, y)) in enumerate(zip(names, pts))]
    # Route: start bottom-left; each next waypoint is drawn uniformly from the 5 nearest unvisited landmarks.
    # (A greedy nearest-landmark walk made the route guessable without any talk: 57% exact on easy maps.)
    start = (30, 570)
    cur, route, left = start, [], list(range(len(lms)))
    for _ in range(cfg["n_waypoints"]):
        j = rng.choice(sorted(left, key=lambda k: math.dist(cur, (lms[k]["x"], lms[k]["y"])))[:5])
        left.remove(j)
        route.append({"landmark": j, "side": rng.choice(["left", "right"])})
        cur = (lms[j]["x"], lms[j]["y"])
    on_route = [r["landmark"] for r in route]
    missing = _pick_missing(rng, lms, on_route, cfg["n_missing"])
    if missing is None:
        return None
    extra = []
    for _ in range(cfg["n_extra"]):
        while True:
            p = (rng.uniform(60, 540), rng.uniform(60, 540))
            if all(math.dist(p, (l["x"], l["y"])) > 85 for l in lms):
                break
        extra.append({"id": 100 + len(extra), "name": rng.choice([n for n in LANDMARKS if n not in names and n not in bases]),
                      "x": p[0], "y": p[1]})
    follower = [l for l in lms if l["id"] not in missing] + extra
    # A landmark the follower lacks is answered as the grid cell where it stands.
    gold = [{"cell": cell_of(lms[r["landmark"]])} if r["landmark"] in missing
            else {"landmark": lms[r["landmark"]]["name"], "side": r["side"]} for r in route]
    return {"level": level, "start": start, "giver_landmarks": lms, "follower_landmarks": follower,
            "route": route, "missing_on_follower": [lms[m]["name"] for m in missing],
            "extra_on_follower": [e["name"] for e in extra], "gold_answer": gold}


def _pick_missing(rng, lms, on_route, n, tries=200):
    """Route landmarks to hide from the follower, chosen so the giver can still locate them from shared ones:
    never the first or last waypoint, never two in a row, and at least two shared landmarks within 2 cells."""
    if not n:
        return []
    for _ in range(tries):
        idx = sorted(rng.sample(range(1, len(on_route) - 1), n))
        if any(b - a < 2 for a, b in zip(idx, idx[1:])):
            continue
        miss = [on_route[i] for i in idx]
        shared = [l for l in lms if l["id"] not in miss]
        if all(sum(math.dist((l["x"], l["y"]), (lms[m]["x"], lms[m]["y"])) <= 2 * CELL for l in shared) >= 2
               for m in miss):
            return miss
    return None  # caller draws a new map


CELL, COLS = 75, "ABCDEFGH"  # the follower's copy carries an 8x8 lettered grid (A-H across, 1-8 down)


def cell_of(p):
    x, y = (p["x"], p["y"]) if isinstance(p, dict) else p
    return f"{COLS[min(7, int(x // CELL))]}{min(7, int(y // CELL)) + 1}"


def cell_center(c):
    return (COLS.index(c[0]) * CELL + CELL / 2, (int(c[1:]) - 1) * CELL + CELL / 2)


def _step_ok(gold, ans):
    if "cell" in gold:
        if "cell" not in ans:
            return False
        (gx, gy), (ax, ay) = cell_center(gold["cell"]), cell_center(ans["cell"])
        return max(abs(gx - ax), abs(gy - ay)) <= CELL + 1  # within one cell, diagonals included
    return ans.get("landmark") == gold["landmark"] and ans.get("side") == gold["side"]


def _offset(prev, p, side):
    dx, dy = p[0] - prev[0], p[1] - prev[1]
    n = math.hypot(dx, dy) or 1
    ox, oy = (-dy / n, dx / n) if side == "left" else (dy / n, -dx / n)
    return (p[0] + 30 * ox, p[1] + 30 * oy)


def answer_points(m, answer):
    """Rebuild the path a follower's step list implies on the follower's own map."""
    pos = {l["name"]: (l["x"], l["y"]) for l in m["follower_landmarks"]}
    pts, prev = [m["start"]], m["start"]
    for a in answer:
        if "cell" in a:
            p = cell_center(a["cell"])
            pts.append(p)
        elif a.get("landmark") in pos:
            p = pos[a["landmark"]]
            pts.append(_offset(prev, p, a.get("side", "left")))
        else:
            continue
        prev = p
    return pts


def _resample(pts, n=200):
    seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total, out = sum(seg) or 1, []
    for k in range(n):
        t, i = total * k / (n - 1), 0
        while i < len(seg) - 1 and t > seg[i]:
            t -= seg[i]
            i += 1
        f = t / (seg[i] or 1)
        out.append((pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])))
    return out


def score_picture(m, answer):
    """Primary: success = every step right, in order. Secondary: per-step accuracy (LCS) and the Map-Task-style
    deviation between drawn and true paths (mean symmetric nearest-point distance, in grid cells)."""
    gold = m["gold_answer"]
    L = [[0] * (len(answer) + 1) for _ in range(len(gold) + 1)]
    for i, g in enumerate(gold):
        for j, a in enumerate(answer):
            L[i + 1][j + 1] = L[i][j] + 1 if _step_ok(g, a) else max(L[i][j + 1], L[i + 1][j])
    lcs = L[-1][-1]
    A, B = _resample(route_points(m)), _resample(answer_points(m, answer))
    dev = (sum(min(math.dist(p, q) for q in B) for p in A) + sum(min(math.dist(p, q) for q in A) for p in B)) / (2 * len(A))
    return {"success": lcs == len(gold) == len(answer), "step_accuracy": round(lcs / len(gold), 3),
            "deviation_cells": round(dev / CELL, 2)}


def route_points(m):
    lms = {l["id"]: l for l in m["giver_landmarks"]}
    pts, prev = [m["start"]], m["start"]
    for r in m["route"]:
        l = lms[r["landmark"]]
        pts.append(_offset(prev, (l["x"], l["y"]), r["side"]))  # pass 30px to the traveller's left or right
        prev = (l["x"], l["y"])
    return pts


def draw_picture(m, who, path, answer=None):
    """who: giver (route, no grid) | follower (grid, no route) | scored (follower map, gold vs an example answer)."""
    im = Image.new("RGB", (600, 640), (250, 247, 235))
    d = ImageDraw.Draw(im)
    f, fb = font(13), font(16)
    title = {"giver": "GIVER (sees route, no grid)", "follower": "FOLLOWER (no route" + ("; map differs)" if m["missing_on_follower"] or m["extra_on_follower"] else ")"),
             "scored": "SCORING EXAMPLE (red = true, blue = answer)"}[who]
    d.text((10, 608), f"{m['level'].upper()} - {title}", fill="black", font=fb)
    if who != "giver":  # the grid exists only on the follower's copy, so the giver cannot read out coordinates
        for k in range(1, 8):
            d.line([(k * CELL, 0), (k * CELL, 600)], fill=(215, 210, 195))
            d.line([(0, k * CELL), (600, k * CELL)], fill=(215, 210, 195))
        for k in range(8):
            d.text((k * CELL + 3, 2), COLS[k], fill=(150, 140, 120), font=f)
            d.text((2, k * CELL + 16), str(k + 1), fill=(150, 140, 120), font=f)
    for l in (m["giver_landmarks"] if who == "giver" else m["follower_landmarks"]):
        extra = l["id"] >= 100
        d.ellipse([l["x"] - 9, l["y"] - 9, l["x"] + 9, l["y"] + 9], fill=(200, 120, 60) if not extra else (90, 140, 200))
        d.text((l["x"] + 11, l["y"] - 8), l["name"], fill="black", font=f)
    d.rectangle([m["start"][0] - 8, m["start"][1] - 8, m["start"][0] + 8, m["start"][1] + 8], fill="green")
    d.text((m["start"][0] + 10, m["start"][1] - 22), "START", fill="green", font=f)
    if who in ("giver", "scored"):
        pts = route_points(m)
        d.line(pts, fill=(200, 30, 30), width=3, joint="curve")
        d.text((pts[-1][0] + 6, pts[-1][1] + 6), "FINISH", fill=(200, 30, 30), font=f)
    if who == "scored":
        if m["missing_on_follower"]:  # where the hidden landmarks really are (the follower never sees these)
            for l in m["giver_landmarks"]:
                if l["name"] in m["missing_on_follower"]:
                    d.ellipse([l["x"] - 9, l["y"] - 9, l["x"] + 9, l["y"] + 9], outline=(120, 120, 120), width=2)
                    d.text((l["x"] + 11, l["y"] - 8), f"({l['name']}: hidden)", fill=(120, 120, 120), font=f)
        d.line(answer_points(m, answer), fill=(30, 80, 220), width=3, joint="curve")
        sc = score_picture(m, answer)
        d.text((250, 585), f"success={sc['success']}  steps={sc['step_accuracy']}  dev={sc['deviation_cells']} cells",
               fill=(30, 80, 220), font=f)
    im.save(path)


def example_answer(m):
    """A plausible imperfect answer: hidden landmarks placed one cell off (still within tolerance), one side wrong."""
    ans = [dict(g) for g in m["gold_answer"]]
    for g in ans:
        if "cell" in g:
            c, r = COLS.index(g["cell"][0]), int(g["cell"][1:])
            g["cell"] = f"{COLS[c + 1 if c < 7 else c - 1]}{r}"
    k = next(i for i in range(len(ans) - 1, -1, -1) if "side" in ans[i])
    ans[k]["side"] = "left" if ans[k]["side"] == "right" else "right"
    return ans


def no_talk_baseline(rng, level, n=200):
    """Follower with no information: walk from START to the nearest landmark on its own map, each time."""
    exact = steps = total = 0
    for _ in range(n):
        m = make_picture_map(rng, level)
        cur, left, guess = m["start"], list(m["follower_landmarks"]), []
        for _ in m["gold_answer"]:
            j = min(left, key=lambda l: math.dist(cur, (l["x"], l["y"])))
            left.remove(j)
            guess.append(j["name"])
            cur = (j["x"], j["y"])
        gold = [g.get("landmark") for g in m["gold_answer"]]
        steps += sum(x == y for x, y in zip(guess, gold))
        total += len(gold)
        exact += guess == gold
    return {"exact_order": exact / n, "per_step": round(steps / total, 3)}


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
