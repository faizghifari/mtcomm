"""Street maps for the directions game (mtcomm.games.wayfind): generation and drawing.

A map is a grid of intersections with some street segments removed, which gives dead ends and three-way intersections.
Small buildings line every segment; some are replaced by labelled landmarks. The route starts at a green arrow that shows
the way the walker faces, runs along streets with a fixed number of turns, and ends at one building.

Candidates are what make the task sensitive to translation: besides the target, the lettered buildings include where the
walker ends up after taking one turn the wrong way (kind "flip") and the target's neighbours (kind "neighbor"), so a
swapped left/right or a lost "second building on your right" lands on a real, wrong answer instead of nothing.
"""

import random

from PIL import Image, ImageDraw

from .routemap import LANDMARKS, font

N = 5  # intersections per side
STEP, MARGIN, SIZE = 120, 60, 600
LEVELS = {"easy": 2, "mid": 3, "hard": 4}  # number of turns
HEADINGS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # clockwise: up, right, down, left (x grows right, y grows down)
LETTERS = "ABCDEFGH"


def _key(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def _make_edges(rng, drop=0.22):
    """All grid segments minus a random share, keeping every intersection connected."""
    nodes = [(i, j) for i in range(N) for j in range(N)]
    edges = {_key((i, j), (i + di, j + dj)) for i, j in nodes for di, dj in ((1, 0), (0, 1)) if i + di < N and j + dj < N}

    def connected(es):
        seen, todo = {(0, 0)}, [(0, 0)]
        while todo:
            n = todo.pop()
            for a, b in es:
                m = b if a == n else a if b == n else None
                if m and m not in seen:
                    seen.add(m)
                    todo.append(m)
        return len(seen) == len(nodes)

    candidates = sorted(edges)
    rng.shuffle(candidates)
    for e in candidates[:int(len(edges) * drop * 2)]:
        if len(edges) <= int((1 - drop) * (2 * N * (N - 1))):
            break
        if connected(edges - {e}):
            edges.remove(e)
    return edges


def _turn(h, side):
    return (h + (1 if side == "right" else 3)) % 4


def _walk(start, h0, moves, edges):
    """Follow moves (ints: blocks straight; 'left'/'right': a turn); returns the nodes, stopping early at a blocked step."""
    node, h, path = tuple(start), h0, [tuple(start)]
    for mv in moves:
        if mv in ("left", "right"):
            h = _turn(h, mv)
            continue
        for _ in range(mv):
            nxt = (node[0] + HEADINGS[h][0], node[1] + HEADINGS[h][1])
            if _key(node, nxt) not in edges:
                return path, h, False
            node = nxt
            path.append(node)
    return path, h, True


def _make_route(rng, edges, n_turns, tries=2000):
    for _ in range(tries):
        start = (rng.randrange(N), rng.randrange(N))
        h = rng.randrange(4)
        moves = [rng.choice((1, 2))]
        for _ in range(n_turns):
            moves += [rng.choice(("left", "right")), rng.choice((1, 2))]
        path, end_h, ok = _walk(start, h, moves, edges)
        if ok and len(set(path)) == len(path):  # never walks through the same intersection twice
            return {"start": start, "heading": h, "moves": moves, "path": path, "end_heading": end_h}
    return None


def _px(node):
    return MARGIN + node[0] * STEP, MARGIN + node[1] * STEP


def building_at(a, b, side, slot):
    """Centre of a building on the segment a->b: side is the walker's left or right, slot 1 or 2 along the walk."""
    (ax, ay), (bx, by) = _px(a), _px(b)
    dx, dy = (bx - ax) / STEP, (by - ay) / STEP
    nx, ny = (-dy, dx) if side == "right" else (dy, -dx)  # y grows downward
    t = STEP * (0.33 if slot == 1 else 0.67)
    return round(ax + dx * t + nx * 21), round(ay + dy * t + ny * 21)


def all_buildings(edges):
    """{id: (x, y)}, one id per building; a building is named from the lower-numbered end of its segment."""
    out = {}
    for a, b in sorted(edges):
        for side in ("left", "right"):
            for slot in (1, 2):
                out[f"{a[0]}{a[1]}-{b[0]}{b[1]}-{side[0]}{slot}"] = building_at(a, b, side, slot)
    return out


def _nearest(buildings, xy, skip=()):
    return min((b for b in buildings if b not in skip),
               key=lambda b: (buildings[b][0] - xy[0]) ** 2 + (buildings[b][1] - xy[1]) ** 2)


def make_street_map(rng, level, map_id, n_landmarks=10, tries=200):
    n_turns = LEVELS[level]
    for _ in range(tries):
        edges = _make_edges(rng)
        route = _make_route(rng, edges, n_turns)
        if route:
            return _finish(rng, level, map_id, edges, route, n_landmarks)
    raise RuntimeError(f"no {level} street map in {tries} tries")


def _finish(rng, level, map_id, edges, route, n_landmarks):
    buildings = all_buildings(edges)
    a, b = route["path"][-2], route["path"][-1]
    side, slot = rng.choice(("left", "right")), rng.choice((1, 2))
    target_xy = building_at(a, b, side, slot)
    target = _nearest(buildings, target_xy)

    kinds = {target: "target"}
    # A wrong turn: flip each turn in turn, keep the rest of the route, and end on the same side and slot.
    turn_idx = [i for i, m in enumerate(route["moves"]) if m in ("left", "right")]
    for i in turn_idx:
        moves = list(route["moves"])
        moves[i] = "left" if moves[i] == "right" else "right"
        path, _, _ = _walk(route["start"], route["heading"], moves, edges)
        if len(path) >= 2:
            c = _nearest(buildings, building_at(path[-2], path[-1], side, slot), skip=kinds)
            kinds[c] = "flip"
    # The other side of the street and the next building along.
    for s, k in ((("left" if side == "right" else "right"), slot), (side, 3 - slot)):
        c = _nearest(buildings, building_at(a, b, s, k), skip=kinds)
        kinds[c] = "neighbor"
    others = sorted(set(buildings) - set(kinds), key=lambda x: (buildings[x][0] - target_xy[0]) ** 2 + (buildings[x][1] - target_xy[1]) ** 2)
    near = others[:30]
    rng.shuffle(near)
    for c in near:
        if len(kinds) >= len(LETTERS):
            break
        kinds[c] = "other"
    letters = list(LETTERS)
    rng.shuffle(letters)
    candidates = {letter: {"building": c, "kind": kinds[c]} for letter, c in zip(letters, kinds)}
    gold = next(k for k, v in candidates.items() if v["kind"] == "target")

    # Landmarks take a building's place. Two sit at the route's turns, since that is what a guide would point to.
    names = list(LANDMARKS)
    rng.shuffle(names)
    free = [x for x in buildings if x not in kinds]
    corners = [route["path"][sum(m for m in route["moves"][:i] if isinstance(m, int))] for i in turn_idx]
    picks = []
    for c in corners[:2]:
        pick = _nearest({k: buildings[k] for k in free if k not in picks}, _px(c))
        picks.append(pick)
    rest = [x for x in free if x not in picks]
    rng.shuffle(rest)
    for x in rest:
        if len(picks) >= n_landmarks:
            break
        if all((buildings[x][0] - buildings[p][0]) ** 2 + (buildings[x][1] - buildings[p][1]) ** 2 > 100 ** 2 for p in picks):
            picks.append(x)
    landmarks = [{"name": n, "building": p, "x": buildings[p][0], "y": buildings[p][1]} for n, p in zip(names, picks)]
    return {"map_id": map_id, "level": level, "edges": [[list(a), list(b)] for a, b in sorted(edges)],
            "start": list(route["start"]), "heading": route["heading"], "moves": route["moves"],
            "path": [list(p) for p in route["path"]], "target": target, "gold": gold, "candidates": candidates,
            "buildings": {k: list(v) for k, v in buildings.items()}, "landmarks": landmarks}


def draw_street_map(m, who, path, label=None):
    """who: guide (route in red, target in red, no letters) | seeker (no route, candidates lettered)."""
    label = label or (lambda name: name)
    cjk = any(ord(ch) > 0x2E80 for l in m["landmarks"] for ch in label(l["name"]))
    f, fb = font(12, cjk), font(13)
    im = Image.new("RGB", (SIZE, SIZE), (246, 244, 236))
    d = ImageDraw.Draw(im)
    for a, b in m["edges"]:
        d.line([_px(a), _px(b)], fill=(190, 190, 190), width=14)
    for a, b in m["edges"]:
        for n in (a, b):
            x, y = _px(n)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(190, 190, 190))
    taken = {l["building"] for l in m["landmarks"]}
    cand = {v["building"]: k for k, v in m["candidates"].items()}
    for b, (x, y) in m["buildings"].items():
        if b not in taken and b not in cand:
            d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=(214, 206, 188), outline=(165, 155, 135))
    if who == "guide":
        # The line stops level with the target, so it does not suggest walking past it.
        pts = [_px(p) for p in m["path"]]
        tx, ty = m["buildings"][m["target"]]
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        pts[-1] = (tx, by) if ay == by else (bx, ty)
        d.line(pts, fill=(205, 30, 30), width=4, joint="curve")
    boxes = [m["buildings"][b] for b in cand]

    def clear(x0, x1, y):  # no candidate under a label
        return not any(x0 - 11 <= cx <= x1 + 11 and abs(cy - y) <= 18 for cx, cy in boxes)

    for l in m["landmarks"]:
        x, y = l["x"], l["y"]
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(225, 130, 50))
        text = label(l["name"])
        w = d.textlength(text, font=f)
        right_ok = x + 10 + w <= SIZE - 2
        left_ok = x - 10 - w >= 2
        tx = x + 10 if right_ok and (clear(x + 10, x + 10 + w, y) or not left_ok) else x - 10 - w if left_ok else x + 10
        d.rectangle([tx - 1, y - 8, tx + w + 1, y + 7], fill=(255, 255, 255))
        d.text((tx, y - 8), text, fill="black", font=f)
    for b, letter in cand.items():  # last, so nothing covers a letter
        x, y = m["buildings"][b]
        if who == "seeker":
            d.rectangle([x - 10, y - 10, x + 10, y + 10], fill=(45, 85, 170))
            d.text((x - 4, y - 8), letter, fill="white", font=fb)
        elif b == m["target"]:
            d.rectangle([x - 10, y - 10, x + 10, y + 10], fill=(205, 30, 30))
        else:
            d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=(214, 206, 188), outline=(165, 155, 135))
    x, y = _px(m["start"])
    hx, hy = HEADINGS[m["heading"]]
    tip, left, right = (x + hx * 16, y + hy * 16), (x - hx * 6 + hy * 11, y - hy * 6 - hx * 11), (x - hx * 6 - hy * 11, y - hy * 6 + hx * 11)
    d.polygon([tip, left, right], fill=(20, 150, 60))
    im.save(path)


def build_maps(seed=0, counts=None):
    rng = random.Random(seed)
    maps = []
    for level, n in (counts or {"easy": 30, "mid": 30, "hard": 30}).items():
        maps += [make_street_map(rng, level, f"w{level[0]}{k + 1:02d}") for k in range(n)]
    return maps
