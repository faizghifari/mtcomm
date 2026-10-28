"""Add same-book distractor pages to the screened Bloom-VIST stories (sequence game, v3).

For each story in manifest_seq_v2.json, pick `--k` pages of the same story that lie outside its
5-page window: same characters and setting, so the narration must carry details that tell pages apart.
A page is skipped when it nearly duplicates a story page (16x16 grey MAE < 4, as in the builder), is
smaller than 200 px, or fails to download. Writes manifest_seq_v3.json with "distractors" per story;
stories without k usable pages are listed and left out.

Usage: python scripts/bloomvist_add_distractors.py [--k 2] [--seed 0]
"""

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bloomvist_build_sets import MAX_SIDE, MIN_SIDE, OUT, OUT_IMG, SRC, fetch, near, thumb, to_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    man = json.load(open(OUT / "manifest_seq_v2.json", encoding="utf-8"))
    d = json.load(open(SRC, encoding="utf-8"))
    images = {i["id"]: i for i in d["images"]}
    anns = defaultdict(list)
    for group in d["annotations"]:
        for a in group:
            anns[a["story_id"]].append(a)
    kept, dropped = [], []
    for s in man["stories"]:
        sid = s["story_id"]
        pages = sorted(anns[sid], key=lambda a: a["worker_arranged_photo_order"])
        window = set(s["worker_order"])
        in_window = {a["photo_flickr_id"] for a in pages if a["worker_arranged_photo_order"] in window}
        cand = [a for a in pages if a["worker_arranged_photo_order"] not in window and a["photo_flickr_id"] not in in_window]
        cand = list({a["photo_flickr_id"]: a for a in cand}.values())  # one entry per photo
        random.Random(f"{args.seed}:{sid}:distractors").shuffle(cand)
        story_thumbs = [thumb(Image.open(OUT_IMG / f)) for f in s["images"]]
        chosen, chosen_thumbs = [], []
        for a in cand:
            if len(chosen) == args.k:
                break
            name = f"{sid}_d{len(chosen) + 1}.jpg"
            raw = fetch(images[a["photo_flickr_id"]]["url_o"])
            if raw is None:
                continue
            try:
                im = to_rgb(Image.open(io.BytesIO(raw)))
            except (OSError, ValueError):
                continue
            t = thumb(im)
            if min(im.size) < MIN_SIDE or any(near(t, x) for x in story_thumbs + chosen_thumbs):
                continue
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            im.save(OUT_IMG / name, "JPEG", quality=90)
            chosen.append(name)
            chosen_thumbs.append(t)
        if len(chosen) == args.k:
            kept.append(dict(s, distractors=chosen))
        else:
            dropped.append((sid, s["title"], len(cand), len(chosen)))
    out = dict(man, version=3, distractors_per_story=args.k, stories=kept)
    json.dump(out, open(OUT / "manifest_seq_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(kept)} stories with {args.k} same-book distractors -> {OUT / 'manifest_seq_v3.json'}")
    for x in dropped:
        print("  dropped (story, title, candidate pages, usable):", x)


if __name__ == "__main__":
    main()
