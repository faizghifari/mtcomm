"""Bloom-VIST picture sequences for an ordering game: 5 consecutive pages of one book, in true order.

Each kept item is one 5-picture window of one Bloom book; the true order is worker_arranged_photo_order.
Players see the pictures shuffled and put them back in order, so text is optional. Text languages are
recorded per book, including translations: a translation story (album not in `albums`) points at the
primary album's photos, and duplicate_ids_list maps primary -> translation.

Filters, in order: licence (no ND on either licence field; no licence record = excluded), quarantine flag,
>= 5 annotated pages, one sequence per album, contiguous page order, 5 distinct photos, download (3 retries),
short side >= 200 px, no near-duplicate pages inside the window (16x16 grey MAE < 4), and no sequence that
re-uses the illustrations of an already kept one (>= 3 of 5 pages match).

Stories are tried in tier order (text in both id and ko, then id or ko, then en, then the rest), with a
seeded shuffle inside each tier, until --n sequences pass.

data.json uses ISO 639-1 codes ('id', 'ko', 'en'), not 'ind'/'kor'. 'kr' is Kanuri, not Korean.

Usage: python scripts/bloomvist_build_sets.py [--n 30] [--seed 0]
"""

import argparse
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import DATA_ROOT  # noqa: E402

SRC = DATA_ROOT / "external" / "bloom_vist" / "data.json"
OUT = DATA_ROOT / "derived" / "bloom_vist"
OUT_IMG = OUT / "images"
N_IMG = 5
MIN_SIDE = 200
MAX_SIDE = 768
DUP_MAE = 4.0
MAX_RETRIES = 3
TIMEOUT = 30


def is_nd(lic):
    return "nd" in lic.lower().split("-")


def tier(text_langs):
    base = {c.split("-")[0] for c in text_langs}
    if {"id", "ko"} <= base:
        return 0
    if base & {"id", "ko"}:
        return 1
    if "en" in base:
        return 2
    return 3


def to_rgb(im):
    # Bloom art is often RGBA or palette-with-transparency: flatten onto white, not black.
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return im.convert("RGB")


def thumb(im):
    return np.asarray(im.convert("L").resize((16, 16), Image.Resampling.BOX), dtype=np.float32)


def near(a, b):
    return float(np.abs(a - b).mean()) < DUP_MAE


def fetch(url):
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2 ** attempt)


def try_story(sid, book, s_anns, images, albums, book_langs, seed, pool, kept_thumbs):
    """Return (reason, record, thumbs). reason is None when the story is kept and its files are written."""
    anns = sorted(s_anns, key=lambda a: a["worker_arranged_photo_order"])
    orders = [a["worker_arranged_photo_order"] for a in anns]
    if any(b - a != 1 for a, b in zip(orders, orders[1:])):
        return "non-contiguous page order", None, None
    pids = [a["photo_flickr_id"] for a in anns]
    windows = [i for i in range(len(pids) - N_IMG + 1) if len(set(pids[i:i + N_IMG])) == N_IMG]
    if not windows:
        return "fewer than 5 distinct photos", None, None
    start = random.Random(f"{seed}:{sid}").choice(windows)
    win = anns[start:start + N_IMG]
    names = [f"{sid}_{k}.jpg" for k in range(1, N_IMG + 1)]
    paths = [OUT_IMG / n for n in names]
    urls = [images[a["photo_flickr_id"]]["url_o"] for a in win]

    if all(p.exists() for p in paths):  # resume: accepted on an earlier run, only thumbs are needed
        thumbs = [thumb(Image.open(p)) for p in paths]
    else:
        raws = list(pool.map(fetch, urls))
        if any(r is None for r in raws):
            return "download failed", None, None
        try:
            ims = [to_rgb(Image.open(io.BytesIO(r))) for r in raws]
        except (OSError, ValueError):
            return "undecodable image", None, None
        if any(min(im.size) < MIN_SIDE for im in ims):
            return "short side < 200 px", None, None
        thumbs = [thumb(im) for im in ims]
        if any(near(thumbs[i], thumbs[j]) for i in range(N_IMG) for j in range(i + 1, N_IMG)):
            return "near-duplicate pages within story", None, None
        for kt in kept_thumbs:
            if sum(any(near(t, k) for k in kt) for t in thumbs) >= 3:
                return "same illustrations as a kept sequence", None, None
        for im, p in zip(ims, paths):
            small = im.copy()
            small.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            small.save(p, "JPEG", quality=90)

    meta = albums[book]["metadata_from_original_json"]
    rec = {
        "story_id": sid,
        "album_id": book,
        "title": albums[book].get("title"),
        "licence": albums[book]["license"],
        "book_licence": meta["license"],
        "copyright": meta.get("copyright"),
        "credits": meta.get("credits"),
        "author": meta.get("author") or meta.get("authors"),
        "publisher": meta.get("publisher") or meta.get("originalPublisher"),
        "text_langs": sorted(book_langs[book]),
        "story_text_langs": sorted({a["lang"] for a in s_anns if a["text"].strip()}),
        "worker_order": orders[start:start + N_IMG],
        "images": names,
        "source_urls": urls,
    }
    return None, rec, thumbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(SRC, encoding="utf-8"))
    albums = {a["id"]: a for a in d["albums"]}
    images = {i["id"]: i for i in d["images"]}
    stories = d["stories"]
    anns = defaultdict(list)
    for group in d["annotations"]:
        for a in group:
            anns[a["story_id"]].append(a)
    alt = {v: k for k, vs in d["duplicate_ids_list"].items() for v in vs}  # translation album -> primary

    def primary(aid):
        return aid if aid in albums else alt.get(aid)

    def lic(book):
        a = albums[book]
        return a.get("license"), a.get("metadata_from_original_json", {}).get("license")

    # Licence distribution, per story (via its book), before and after the ND filter.
    before_top, before_meta, after_top, after_meta = Counter(), Counter(), Counter(), Counter()
    for s in stories.values():
        book = primary(s["album_id"])
        if book is None:
            before_top["MISSING"] += 1
            before_meta["MISSING"] += 1
            continue
        top, meta = lic(book)
        before_top[top or "MISSING"] += 1
        before_meta[meta or "MISSING"] += 1
        if top is None or meta is None or is_nd(top) or is_nd(meta):
            continue
        after_top[top] += 1
        after_meta[meta] += 1
    print(f"stories: {len(stories)}, translation stories: {sum(1 for s in stories.values() if s['album_id'] not in albums)}")
    print(f"licence (album 'license') before: {dict(before_top)}")
    print(f"licence (album 'license') after : {dict(after_top)}")
    print(f"licence (metadata)        before: {dict(before_meta)}")
    print(f"licence (metadata)        after : {dict(after_meta)}")

    # Text languages per book, over primary and translation stories (quarantined stories ignored).
    book_langs = defaultdict(set)
    for sid, s in stories.items():
        book = primary(s["album_id"])
        if book and not s["quarantine"]:
            book_langs[book] |= {a["lang"] for a in anns.get(sid, []) if a["text"].strip()}

    pool_stats = Counter()
    cands = []
    for sid, s in stories.items():
        book = s["album_id"]
        if book not in albums:
            pool_stats["translation story (text only)"] += 1
            continue
        if s["quarantine"]:
            pool_stats["quarantined"] += 1
            continue
        top, meta = lic(book)
        if top is None or meta is None or is_nd(top) or is_nd(meta):
            pool_stats["licence excluded"] += 1
            continue
        if len(anns.get(sid, [])) < N_IMG:
            pool_stats["fewer than 5 pages"] += 1
            continue
        cands.append((sid, book))
    print(f"candidate pool: {len(cands)} stories; excluded before download: {dict(pool_stats)}")

    rng = random.Random(args.seed)
    rng.shuffle(cands)
    cands.sort(key=lambda c: tier(book_langs[c[1]]))  # stable: seeded shuffle kept within each tier

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    kept, kept_thumbs, used_books = [], [], set()
    reasons = Counter()
    tried = 0
    with ThreadPoolExecutor(max_workers=4) as pool:  # at most 4 concurrent downloads
        for sid, book in cands:
            if len(kept) >= args.n:
                break
            tried += 1
            title = (albums[book].get("title") or "")[:40]
            if book in used_books:
                reasons["same book already kept"] += 1
                print(f"[{tried:3d}] skip {sid[:8]} same book already kept", flush=True)
                continue
            reason, rec, thumbs = try_story(sid, book, anns[sid], images, albums, book_langs,
                                            args.seed, pool, kept_thumbs)
            if reason:
                reasons[reason] += 1
                print(f"[{tried:3d}] skip {sid[:8]} {reason}", flush=True)
                continue
            kept.append(rec)
            kept_thumbs.append(thumbs)
            used_books.add(book)
            print(f"[{tried:3d}] KEEP {sid[:8]} tier {tier(book_langs[book])} {title!r}", flush=True)

    manifest = {
        "version": "seq_v1",
        "source": f"Bloom-VIST data.json ({SRC})",
        "n_images": N_IMG,
        "seed": args.seed,
        "stories": kept,
    }
    json.dump(manifest, open(OUT / "manifest_seq_v1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\ntried {tried}, kept {len(kept)}")
    print(f"rejections: {dict(reasons)}")
    cov = Counter()
    for r in kept:
        base = {c.split("-")[0] for c in r["text_langs"]}
        cov["id+ko" if {"id", "ko"} <= base else "id" if "id" in base else "ko" if "ko" in base
            else "en" if "en" in base else "other" if base else "no text"] += 1
    print(f"text coverage of kept: {dict(cov)}")
    print(f"kept tiers: {dict(Counter(tier(set(r['text_langs'])) for r in kept))}")
    if len(kept) < args.n:
        print(f"WARNING: only {len(kept)} of {args.n} sequences passed the filters")
    print(f"wrote {len(kept)} sequences to {OUT / 'manifest_seq_v1.json'}")


if __name__ == "__main__":
    main()
