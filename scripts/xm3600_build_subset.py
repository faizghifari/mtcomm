"""Select the PoC-1 target subset from XM3600 and mine distractors with SigLIP.

Steps:
1. Embed all 3,600 images (and each image's first English caption) with SigLIP; cache the
   embeddings outside the repo.
2. Sample targets across locales (at most --per-locale per locale, seeded).
3. For each target, build two 5-distractor tiers: random, and hard (nearest image
   neighbours, skipping near-duplicates above --dup-threshold).
4. Keep a target only if SigLIP text->image retrieval with its English caption ranks it
   first in both 6-image sets (a free, local solvability check; it biases the subset toward
   items SigLIP finds easy, which is reported).
5. Freeze a manifest, then export a captioning folder holding only the target images under
   neutral IDs (no captions, no distractors), for writing native su/jv captions.

Usage: python scripts/xm3600_build_subset.py [--n-targets 60] [--seed 0]
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import ANNOTATION_DIR, DATA_ROOT, XM3600_DIR  # noqa: E402

MODEL_ID = "google/siglip-so400m-patch14-384"
DERIVED = DATA_ROOT / "derived" / "xm3600"


def load_captions():
    rows = {}
    with open(XM3600_DIR / "captions" / "captions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["image/key"]] = r
    return rows


def _features(out):
    """transformers 5 returns a model output object; older versions return the tensor."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


@torch.no_grad()
def embed(keys, captions, batch=32):
    cache = DERIVED / "siglip_so400m.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["keys"]) == keys:
            return z["img"], z["txt"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    img_out, txt_out = [], []
    for i in range(0, len(keys), batch):
        ks = keys[i:i + batch]
        ims = [Image.open(XM3600_DIR / "images" / f"{k}.jpg").convert("RGB") for k in ks]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, model.dtype)
        img_out.append(torch.nn.functional.normalize(_features(model.get_image_features(pixel_values=px)).float(), dim=-1).cpu())
        tx = proc(text=[captions[k]["en"]["caption"][0] for k in ks], padding="max_length", max_length=64,
                  truncation=True, return_tensors="pt")["input_ids"].to(device)
        txt_out.append(torch.nn.functional.normalize(_features(model.get_text_features(input_ids=tx)).float(), dim=-1).cpu())
        print(f"embedded {i + len(ks)}/{len(keys)}", end="\r")
    img, txt = torch.cat(img_out).numpy(), torch.cat(txt_out).numpy()
    DERIVED.mkdir(parents=True, exist_ok=True)
    np.savez(cache, keys=np.array(keys), img=img, txt=txt)
    return img, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=60)
    ap.add_argument("--per-locale", type=int, default=2)
    ap.add_argument("--n-distractors", type=int, default=5)
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    caps = load_captions()
    keys = sorted(caps)
    img, txt = embed(keys, caps)
    sim = img @ img.T
    np.fill_diagonal(sim, -1.0)

    rng = random.Random(args.seed)
    by_locale = defaultdict(list)
    for i, k in enumerate(keys):
        by_locale[caps[k]["image/locale"]].append(i)
    for idx in by_locale.values():
        rng.shuffle(idx)

    def solvable(t, ds):
        cand = [t] + ds
        scores = img[cand] @ txt[t]
        return int(np.argmax(scores) == 0), float(scores[0] - np.max(scores[1:]))

    targets, rejected = [], 0
    used = set()
    locales = sorted(by_locale)
    rng.shuffle(locales)
    # Images taken in Indonesia and Korea are always included: they matter for the su/jv/id/ko players.
    locales = ["id", "ko"] + [loc for loc in locales if loc not in ("id", "ko")]
    taken = defaultdict(int)
    while len(targets) < args.n_targets:
        progress = False
        for loc in locales:
            if len(targets) >= args.n_targets or taken[loc] >= args.per_locale or not by_locale[loc]:
                continue
            t = by_locale[loc].pop()
            progress = True
            if t in used:
                continue
            order = [j for j in np.argsort(-sim[t]) if sim[t, j] < args.dup_threshold and j != t]
            hard = [int(j) for j in order[:args.n_distractors]]
            pool = [j for j in range(len(keys)) if j != t and j not in hard]
            rand = rng.sample(pool, args.n_distractors)
            ok_h, margin_h = solvable(t, hard)
            ok_r, margin_r = solvable(t, rand)
            if not (ok_h and ok_r):
                rejected += 1
                continue
            used.add(t)
            taken[loc] += 1
            targets.append({
                "item_id": f"x{len(targets) + 1:03d}",
                "image_key": keys[t],
                "locale": loc,
                "distractors": {"random": [keys[j] for j in rand], "hard": [keys[j] for j in hard]},
                "hard_max_sim": float(sim[t, hard].max()),
                "siglip_margin": {"random": margin_r, "hard": margin_h},
            })
        if not progress:
            break

    manifest = {
        "version": "poc1_v1", "seed": args.seed, "model": MODEL_ID, "n_targets": len(targets),
        "per_locale": args.per_locale, "dup_threshold": args.dup_threshold,
        "rejected_by_solvability_check": rejected,
        "note": "Targets kept only if SigLIP text->image retrieval with the English caption ranks them "
                "first in both tiers; this biases the subset toward SigLIP-easy items.",
        "items": targets,
    }
    DERIVED.mkdir(parents=True, exist_ok=True)
    with open(DERIVED / "manifest_poc1_v1.json", "w") as f:
        json.dump(manifest, f, indent=1)

    out = ANNOTATION_DIR / "xm3600_captioning"
    (out / "images").mkdir(parents=True, exist_ok=True)
    order = targets[:]
    random.Random(args.seed + 1).shuffle(order)
    with open(out / "captions_to_write.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "image_file", "caption_su", "caption_jv", "notes"])
        for t in order:
            shutil.copy(XM3600_DIR / "images" / f"{t['image_key']}.jpg", out / "images" / f"{t['item_id']}.jpg")
            w.writerow([t["item_id"], f"images/{t['item_id']}.jpg", "", "", ""])

    print(f"\nkept {len(targets)} targets from {len({t['locale'] for t in targets})} locales; rejected {rejected} by the solvability check")
    print(f"hard-tier max similarity: median {np.median([t['hard_max_sim'] for t in targets]):.3f}")
    print(f"manifest: {DERIVED / 'manifest_poc1_v1.json'}")
    print(f"captioning folder: {out}")


if __name__ == "__main__":
    main()
