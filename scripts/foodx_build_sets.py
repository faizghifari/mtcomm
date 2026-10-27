"""MD3-style image sets from FoodX-251 (option B): six photos of the SAME dish per set.

MD3 shows the guesser images of one fine-grained class (one dog breed, bird species or dish),
so a one-word answer never works and the pair must talk about plating, colours, background.
Here: sample --n-sets dish classes; in each, pick a target and its 5 most similar same-dish
photos by SigLIP, skipping near-duplicates (similarity above --dup-threshold) and photos
with readable text (EasyOCR). No solvability filter, unlike the XM3600 v1/v2 subsets.

FoodX-251 terms: non-commercial research only, no redistribution of the images; a released
benchmark should reference image IDs.

Usage: python scripts/foodx_build_sets.py [--n-sets 60] [--seed 0]
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import DATA_ROOT  # noqa: E402

FOODX = DATA_ROOT / "external" / "foodx251"
OUT = DATA_ROOT / "derived" / "foodx251"
MODEL_ID = "google/siglip-so400m-patch14-384"


def load_classes():
    names = {}
    for line in open(FOODX / "class_list.txt"):
        idx, name = line.split(maxsplit=1)
        names[int(idx)] = name.strip()
    by_class = defaultdict(list)
    for row in csv.reader(open(FOODX / "val_info.csv")):
        by_class[int(row[1])].append(row[0])
    return names, by_class


@torch.no_grad()
def embed(paths, batch=32):
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        f = model.get_image_features(pixel_values=px)
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def has_text(reader, path, min_conf=0.3, min_area=0.0005):
    w, h = Image.open(path).size
    for box, text, conf in reader.readtext(str(path)):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        if conf >= min_conf and (max(xs) - min(xs)) * (max(ys) - min(ys)) / (w * h) >= min_area \
                and sum(ch.isalnum() for ch in text) >= 2:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sets", type=int, default=60)
    ap.add_argument("--n-distractors", type=int, default=5)
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import easyocr

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    names, by_class = load_classes()
    rng = random.Random(args.seed)
    classes = sorted(c for c, imgs in by_class.items() if len(imgs) >= 20)
    rng.shuffle(classes)
    img_dir = FOODX / "val_set"
    items = []
    for c in classes:
        if len(items) >= args.n_sets:
            break
        files = sorted(by_class[c])
        files = [f for f in files if not has_text(reader, img_dir / f)]
        if len(files) < args.n_distractors + 1:
            continue
        emb = embed([img_dir / f for f in files])
        sim = emb @ emb.T
        t = rng.randrange(len(files))
        order = [j for j in np.argsort(-sim[t]) if j != t and sim[t, j] < args.dup_threshold]
        hard = order[: args.n_distractors]
        if len(hard) < args.n_distractors:
            continue
        items.append({"item_id": f"f{len(items) + 1:03d}", "image_key": Path(files[t]).stem, "class": names[c],
                      "locale": "foodx251",
                      "distractors": {"hard": [Path(files[j]).stem for j in hard],
                                      "random": [Path(files[j]).stem for j in rng.sample(
                                          [k for k in range(len(files)) if k != t], args.n_distractors)]},
                      "hard_max_sim": float(sim[t, hard].max()), "hard_min_sim": float(sim[t, hard].min())})
        print(f"{items[-1]['item_id']} {names[c]:28s} sims {sim[t, hard].min():.3f}-{sim[t, hard].max():.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "food_v1", "source": "FoodX-251 val", "image_dir": str(img_dir), "image_ext": ".jpg",
                "seed": args.seed, "model": MODEL_ID, "dup_threshold": args.dup_threshold,
                "note": "Same-dish sets (MD3 style). 'random' distractors are random photos of the same dish.",
                "items": items}
    json.dump(manifest, open(OUT / "manifest_food_v1.json", "w"), indent=1)
    print(f"wrote {len(items)} sets to {OUT / 'manifest_food_v1.json'}")


if __name__ == "__main__":
    main()
