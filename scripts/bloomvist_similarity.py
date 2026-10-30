"""Within-story visual similarity of Bloom-VIST stories (SigLIP image embeddings).

A story whose pages look alike (same characters, same setting) can only be ordered from what changes
between pages, so its order has to travel as language about actions and states, not as a picture
inventory. Writes, per story in the pool, the mean pairwise cosine similarity of its pages.

Usage: python scripts/bloomvist_similarity.py   -> data/derived/bloom_vist/similarity_pool120.json
"""

import itertools
import json

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from mtcomm.paths import DATA_ROOT

D = DATA_ROOT / "derived" / "bloom_vist"
MODEL = "google/siglip-so400m-patch14-384"


def main():
    pool = json.load(open(D / "manifest_seq_pool120.json"))["stories"]
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    out = {}
    for s in pool:
        imgs = [Image.open(D / "images" / f).convert("RGB") for f in s["images"]]
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
            f = model.get_image_features(pixel_values=x)
            f = getattr(f, "pooler_output", f)  # transformers 5 returns an output object
            e = torch.nn.functional.normalize(f.float(), dim=-1)
        sims = [float(e[i] @ e[j]) for i, j in itertools.combinations(range(len(imgs)), 2)]
        out[s["story_id"]] = round(sum(sims) / len(sims), 4)
    json.dump(out, open(D / "similarity_pool120.json", "w"), indent=1)
    vals = sorted(out.values())
    print(f"{len(out)} stories; similarity min {vals[0]}, median {vals[len(vals) // 2]}, max {vals[-1]}")


if __name__ == "__main__":
    main()
