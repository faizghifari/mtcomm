"""Controlled near-twin image sets: each target plus 5 single-attribute edits (option A).

Stages (each can be re-run; results are cached on disk):
  propose   Gemini 3.8 Flash looks at the target and proposes 5 single-attribute edits
            (colour, count, position, add/remove, size/state), as JSON. Synchronous (one call per target).
  edit      Gemini 3.1 Flash Image renders, in one batch: the target passed through unchanged (so all
            six images carry the same generation artefacts) and the 5 edits.
  validate  Gemini 3.8 Flash compares each edit with the pass-through (is the change visible, as
            instructed, and the only change?), in one batch; SigLIP similarity is recorded too.
  sheet     Contact sheets for visual inspection.

Outputs go to <DATA_ROOT>/derived/perturb/<run>/.

Usage: python scripts/perturb_build.py {propose,edit,validate,sheet,all} --n 3 [--run test3]
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.backends import load_env  # noqa: E402
from mtcomm.paths import DATA_ROOT, XM3600_DIR  # noqa: E402

PROPOSER = "gemini-3.8-flash"
EDITOR = "gemini-3.1-flash-image"
ATTRIBUTES = ["color", "count", "position", "add_remove", "size_state"]

PROPOSE_PROMPT = """You are preparing a visual reference game. Look at this photo (an English caption is given for context: "{caption}").
Propose exactly 5 edits to this photo, one per attribute type, in this order: color, count, position, add_remove, size_state.
Each edit must:
- change exactly ONE clearly visible thing, so the edited photo is close to the original but certainly not identical;
- be easy to state in one short phrase (e.g. "the umbrella is green instead of red", "three cups instead of two");
- keep the camera, framing, lighting and everything else the same;
- be physically plausible.
Reply with JSON only: {{"edits": [{{"attribute": "...", "instruction": "imperative edit instruction for an image editor", "difference": "short phrase stating what differs from the original"}}, ...]}}"""

PASS_PROMPT = ("Reproduce this exact photo. Do not change, add or remove anything: keep every object, colour, "
               "position, the framing, lighting and style identical.")
EDIT_PROMPT = ("Edit this photo: {instruction}. Change nothing else: keep every other object, colour and position, "
               "the framing, lighting and photographic style identical.")
VALIDATE_PROMPT = """Photo 1 is the reference. Photo 2 is supposed to differ from it in exactly one way: "{difference}".
Answer with JSON only: {{"change_visible": true/false (is that difference clearly visible?), "matches_instruction": true/false, "other_changes": true/false (any other noticeable difference?), "notes": "short"}}"""


def client():
    from google import genai

    load_env()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def aspect(path):
    w, h = Image.open(path).size
    ratios = {"1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}
    return min(ratios, key=lambda k: abs(ratios[k] - w / h))


def run_batch(model, requests, workdir, name):
    """Submit {key: request} as one Gemini batch, wait, return {key: response dict or {'error': ...}}."""
    from google.genai import types

    c = client()
    path = workdir / f"{name}.jsonl"
    with open(path, "w") as f:
        for k, r in requests.items():
            f.write(json.dumps({"key": k, "request": r}) + "\n")
    up = c.files.upload(file=str(path), config=types.UploadFileConfig(mime_type="jsonl", display_name=path.name))
    job = c.batches.create(model=model, src=up.name, config={"display_name": f"mtcomm-perturb-{name}"})
    print(f"submitted {len(requests)} to {model}: {job.name}", flush=True)
    while True:
        st = str(c.batches.get(name=job.name).state)
        if any(s in st for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            break
        time.sleep(30)
    print(f"batch {name}: {st}", flush=True)
    out = {}
    if "SUCCEEDED" in st:
        raw = c.files.download(file=c.batches.get(name=job.name).dest.file_name).decode()
        for line in raw.splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r.get("response") or {"error": r.get("error")}
    return out


def parts(resp):
    cands = resp.get("candidates") or []
    return ((cands[0].get("content") or {}).get("parts") or []) if cands else []


def stage_propose(items, run_dir):
    c = client()
    from google.genai import types

    for it in items:
        out = run_dir / it["item_id"] / "proposals.json"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = XM3600_DIR / "images" / f"{it['image_key']}.jpg"
        resp = c.models.generate_content(
            model=PROPOSER,
            contents=[types.Part.from_bytes(data=img.read_bytes(), mime_type="image/jpeg"),
                      PROPOSE_PROMPT.format(caption=it["caption_en"])],
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               thinking_config=types.ThinkingConfig(thinking_level="low")))
        edits = json.loads(resp.text)["edits"]
        json.dump({"target": it, "edits": edits}, open(out, "w"), indent=1, ensure_ascii=False)
        print(it["item_id"], [e["difference"] for e in edits], flush=True)


def stage_edit(items, run_dir):
    reqs = {}
    for it in items:
        d = run_dir / it["item_id"]
        prop = json.load(open(d / "proposals.json"))
        img = XM3600_DIR / "images" / f"{it['image_key']}.jpg"
        cfg = {"response_modalities": ["IMAGE"], "image_config": {"aspect_ratio": aspect(img), "image_size": "1K"}}
        jobs = [("pass", PASS_PROMPT)] + [(f"e{i + 1}", EDIT_PROMPT.format(instruction=e["instruction"]))
                                          for i, e in enumerate(prop["edits"])]
        for name, prompt in jobs:
            if (d / f"{name}.png").exists():
                continue
            reqs[f"{it['item_id']}|{name}"] = {
                "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": b64(img)}},
                                                        {"text": prompt}]}],
                "generation_config": cfg}
    if not reqs:
        return
    res = run_batch(EDITOR, reqs, run_dir, "edit")
    for key, resp in res.items():
        item_id, name = key.split("|")
        imgs = [p["inlineData"]["data"] for p in parts(resp) if "inlineData" in p] or \
               [p["inline_data"]["data"] for p in parts(resp) if "inline_data" in p]
        if not imgs:
            print(f"{key}: no image ({json.dumps(resp)[:200]})")
            continue
        Image.open(io.BytesIO(base64.b64decode(imgs[0]))).save(run_dir / item_id / f"{name}.png")


def stage_validate(items, run_dir):
    from mtcomm.backends import GeminiBackend

    reqs = {}
    for it in items:
        d = run_dir / it["item_id"]
        prop = json.load(open(d / "proposals.json"))
        if not (d / "pass.png").exists():
            continue
        for i, e in enumerate(prop["edits"]):
            if not (d / f"e{i + 1}.png").exists():
                continue
            reqs[f"{it['item_id']}|e{i + 1}"] = {
                "contents": [{"role": "user", "parts": [
                    {"text": "Photo 1:"}, {"inline_data": {"mime_type": "image/png", "data": b64(d / "pass.png")}},
                    {"text": "Photo 2:"}, {"inline_data": {"mime_type": "image/png", "data": b64(d / f"e{i + 1}.png")}},
                    {"text": VALIDATE_PROMPT.format(difference=e["difference"])}]}],
                "generation_config": {"response_mime_type": "application/json",
                                      "thinking_config": {"thinking_level": "low"}}}
    res = run_batch(PROPOSER, reqs, run_dir, "validate")
    parse = GeminiBackend(PROPOSER).parse
    by_item = {}
    for key, resp in res.items():
        item_id, name = key.split("|")
        try:
            by_item.setdefault(item_id, {})[name] = json.loads(parse(resp))
        except Exception as e:
            by_item.setdefault(item_id, {})[name] = {"error": str(e)[:200]}
    for item_id, v in by_item.items():
        json.dump(v, open(run_dir / item_id / "validation.json", "w"), indent=1)
        print(item_id, {k: (x.get("change_visible"), x.get("other_changes")) for k, x in sorted(v.items())})


def stage_sheet(items, run_dir):
    from PIL import ImageDraw

    for it in items:
        d = run_dir / it["item_id"]
        names = ["orig", "pass"] + [f"e{i}" for i in range(1, 6)]
        prop = json.load(open(d / "proposals.json"))
        val = json.load(open(d / "validation.json")) if (d / "validation.json").exists() else {}
        tiles = []
        for n in names:
            p = XM3600_DIR / "images" / f"{it['image_key']}.jpg" if n == "orig" else d / f"{n}.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((360, 360))
            canvas = Image.new("RGB", (360, 410), "white")
            canvas.paste(im, (0, 0))
            label = n
            if n.startswith("e"):
                e = prop["edits"][int(n[1]) - 1]
                v = val.get(n, {})
                label = f"{n} {e['attribute']}: {e['difference'][:40]}  vis={v.get('change_visible')} other={v.get('other_changes')}"
            ImageDraw.Draw(canvas).text((4, 365), label[:70], fill="black")
            tiles.append(canvas)
        sheet = Image.new("RGB", (360 * 4, 410 * 2), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * 360, (i // 4) * 410))
        sheet.save(d / "sheet.jpg", quality=85)
        print(d / "sheet.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["propose", "edit", "validate", "sheet", "all"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--run", default="test3")
    ap.add_argument("--manifest", default=str(DATA_ROOT / "derived" / "xm3600" / "manifest_poc1_v2.json"))
    args = ap.parse_args()
    run_dir = DATA_ROOT / "derived" / "perturb" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    for line in open(XM3600_DIR / "captions" / "captions.jsonl"):
        r = json.loads(line)
        caps[r["image/key"]] = r["en"]["caption"][0]
    items = json.load(open(args.manifest))["items"][: args.n]
    for it in items:
        it["caption_en"] = caps[it["image_key"]]
    stages = ["propose", "edit", "validate", "sheet"] if args.stage == "all" else [args.stage]
    for s in stages:
        globals()[f"stage_{s}"](items, run_dir)


if __name__ == "__main__":
    main()
