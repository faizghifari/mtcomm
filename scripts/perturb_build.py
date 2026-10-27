"""Controlled near-twin image sets: each target plus 5 single-attribute edits (option A).

Pipeline (all model calls in Gemini batches; every stage is cached on disk and resumable):
  1. propose   Gemini 3.8 Flash proposes 5 edits for the target, one per attribute type
               (colour, count, add/remove, size/state, pattern/material). Position edits were
               dropped: the editor ignored them in 2 of 3 test cases.
  2. render    Gemini 3.1 Flash Image renders the target passed through unchanged ("pass", which is
               the target shown in the game, so all six images share the generator's look) and each edit.
  3. check     an edit is kept only if
                 - Gemini 3.8 Flash, comparing it with "pass", says the stated difference is visible, matches
                   the instruction, and nothing else changed; and
                 - pixel checks against "pass" pass: changed area >= MIN_AREA (not invisibly small),
                   at most MAX_GLOBAL_FRAC of the image changed and mean brightness shift <= MAX_SHIFT
                   (no global relighting, tint or reframing, which would give the odd one out away).
  4. repair    rejected slots get a new proposal (same attribute, avoiding earlier ones), then render
               and check again; up to --repair-rounds rounds.
  5. manifest  targets with 5 valid edits become a manifest (target = pass, hard distractors = edits).

Outputs: <DATA_ROOT>/derived/perturb/<run>/<item_id>/{pass,e1..e5}.png, edits.json, sheet.jpg, and
<DATA_ROOT>/derived/perturb/<run>/manifest_perturb_<run>.json.

Usage: python scripts/perturb_build.py --run full60 [--n 60] [--repair-rounds 2]
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.backends import GeminiBackend, load_env  # noqa: E402
from mtcomm.paths import DATA_ROOT, XM3600_DIR  # noqa: E402

PROPOSER = "gemini-3.8-flash"
EDITOR = "gemini-3.1-flash-image"
ATTRIBUTES = ["color", "count", "add_remove", "size_state", "pattern_material"]
MIN_AREA = 0.001  # fraction of pixels (256px grid) that must visibly change
MAX_GLOBAL_FRAC = 0.25  # fraction of a 64px grid allowed to change (more = relighting / reframing)
MAX_SHIFT = 0.03  # mean brightness shift allowed

RULES = """Each edit must:
- change exactly ONE thing, and that thing must be large enough to see clearly in a small thumbnail
  (an object covering at least a few percent of the photo, not a distant speck);
- be easy to state in one short phrase (e.g. "the umbrella is green instead of red", "three cups instead of two");
- keep the camera, framing, lighting, colours of everything else, and all other objects the same;
- be physically plausible."""

PROPOSE_PROMPT = """You are preparing a visual reference game. Look at this photo (English caption for context: "{caption}").
Propose exactly 5 edits, one per attribute type, in this order: color, count, add_remove, size_state, pattern_material.
""" + RULES + """
Reply with JSON only: {{"edits": [{{"attribute": "...", "instruction": "imperative edit instruction for an image editor", "difference": "short phrase stating what differs from the original"}}, ...]}}"""

REPAIR_PROMPT = """You are preparing a visual reference game. Look at this photo (English caption for context: "{caption}").
Propose ONE new edit of attribute type "{attribute}". Do not repeat any of these earlier attempts: {avoid}.
""" + RULES + """
Reply with JSON only: {{"attribute": "{attribute}", "instruction": "...", "difference": "..."}}"""

PASS_PROMPT = ("Reproduce this exact photo. Do not change, add or remove anything: keep every object, colour, "
               "position, the framing, lighting and style identical.")
EDIT_PROMPT = ("Edit this photo: {instruction}. Change nothing else: keep every other object, colour and position, "
               "the framing, lighting, white balance and photographic style identical.")
VALIDATE_PROMPT = """Photo 1 is the reference. Photo 2 is supposed to differ from it in exactly one way: "{difference}".
Answer with JSON only: {{"change_visible": true/false (is that difference clearly visible?), "matches_instruction": true/false, "other_changes": true/false (any other noticeable difference, including lighting, colour tint or framing?), "notes": "short"}}"""


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


def run_batch(model, requests, run_dir, name):
    """Submit {key: request} as one Gemini batch (job id saved for resumption), wait, return {key: response}."""
    from google.genai import types

    if not requests:
        return {}
    c = client()
    state_path = run_dir / f"batch_{name}.json"
    if state_path.exists():
        job_name = json.load(open(state_path))["job"]
    else:
        path = run_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for k, r in requests.items():
                f.write(json.dumps({"key": k, "request": r}) + "\n")
        up = c.files.upload(file=str(path), config=types.UploadFileConfig(mime_type="jsonl", display_name=path.name))
        job_name = c.batches.create(model=model, src=up.name, config={"display_name": f"mtcomm-perturb-{name}"}).name
        json.dump({"job": job_name, "n": len(requests)}, open(state_path, "w"))
    print(f"[{time.strftime('%H:%M:%S')}] batch {name}: {len(requests)} requests -> {job_name}", flush=True)
    while True:
        st = str(c.batches.get(name=job_name).state)
        if any(s in st for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
            break
        time.sleep(30)
    print(f"[{time.strftime('%H:%M:%S')}] batch {name}: {st}", flush=True)
    out = {}
    if "SUCCEEDED" in st:
        raw = c.files.download(file=c.batches.get(name=job_name).dest.file_name).decode()
        for line in raw.splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r.get("response") or {"error": r.get("error")}
    return out


def parts(resp):
    cands = resp.get("candidates") or []
    return ((cands[0].get("content") or {}).get("parts") or []) if cands else []


def json_text(resp):
    return json.loads(GeminiBackend(PROPOSER).parse(resp))


def image_part(path, mime="image/jpeg"):
    return {"inline_data": {"mime_type": mime, "data": b64(path)}}


def text_request(img_path, prompt, mime="image/jpeg"):
    return {"contents": [{"role": "user", "parts": [image_part(img_path, mime), {"text": prompt}]}],
            "generation_config": {"response_mime_type": "application/json", "thinking_config": {"thinking_level": "low"}}}


def pixel_checks(pass_path, edit_path):
    def load(p, s):
        return np.asarray(Image.open(p).convert("RGB").resize((s, s)), dtype=float) / 255

    a64, b64_ = load(pass_path, 64), load(edit_path, 64)
    a256, b256 = load(pass_path, 256), load(edit_path, 256)
    area = float((np.abs(a256 - b256).mean(2) > 0.12).mean())
    global_frac = float((np.abs(a64 - b64_).mean(2) > 0.05).mean())
    shift = float(abs(a64.mean() - b64_.mean()))
    ok = area >= MIN_AREA and global_frac <= MAX_GLOBAL_FRAC and shift <= MAX_SHIFT
    return {"area": round(area, 5), "global_frac": round(global_frac, 3), "shift": round(shift, 4), "ok": ok}


def render_requests(items, run_dir, slots):
    """slots: list of (item_id, name, instruction or None for pass)."""
    reqs = {}
    for item_id, name, instruction in slots:
        it = items[item_id]
        img = XM3600_DIR / "images" / f"{it['image_key']}.jpg"
        prompt = PASS_PROMPT if instruction is None else EDIT_PROMPT.format(instruction=instruction)
        reqs[f"{item_id}|{name}"] = {
            "contents": [{"role": "user", "parts": [image_part(img), {"text": prompt}]}],
            "generation_config": {"response_modalities": ["IMAGE"],
                                  "image_config": {"aspect_ratio": aspect(img), "image_size": "1K"}}}
    return reqs


def save_renders(res, run_dir, suffix):
    for key, resp in res.items():
        item_id, name = key.split("|")
        imgs = [p.get("inlineData", p.get("inline_data", {})).get("data") for p in parts(resp)
                if "inlineData" in p or "inline_data" in p]
        if imgs and imgs[0]:
            Image.open(io.BytesIO(base64.b64decode(imgs[0]))).save(run_dir / item_id / f"{name}{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full60")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "derived" / "xm3600" / "manifest_poc1_v2.json"))
    args = ap.parse_args()
    run_dir = DATA_ROOT / "derived" / "perturb" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    for line in open(XM3600_DIR / "captions" / "captions.jsonl"):
        r = json.loads(line)
        caps[r["image/key"]] = r["en"]["caption"][0]
    items = {it["item_id"]: {**it, "caption_en": caps[it["image_key"]]}
             for it in json.load(open(args.manifest))["items"][: args.n]}
    for item_id in items:
        (run_dir / item_id).mkdir(exist_ok=True)

    def edits_path(i):
        return run_dir / i / "edits.json"

    # 1. propose
    todo = {i: it for i, it in items.items() if not edits_path(i).exists()}
    res = run_batch(PROPOSER, {i: text_request(XM3600_DIR / "images" / f"{it['image_key']}.jpg",
                                               PROPOSE_PROMPT.format(caption=it["caption_en"]))
                               for i, it in todo.items()}, run_dir, "propose")
    for i, resp in res.items():
        try:
            edits = json_text(resp)["edits"][:5]
            slots = {f"e{k + 1}": {**e, "attempts": [e["difference"]], "status": "pending", "round": 0}
                     for k, e in enumerate(edits)}
            json.dump({"slots": slots}, open(edits_path(i), "w"), indent=1, ensure_ascii=False)
        except Exception as e:
            print(f"{i}: proposal failed: {e}")

    # 2-4. render, check, repair
    for rnd in range(args.repair_rounds + 1):
        slots = []
        for i in items:
            if not edits_path(i).exists():
                continue
            if not (run_dir / i / "pass.png").exists():
                slots.append((i, "pass", None))
            for name, s in json.load(open(edits_path(i)))["slots"].items():
                if s["status"] == "pending":
                    slots.append((i, name, s["instruction"]))
        save_renders(run_batch(EDITOR, render_requests(items, run_dir, slots), run_dir, f"render_r{rnd}"), run_dir, "")

        checks = {}
        for i, name, instr in slots:
            if name == "pass" or not (run_dir / i / f"{name}.png").exists() or not (run_dir / i / "pass.png").exists():
                continue
            s = json.load(open(edits_path(i)))["slots"][name]
            checks[f"{i}|{name}"] = {"contents": [{"role": "user", "parts": [
                {"text": "Photo 1:"}, image_part(run_dir / i / "pass.png", "image/png"),
                {"text": "Photo 2:"}, image_part(run_dir / i / f"{name}.png", "image/png"),
                {"text": VALIDATE_PROMPT.format(difference=s["difference"])}]}],
                "generation_config": {"response_mime_type": "application/json",
                                      "thinking_config": {"thinking_level": "low"}}}
        verdicts = run_batch(PROPOSER, checks, run_dir, f"check_r{rnd}")

        repairs = {}
        for i in items:
            if not edits_path(i).exists():
                continue
            doc = json.load(open(edits_path(i)))
            for name, s in doc["slots"].items():
                if s["status"] != "pending":
                    continue
                img = run_dir / i / f"{name}.png"
                v = {}
                if f"{i}|{name}" in verdicts:
                    try:
                        v = json_text(verdicts[f"{i}|{name}"])
                    except Exception as e:
                        v = {"error": str(e)[:200]}
                px = pixel_checks(run_dir / i / "pass.png", img) if img.exists() and (run_dir / i / "pass.png").exists() else {"ok": False}
                ok = bool(v.get("change_visible")) and bool(v.get("matches_instruction")) and \
                    not v.get("other_changes", True) and px["ok"]
                s.update(validator=v, pixels=px, round=rnd, status="valid" if ok else "rejected")
                if not ok and rnd < args.repair_rounds:
                    repairs[f"{i}|{name}"] = text_request(
                        XM3600_DIR / "images" / f"{items[i]['image_key']}.jpg",
                        REPAIR_PROMPT.format(caption=items[i]["caption_en"], attribute=s["attribute"],
                                             avoid=json.dumps(s["attempts"])))
            json.dump(doc, open(edits_path(i), "w"), indent=1, ensure_ascii=False)
        n_valid = sum(s["status"] == "valid" for i in items if edits_path(i).exists()
                      for s in json.load(open(edits_path(i)))["slots"].values())
        print(f"round {rnd}: {n_valid} valid edits; {len(repairs)} to repair", flush=True)
        if not repairs:
            break
        for key, resp in run_batch(PROPOSER, repairs, run_dir, f"repair_r{rnd}").items():
            i, name = key.split("|")
            doc = json.load(open(edits_path(i)))
            try:
                e = json_text(resp)
                old = doc["slots"][name]
                doc["slots"][name] = {**e, "attempts": old["attempts"] + [e["difference"]], "status": "pending",
                                      "round": rnd + 1}
                (run_dir / i / f"{name}.png").rename(run_dir / i / f"{name}_rejected_r{rnd}.png")
            except Exception as ex:
                print(f"{key}: repair proposal failed: {ex}")
            json.dump(doc, open(edits_path(i), "w"), indent=1, ensure_ascii=False)

    # 5. manifest
    out_items = []
    for i, it in items.items():
        if not edits_path(i).exists():
            continue
        slots = json.load(open(edits_path(i)))["slots"]
        valid = [n for n, s in slots.items() if s["status"] == "valid"]
        if len(valid) == 5 and (run_dir / i / "pass.png").exists():
            out_items.append({"item_id": i, "image_key": f"{i}/pass", "locale": it["locale"],
                              "source_image_key": it["image_key"],
                              "distractors": {"hard": [f"{i}/{n}" for n in valid]},
                              "differences": {f"{i}/{n}": {"attribute": slots[n]["attribute"],
                                                           "difference": slots[n]["difference"]} for n in valid}})
    manifest = {"version": f"perturb_{args.run}", "image_dir": str(run_dir), "image_ext": ".png",
                "editor": EDITOR, "proposer": PROPOSER,
                "checks": {"min_area": MIN_AREA, "max_global_frac": MAX_GLOBAL_FRAC, "max_shift": MAX_SHIFT},
                "note": "Target is the unchanged re-render ('pass'); each distractor differs in one stated attribute.",
                "items": out_items}
    json.dump(manifest, open(run_dir / f"manifest_perturb_{args.run}.json", "w"), indent=1, ensure_ascii=False)
    print(f"manifest: {len(out_items)} of {len(items)} targets have 5 valid edits")


if __name__ == "__main__":
    main()
