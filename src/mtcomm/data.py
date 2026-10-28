"""Data loaders for the PoC-1 tasks: image-reference items (XM3600) and local-visitor items (BLEnD).

Both loaders return plain dicts in the shape the game classes expect (games/image_ref.py and
games/blend_lv.py). Nothing here calls a network or a model.

Renderings: a visitor-language string is the "final" cell of the post-edit workbook
(blend_render_<lang>.xlsx) when that cell is non-empty, else the draft in <lang>_draft.json.

Usage: python -m mtcomm.data     (prints item counts)
"""

import csv
import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openpyxl

from .paths import ANNOTATION_DIR, BLEND_DIR, DATA_ROOT, OUTPUT_ROOT, XM3600_DIR

TIERS = ("random", "hard")
DEFAULT_CAPTION_LANGS = ("en", "ko", "id", "su", "jv")
USER_CAPTION_LANGS = ("su", "jv")  # written by hand in captions_to_write.csv, not in XM3600

MANIFEST = DATA_ROOT / "derived" / "xm3600" / "manifest_poc1_v2.json"  # v2: text-free targets
XM3600_CAPTIONS = XM3600_DIR / "captions" / "captions.jsonl"
XM3600_IMAGES = XM3600_DIR / "images"
USER_CAPTIONS_CSV = ANNOTATION_DIR / "xm3600_captioning" / "captions_to_write.csv"

BLEND_ITEMS = OUTPUT_ROOT / "blend" / "local_visitor_items.jsonl"
BLEND_RENDER_DIR = ANNOTATION_DIR / "blend_render"

LOCAL_PLACE = {
    "West_Java": "West Java",
    "South_Korea": "South Korea",
    "Indonesia": "Indonesia",
    "US": "the United States",
}


# ---------------------------------------------------------------- image reference (XM3600)

def _image_path(key: str, image_dir=None, ext: str = ".jpg") -> str:
    return str(Path(image_dir or XM3600_IMAGES) / f"{key}{ext}")


@lru_cache(maxsize=None)
def _xm3600_captions(langs: Tuple[str, ...]) -> Dict[str, Dict[str, Optional[str]]]:
    """image key -> {lang: first caption}, for the requested XM3600 languages only."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    with open(XM3600_CAPTIONS, encoding="utf-8") as f:
        first = True
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if first:
                unknown = [lang for lang in langs if lang not in rec]
                if unknown:
                    raise ValueError(f"not XM3600 languages: {unknown}")
                first = False
            caps = {}
            for lang in langs:
                texts = rec[lang].get("caption") or []
                caps[lang] = (texts[0].strip() or None) if texts else None
            out[rec["image/key"]] = caps
    return out


def _user_captions(langs: Tuple[str, ...]) -> Dict[str, Dict[str, Optional[str]]]:
    """item_id -> {lang: hand-written caption or None if the cell is empty."""
    if not langs:
        return {}
    with open(USER_CAPTIONS_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for lang in langs:
            if f"caption_{lang}" not in cols:
                raise ValueError(f"{USER_CAPTIONS_CSV.name} has no column caption_{lang}")
        out = {}
        for row in reader:
            out[row["item_id"].strip()] = {
                lang: (row[f"caption_{lang}"] or "").strip() or None for lang in langs
            }
    return out


def load_image_items(tier: str, caption_langs=DEFAULT_CAPTION_LANGS, n_candidates: int = 6, seed: int = 0,
                     manifest_path=None) -> List[dict]:
    """One item per manifest entry: the target plus the first n_candidates-1 distractors of `tier`,
    shuffled deterministically. `gold` is the 1-based position of the target in `candidates`."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    caption_langs = tuple(caption_langs)
    xm_langs = tuple(lang for lang in caption_langs if lang not in USER_CAPTION_LANGS)
    xm = _xm3600_captions(xm_langs) if xm_langs else {}
    user = _user_captions(tuple(lang for lang in caption_langs if lang in USER_CAPTION_LANGS))

    manifest_doc = json.loads(Path(manifest_path or MANIFEST).read_text(encoding="utf-8"))
    manifest = manifest_doc["items"]
    # Non-XM3600 sets (e.g. FoodX-251) name their image folder; they have no XM3600 captions.
    image_dir, image_ext = manifest_doc.get("image_dir"), manifest_doc.get("image_ext", ".jpg")
    items = []
    for it in manifest:
        item_id, target_key = it["item_id"], it["image_key"]
        distractors = it["distractors"][tier]
        if n_candidates - 1 > len(distractors):
            raise ValueError(f"{item_id}: {tier} tier has {len(distractors)} distractors, need {n_candidates - 1}")
        keys = [target_key] + list(distractors[: n_candidates - 1])
        if len(set(keys)) != len(keys):
            raise ValueError(f"{item_id}/{tier}: duplicate candidate keys {keys}")
        random.Random(f"{seed}-{item_id}-{tier}").shuffle(keys)

        captions = {}
        for lang in caption_langs:
            if lang in USER_CAPTION_LANGS:
                captions[lang] = user.get(item_id, {}).get(lang)
            else:
                captions[lang] = xm.get(target_key, {}).get(lang)

        items.append({
            "item_id": item_id,
            "tier": tier,
            "locale": it["locale"],
            "target": _image_path(target_key, image_dir, image_ext),
            "candidates": [_image_path(k, image_dir, image_ext) for k in keys],
            "gold": keys.index(target_key) + 1,
            "captions": captions,
        })
    return items


# ---------------------------------------------------------------- local visitor (BLEnD)

def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _xlsx_finals(lang: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Non-empty 'final' cells of the post-edit workbook: (questions by key, options by English)."""
    path = BLEND_RENDER_DIR / f"blend_render_{lang}.xlsx"
    if not path.exists():
        return {}, {}
    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        def finals(sheet: str, key_col: str) -> Dict[str, str]:
            rows = wb[sheet].iter_rows(values_only=True)
            header = [_cell(h) for h in next(rows)]
            ki, fi = header.index(key_col), header.index("final")
            out = {}
            for row in rows:
                key, final = _cell(row[ki]), _cell(row[fi])
                if key and final:
                    out[key] = final
            return out

        return finals("questions", "key"), finals("options", "english")
    finally:
        wb.close()


def _load_renderings(lang: str) -> dict:
    draft = json.loads((BLEND_RENDER_DIR / f"{lang}_draft.json").read_text(encoding="utf-8"))
    q_final, o_final = _xlsx_finals(lang)
    q_draft = {k: v.get("draft") for k, v in draft["questions"].items()}
    return {"q": (q_final, q_draft), "o": (o_final, draft["options"]), "lang": lang}


def _pick(rend: dict, which: str, key: str) -> Tuple[str, bool]:
    """(text, is_final): the final cell if non-empty, else the draft."""
    finals, drafts = rend[which]
    if key in finals:
        return finals[key], True
    text = drafts.get(key)
    if not text or not str(text).strip():
        raise KeyError(f"no {rend['lang']} rendering for {which} {key!r}")
    return text, False


@lru_cache(maxsize=None)
def _load_culture(culture: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """(native annotation entries by BLEnD ID, topic by BLEnD ID) for one culture."""
    annots = json.loads((BLEND_DIR / "annotations" / f"{culture}_data.json").read_text(encoding="utf-8"))
    topics = json.loads((BLEND_DIR / "questions" / f"{culture}_questions.json").read_text(encoding="utf-8"))
    return {a["ID"]: a for a in annots}, {q["ID"]: q["Topic"] for q in topics}


def _top_answers(entries: List[dict], n: int) -> List[dict]:
    ranked = sorted(entries, key=lambda a: -a["count"])[:n]
    return [{"answers": list(a["answers"]), "count": a["count"]} for a in ranked]


def load_blend_items(visitor_lang: Optional[str] = None, n_distractor_questions: int = 4,
                     max_answers: int = 5, seed: int = 0, items_path=None) -> List[dict]:
    """Local-visitor items in the shape games/blend_lv.py expects. local_knowledge holds the target
    question plus up to n_distractor_questions same-topic questions from the same culture, shuffled."""
    rows = [json.loads(line) for line in Path(items_path or BLEND_ITEMS).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if visitor_lang is not None:
        rows = [r for r in rows if r["visitor_lang"] == visitor_lang]

    renders: Dict[str, dict] = {}
    items = []
    for r in rows:
        vlang, culture, qid, item_id = r["visitor_lang"], r["local_culture"], r["qid"], r["item_id"]
        if vlang not in renders:
            renders[vlang] = _load_renderings(vlang)
        annot, topic = _load_culture(culture)
        if qid not in annot or qid not in topic:
            raise KeyError(f"{item_id}: {qid} missing from {culture} annotations or topics")

        question, q_final = _pick(renders[vlang], "q", f"{qid}@{culture}")
        all_final = q_final
        options = {}
        for letter, english in r["choices_en"].items():
            options[letter], o_final = _pick(renders[vlang], "o", english)
            all_final = all_final and o_final

        pool = sorted(i for i, a in annot.items() if i != qid and topic.get(i) == topic[qid] and a["annotations"])
        rng = random.Random(f"{seed}-{item_id}")
        order = [qid] + rng.sample(pool, min(n_distractor_questions, len(pool)))
        rng.shuffle(order)
        local_knowledge = [
            {"question": annot[i]["question"], "answers": _top_answers(annot[i]["annotations"], max_answers)}
            for i in order
        ]

        items.append({
            "item_id": item_id,
            "qid": qid,
            "local_culture": culture,
            "local_lang": r["local_lang"],
            "visitor_lang": vlang,
            "visitor_own_culture": r["visitor_own_culture"],
            "question": question,
            "options": options,
            "answer_idx": r["answer_idx"],
            "own_culture_idx": r["own_culture_idx"],
            "local_place": LOCAL_PLACE[culture],
            "local_knowledge": local_knowledge,
            "rendering_source": "final" if all_final else "draft",
        })
    return items


# ---------------------------------------------------------------- counts

def summarize() -> None:
    """Print item counts: image items per tier, caption coverage, blend items per (culture, visitor lang)."""
    for tier in TIERS:
        print(f"image items, tier={tier}: {len(load_image_items(tier))}")
    img = load_image_items("random")
    for lang in USER_CAPTION_LANGS:
        n = sum(1 for it in img if it["captions"][lang])
        print(f"  hand-written {lang} captions: {n}/{len(img)}")

    blend = load_blend_items()
    counts = Counter((it["local_culture"], it["visitor_lang"]) for it in blend)
    for (culture, vlang), n in sorted(counts.items()):
        print(f"blend items, {culture} -> visitor {vlang}: {n}")
    print(f"blend items total: {len(blend)}")


if __name__ == "__main__":
    summarize()


# ---------------------------------------------------------------- sequence ordering (Bloom-VIST)

# v3 = v2 plus two same-book distractor pages per story (scripts/bloomvist_add_distractors.py);
# v2 = stories the no-talk prior cannot put in order (scripts/sequence_prior_screen.py); v1 = first 30 built.
SEQUENCE_MANIFEST = next((DATA_ROOT / "derived" / "bloom_vist" / f"manifest_seq_v{v}.json" for v in (3, 2, 1)
                          if (DATA_ROOT / "derived" / "bloom_vist" / f"manifest_seq_v{v}.json").exists()),
                         DATA_ROOT / "derived" / "bloom_vist" / "manifest_seq_v1.json")


def load_sequence_items(seed: int = 0, manifest_path=None) -> List[dict]:
    """One item per story: `story` in true order, `shuffled` (the arranger's photos: the story plus any
    same-book `distractors`, never in story order), and `gold`, the arranger's photo number for each
    story position."""
    path = Path(manifest_path or SEQUENCE_MANIFEST)
    doc = json.loads(path.read_text(encoding="utf-8"))
    image_dir = path.parent / "images"
    items = []
    for i, s in enumerate(doc["stories"]):
        story = [str(image_dir / f) for f in s["images"]]
        distractors = [str(image_dir / f) for f in s.get("distractors", [])]
        rng = random.Random(f"{seed}-{s['story_id']}")
        shuffled = story + distractors
        while [p for p in shuffled if p in story] == story:
            rng.shuffle(shuffled)
        items.append({"item_id": f"S{i + 1:02d}", "story_id": s["story_id"], "title": s["title"],
                      "licence": s["licence"], "story": story, "distractors": distractors, "shuffled": shuffled,
                      "gold": [shuffled.index(p) + 1 for p in story]})
    return items


# ---------------------------------------------------------------- route redrawing (synthetic maps)

ROUTE_MAPS = DATA_ROOT / "derived" / "route" / "maps_v1.json"
ROUTE_LEXICON_DIR = ANNOTATION_DIR / "route_lexicon"


def _route_lexicon() -> dict:
    """The checked lexicon when it exists, else the draft (English needs neither)."""
    for name in ["lexicon_final.json", "lexicon_draft.json"]:
        p = ROUTE_LEXICON_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"landmarks": {}, "attributes": {}, "directions": {}}


def route_label(name: str, lang: str, lex: dict) -> str:
    """'well (broken)' -> the landmark and attribute words in `lang`."""
    if lang == "en":
        return name
    base, _, attr = name.partition(" (")
    out = lex["landmarks"][base][lang]
    return f"{out} ({lex['attributes'][attr.rstrip(')')][lang]})" if attr else out


def load_route_items(langs=("en",), maps_path=None) -> List[dict]:
    """Route items with labels and map images for each language in `langs` (images drawn on first use)."""
    from .routemap import draw_picture

    doc = json.loads(Path(maps_path or ROUTE_MAPS).read_text(encoding="utf-8"))
    lex = _route_lexicon()
    img_dir = Path(maps_path or ROUTE_MAPS).parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for m in doc["maps"]:
        names = {l["name"] for l in m["giver_landmarks"] + m["follower_landmarks"]}
        it = {"item_id": m["map_id"], "map": m, "labels": {}, "giver_image": {}, "follower_image": {},
              "directions": {}}
        for lang in langs:
            labels = {n: route_label(n, lang, lex) for n in names}
            it["labels"][lang] = labels
            it["directions"][lang] = ({"left": "left", "right": "right"} if lang == "en" else
                                      {k: lex["directions"][k][lang] for k in ("left", "right")})
            # The label hash in the name keeps images drawn from an older lexicon from being reused.
            tag = hashlib.sha256(json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]
            for who in ("giver", "follower"):
                p = img_dir / f"{m['map_id']}_{who}_{lang}_{tag}.png"
                if not p.exists():
                    draw_picture(m, who, p, label=labels.get, plain=True)
                it[f"{who}_image"][lang] = str(p)
        items.append(it)
    return items
