"""Data loaders for the MuG-Eval tasks: Easy Twenty Questions (entity words) and MCQ Conversation (Belebele).

Entity words: idx is the 1-based position in seed_en.jsonl. The Google lists are numbered 1..199 in
an order that differs from the seed, so id/ko are aligned to the seed by the English word
(exact string match; see read_google_list). Sundanese and Javanese come from the annotation
workbook when present (final cell if non-empty, else draft), else from the draft JSON.

MCQ items: Belebele rows keyed by (link, question_number), which are parallel across languages.
Only items present in all of MCQ_LANGS are sampled, so the same items appear for every pair.

Usage: python -m mtcomm.data_mugeval     (prints item counts)
"""

import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import openpyxl

from . import languages
from .paths import ANNOTATION_DIR, DATA_ROOT

WORDS_DIR = DATA_ROOT / "external" / "mugeval" / "words"
SEED_PATH = WORDS_DIR / "seed_en.jsonl"
GOOGLE_FILES = {"en": "google_eng_Latn.txt", "id": "google_ind_Latn.txt", "ko": "google_kor_Hang.txt"}
SU_JV_DIR = ANNOTATION_DIR / "mugeval_words"
SU_JV_DRAFT = SU_JV_DIR / "words_draft.json"
SU_JV_WORKBOOK = SU_JV_DIR / "words_su_jv.xlsx"

WORD_LANGS = ("en", "id", "ko", "su", "jv")
N_WORDS = 199
MCQ_LANGS = ("en", "id", "ko", "su", "jv")
BELEBELE = "facebook/belebele"

_NUMBERED = re.compile(r"^(\d+):\s*(.*?)\s*$")


# ---------------------------------------------------------------- entity words

def read_google_list(filename: str) -> Dict[int, str]:
    """number -> word for one Google-translated list. Line 1 is a URL; numbered lines are 'N: word'."""
    path = WORDS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Google word list not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("http"):
        raise ValueError(f"{path}: expected a URL on line 1")
    out: Dict[int, str] = {}
    for line in lines[1:]:
        m = _NUMBERED.match(line)
        if not m:
            continue  # blank lines and '---' separators
        num, word = int(m.group(1)), m.group(2)
        if num in out:
            raise ValueError(f"{path}: duplicate number {num}")
        if not word:
            raise ValueError(f"{path}: empty word for number {num}")
        out[num] = word
    if sorted(out) != list(range(1, N_WORDS + 1)):
        raise ValueError(f"{path}: expected numbers 1..{N_WORDS}, got {len(out)} entries")
    return out


@lru_cache(maxsize=None)
def _seed() -> Tuple[str, ...]:
    """English seed words in seed_en.jsonl order; position + 1 is idx."""
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"seed word list not found: {SEED_PATH}")
    words = []
    with open(SEED_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words.append(json.loads(line)["word"])
    if len(words) != N_WORDS:
        raise ValueError(f"{SEED_PATH}: expected {N_WORDS} words, got {len(words)}")
    return tuple(words)


@lru_cache(maxsize=None)
def _google_aligned(lang: str) -> Tuple[str, ...]:
    """Words for lang (id or ko) in seed order, matched through the English Google list."""
    eng = read_google_list(GOOGLE_FILES["en"])
    by_en: Dict[str, int] = {}
    for num, word in eng.items():
        key = word.casefold()
        if key in by_en:
            raise ValueError(f"{GOOGLE_FILES['en']}: duplicate English word {word!r}")
        by_en[key] = num
    target = read_google_list(GOOGLE_FILES[lang])
    out = []
    for word in _seed():
        if word.casefold() not in by_en:
            raise ValueError(f"seed word {word!r} not found in {GOOGLE_FILES['en']}")
        out.append(target[by_en[word.casefold()]])
    return tuple(out)


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


@lru_cache(maxsize=None)
def _su_jv(lang: str) -> Tuple[str, ...]:
    """Sundanese or Javanese words in seed order: *_final if non-empty, else draft. Keyed by English."""
    if not SU_JV_DRAFT.exists() and not SU_JV_WORKBOOK.exists():
        raise FileNotFoundError(
            f"{lang} words not found: neither {SU_JV_DRAFT} nor {SU_JV_WORKBOOK} exists yet")
    seed_keys = [w.casefold() for w in _seed()]
    drafts: Dict[str, str] = {}
    if SU_JV_DRAFT.exists():
        doc = json.loads(SU_JV_DRAFT.read_text(encoding="utf-8"))
        for row in doc["words"]:
            drafts[_cell(row["en"]).casefold()] = _cell(row.get(lang))
    finals: Dict[str, str] = {}
    wb_drafts: Dict[str, str] = {}
    if SU_JV_WORKBOOK.exists():
        wb = openpyxl.load_workbook(SU_JV_WORKBOOK, data_only=True)
        try:
            if "words" not in wb.sheetnames:
                raise ValueError(f"{SU_JV_WORKBOOK.name} has no sheet 'words'")
            rows = wb["words"].iter_rows(values_only=True)
            header = [_cell(h) for h in next(rows)]
            for col in ("en", f"{lang}_draft", f"{lang}_final"):
                if col not in header:
                    raise ValueError(f"{SU_JV_WORKBOOK.name} sheet 'words' has no column {col!r}")
            ei, di, fi = header.index("en"), header.index(f"{lang}_draft"), header.index(f"{lang}_final")
            for row in rows:
                en = _cell(row[ei]).casefold()
                if not en:
                    continue
                wb_drafts[en] = _cell(row[di])
                finals[en] = _cell(row[fi])
        finally:
            wb.close()

    out = []
    for key, word in zip(seed_keys, _seed()):
        value = finals.get(key) or drafts.get(key) or wb_drafts.get(key) or ""
        if not value:
            raise ValueError(f"no {lang} word for en={word!r} (seed idx {seed_keys.index(key) + 1})")
        out.append(value)
    return tuple(out)


def word_list(lang: str) -> List[str]:
    """The 199 entity names in seed order. Returns a fresh list each call."""
    if lang not in WORD_LANGS:
        raise ValueError(f"word_list language must be one of {WORD_LANGS}, got {lang!r}")
    if lang == "en":
        return list(_seed())
    if lang in ("id", "ko"):
        return list(_google_aligned(lang))
    return list(_su_jv(lang))


def load_twentyq_items(lang_q: str, lang_a: str, n: int = 30, n_candidates: int = 100, seed: int = 0) -> List[dict]:
    """Easy Twenty Questions items. Targets are the same for every language pair; the candidate list
    holds the target plus n_candidates-1 distractors, all in lang_q, shuffled per item."""
    if not 1 <= n <= N_WORDS:
        raise ValueError(f"n must be in 1..{N_WORDS}, got {n}")
    if not 2 <= n_candidates <= N_WORDS:
        raise ValueError(f"n_candidates must be in 2..{N_WORDS}, got {n_candidates}")
    words_q = word_list(lang_q)
    words_a = word_list(lang_a)

    perm = list(range(1, N_WORDS + 1))
    random.Random(seed).shuffle(perm)
    targets = perm[:n]

    items = []
    for idx in targets:
        rng = random.Random(f"{seed}-{idx}")
        others = [j for j in range(1, N_WORDS + 1) if j != idx]
        cand = [idx] + rng.sample(others, n_candidates - 1)
        rng.shuffle(cand)
        items.append({
            "item_id": f"tq-{idx}",
            "idx": idx,
            "target_q": words_q[idx - 1],
            "target_a": words_a[idx - 1],
            "candidates_q": [words_q[j - 1] for j in cand],
        })
    return items


# ---------------------------------------------------------------- MCQ (Belebele)

@lru_cache(maxsize=None)
def _belebele(lang: str) -> Dict[Tuple[str, int], dict]:
    """(link, question_number) -> Belebele row for one language. Cached; do not mutate."""
    import datasets

    nllb = languages.get(lang).nllb
    ds = datasets.load_dataset(BELEBELE, nllb, split="test")
    rows: Dict[Tuple[str, int], dict] = {}
    for row in ds:
        key = (row["link"], int(row["question_number"]))
        if key in rows:
            raise ValueError(f"duplicate Belebele key {key} in {nllb}")
        rows[key] = row
    return rows


def _item_id(link: str, question_number: int) -> str:
    return f"bb-{hashlib.sha256(link.encode('utf-8')).hexdigest()[:8]}-{question_number}"


def load_mcq_items(lang_q: str, lang_a: str, n: int = 30, seed: int = 0) -> List[dict]:
    """MCQ Conversation items: the passage in lang_a, the question and four options in lang_q, and the
    gold choice (1-4). The n items are sampled from the keys present in every language in MCQ_LANGS."""
    for lang in (lang_q, lang_a):
        if lang not in MCQ_LANGS:
            raise ValueError(f"MCQ languages must be in {MCQ_LANGS}, got {lang!r}")
    tables = {lang: _belebele(lang) for lang in MCQ_LANGS}
    keys = sorted(set.intersection(*(set(t) for t in tables.values())))
    if not 1 <= n <= len(keys):
        raise ValueError(f"n must be in 1..{len(keys)}, got {n}")
    chosen = random.Random(seed).sample(keys, n)

    items = []
    for key in chosen:
        link, qn = key
        q_row, a_row = tables[lang_q][key], tables[lang_a][key]
        items.append({
            "item_id": _item_id(link, qn),
            "link": link,
            "question_number": qn,
            "passage_a": a_row["flores_passage"],
            "question_q": q_row["question"],
            "options_q": [q_row[f"mc_answer{i}"] for i in range(1, 5)],
            "gold": int(q_row["correct_answer_num"]),
        })
    return items


def summarize() -> None:
    """Print item counts for the default settings."""
    print(f"entity words: {len(word_list('en'))} en, {len(word_list('id'))} id, {len(word_list('ko'))} ko")
    try:
        print(f"  su/jv: {len(word_list('su'))} su, {len(word_list('jv'))} jv")
    except FileNotFoundError as e:
        print(f"  su/jv: not available ({e})")
    print(f"twentyq items en->id: {len(load_twentyq_items('en', 'id'))}")
    print(f"mcq items en->id: {len(load_mcq_items('en', 'id'))}")


if __name__ == "__main__":
    summarize()
