"""Build local-visitor MCQ items from BLEnD annotations for the PoC pairs.

Options are built so the visitor cannot answer from world knowledge (pilot finding: with
BLEnD's own MCQs, whose distractors come from other countries, visitors answered 65-100%
correctly without asking anything). Each item has four options:
  - the local culture's top answer (gold), kept only if it has at least twice the votes of
    the runner-up, so the gold is unambiguous;
  - two lower-ranked answers from the SAME local culture (plausible for that place; only the
    local's vote counts say which is most common);
  - the visitor's own culture's top answer, for the own-culture fallback rate (it must differ
    from all three local options).
Option order is shuffled per item. Reports per-pair survival counts and the number of
unique option strings each visitor language must render.

Usage: python scripts/blend_build_pairs.py [--n-questions 40] [--seed 0]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import BLEND_DIR, OUTPUT_ROOT  # noqa: E402

# (local culture, local lang, visitor lang, visitor's own culture for the fallback option)
PAIRS = [
    ("West_Java", "su", "jv", "Indonesia"),  # no Javanese culture in BLEnD: Indonesia is a proxy
    ("West_Java", "su", "ko", "South_Korea"),
    ("South_Korea", "ko", "su", "West_Java"),
    ("Indonesia", "id", "ko", "South_Korea"),
    ("South_Korea", "ko", "id", "Indonesia"),
    ("US", "en", "ko", "South_Korea"),
    ("West_Java", "su", "id", "Indonesia"),  # optional nested-culture pair
]

# Pairs in a group share one question sample: reverse directions (asymmetry) and
# same-local different-visitor pairs (jv vs id visiting West Java).
GROUPS = [
    [("West_Java", "ko"), ("South_Korea", "su")],
    [("Indonesia", "ko"), ("South_Korea", "id")],
    [("West_Java", "jv"), ("West_Java", "id")],
    [("US", "ko")],
]

# BLEnD placeholders; never a usable answer.
SENTINELS = {"not-applicable", "not applicable", "idk", "no-answer", "i don't know", ""}
MIN_MARGIN = 2.0
# Questions whose annotations are degenerate (e.g. single-letter answers); dropped after
# sampling so the other samples stay unchanged.
BLOCKLIST = {"Th-en-48"}  # gold must have at least this many times the runner-up's votes


def load_annotations(culture):
    with open(BLEND_DIR / "annotations" / f"{culture}_data.json") as f:
        return {q["ID"]: q for q in json.load(f)}


def ranked_answers(q):
    """[(english answer, count)] sorted by votes, sentinels and duplicates dropped."""
    out, seen = [], set()
    for a in sorted(q["annotations"], key=lambda a: -a["count"]):
        en = (a.get("en_answers") or a.get("answers") or [""])[0].strip()
        if en.lower() in SENTINELS or en.lower() in seen:
            continue
        seen.add(en.lower())
        out.append((en, a["count"]))
    return out


def content_words(s):
    return {w for w in s.lower().replace("-", " ").split() if len(w) >= 4}


def overlaps(a, b):
    """Same answer in different words, e.g. 'valentino jebret' vs 'valentino simanjuntak'."""
    return a.lower() == b.lower() or bool(content_words(a) & content_words(b))


def build_options(local_q, visitor_q):
    """Return {"gold", "local_distractors", "own"} or None if the question is unusable for this pair.
    Distractors are the highest-ranked local answers that share no content word with the gold or
    each other; the visitor's own-culture answer must share none with any of the three."""
    local = ranked_answers(local_q)
    if len(local) < 3 or local[0][1] < MIN_MARGIN * local[1][1]:
        return None
    visitor = ranked_answers(visitor_q) if visitor_q else []
    if not visitor:
        return None
    gold = local[0][0]
    distractors = []
    for ans, _ in local[1:]:
        if not overlaps(ans, gold) and not any(overlaps(ans, d) for d in distractors):
            distractors.append(ans)
        if len(distractors) == 2:
            break
    if len(distractors) < 2:
        return None
    own = visitor[0][0]
    if any(overlaps(own, o) for o in [gold] + distractors):
        return None
    return {"gold": gold, "local_distractors": distractors, "own": own}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = {c: load_annotations(c) for c in {p[0] for p in PAIRS} | {p[3] for p in PAIRS}}
    valid = {}
    for local, _, vlang, vcult in PAIRS:
        valid[(local, vlang)] = {q for q in ann[local]
                                 if build_options(ann[local][q], ann[vcult].get(q)) is not None}
        print(f"{local:12s} -> visitor {vlang}: {len(valid[(local, vlang)])} usable questions")

    rng = random.Random(args.seed)
    chosen = {}
    for group in GROUPS:
        pool = sorted(set.intersection(*(valid[p] for p in group)))
        sample = sorted(rng.sample(pool, min(args.n_questions, len(pool))))
        print(f"group {group}: {len(pool)} shared, sampled {len(sample)}")
        for p in group:
            chosen[p] = sample

    strings = defaultdict(set)
    items = []
    for local, llang, vlang, vcult in PAIRS:
        for qid in chosen[(local, vlang)]:
            if qid in BLOCKLIST:
                continue
            opts = build_options(ann[local][qid], ann[vcult][qid])
            entries = [(opts["gold"], "local#1"), (opts["local_distractors"][0], "local#2"),
                       (opts["local_distractors"][1], "local#3"), (opts["own"], f"visitor_own:{vcult}")]
            random.Random(f"{args.seed}-{qid}-{local}-{vlang}").shuffle(entries)
            letters = "ABCD"
            choices = {letters[i]: e[0] for i, e in enumerate(entries)}
            source = {letters[i]: e[1] for i, e in enumerate(entries)}
            strings[vlang] |= set(choices.values())
            items.append({
                "item_id": f"{qid}@{local}__{vlang}",
                "qid": qid,
                "local_culture": local,
                "local_lang": llang,
                "visitor_lang": vlang,
                "visitor_own_culture": vcult,
                "visitor_culture_is_proxy": vlang == "jv",
                "choices_en": choices,
                "choice_source": source,
                "answer_idx": next(k for k, v in source.items() if v == "local#1"),
                "own_culture_idx": next(k for k, v in source.items() if v.startswith("visitor_own")),
            })

    print("unique option strings to render per visitor language:")
    for vlang, s in sorted(strings.items()):
        print(f"  {vlang}: {len(s)}")

    out_dir = OUTPUT_ROOT / "blend"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "local_visitor_items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out_dir / "chosen_qids.json", "w") as f:
        json.dump({"seed": args.seed, "qids": {f"{l}->{v}": q for (l, v), q in chosen.items()}}, f, indent=1)
    print(f"wrote {len(items)} items to {out_dir}")


if __name__ == "__main__":
    main()
