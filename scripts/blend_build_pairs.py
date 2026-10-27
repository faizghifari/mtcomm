"""Build local-visitor MCQ items from BLEnD for the PoC pairs.

For each (local culture, visitor) pair, keep BLEnD MCQ variants whose correct answer
is the local culture's and whose options include the visitor's own-culture answer
(so the own-culture default rate is defined). Reports per-pair survival counts and
the number of unique option strings each visitor language must render.

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

# (local culture, local lang, visitor lang, visitor's own culture for the default-rate option)
PAIRS = [
    ("West_Java", "su", "jv", "Indonesia"),  # no Javanese culture in BLEnD: Indonesia is a proxy
    ("West_Java", "su", "ko", "South_Korea"),
    ("South_Korea", "ko", "su", "West_Java"),
    ("Indonesia", "id", "ko", "South_Korea"),
    ("South_Korea", "ko", "id", "Indonesia"),
    ("US", "en", "ko", "South_Korea"),
    ("West_Java", "su", "id", "Indonesia"),  # optional nested-culture pair
]

# BLEnD placeholders that occasionally appear as MCQ options; never a usable answer.
SENTINELS = {"not-applicable", "not applicable", "idk", "no-answer", "i don't know"}

GROUPS = [
    [("West_Java", "ko"), ("South_Korea", "su")],
    [("Indonesia", "ko"), ("South_Korea", "id")],
    [("West_Java", "jv"), ("West_Java", "id")],
    [("US", "ko")],
]


def load_mcq():
    with open(BLEND_DIR / "mcq" / "mc_questions_file_v1.1.json") as f:
        rows = json.load(f)
    by_key = defaultdict(list)
    for r in rows:
        r["choices"] = json.loads(r["choices"])
        r["choice_countries"] = json.loads(r["choice_countries"])
        by_key[(r["ID"], r["country"])].append(r)
    return by_key


def candidates(by_key, qid, local, visitor_culture):
    """MCQ variants for (qid, local) that contain an option from the visitor's own culture."""
    out = []
    for r in by_key.get((qid, local), []):
        countries = r["choice_countries"]
        if countries[r["answer_idx"]] != local:
            continue
        if any(v.strip().lower() in SENTINELS for v in r["choices"].values()):
            continue
        if visitor_culture not in countries.values():
            continue
        own = [k for k, c in countries.items() if c == visitor_culture][0]
        if r["choices"][own].strip().lower() == r["choices"][r["answer_idx"]].strip().lower():
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_key = load_mcq()
    qids = sorted({q for q, _ in by_key})

    valid = {}
    for local, _, vlang, vcult in PAIRS:
        valid[(local, vlang)] = {q for q in qids if candidates(by_key, q, local, vcult)}
        print(f"{local:12s} -> visitor {vlang}: {len(valid[(local, vlang)])} questions with an own-culture option")

    # Pairs in a group share one question sample: reverse directions (asymmetry) and
    # same-local different-visitor pairs (jv vs id visiting West Java).
    rng = random.Random(args.seed)
    chosen = {}
    for group in GROUPS:
        pool = sorted(set.intersection(*(valid[p] for p in group)))
        sample = sorted(rng.sample(pool, min(args.n_questions, len(pool))))
        print(f"group {group}: {len(pool)} shared, sampled {len(sample)}")
        for p in group:
            chosen[p] = sample

    # Greedy variant choice per (question, pair): reuse option strings already in the visitor's set.
    seen_strings = defaultdict(set)
    items = []
    for local, llang, vlang, vcult in PAIRS:
        for qid in chosen[(local, vlang)]:
            cands = candidates(by_key, qid, local, vcult)
            best = min(cands, key=lambda r: (len(set(r["choices"].values()) - seen_strings[vlang]), r["MCQID"]))
            seen_strings[vlang] |= set(best["choices"].values())
            own = [k for k, c in best["choice_countries"].items() if c == vcult][0]
            items.append({
                "item_id": f"{best['MCQID']}__{vlang}",
                "qid": qid,
                "local_culture": local,
                "local_lang": llang,
                "visitor_lang": vlang,
                "visitor_own_culture": vcult,
                "visitor_culture_is_proxy": vlang == "jv",
                "mcq_id": best["MCQID"],
                "choices_en": best["choices"],
                "choice_countries": best["choice_countries"],
                "answer_idx": best["answer_idx"],
                "own_culture_idx": own,
            })

    print("unique option strings to render per visitor language:")
    for vlang, s in sorted(seen_strings.items()):
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
