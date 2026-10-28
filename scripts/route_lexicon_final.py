"""Merge the native-speaker check of the route landmark lexicon into lexicon_final.json.

Reads data/annotation/route_lexicon/lexicon_check.xlsx (one sheet per language: key, draft, correct?,
better word, note) over lexicon_draft.json. A filled "better word" replaces the draft; a language whose
sheet has no marks yet keeps its draft and is listed as unchecked. Then checks that within a language
no two names are equal and no landmark name is contained in another (labels are matched by substring).

Usage: python scripts/route_lexicon_final.py   (re-run after more sheets are checked)
"""

import json

import openpyxl

from mtcomm.paths import ANNOTATION_DIR

DIR = ANNOTATION_DIR / "route_lexicon"


def main():
    lex = json.loads((DIR / "lexicon_draft.json").read_text(encoding="utf-8"))
    wb = openpyxl.load_workbook(DIR / "lexicon_check.xlsx")
    checked, changes = [], {}
    for ws in wb.worksheets:
        lang = ws.title
        rows = [r for r in ws.iter_rows(min_row=3, values_only=True) if r and r[0]]
        if not any(r[2] not in (None, "") for r in rows):
            continue
        checked.append(lang)
        for key, draft, ok, better, _ in [(tuple(r) + (None,) * 5)[:5] for r in rows]:
            if better not in (None, ""):
                section = next(s for s in ("landmarks", "attributes", "directions") if key in lex[s])
                lex[section][key][lang] = str(better).strip()
                changes.setdefault(lang, []).append(f"{key}: {draft} -> {str(better).strip()}")
    problems = []
    for lang in ["en", "id", "su", "jv", "ko"]:
        names = [v[lang] for v in lex["landmarks"].values()]
        everything = names + [v[lang] for s in ("attributes", "directions") for v in lex[s].values()]
        if len(set(everything)) != len(everything):
            problems.append(f"{lang}: duplicate words")
        for a in names:
            for b in names:
                if a != b and a.casefold() in b.casefold():
                    problems.append(f"{lang}: '{a}' is inside '{b}'")
    lex["checked"] = checked
    lex["unchecked"] = [lang for lang in ["id", "su", "jv", "ko"] if lang not in checked]
    (DIR / "lexicon_final.json").write_text(json.dumps(lex, ensure_ascii=False, indent=1), encoding="utf-8")
    for lang, ch in changes.items():
        print(lang, ch)
    print("checked:", checked, "unchecked (draft kept):", lex["unchecked"])
    print("problems:", problems or "none")


if __name__ == "__main__":
    main()
