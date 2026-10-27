"""List the strings each visitor language must render for the BLEnD local-visitor items.

Questions: where BLEnD has a native template in the visitor's language (su, id, ko) the
template is reused with the visitor's own country name swapped for the local's; Javanese
has no template, so its questions need a full draft. Options exist only in English.

Writes <ANNOTATION_DIR>/blend_render/<lang>_todo.json, the input for drafting.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import ANNOTATION_DIR, BLEND_DIR, OUTPUT_ROOT  # noqa: E402

# Visitor language -> (BLEnD culture whose native template it uses, name of that culture in the template)
TEMPLATE_SOURCE = {
    "su": ("West_Java", ["Jawa Barat"]),
    "id": ("Indonesia", ["Indonesia"]),
    "ko": ("South_Korea", ["대한민국", "한국"]),
}

# Local culture's name in each visitor language (draft; the native checker confirms).
COUNTRY_NAME = {
    "su": {"South_Korea": "Koréa Kidul"},
    "id": {"South_Korea": "Korea Selatan", "West_Java": "Jawa Barat"},
    "ko": {"West_Java": "서자바", "Indonesia": "인도네시아", "US": "미국"},
}

# Demonyms that survive the country-name swap and make the question inconsistent.
RESIDUAL_TERMS = {"su": ["Sunda"], "id": ["orang Indonesia"], "ko": ["한국인", "한국 사람", "한국의", "우리나라"]}

EN_COUNTRY = {"West_Java": "West Java", "Indonesia": "Indonesia", "South_Korea": "South Korea", "US": "US"}


def load_templates(culture):
    with open(BLEND_DIR / "questions" / f"{culture}_questions.json") as f:
        return {q["ID"]: q for q in json.load(f)}


def main():
    items = [json.loads(line) for line in open(OUTPUT_ROOT / "blend" / "local_visitor_items.jsonl")]
    en_templates = load_templates("US")

    questions = defaultdict(dict)
    options = defaultdict(set)
    for it in items:
        vlang, local = it["visitor_lang"], it["local_culture"]
        key = f"{it['qid']}@{local}"
        en_q = re.sub(r"\bUS\b", EN_COUNTRY[local], en_templates[it["qid"]]["Translation"])
        entry = {"key": key, "qid": it["qid"], "local_culture": local, "english": en_q}
        if vlang in TEMPLATE_SOURCE:
            culture, names = TEMPLATE_SOURCE[vlang]
            native = load_templates(culture)[it["qid"]]["Question"]
            target = COUNTRY_NAME[vlang][local]
            swapped = native
            for n in names:
                swapped = swapped.replace(n, target)
            entry.update(draft=swapped, draft_source=f"BLEnD {culture} template, country name swapped",
                         needs_review=swapped == native or any(t in swapped for t in RESIDUAL_TERMS[vlang]))
        else:
            entry.update(draft=None, draft_source="needs full draft", needs_review=True)
        questions[vlang][key] = entry
        options[vlang] |= set(it["choices_en"].values())

    out = ANNOTATION_DIR / "blend_render"
    out.mkdir(parents=True, exist_ok=True)
    for vlang in sorted(questions):
        todo = {"language": vlang, "questions": list(questions[vlang].values()),
                "options": sorted(options[vlang])}
        with open(out / f"{vlang}_todo.json", "w") as f:
            json.dump(todo, f, ensure_ascii=False, indent=1)
        flagged = sum(q["needs_review"] for q in todo["questions"])
        print(f"{vlang}: {len(todo['questions'])} questions ({flagged} need a full draft or review), "
              f"{len(todo['options'])} options")


if __name__ == "__main__":
    main()
