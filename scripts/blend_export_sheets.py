"""Export one post-editing workbook per visitor language from the todo lists and drafts.

Each workbook has a "questions" sheet and an "options" sheet. Options carry the questions
they appear in, because short answers are ambiguous out of context ("date", "turkey").
Items whose drafted options collide (two options rendered identically) are flagged, since
the MCQ would be unanswerable in that language.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtcomm.paths import ANNOTATION_DIR, OUTPUT_ROOT  # noqa: E402

RENDER_DIR = ANNOTATION_DIR / "blend_render"
LANG_NAMES = {"su": "Sundanese", "jv": "Javanese", "id": "Indonesian", "ko": "Korean"}
FLAG = PatternFill("solid", fgColor="FFF2CC")


def sheet(wb, title, header, rows, widths):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append(r)
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    return ws


def main():
    items = [json.loads(line) for line in open(OUTPUT_ROOT / "blend" / "local_visitor_items.jsonl")]
    for lang in ["su", "jv", "id", "ko"]:
        todo = json.load(open(RENDER_DIR / f"{lang}_todo.json"))
        draft = json.load(open(RENDER_DIR / f"{lang}_draft.json"))
        en_q = {q["key"]: q["english"] for q in todo["questions"]}

        used_in = defaultdict(list)
        collisions = defaultdict(list)
        for it in items:
            if it["visitor_lang"] != lang:
                continue
            key = f"{it['qid']}@{it['local_culture']}"
            for v in it["choices_en"].values():
                used_in[v].append(en_q[key])
            rendered = [draft["options"][v].strip().lower() for v in it["choices_en"].values()]
            if len(set(rendered)) < len(rendered):
                for v in it["choices_en"].values():
                    collisions[v].append(it["item_id"])

        wb = Workbook()
        wb.remove(wb.active)
        q_rows = []
        for q in todo["questions"]:
            d = draft["questions"][q["key"]]
            q_rows.append([q["key"], q["local_culture"], q["english"], d["draft"],
                           q["draft_source"] + (f"; edited: {d.get('note')}" if d.get("changed") else ""), "", ""])
        ws = sheet(wb, "questions", ["key", "local_culture", "english", "draft", "draft_source", "final", "comment"],
                   q_rows, [22, 14, 50, 50, 40, 50, 30])
        o_rows = []
        for o in todo["options"]:
            ctx = sorted(set(used_in[o]))
            o_rows.append([o, draft["options"][o], "", "; ".join(ctx[:3]) + (f" (+{len(ctx) - 3} more)" if len(ctx) > 3 else ""),
                           "SAME AS ANOTHER OPTION in " + ", ".join(collisions[o]) if collisions[o] else "", ""])
        wo = sheet(wb, "options", ["english", "draft", "final", "used_in_questions", "flag", "comment"],
                   o_rows, [28, 28, 28, 70, 30, 30])
        for row in wo.iter_rows(min_row=2):
            if row[4].value:
                for c in row:
                    c.fill = FLAG

        readme = wb.create_sheet("README", 0)
        for line in [
            f"BLEnD local-visitor renderings: {LANG_NAMES[lang]} ({lang})",
            "Drafts are by Claude Haiku (outside the translator set under test). Please post-edit into 'final'.",
            "Leave 'final' empty if the draft is already correct; it will be used as-is.",
            "questions: the visitor's question about the LOCAL culture named in local_culture, in your language.",
            "options: short answers. Check the meaning against used_in_questions (e.g. 'date' the fruit vs. the day).",
            "Highlighted rows: two options of one item have the same draft, so that item is unanswerable; please make them distinct.",
            "Keep culture-specific names (dishes, places) as they are normally written in your language.",
        ]:
            readme.append([line])
        readme.column_dimensions["A"].width = 120
        path = RENDER_DIR / f"blend_render_{lang}.xlsx"
        wb.save(path)
        n_coll = sum(1 for o in todo["options"] if collisions[o])
        print(f"{lang}: {len(q_rows)} questions, {len(o_rows)} options, {n_coll} options flagged for collisions -> {path}")


if __name__ == "__main__":
    main()
