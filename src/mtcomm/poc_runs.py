"""The runs behind the PoC results, with one player (3.8 Flash) on every task. Analysis scripts read this list.

Sequence and image keep their main2 runs, which already used 3.8 Flash. Route is the easy game on 30 maps: main2's first 15
and the 15 of main3_route_e16, merged by scripts/merge_route.py; a condition goes in only when it has all 30 maps. MCQ and Twenty
Questions are new runs. Culture is a control until its translators run.
"""

from .paths import OUTPUT_ROOT

ANALYSIS = OUTPUT_ROOT / "analysis_poc3"
# title, run, task
FINAL = [("MCQ conversation", "main3_mcq", "mcq"), ("Twenty Questions", "main3_twentyq", "twentyq"),
         ("Sequence ordering", "main2x_sequence", "sequence"), ("Image near-twin", "main2xf_image", "image"),
         ("Route, easy maps", "poc3_route", "route"), ("Culture (BLEnD)", "main3_culture", "culture")]
RUNS = [run for _, run, _ in FINAL]


def available():
    """FINAL without runs that have not finished."""
    return [t for t in FINAL if (OUTPUT_ROOT / t[1] / "records.jsonl").exists()]
