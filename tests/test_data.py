"""Tests for the data loaders in mtcomm.data, run against the real data files (read-only).

No network, no GPU. These need the data directory to be present.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from mtcomm import data  # noqa: E402
from mtcomm.paths import BLEND_DIR  # noqa: E402


@pytest.fixture(scope="module")
def hard_items():
    return data.load_image_items("hard")


@pytest.fixture(scope="module")
def blend_items():
    return data.load_blend_items()


@pytest.fixture(scope="module")
def native_questions():
    """(culture, BLEnD ID) -> native question text, read straight from the annotation files."""
    out = {}
    for path in (BLEND_DIR / "annotations").glob("*_data.json"):
        culture = path.name[: -len("_data.json")]
        for entry in json.loads(path.read_text(encoding="utf-8")):
            out[(culture, entry["ID"])] = entry["question"]
    return out


def test_hard_tier_has_60_items(hard_items):
    assert len(hard_items) == 60


def test_hard_candidates_are_unique_and_gold_points_at_target(hard_items):
    for it in hard_items:
        assert len(it["candidates"]) == 6, it["item_id"]
        assert len(set(it["candidates"])) == 6, it["item_id"]
        assert it["candidates"][it["gold"] - 1] == it["target"], it["item_id"]
        assert 1 <= it["gold"] <= 6, it["item_id"]


def test_hard_candidate_files_exist(hard_items):
    for it in hard_items:
        assert Path(it["target"]).is_file(), it["target"]
        for path in it["candidates"]:
            assert Path(path).is_file(), path


def test_image_items_are_deterministic_for_a_seed():
    first = data.load_image_items("hard", seed=0)
    again = data.load_image_items("hard", seed=0)
    other = data.load_image_items("hard", seed=1)
    assert [it["candidates"] for it in first] == [it["candidates"] for it in again]
    assert [it["candidates"] for it in first] != [it["candidates"] for it in other]


def test_image_captions_cover_requested_languages(hard_items):
    for it in hard_items:
        assert set(it["captions"]) == {"en", "ko", "id", "su", "jv"}
        for lang in ("en", "ko", "id"):
            assert isinstance(it["captions"][lang], str) and it["captions"][lang], (it["item_id"], lang)


def test_blend_has_280_items(blend_items):
    assert len(blend_items) == 280


def test_blend_visitor_lang_filter():
    jv = data.load_blend_items("jv")
    assert len(jv) == 40
    assert all(it["visitor_lang"] == "jv" for it in jv)


def test_blend_options_are_four_nonempty_strings(blend_items):
    for it in blend_items:
        assert sorted(it["options"]) == ["A", "B", "C", "D"], it["item_id"]
        for letter, text in it["options"].items():
            assert isinstance(text, str) and text.strip(), (it["item_id"], letter)


def test_blend_local_knowledge_contains_target_and_fits_budget(blend_items, native_questions):
    for it in blend_items:
        lk = it["local_knowledge"]
        assert 1 <= len(lk) <= 5, it["item_id"]
        target_native = native_questions[(it["local_culture"], it["qid"])]
        assert target_native in [k["question"] for k in lk], it["item_id"]


def test_blend_distractors_share_the_target_topic(blend_items):
    topics = {}
    for it in blend_items:
        culture = it["local_culture"]
        if culture not in topics:
            qs = json.loads((BLEND_DIR / "questions" / f"{culture}_questions.json").read_text(encoding="utf-8"))
            topics[culture] = {q["ID"]: q["Topic"] for q in qs}
        native_to_id = {}
        for entry in json.loads((BLEND_DIR / "annotations" / f"{culture}_data.json").read_text(encoding="utf-8")):
            native_to_id[entry["question"]] = entry["ID"]
        target_topic = topics[culture][it["qid"]]
        for k in it["local_knowledge"]:
            assert topics[culture][native_to_id[k["question"]]] == target_topic, it["item_id"]
