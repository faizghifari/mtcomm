"""Tests for mtcomm.data_mugeval and the MuG-Eval games on real items.

Word lists and Twenty Questions items come from the local files only. MCQ items load Belebele from
the Hugging Face hub or the local HF cache; if that fails, the MCQ tests skip. The su/jv word tests
skip until the draft file exists.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from mtcomm import data_mugeval as dm  # noqa: E402
from mtcomm.channels import IdentityChannel  # noqa: E402
from mtcomm.game import Move, run_episode  # noqa: E402
from mtcomm.games.mugeval import MCQConversationGame, TwentyQuestionsGame  # noqa: E402
from mtcomm.players import ScriptedPlayer  # noqa: E402

needs_su_jv = pytest.mark.skipif(not dm.SU_JV_DRAFT.exists(),
                                 reason=f"su/jv draft not written yet: {dm.SU_JV_DRAFT}")


def _seed_words_from_file():
    with open(dm.SEED_PATH, encoding="utf-8") as f:
        return [json.loads(line)["word"] for line in f if line.strip()]


def _google_eng_number_by_word():
    eng = dm.read_google_list(dm.GOOGLE_FILES["en"])
    return {word: num for num, word in eng.items()}


# ---------------------------------------------------------------- word lists

def test_en_list_is_seed_order():
    seed = _seed_words_from_file()
    assert len(seed) == 199
    assert dm.word_list("en") == seed


def test_google_english_list_matches_seed_words_exactly():
    # Same set of words, exact strings, each once. The Google order differs from the seed order.
    eng = dm.read_google_list(dm.GOOGLE_FILES["en"])
    assert sorted(eng.values()) == sorted(_seed_words_from_file())


@pytest.mark.parametrize("lang", ["id", "ko"])
def test_google_aligned_list_has_199_nonempty_entries(lang):
    words = dm.word_list(lang)
    assert len(words) == 199
    assert all(isinstance(w, str) and w.strip() for w in words)


@pytest.mark.parametrize("lang, filename", [("id", dm.GOOGLE_FILES["id"]), ("ko", dm.GOOGLE_FILES["ko"])])
def test_google_aligned_list_uses_english_number_for_each_seed_word(lang, filename):
    # Independent of word_list: look up each seed word's number in the English Google list and read the
    # same number from the target list.
    number = _google_eng_number_by_word()
    target = dm.read_google_list(filename)
    expected = [target[number[w]] for w in _seed_words_from_file()]
    assert dm.word_list(lang) == expected


@pytest.mark.xfail(strict=True, reason="data issue: duplicate Indonesian translations (rusa, sendok); "
                                       "reported, not fixed")
def test_id_translations_are_unique():
    words = [w.casefold() for w in dm.word_list("id")]
    assert len(set(words)) == len(words)


def test_word_list_returns_fresh_copy():
    first = dm.word_list("id")
    first.clear()
    assert len(dm.word_list("id")) == 199


def test_word_list_rejects_unknown_language():
    with pytest.raises(ValueError, match="word_list language"):
        dm.word_list("min")


@needs_su_jv
@pytest.mark.parametrize("lang", ["su", "jv"])
def test_su_jv_list_has_199_nonempty_entries(lang):
    words = dm.word_list(lang)
    assert len(words) == 199
    assert all(isinstance(w, str) and w.strip() for w in words)


# ---------------------------------------------------------------- Twenty Questions items

def test_twentyq_shape():
    items = dm.load_twentyq_items("en", "id", n=30, n_candidates=100)
    en, idw = dm.word_list("en"), dm.word_list("id")
    assert len(items) == 30
    for item in items:
        assert item["item_id"] == f"tq-{item['idx']}"
        assert len(item["candidates_q"]) == 100
        assert len(set(item["candidates_q"])) == 100
        assert item["target_q"] in item["candidates_q"]
        assert item["target_q"] == en[item["idx"] - 1]
        assert item["target_a"] == idw[item["idx"] - 1]


def test_twentyq_same_targets_across_pairs():
    def targets(lq, la):
        return [item["idx"] for item in dm.load_twentyq_items(lq, la, n=30)]

    base = targets("en", "id")
    assert len(set(base)) == 30
    assert targets("id", "ko") == base
    assert targets("en", "ko") == base
    assert targets("ko", "en") == base


def test_twentyq_candidates_are_shared_per_question_language():
    a = dm.load_twentyq_items("en", "id", n=10)
    b = dm.load_twentyq_items("en", "ko", n=10)
    assert [x["candidates_q"] for x in a] == [x["candidates_q"] for x in b]


def test_twentyq_deterministic_and_seed_sensitive():
    first = dm.load_twentyq_items("en", "id", n=30, seed=0)
    assert first == dm.load_twentyq_items("en", "id", n=30, seed=0)
    assert first != dm.load_twentyq_items("en", "id", n=30, seed=1)


@needs_su_jv
def test_twentyq_su_jv_pair_has_same_targets_as_en_pair():
    base = [x["idx"] for x in dm.load_twentyq_items("en", "id", n=30)]
    su_jv = dm.load_twentyq_items("su", "jv", n=30)
    assert [x["idx"] for x in su_jv] == base
    assert all(x["target_q"] in x["candidates_q"] and len(x["candidates_q"]) == 100 for x in su_jv)


# ---------------------------------------------------------------- MCQ items (Belebele)

@pytest.fixture(scope="module")
def mcq_en_id():
    try:
        return dm.load_mcq_items("en", "id", n=30)
    except Exception as e:  # network or HF cache unavailable
        pytest.skip(f"Belebele unavailable: {type(e).__name__}: {e}")


def test_mcq_shape(mcq_en_id):
    assert len(mcq_en_id) == 30
    assert len({x["item_id"] for x in mcq_en_id}) == 30
    for item in mcq_en_id:
        assert len(item["options_q"]) == 4
        assert all(isinstance(o, str) and o.strip() for o in item["options_q"])
        assert item["gold"] in (1, 2, 3, 4)
        assert item["passage_a"].strip() and item["question_q"].strip()
        assert re.fullmatch(r"bb-[0-9a-f]{8}-\d+", item["item_id"])


def test_mcq_same_items_across_pairs():
    su_jv = dm.load_mcq_items("su", "jv", n=30)
    ko_su = dm.load_mcq_items("ko", "su", n=30)
    assert [x["item_id"] for x in su_jv] == [x["item_id"] for x in ko_su]


def test_mcq_gold_and_options_match_belebele_rows(mcq_en_id):
    for item in mcq_en_id:
        key = (item["link"], item["question_number"])
        rows = {lang: dm._belebele(lang)[key] for lang in dm.MCQ_LANGS}
        assert {int(r["correct_answer_num"]) for r in rows.values()} == {item["gold"]}
        assert item["options_q"] == [rows["en"][f"mc_answer{i}"] for i in range(1, 5)]


def test_mcq_passage_is_in_answerer_language(mcq_en_id):
    # Pair (en, id): passage_a is the Indonesian passage, not the English one.
    for item in mcq_en_id:
        key = (item["link"], item["question_number"])
        assert item["passage_a"] == dm._belebele("id")[key]["flores_passage"]
        assert item["passage_a"] != dm._belebele("en")[key]["flores_passage"]


# ---------------------------------------------------------------- game smoke tests

@pytest.fixture(scope="module")
def twentyq_item():
    return dm.load_twentyq_items("en", "id", n=1)[0]


def _run(gm, questioner_moves, answerer_moves, max_moves=20):
    players = {"questioner": ScriptedPlayer(questioner_moves), "answerer": ScriptedPlayer(answerer_moves)}
    return run_episode(gm, players, IdentityChannel(), max_moves=max_moves)


def test_twentyq_correct_guess_scores_and_counts_questions(twentyq_item):
    item = twentyq_item
    gm = TwentyQuestionsGame(item, "en", "id")
    q = ["Is it an animal?", "Is it bigger than a cat?", "Can you eat it?"]
    rec = _run(gm,
               [Move("message", x) for x in q] + [Move("answer", f"[[{item['target_q']}]]")],
               [Move("message", "Ya"), Move("message", "Tidak"), Move("message", "Ya")])

    assert rec["finished"] is True
    assert rec["channel"] == "identity"
    assert rec["score"] == {"guess": item["target_q"], "gold": item["target_q"], "correct": True,
                            "questions_asked": 3, "valid_format": True}


def test_twentyq_wrong_guess_is_incorrect(twentyq_item):
    item = twentyq_item
    wrong = next(c for c in item["candidates_q"] if c != item["target_q"])
    gm = TwentyQuestionsGame(item, "en", "id")
    rec = _run(gm, [Move("answer", f"[[{wrong}]]")], [])

    assert rec["score"]["guess"] == wrong
    assert rec["score"]["correct"] is False
    assert rec["score"]["questions_asked"] == 0


def test_twentyq_question_budget_is_twenty(twentyq_item):
    item = twentyq_item
    gm = TwentyQuestionsGame(item, "en", "id")
    q_moves = [Move("message", f"Question {i}?") for i in range(20)] + [Move("answer", f"[[{item['target_q']}]]")]
    a_moves = [Move("message", "Ya") for _ in range(20)]
    rec = _run(gm, q_moves, a_moves, max_moves=60)  # 20 questions + 20 replies + final = 41 moves

    assert rec["score"]["questions_asked"] == 20
    assert rec["score"]["correct"] is True
    assert gm.observe("questioner").can_message is False


@pytest.fixture(scope="module")
def mcq_item(mcq_en_id):
    return mcq_en_id[0]


def test_mcq_correct_choice_scores_and_counts_questions(mcq_item):
    item = mcq_item
    gm = MCQConversationGame(item, "en", "id")
    rec = _run(gm,
               [Move("message", f"Question {i}?") for i in range(1, 5)] + [Move("answer", f"[[{item['gold']}]]")],
               [Move("message", "Ya") for _ in range(4)])

    assert rec["finished"] is True
    assert rec["score"] == {"guess": item["gold"], "gold": item["gold"], "correct": True,
                            "questions_asked": 4, "valid_format": True}


def test_mcq_wrong_choice_is_incorrect(mcq_item):
    item = mcq_item
    wrong = item["gold"] % 4 + 1
    gm = MCQConversationGame(item, "en", "id")
    rec = _run(gm,
               [Move("message", f"Question {i}?") for i in range(1, 5)] + [Move("answer", f"[[{wrong}]]")],
               [Move("message", "Mungkin") for _ in range(4)])

    assert rec["score"]["guess"] == wrong
    assert rec["score"]["correct"] is False
    assert rec["score"]["questions_asked"] == 4


def test_mcq_english_reply_is_rewritten(mcq_item):
    gm = MCQConversationGame(mcq_item, "en", "id")
    rec = _run(gm, [Move("message", f"Question {i}?") for i in range(1, 5)] + [Move("answer", "[[1]]")],
               [Move("message", "Maybe."), Move("message", "Mungkin.")] + [Move("message", "Ya") for _ in range(3)])
    first = rec["turns"][1]
    assert first["source"] == "Mungkin." and first["rejected"][0]["text"] == "Maybe."
