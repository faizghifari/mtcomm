"""Unit tests for the game loop, the games, the action parser and PivotChannel.

No network, no GPU: every player is scripted and every channel is local.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from mtcomm.channels import Channel, ChannelOutput, IdentityChannel, NoChannel, PivotChannel  # noqa: E402
from mtcomm.game import Move, Observation, run_episode  # noqa: E402
from mtcomm.games.blend_lv import BlendLocalVisitorGame  # noqa: E402
from mtcomm.games.image_ref import ImageRefGame  # noqa: E402
from mtcomm.games.mugeval import MCQConversationGame, TwentyQuestionsGame  # noqa: E402
from mtcomm.players import LLMPlayer, Player, ScriptedPlayer, parse_action  # noqa: E402


def _obs(can_message=True, can_answer=False):
    return Observation("describer", "en", "instructions", can_message=can_message, can_answer=can_answer)


# ---------------------------------------------------------------- parse_action

def test_parse_action_valid_message():
    move = parse_action('{"action": "message", "content": " a red bus "}', _obs())
    assert move == Move("message", "a red bus")


def test_parse_action_valid_answer():
    move = parse_action('{"action": "answer", "content": "3"}', _obs(can_message=False, can_answer=True))
    assert move == Move("answer", "3")


def test_parse_action_json_wrapped_in_prose():
    text = 'Sure, here is my reply: {"action": "answer", "content": "B"}. Hope that helps.'
    assert parse_action(text, _obs(can_message=False, can_answer=True)) == Move("answer", "B")


@pytest.mark.parametrize("text", ["a red bus", "{not json}", '{"action": "message", "content": '])
def test_parse_action_non_json_falls_back_to_raw_text(text):
    assert parse_action(text, _obs(can_message=True)) == Move("message", text.strip())


@pytest.mark.parametrize("text", ["a red bus", "{not json}"])
def test_parse_action_non_json_falls_back_to_answer_when_no_message_allowed(text):
    assert parse_action(text, _obs(can_message=False, can_answer=True)) == Move("answer", text.strip())


def test_parse_action_message_coerced_to_answer_when_cannot_message():
    move = parse_action('{"action": "message", "content": "is it B?"}', _obs(can_message=False, can_answer=True))
    assert move == Move("answer", "is it B?")


# ---------------------------------------------------------------- ImageRefGame

IMAGE_ITEM = {
    "target": "/img/target.jpg",
    "candidates": ["/img/a.jpg", "/img/target.jpg", "/img/b.jpg", "/img/c.jpg", "/img/d.jpg", "/img/e.jpg"],
    "gold": 2,
}


def test_image_ref_correct_guess_and_message_delivered():
    gm = ImageRefGame(IMAGE_ITEM, describer_lang="en", guesser_lang="en")
    players = {
        "describer": ScriptedPlayer([Move("message", "a red bus")]),
        "guesser": ScriptedPlayer([Move("answer", "2")]),
    }
    rec = run_episode(gm, players, IdentityChannel())

    assert rec["finished"] is True
    assert rec["channel"] == "identity"
    assert rec["score"] == {"guess": 2, "gold": 2, "correct": True, "describer_words": 3, "questions_asked": 0,
                            "over_word_limit": False}
    assert [(t.role, t.kind) for t in gm.turns] == [("describer", "message"), ("guesser", "answer")]
    assert gm.inbox["guesser"] == ["a red bus"]
    assert gm.observe("guesser").inbox == ["a red bus"]


def test_image_ref_no_channel_delivers_none():
    gm = ImageRefGame(IMAGE_ITEM, describer_lang="en", guesser_lang="en")
    players = {
        "describer": ScriptedPlayer([Move("message", "a red bus")]),
        "guesser": ScriptedPlayer([Move("answer", "1")]),
    }
    rec = run_episode(gm, players, NoChannel())

    assert gm.turns[0].delivered is None
    assert gm.inbox["guesser"] == [None]
    assert rec["channel"] == "none"
    assert rec["score"]["correct"] is False


def test_image_ref_without_message_describer_never_moves():
    gm = ImageRefGame(IMAGE_ITEM, describer_lang="en", guesser_lang="en", allow_message=False)
    # An empty ScriptedPlayer raises if asked to act, so a finished episode proves it never did.
    players = {
        "describer": ScriptedPlayer([]),
        "guesser": ScriptedPlayer([Move("answer", "2")]),
    }
    rec = run_episode(gm, players, IdentityChannel())

    assert [t.role for t in gm.turns] == ["guesser"]
    assert gm.inbox["guesser"] == []
    assert rec["score"]["correct"] is True


# ---------------------------------------------------------------- BlendLocalVisitorGame

BLEND_ITEM = {
    "item_id": "test-item",
    "question": "What do locals eat for breakfast?",
    "options": {"A": "bread", "B": "rice", "C": "cereal", "D": "soup"},
    "answer_idx": "B",
    "own_culture_idx": "A",
    "local_place": "West Java",
    "local_knowledge": [
        {"question": "Apa anu didahar isuk-isuk?", "answers": [{"answers": ["nasi"], "count": 3}]},
    ],
}


def test_blend_turn_order_visitor_local_visitor():
    gm = BlendLocalVisitorGame(BLEND_ITEM, local_lang="su", visitor_lang="ko", max_questions=3)
    players = {
        "visitor": ScriptedPlayer([Move("message", "What do locals eat?"), Move("answer", "B")]),
        "local": ScriptedPlayer([Move("message", "Rice, usually.")]),
    }
    rec = run_episode(gm, players, IdentityChannel())

    assert [(t.role, t.kind) for t in gm.turns] == [
        ("visitor", "message"), ("local", "message"), ("visitor", "answer"),
    ]
    assert gm.inbox["local"] == ["What do locals eat?"]
    assert gm.inbox["visitor"] == ["Rice, usually."]
    assert rec["score"] == {
        "guess": "B", "gold": "B", "correct": True, "own_culture": False, "questions_asked": 1,
    }


@pytest.mark.parametrize("answer, correct, own_culture", [("B", True, False), ("A", False, True)])
def test_blend_flags_correct_and_own_culture(answer, correct, own_culture):
    gm = BlendLocalVisitorGame(BLEND_ITEM, local_lang="su", visitor_lang="ko", max_questions=3)
    players = {
        "visitor": ScriptedPlayer([Move("message", "What do locals eat?"), Move("answer", answer)]),
        "local": ScriptedPlayer([Move("message", "Rice, usually.")]),
    }
    rec = run_episode(gm, players, IdentityChannel())

    assert rec["score"]["correct"] is correct
    assert rec["score"]["own_culture"] is own_culture


def test_blend_zero_questions_visitor_answers_immediately():
    gm = BlendLocalVisitorGame(BLEND_ITEM, local_lang="su", visitor_lang="ko", max_questions=0)
    players = {
        "visitor": ScriptedPlayer([Move("answer", "B")]),
        "local": ScriptedPlayer([]),  # never asked to act
    }
    rec = run_episode(gm, players, IdentityChannel())

    assert gm.observe("visitor").can_message is False
    assert [t.role for t in gm.turns] == ["visitor"]
    assert rec["score"] == {
        "guess": "B", "gold": "B", "correct": True, "own_culture": False, "questions_asked": 0,
    }


# ---------------------------------------------------------------- PivotChannel

class Leg1(Channel):
    name = "leg1"

    def translate(self, text, src, tgt, history=None):
        return ChannelOutput(text + "|1", {"channel": self.name})


class Leg2(Channel):
    name = "leg2"

    def translate(self, text, src, tgt, history=None):
        return ChannelOutput(text + "|2", {"channel": self.name})


def test_pivot_channel_composes_two_legs():
    channel = PivotChannel(Leg1(), pivot="en", second=Leg2())
    out = channel.translate("hola", "su", "ko")

    assert out.text == "hola|1|2"
    assert out.meta["pivot_text"] == "hola|1"
    assert out.meta["leg1"] == {"channel": "leg1"}
    assert out.meta["leg2"] == {"channel": "leg2"}
    assert channel.name == "pivot[en]:leg1+leg2"


def test_pivot_channel_single_channel_reuses_first_leg():
    out = PivotChannel(Leg1(), pivot="en").translate("hola", "su", "ko")

    assert out.text == "hola|1|1"
    assert out.meta["pivot_text"] == "hola|1"


def test_image_ref_dialogue_turn_order_and_budget():
    item = {"target": "t.jpg", "candidates": ["a.jpg", "t.jpg", "c.jpg"], "gold": 2}
    gm = ImageRefGame(item, "ko", "id", max_questions=2)
    describer = ScriptedPlayer([Move("message", "a red bus"), Move("message", "two windows"),
                                Move("message", "no people")])
    guesser = ScriptedPlayer([Move("message", "how many windows?"), Move("message", "any people?"),
                              Move("answer", "2")])
    rec = run_episode(gm, {"describer": describer, "guesser": guesser}, IdentityChannel())
    roles = [(t["role"], t["kind"]) for t in rec["turns"]]
    assert roles == [("describer", "message"), ("guesser", "message"), ("describer", "message"),
                     ("guesser", "message"), ("describer", "message"), ("guesser", "answer")]
    assert rec["score"]["correct"] is True and rec["score"]["questions_asked"] == 2
    # Budget used up: the guesser can no longer ask.
    assert gm.observe("guesser").can_message is False


def test_image_ref_static_prompt_unchanged_by_dialogue_option():
    item = {"target": "t.jpg", "candidates": ["a.jpg", "t.jpg"], "gold": 2}
    static = ImageRefGame(item, "ko", "id")
    assert "ask" not in static.observe("guesser").instructions
    assert static.observe("guesser").can_message is False


def test_players_see_their_own_earlier_messages():
    """Regression: the 20Q questioner saw only the answers and asked the same question 15 times."""
    item = {"target_q": "kursi", "target_a": "kursi", "candidates_q": ["kursi", "meja"]}
    gm = TwentyQuestionsGame(item, "su", "su")
    seen = []

    class Recorder(Player):
        def __init__(self, moves):
            self.moves = moves

        def act(self, obs):
            seen.append((obs.role, list(obs.dialogue)))
            return self.moves.pop(0)

    players = {"questioner": Recorder([Move("message", "Q1?"), Move("message", "Q2?"), Move("answer", "[[kursi]]")]),
               "answerer": Recorder([Move("message", "Enya"), Move("message", "Henteu")])}
    run_episode(gm, players, IdentityChannel())
    third_q = [d for r, d in seen if r == "questioner"][2]
    assert third_q == [{"speaker": "you", "text": "Q1?"}, {"speaker": "partner", "text": "Enya"},
                       {"speaker": "you", "text": "Q2?"}, {"speaker": "partner", "text": "Henteu"}]
    second_a = [d for r, d in seen if r == "answerer"][1]  # MuG-Eval's 20Q answerer keeps its history
    assert second_a == [{"speaker": "partner", "text": "Q1?"}, {"speaker": "you", "text": "Enya"},
                        {"speaker": "partner", "text": "Q2?"}]
    prompt = LLMPlayer(lambda *a: "", "x").prompt(Observation(
        "questioner", "su", "intro", dialogue=third_q, inbox=["Enya", "Henteu"], can_answer=True))
    assert "You: Q1?" in prompt and "Partner: Henteu" in prompt


@pytest.mark.parametrize("game", ["image", "culture", "mcq", "twentyq"])
def test_every_dialogue_game_shows_history(game):
    """At every turn a player sees every message sent so far (its own included); only MuG-Eval's MCQ
    answerer is reset per question, as in MuG-Eval's mcqa_gen_answer.py."""
    seen = []

    class Talker(Player):
        def __init__(self, role):
            self.role, self.k = role, 0

        def act(self, obs):
            sent = [t.source for t in gm.turns if t.kind == "message"]
            seen.append((obs.role, [d["text"] for d in obs.dialogue], sent))
            self.k += 1
            if obs.can_message and self.k <= 3:
                return Move("message", f"{self.role}{self.k}")
            return Move("answer", "A" if game == "culture" else "1")

    if game == "image":
        item = {"item_id": "x", "target": "t.jpg", "candidates": ["a.jpg", "t.jpg"], "gold": 2}
        gm, roles = ImageRefGame(item, "su", "jv", max_questions=3), ["describer", "guesser"]
    elif game == "culture":
        item = {"item_id": "x", "question": "q", "options": {"A": "a", "B": "b"}, "answer_idx": "A",
                "own_culture_idx": "B", "local_place": "Bandung", "visitor_own_culture": "Indonesia",
                "local_knowledge": [{"question": "q", "answers": [{"answers": ["a"], "count": 3}]}]}
        gm, roles = BlendLocalVisitorGame(item, "su", "jv", max_questions=3, min_questions=1), ["visitor", "local"]
    elif game == "mcq":
        item = {"passage_a": "p", "question_q": "q", "options_q": ["a", "b", "c", "d"], "gold": 1}
        gm, roles = MCQConversationGame(item, "su", "jv"), ["questioner", "answerer"]
    else:
        item = {"target_q": "kursi", "target_a": "kursi", "candidates_q": ["kursi", "meja"]}
        gm, roles = TwentyQuestionsGame(item, "su", "jv"), ["questioner", "answerer"]
    run_episode(gm, {r: Talker(r) for r in roles}, IdentityChannel())
    assert sum(1 for _, _, sent in seen if len(sent) >= 2) >= 2  # the game really went several turns
    for role, dialogue, sent in seen:
        assert dialogue == ([] if (game == "mcq" and role == "answerer") else sent), (role, dialogue, sent)
