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
from mtcomm.players import ScriptedPlayer, parse_action  # noqa: E402


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
    assert rec["score"] == {"guess": 2, "gold": 2, "correct": True, "describer_words": 3, "over_word_limit": False}
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
