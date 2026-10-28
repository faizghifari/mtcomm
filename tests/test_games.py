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


@pytest.mark.parametrize("game", ["image", "culture", "mcq", "twentyq", "sequence", "sequence_step", "route"])
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
    elif game == "route":
        from mtcomm.games.route import RouteGame

        gm, roles = RouteGame(_route_item(), "en", "en"), ["giver", "follower"]
    elif game.startswith("sequence"):
        from mtcomm.games.sequence import SequenceGame

        item = {"story": ["p1", "p2"], "shuffled": ["p2", "p1"], "gold": [2, 1]}
        gm, roles = SequenceGame(item, "su", "jv", max_questions=3, stepwise=game == "sequence_step"), ["narrator", "arranger"]
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


def test_sequence_game_order_scoring_and_turns():
    from mtcomm.games.sequence import SequenceGame, kendall_tau

    item = {"story": ["p1", "p2", "p3"], "shuffled": ["p3", "p1", "p2"], "gold": [2, 3, 1]}
    gm = SequenceGame(item, "su", "jv", max_questions=1)
    narrator = ScriptedPlayer([Move("message", "first a cat, then a dog, then a bird"), Move("message", "yes")])
    arranger = ScriptedPlayer([Move("message", "is the bird last?"), Move("answer", "Order: 2, 3, 1")])
    rec = run_episode(gm, {"narrator": narrator, "arranger": arranger}, IdentityChannel())
    assert [(t["role"], t["kind"]) for t in rec["turns"]] == [
        ("narrator", "message"), ("arranger", "message"), ("narrator", "message"), ("arranger", "answer")]
    assert rec["score"]["correct"] is True and rec["score"]["kendall_tau"] == 1.0
    assert gm.observe("arranger").can_message is False
    # Partly right and malformed answers.
    assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    gm2 = SequenceGame(item, "su", "jv", allow_message=False)
    assert gm2.next_role() == "arranger"
    gm2.apply("arranger", Move("answer", "2, 1, 3"))
    s = gm2.score()
    assert s["correct"] is False and s["positions_correct"] == 1 and s["valid_format"] is True
    gm3 = SequenceGame(item, "su", "jv", allow_message=False)
    gm3.apply("arranger", Move("answer", "2, 2, 1"))
    assert gm3.score()["valid_format"] is False and gm3.score()["correct"] is False


def test_sequence_items_shuffled_and_gold_consistent():
    from mtcomm.data import SEQUENCE_MANIFEST, load_sequence_items

    if not SEQUENCE_MANIFEST.exists():
        pytest.skip("Bloom-VIST sets not built")
    items = load_sequence_items()
    assert len(items) == 30
    for it in items:
        assert it["shuffled"] != it["story"] and sorted(it["shuffled"]) == sorted(it["story"])
        assert [it["shuffled"][g - 1] for g in it["gold"]] == it["story"]
        assert all(Path(p).exists() for p in it["story"])


def test_sequence_stepwise_turns_and_answer_gate():
    from mtcomm.games.sequence import SequenceGame

    item = {"story": ["p1", "p2", "p3"], "shuffled": ["p3", "p1", "p2"], "gold": [2, 3, 1]}
    gm = SequenceGame(item, "su", "jv", stepwise=True, max_questions=1)
    assert gm.next_role() == "narrator"
    gm.apply("narrator", Move("message", "a cat"), ChannelOutput("a cat"))
    obs = gm.observe("arranger")
    assert obs.can_message and not obs.can_answer  # must hear all three pictures first
    gm.apply("arranger", Move("message", "the grey cat?"), ChannelOutput("the grey cat?"))
    assert gm.next_role() == "narrator"
    for m in ["yes, grey", "a dog", "a bird"]:
        gm.apply("narrator", Move("message", m), ChannelOutput(m))
        if gm._narrator_left() > 0 or m != "a bird":
            gm.apply("arranger", Move("message", "ok"), ChannelOutput("ok"))
    # Narrator budget (3 pictures + 1 question) is used up: the arranger must answer now.
    assert gm.next_role() == "arranger"
    obs = gm.observe("arranger")
    assert obs.can_answer and not obs.can_message
    gm.apply("arranger", Move("answer", "2, 3, 1"))
    s = gm.score()
    assert s["correct"] and s["narrator_messages"] == 4 and s["messages_over_word_limit"] == 0


def _route_item():
    from mtcomm.data import load_route_items

    items = load_route_items()
    return next(it for it in items if it["map"]["level"] == "hard")


def test_route_giver_filter_rejects_lists_and_accepts_narration():
    from mtcomm.games.route import RouteGame

    it = _route_item()
    gm = RouteGame(it, "en", "en")
    names = [l["name"] for l in it["map"]["giver_landmarks"]]
    assert gm.validate("giver", f"{names[0]} -> {names[1]} -> {names[2]}") is not None  # arrows
    assert gm.validate("giver", f"1. {names[0]}\n2. {names[1]}") is not None  # list
    assert gm.validate("giver", "go to C7 then up") is not None  # grid code
    assert gm.validate("giver", f"{names[0]}, {names[1]}, {names[2]}, {names[3]}") is not None  # too many names
    assert gm.validate("giver", " ".join(["walk"] * 30)) is not None  # too long
    ok = f"From the start walk up and keep the {names[0]} on your left, then turn right."
    assert gm.validate("giver", ok) is None
    assert gm.validate("follower", "a -> b") is None  # only the giver is restricted


def test_route_rewrite_loop_and_scoring():
    from mtcomm.games.route import RouteGame

    it = _route_item()
    gm = RouteGame(it, "en", "en", extra_messages=0)
    gold = it["map"]["gold_answer"]
    answer = "\n".join(f"cell {g['cell']}" if "cell" in g else f"{g['landmark']}, {g['side']}" for g in gold)
    n = gm.n_steps
    giver = ScriptedPlayer([Move("message", "a -> b"), Move("message", "Go up and keep the pond on your left.")]
                           + [Move("message", "Keep going the same way.")] * (n - 1))
    follower = ScriptedPlayer([Move("message", "ok")] * (n - 1) + [Move("answer", answer)])
    rec = run_episode(gm, {"giver": giver, "follower": follower}, IdentityChannel(), max_moves=60)
    first = rec["turns"][0]
    assert first["source"].startswith("Go up") and first["rejected"][0]["text"] == "a -> b"
    assert all("->" not in (t["delivered"] or "") for t in rec["turns"] if t["kind"] == "message")
    assert rec["score"]["correct"] is True and rec["score"]["giver_rewrites"] == 1
    assert rec["score"]["giver_violations"] == 0
    # One side flipped -> not a success, but most steps right.
    flipped = answer.replace(", left", ", RIGHTX", 1).replace("RIGHTX", "right")
    gm2 = RouteGame(it, "en", "en", allow_message=False)
    gm2.apply("follower", Move("answer", flipped))
    assert gm2.score()["correct"] is False and gm2.score()["step_accuracy"] >= 0.8
