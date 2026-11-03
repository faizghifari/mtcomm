"""Unit tests for the game loop, the games, the action parser and PivotChannel.

No network, no GPU: every player is scripted and every channel is local.
"""

import os
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
    assert len(items) >= 30
    for it in items:
        assert [p for p in it["shuffled"] if p in it["story"]] != it["story"]  # never already in order
        assert sorted(it["shuffled"]) == sorted(it["story"] + it["distractors"])
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
    s2 = gm2.score()  # a wrong side is not a failure of the route, but is reported
    assert s2["correct"] is True and s2["strict_success"] is False and s2["side_accuracy"] < 1


def test_route_success_tolerates_passed_landmarks_only():
    import math

    from mtcomm.routemap import CELL, _resample, cell_of, route_points, route_success

    it = _route_item()
    m = it["map"]
    gold = [dict(g) for g in m["gold_answer"]]
    assert route_success(m, gold)
    true = _resample(route_points(m))
    gold_names = {g.get("landmark") for g in gold}
    dist = {l["name"]: min(math.dist((l["x"], l["y"]), q) for q in true) for l in m["follower_landmarks"]
            if l["name"] not in gold_names}
    far = max(dist, key=dist.get)
    assert dist[far] > 0.75 * CELL
    assert not route_success(m, gold[:1] + [{"landmark": far, "side": "left"}] + gold[1:])  # off the path
    assert not route_success(m, gold[:-1])  # a waypoint missing
    assert not route_success(m, list(reversed(gold)))  # wrong order
    near = [n for n, d in dist.items() if d <= 0.75 * CELL]
    if near:
        assert route_success(m, [{"landmark": near[0], "side": "left"}] + gold)  # passed on the way
    # A landmark the follower has can also be placed by its cell.
    first = next(g for g in gold if "landmark" in g)
    lm = next(l for l in m["giver_landmarks"] if l["name"] == first["landmark"])
    by_cell = [{"cell": cell_of(lm)} if g is first else g for g in gold]
    assert route_success(m, by_cell)


def test_sequence_with_distractors_pick_and_order():
    from mtcomm.games.sequence import SequenceGame

    item = {"story": ["p1", "p2", "p3"], "distractors": ["d1", "d2"], "shuffled": ["d1", "p3", "p1", "d2", "p2"],
            "gold": [3, 5, 2]}
    gm = SequenceGame(item, "su", "jv", allow_message=False)
    assert "same book" in gm.observe("arranger").instructions
    gm.apply("arranger", Move("answer", "3, 5, 2"))
    assert gm.score()["correct"] and gm.score()["picked"] == 3
    gm2 = SequenceGame(item, "su", "jv", allow_message=False)
    gm2.apply("arranger", Move("answer", "3, 4, 2"))  # a distractor instead of p2
    s = gm2.score()
    assert s["valid_format"] and not s["correct"] and s["picked"] == 2 and s["kendall_tau"] == 1.0
    gm3 = SequenceGame(item, "su", "jv", stepwise=True)
    assert "mixed with 2 other pages" in gm3.observe("narrator").instructions


def test_sequence_items_include_distractors_when_built():
    from mtcomm.data import SEQUENCE_MANIFEST, load_sequence_items

    items = load_sequence_items()
    if "v3" not in SEQUENCE_MANIFEST.name:
        pytest.skip("no distractor manifest")
    for it in items:
        assert len(it["distractors"]) == 2 and len(it["shuffled"]) == 7
        assert [it["shuffled"][g - 1] for g in it["gold"]] == it["story"]
        assert all(Path(p).exists() for p in it["distractors"])


def test_gemini_request_unchanged_without_temperature():
    """Existing caches stay valid: the request (and its hash) only changes when a temperature is set."""
    from mtcomm.backends import GeminiBackend

    plain = GeminiBackend("gemini-3.1-flash-lite").build("hi", None, [])
    assert "temperature" not in plain["generation_config"]
    assert GeminiBackend("gemini-3.1-flash-lite", temperature=0.7).build("hi", None, [])["generation_config"]["temperature"] == 0.7


def test_opening_refusal_is_rewritten():
    """A refusal before the partner said anything is rejected and rewritten; it is not delivered."""
    from mtcomm.channels import IdentityChannel
    from mtcomm.game import run_episode
    from mtcomm.games.sequence import SequenceGame
    from mtcomm.players import NOT_UNDERSTOOD, is_not_understood

    item = {"item_id": "x", "story": ["a", "b"], "distractors": [], "shuffled": ["b", "a"], "gold": [2, 1]}
    gm = SequenceGame(item, "ko", "ko", allow_message=True, max_words=20, max_questions=0)
    players = {"narrator": ScriptedPlayer([Move("message", NOT_UNDERSTOOD["ko"]), Move("message", "소년이 달린다")]),
               "arranger": ScriptedPlayer([Move("answer", "2, 1")])}
    rec = run_episode(gm, players, IdentityChannel(), refused=is_not_understood)
    first = rec["turns"][0]
    assert first["source"] == "소년이 달린다"
    assert first["rejected"] and first["rejected"][0]["text"] == NOT_UNDERSTOOD["ko"]


def test_targeted_culture_prompts():
    from mtcomm.games.blend_lv import BlendLocalVisitorGame

    item = {"item_id": "c", "question": "Q?", "options": {"A": "x", "B": "y"}, "answer_idx": "A", "own_culture_idx": "B",
            "local_place": "West Java", "local_knowledge": [{"question": "Q?", "answers": [{"answers": ["x"], "count": 3}]}]}
    gm = BlendLocalVisitorGame(item, "su", "ko", max_questions=3, min_questions=1, prompt_style="targeted")
    assert "no greeting" in str(gm.observe("visitor"))
    assert "Answer only what the visitor asks" in str(gm.observe("local"))


# ---------------------------------------------------------------- directions game

def _wayfind_item(level="mid", seed=1):
    import random

    from mtcomm.streetmap import make_street_map
    m = make_street_map(random.Random(seed), level, "t01")
    return {"item_id": "t01", "map": m, "guide_image": {"en": "g.png", "id": "g.png"}, "seeker_image": {"en": "s.png", "id": "s.png"}}


def test_street_map_route_has_the_level_turns_and_a_labelled_target():
    from mtcomm.streetmap import LEVELS
    for level, turns in LEVELS.items():
        m = _wayfind_item(level)["map"]
        assert sum(isinstance(x, str) for x in m["moves"]) == turns
        assert len({tuple(p) for p in m["path"]}) == len(m["path"])  # no intersection twice
        assert len(m["candidates"]) == 8
        assert m["candidates"][m["gold"]]["building"] == m["target"]
        assert m["candidates"][m["gold"]]["kind"] == "target"
        assert any(c["kind"] == "flip" for c in m["candidates"].values())
        taken = {l["building"] for l in m["landmarks"]}
        assert not taken & {c["building"] for c in m["candidates"].values()}  # a landmark never hides a candidate


def test_street_map_keeps_every_route_step_on_a_street():
    m = _wayfind_item("hard", seed=3)["map"]
    edges = {tuple(sorted((tuple(a), tuple(b)))) for a, b in m["edges"]}
    assert all(tuple(sorted((tuple(a), tuple(b)))) in edges for a, b in zip(m["path"], m["path"][1:]))


def test_wayfind_flow_question_then_answer_and_score_kinds():
    from mtcomm.games.wayfind import WayfindGame
    it = _wayfind_item()
    wrong = next(k for k, v in it["map"]["candidates"].items() if v["kind"] == "flip")
    gm = WayfindGame(it, "en", "id")
    rec = run_episode(gm, {"guide": ScriptedPlayer([Move("message", "Go straight, then left."), Move("message", "Yes.")]),
                           "seeker": ScriptedPlayer([Move("message", "Which side?"), Move("answer", f"[[{wrong}]]")])},
                      IdentityChannel(), max_moves=10)
    assert rec["finished"] and rec["score"]["correct"] is False and rec["score"]["guess_kind"] == "flip"
    assert rec["score"]["questions_asked"] == 1


def test_wayfind_prior_needs_no_message_and_the_guide_is_held_to_the_limits():
    from mtcomm.games.wayfind import WayfindGame
    it = _wayfind_item()
    prior = WayfindGame(it, "en", "en", allow_message=False)
    assert prior.next_role() == "seeker" and prior.observe("seeker").can_message is False
    gm = WayfindGame(it, "en", "id", max_words=5)
    assert "words" in gm.validate("guide", "one two three four five six")
    assert "line breaks" in gm.validate("guide", "go on\nthen left")
    assert gm.validate("guide", "go straight then left") is None


def test_lid_gate_fires_in_a_real_mcq_game_and_only_on_other_language_messages():
    from mtcomm.games.mugeval import MCQConversationGame
    from mtcomm.players import NOT_UNDERSTOOD, LidGatedPlayer
    item = {"item_id": "x", "passage_a": "p", "question_q": "q", "options_q": ["a", "b", "c", "d"], "gold": 1}

    def play(check, question="this is a question"):
        gm = MCQConversationGame(item, "en", "su")
        inner = ScriptedPlayer([Move("message", "Enya.")] * 4)
        answerer = LidGatedPlayer(inner, "su", check, lambda: gm.inbox["answerer"][-1] if gm.inbox["answerer"] else None)
        moves = [Move("message", question) for _ in range(4)] + [Move("answer", "[[1]]")]
        rec = run_episode(gm, {"questioner": ScriptedPlayer(moves), "answerer": answerer}, IdentityChannel(), max_moves=20)
        return [t["source"] for t in rec["turns"] if t["role"] == "answerer"]

    foreign = lambda text, code: {"ok": False, "needs_review": False}  # noqa: E731
    assert play(foreign) == [NOT_UNDERSTOOD["su"]] * 4
    assert play(lambda text, code: {"ok": True, "needs_review": False}) == ["Enya."] * 4
    assert play(lambda text, code: {"ok": False, "needs_review": True}) == ["Enya."] * 4  # unsure: not gated
    assert play(foreign, question="two words") == ["Enya."] * 4  # too short to judge


# ---------------------------------------------------------------- analysis

def test_paired_uses_the_ceiling_of_the_language_the_task_scores():
    import importlib.util
    spec = importlib.util.spec_from_file_location("paired", Path(__file__).resolve().parents[1] / "scripts" / "paired_main2.py")
    paired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paired)
    # sol won on x; su's ceiling lost it and ko's ceiling won it.
    d = {("su-ko", "sol", "x"): 1, ("su", "ceiling", "x"): 0, ("ko", "ceiling", "x"): 1}
    assert paired.vs_ceiling(d, "mcq", "sol", ["su-ko"]) == (1, 0, 1)  # MCQ: su asks and is scored
    assert paired.vs_ceiling(d, "twentyq", "sol", ["su-ko"]) == (1, 0, 1)
    assert paired.vs_ceiling(d, "sequence", "sol", ["su-ko"]) == (0, 0, 1)  # sequence: ko receives and is scored


def test_pivots_on_every_pair_only_when_asked():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_main", Path(__file__).resolve().parents[1] / "scripts" / "run_main.py")
    run_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_main)
    assert "nllb-id" not in run_main.conditions("su", "ko") and "nllb-id" in run_main.conditions("su", "jv")
    run_main.PIVOTS_ALL = True
    try:
        assert {"nllb-id", "google-id"} <= set(run_main.conditions("ko", "su"))
    finally:
        run_main.PIVOTS_ALL = False


def _load_script(name):
    import importlib.util
    import sys
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cell_flags_come_from_the_parent_config():
    cells = _load_script("run_cells")
    route = {"tasks": ["route"], "n": 15, "route_level": "easy", "route_start": 15, "items": {}}
    assert cells.flags_from(route) == ["--n", "15", "--route-level", "easy", "--route-start", "15"]
    seq = {"tasks": ["sequence"], "n": 30, "sequence": {"manifest": "m.json", "max_words": 20, "max_questions": 0}}
    assert cells.flags_from(seq) == ["--n", "30", "--seq-words", "20", "--seq-questions", "0", "--seq-manifest", "m.json"]
    mcq = {"tasks": ["mcq"], "n": 30, "items": {"mcq_lid_gate": True}}
    assert cells.flags_from(mcq) == ["--n", "30", "--mcq-lid-gate"]


def test_repeat_consistency_statistics():
    rc = _load_script("repeat_consistency")
    assert rc.kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1
    assert rc.kappa([1, 1, 1, 1], [1, 1, 1, 1]) is None
    assert abs(rc.kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
    # every item always the same in its repeats: no residual, all item
    v = rc.components({"a": [1, 1, 1], "b": [0, 0, 0]})
    assert v["resid"] == 0 and abs(v["item"] - 0.5) < 1e-9
    # coin-flip repeats on identical items: no item variance
    v = rc.components({"a": [1, 0], "b": [1, 0]})
    assert v["item"] == 0 and abs(v["resid"] - 0.5) < 1e-9
    # more repeats shrink only the residual part
    assert rc.se_design(0.1, 0.1, 100, 3) < rc.se_design(0.1, 0.1, 100, 1)
    assert rc.se_design(0.1, 0.0, 100, 3) == rc.se_design(0.1, 0.0, 100, 1)


def test_nllb_server_round_trip():
    import threading

    from mtcomm.channels import NLLBRemote, nllb_channel

    server_mod = _load_script("nllb_server")

    class Fake(Channel):
        name = "fake"

        def translate(self, text, src, tgt, history=None):
            return ChannelOutput(f"{text}>{tgt}", {"channel": "nllb:nllb-200-3.3B"})

    srv = server_mod.make_server(Fake(), "facebook/nllb-200-3.3B", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        os.environ["NLLB_URL"] = url
        ch = nllb_channel()
        assert isinstance(ch, NLLBRemote) and ch.name == "nllb:nllb-200-3.3B"  # same cache keys as a local copy
        out = ch.translate("halo", "su", "jv")
        assert out.text == "halo>jv" and out.meta["channel"] == "nllb:nllb-200-3.3B"
        # a server that is down at start is left out; one that goes down later is skipped
        os.environ["NLLB_URL"] = f"http://127.0.0.1:9,{url}"
        assert nllb_channel().urls == [url]
        assert NLLBRemote(["http://127.0.0.1:9", url]).translate("halo", "su", "jv").text == "halo>jv"
    finally:
        del os.environ["NLLB_URL"]
        srv.shutdown()
