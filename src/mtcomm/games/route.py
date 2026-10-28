"""Route redrawing game (Map Task style): the giver narrates a route, the follower rebuilds it.

Item fields: map (a mtcomm.routemap picture map, landmark names in English), labels ({lang: {name:
label}}, the native-checked lexicon), giver_image / follower_image ({lang: path}, maps drawn with that
language's labels), directions ({lang: {"left": ..., "right": ...}}).

The giver sees the route and no grid; the follower sees a lettered grid and no route, and on hard maps
its map differs (two route landmarks hidden, one extra). Both are told the maps may differ. Turns
alternate: the giver tells the route a step or two at a time; after each message the follower replies
(confirms where it is, or asks, e.g. where a landmark it does not have is). The giver has
n_steps + extra_messages messages; the follower may answer once the giver has sent half of n_steps.

Giver messages must narrate, not list: validate() refuses messages over max_words words, naming more
than max_landmarks landmarks, or shaped like a list (arrows, numbering, line breaks, grid codes,
measurements, brackets), and the runner asks for a rewrite (game.run_episode).

The follower answers one step per line: "<landmark as written on its map>, left|right" or "cell D7".
Scoring is mtcomm.routemap.score_picture: success = every step right in order.
"""

import math
import re
from typing import List, Optional

from ..game import GameMaster, Observation
from ..languages import get as lang
from ..routemap import COLS, score_picture

LIST_SHAPES = [
    (re.compile(r"->|→|=>|⇒|>"), "uses arrows"),
    (re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s", re.M), "is a numbered or bulleted list"),
    (re.compile(r"\n"), "has line breaks"),
    (re.compile(rf"\b[{COLS}{COLS.lower()}]\s?[1-8]\b"), "uses grid coordinates"),
    (re.compile(r"\d+\s*(px|%|°|cm|mm|m\b)"), "uses measurements"),
    (re.compile(r"[\[\]{}|]"), "uses brackets or symbols"),
]


def parse_route_answer(text: str, labels: dict, directions: dict) -> List[dict]:
    """One step per line (or ';'-separated): a cell code, or a landmark label plus a side."""
    inv = sorted(((lab.casefold(), name) for name, lab in labels.items()), key=lambda x: -len(x[0]))
    left = {"left", directions.get("left", "left").casefold()}
    right = {"right", directions.get("right", "right").casefold()}
    steps = []
    for line in re.split(r"[\n;]+", text or ""):
        low = line.casefold().strip()
        if not low:
            continue
        name = next((n for lab, n in inv if lab in low), None)
        if name is None:
            m = re.search(rf"\b([{COLS.lower()}])\s?([1-8])\b", low)
            if m:
                steps.append({"cell": f"{m.group(1).upper()}{m.group(2)}"})
            continue
        rest = low.replace(next(lab for lab, n in inv if n == name), " ")
        side = ("left" if any(w in rest for w in left) else "right" if any(w in rest for w in right) else None)
        steps.append({"landmark": name, "side": side})
    return steps


class RouteGame(GameMaster):
    def __init__(self, item: dict, giver_lang: str, follower_lang: str, allow_message: bool = True,
                 extra_messages: int = 4, max_words: int = 25, max_landmarks: int = 3):
        super().__init__(item, {"giver": giver_lang, "follower": follower_lang})
        self.allow_message = allow_message  # False = no-communication prior (follower guesses a route)
        self.n_steps = len(item["map"]["gold_answer"])
        self.giver_budget = self.n_steps + extra_messages if allow_message else 0
        self.max_words, self.max_landmarks = max_words, max_landmarks

    def _sent(self, role):
        return sum(1 for t in self.turns if t.role == role and t.kind == "message")

    def next_role(self):
        if self.allow_message and not self.turns:
            return "giver"
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "giver":
            return "follower"
        return "giver" if self._sent("giver") < self.giver_budget else "follower"

    def _labels(self, role):
        return self.item["labels"][self.roles[role]]

    def validate(self, role, text):
        """Why a giver message would turn the route into a list, or None."""
        if role != "giver":
            return None
        words = text.split()
        if len(words) > self.max_words:
            return f"it has {len(words)} words; use at most {self.max_words}."
        for pattern, why in LIST_SHAPES:
            if pattern.search(text):
                return f"it {why}; describe the way in ordinary sentences."
        low = text.casefold()
        bases = {lab.split(" (")[0].casefold() for lab in self._labels(role).values()}
        named = [b for b in bases if b in low]
        if len(named) > self.max_landmarks:
            return f"it names {len(named)} landmarks; mention at most {self.max_landmarks} per message."
        if words and sum(len(b.split()) for b in named) / len(words) > 0.6:
            return "it is mostly landmark names; describe how to move between them."
        return None

    def observe(self, role):
        name = lang(self.roles[role]).name
        if role == "giver":
            left = self.giver_budget - self._sent("giver")
            return Observation(
                role, self.roles[role],
                f"You and your partner each have a map of the same village, but the maps may not be identical. "
                f"Yours shows a route in red, starting at the green square and ending at the red dot; your "
                f"partner's map has no route. Guide your partner along the route by talking, in {name}, the way "
                f"you would guide a friend over the phone: say where to go and which side of each landmark to "
                f"pass. Give one or two steps per message, then wait for your partner's reply, and answer their "
                f"questions. Rules: ordinary sentences only, at most {self.max_words} words per message, at most "
                f"{self.max_landmarks} landmarks named per message; no lists, arrows, numbering, coordinates or "
                f"measurements. You have {left} message(s) left. Write only in {name}.",
                images=[self.item["giver_image"][self.roles[role]]], inbox=list(self.inbox[role]),
                can_message=True, can_answer=False)
        heard = self._sent("giver")
        can_msg = heard < self.giver_budget
        can_answer = not self.allow_message or heard >= math.ceil(self.n_steps / 2) or not can_msg
        dirs = self.item["directions"][self.roles[role]]
        talk = (f"Your partner sees the route and is guiding you along it, in {name}. Reply after each message: "
                f"say where you are on your map, or ask a question (for example, where a landmark is that you do "
                f"not have). The maps may not be identical. " if self.allow_message else
                "Nobody can tell you the route; guess it. ")
        when = ("" if not self.allow_message else
                " Your partner has no messages left: give your answer now." if not can_msg else
                " You may give your answer when you think you know the whole route." if can_answer else
                " You can answer after hearing more of the route.")
        return Observation(
            role, self.roles[role],
            f"You have a map of a village with a lettered grid (columns A-H, rows 1-8) and a green square where "
            f"a route starts. {talk}Your answer is the route as steps in order, one per line: the landmark the "
            f"route passes, written exactly as on your map, then the side it passes on ({dirs['left']} or "
            f"{dirs['right']}), e.g. \"<landmark>, {dirs['left']}\"; or, for a place where your map has no "
            f"landmark, the grid cell, e.g. \"cell D7\".{when}",
            images=[self.item["follower_image"][self.roles[role]]], inbox=list(self.inbox[role]),
            can_message=can_msg and self.allow_message, can_answer=can_answer)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "follower" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "follower"]
        if not answers:
            return {"correct": None}
        fl = self.roles["follower"]
        steps = parse_route_answer(answers[-1], self.item["labels"][fl], self.item["directions"][fl])
        sc = score_picture(self.item["map"], steps)
        giver = [t for t in self.turns if t.role == "giver" and t.kind == "message"]
        return {"correct": sc["success"], "step_accuracy": sc["step_accuracy"],
                "deviation_cells": sc["deviation_cells"], "answer_steps": steps,
                "gold": self.item["map"]["gold_answer"], "level": self.item["map"]["level"],
                "giver_messages": len(giver), "follower_messages": self._sent("follower"),
                "giver_rewrites": sum(len(t.rejected) for t in giver),
                "giver_violations": sum(1 for t in giver if t.violation)}
