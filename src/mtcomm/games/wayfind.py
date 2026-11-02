"""Directions game: a visitor asks a local the way to a place, in an unfamiliar area.

The guide's map shows the route from the green arrow to a red building; the seeker's map has the same streets and landmarks,
no route, and eight lettered buildings. The guide gives the directions in one message, from the walker's point of view: go
straight, which intersection, left or right, what you pass, which building it is. The seeker may ask one question, gets
one short answer, and picks a letter. Success is the right building; chance is 1 in 8. The kind of wrong building (a turn
taken the wrong way, a neighbour of the target) is recorded, so errors can be traced to what the translation lost.

A later variant of the route task, not part of the PoC comparison, which uses the route game on easy maps.
"""

import re

from ..game import GameMaster, Observation
from ..languages import get as lang

LIST_SHAPES = [(re.compile(r"->|→|=>|⇒"), "uses arrows"),
               (re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s", re.M), "is a numbered or bulleted list"),
               (re.compile(r"\n"), "has line breaks")]
ANSWER = re.compile(r"\[\[\s*([A-H])\s*\]\]|\b([A-H])\b")


class WayfindGame(GameMaster):
    def __init__(self, item: dict, guide_lang: str, seeker_lang: str, allow_message: bool = True, max_words: int = 70,
                 reply_words: int = 25, max_questions: int = 1):
        super().__init__(item, {"guide": guide_lang, "seeker": seeker_lang})
        self.allow_message = allow_message  # False = no-communication prior: chance is 1 in 8
        self.max_words, self.reply_words = max_words, reply_words
        self.max_questions = max_questions if allow_message else 0

    def _questions(self):
        return sum(1 for t in self.turns if t.role == "seeker" and t.kind == "message")

    def next_role(self):
        if self.allow_message and not self.turns:
            return "guide"
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "seeker" and last.kind == "message":
            return "guide"
        return "seeker"

    def validate(self, role, text):
        if role != "guide":
            return None
        words = text.split()
        limit = self.reply_words if self._questions() else self.max_words
        if len(words) > limit:
            return f"it has {len(words)} words; use at most {limit}."
        for pattern, why in LIST_SHAPES:
            if pattern.search(text):
                return f"it {why}; describe the way in ordinary sentences."
        return None

    def observe(self, role):
        code = self.roles[role]
        name = lang(code).name
        if role == "guide":
            image = [self.item["guide_image"][code]]
            if not self._questions():
                return Observation(
                    role, code,
                    f"A visitor who does not know this area asks you the way. Your map shows the route in red, from the "
                    f"green arrow, where the visitor stands facing the way the arrow points, to the red building they are "
                    f"looking for. Tell them how to get there in {name}, in one message of at most {self.max_words} words, "
                    f"the way you would to a stranger in the street: from the walker's point of view, using straight on, "
                    f"left and right, the intersections they reach and the landmarks they pass, and say exactly which "
                    f"building it is at the end. The visitor's map has no route. Do not use compass directions, map "
                    f"directions such as up or down, grid positions or distances in metres. Ordinary sentences only. "
                    f"Write only in {name}.",
                    images=image, can_message=True, can_answer=False)
            return Observation(
                role, code,
                f"You gave a visitor directions to the red building on your map, and they asked a question. Answer it in "
                f"{name}, in at most {self.reply_words} words, from the walker's point of view.",
                images=image, inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        can_ask = self._questions() < self.max_questions
        ask = (f" You may ask one question in {name} if something is unclear, or answer right away." if can_ask else "")
        told = ("A local has told you the way." if self.allow_message else "Nobody can tell you the way; guess.")
        return Observation(
            role, code,
            f"You are a visitor in an area you do not know, looking for a building. On your map, the green arrow is where "
            f"you stand and the way you are facing. The blue buildings, lettered A to H, are the possible destinations. "
            f"{told}{ask} Follow the directions on the map from where you stand and answer with the letter of the "
            f"building, in double brackets, e.g. [[C]].",
            images=[self.item["seeker_image"][code]], inbox=list(self.inbox[role]), can_message=can_ask, can_answer=True)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "seeker" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "seeker"]
        m = ANSWER.search(answers[-1]) if answers else None
        guess = (m.group(1) or m.group(2)) if m else None
        kind = self.item["map"]["candidates"].get(guess, {}).get("kind") if guess else None
        return {"guess": guess, "gold": self.item["map"]["gold"], "correct": guess == self.item["map"]["gold"], "guess_kind": kind,
                "questions_asked": self._questions(), "valid_format": guess is not None, "n_turns": sum(isinstance(x, str) for x in self.item["map"]["moves"])}
