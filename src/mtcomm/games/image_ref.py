"""Image reference game: the describer sees one image, the guesser picks it out of N.

Item fields: target (image path), candidates (list of image paths, target included),
gold (1-based index of the target among candidates).

Static (max_questions=0, default): one describer message, then one guess; the static variant
can script the describer with a native human caption.
Dialogue (max_questions>0): after the description, the guesser may ask up to max_questions
questions (as in MD3), each answered by the describer, before choosing. Every message passes
through the channel, so near-identical candidates make the game depend on the translation of
the distinguishing detail.

Each describer message is limited to one sentence of at most `max_words` words (default 14,
the length of XM3600's human captions). The limit is instructed, not enforced; the score
records the first description's length.
"""

import re
from typing import Optional

from ..game import GameMaster, Observation
from ..languages import get as lang


class ImageRefGame(GameMaster):
    def __init__(self, item: dict, describer_lang: str, guesser_lang: str, allow_message: bool = True,
                 max_words: Optional[int] = 14, max_questions: int = 0):
        super().__init__(item, {"describer": describer_lang, "guesser": guesser_lang})
        self.allow_message = allow_message  # False = no-communication prior (guesser answers blind)
        self.max_words = max_words
        self.max_questions = max_questions if allow_message else 0

    def _questions(self):
        return sum(1 for t in self.turns if t.role == "guesser" and t.kind == "message")

    def next_role(self):
        if self.allow_message and not self.turns:
            return "describer"
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "guesser" and last.kind == "message":
            return "describer"
        return "guesser"

    def _limit(self):
        return "" if self.max_words is None else f"a single sentence of at most {self.max_words} words"

    def observe(self, role):
        name = lang(self.roles[role]).name
        n = len(self.item["candidates"])
        if role == "describer":
            if not self.turns:
                return Observation(
                    role, self.roles[role],
                    f"You see one photo. Describe it in {name} so that your partner, who sees this photo among "
                    f"{n} similar photos, can pick it out. Write only in {name}. " +
                    ("You have one message." if self.max_words is None else f"You have one message: {self._limit()}.") +
                    (f" Your partner may then ask you up to {self.max_questions} questions." if self.max_questions else ""),
                    images=[self.item["target"]], can_message=True, can_answer=False)
            return Observation(
                role, self.roles[role],
                f"You see one photo, which your partner must pick out among {n} similar photos. You described it "
                f"and your partner has asked a question. Answer it about your photo in {name}" +
                (f", in {self._limit()}." if self.max_words is not None else "."),
                images=[self.item["target"]], inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        asked = self._questions()
        can_ask = asked < self.max_questions
        ask = (f" Before choosing, you may ask your partner up to {self.max_questions - asked} more question(s), "
               f"in {name}." if can_ask else "")
        return Observation(
            role, self.roles[role],
            f"You see {n} photos, numbered 1 to {n} in the order given. Your partner has described one of "
            f"them. Pick the photo your partner described.{ask} Your answer is only the number (1-{n}).",
            images=list(self.item["candidates"]), inbox=list(self.inbox[role]),
            can_message=can_ask, can_answer=True)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "guesser" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "guesser"]
        if not answers:
            return {"correct": None}
        n = len(self.item["candidates"])
        nums = [int(x) for x in re.findall(r"\d+", answers[-1]) if 1 <= int(x) <= n]
        guess = nums[0] if nums else None
        msgs = [t.source for t in self.turns if t.kind == "message" and t.role == "describer"]
        words = len(msgs[0].split()) if msgs else 0
        return {"guess": guess, "gold": self.item["gold"], "correct": guess == self.item["gold"],
                "describer_words": words, "questions_asked": self._questions(),
                "over_word_limit": self.max_words is not None and words > self.max_words}
