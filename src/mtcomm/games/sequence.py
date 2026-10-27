"""Sequence ordering game: the narrator sees a picture story in order, the arranger puts it back in order.

Item fields: story (image paths in true order), shuffled (the same paths in the arranger's order),
gold (for each story position, the arranger's 1-based photo number).

The narrator speaks first and tells the story picture by picture; the arranger may then ask up to
max_questions questions, each answered, before giving the order. The narrator's photo numbers are
not the arranger's (the arranger's copy is shuffled), so the order has to travel as descriptions
of picture content, through the channel. Messages are limited by instruction to max_words words
(the opening story) and reply_words (answers); the score records the opening length.

Success is the exact order; the score also gives the number of pictures in the right position and
Kendall's tau, since a story that is half right still carried information.
"""

import re
from typing import List, Optional

from ..game import GameMaster, Observation
from ..languages import get as lang


def kendall_tau(guess: List[int], gold: List[int]) -> float:
    """Rank agreement between two orderings of the same items, in [-1, 1]."""
    pos = {x: i for i, x in enumerate(guess)}
    ranks = [pos[x] for x in gold]
    n = len(ranks)
    pairs = n * (n - 1) / 2
    concordant = sum(1 for i in range(n) for j in range(i + 1, n) if ranks[i] < ranks[j])
    return round((2 * concordant - pairs) / pairs, 3) if pairs else 1.0


class SequenceGame(GameMaster):
    def __init__(self, item: dict, narrator_lang: str, arranger_lang: str, allow_message: bool = True,
                 max_words: Optional[int] = 60, reply_words: Optional[int] = 20, max_questions: int = 3):
        super().__init__(item, {"narrator": narrator_lang, "arranger": arranger_lang})
        self.allow_message = allow_message  # False = no-communication prior (story order guessed from pictures)
        self.max_words, self.reply_words = max_words, reply_words
        self.max_questions = max_questions if allow_message else 0

    def _questions(self):
        return sum(1 for t in self.turns if t.role == "arranger" and t.kind == "message")

    def next_role(self):
        if self.allow_message and not self.turns:
            return "narrator"
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "arranger" and last.kind == "message":
            return "narrator"
        return "arranger"

    def observe(self, role):
        name = lang(self.roles[role]).name
        n = len(self.item["story"])
        if role == "narrator":
            if not self.turns:
                limit = "" if self.max_words is None else f" Use at most {self.max_words} words in total."
                return Observation(
                    role, self.roles[role],
                    f"You see {n} photos that tell a story, in their correct order (photo 1 comes first). Your "
                    f"partner has the same {n} photos in a different, shuffled order and numbering, so your photo "
                    f"numbers mean nothing to them. Tell your partner the story picture by picture in {name}, so "
                    f"they can put their photos in the right order. Write only in {name}.{limit}" +
                    (f" Your partner may then ask you up to {self.max_questions} questions." if self.max_questions else ""),
                    images=list(self.item["story"]), can_message=True, can_answer=False)
            limit = "" if self.reply_words is None else f", in at most {self.reply_words} words"
            return Observation(
                role, self.roles[role],
                f"You see {n} photos that tell a story, in their correct order. Your partner has them shuffled and "
                f"must put them in order. You told the story and your partner has asked a question. Answer it in "
                f"{name}{limit}. Your photo numbers mean nothing to your partner; describe what is in the picture.",
                images=list(self.item["story"]), inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        asked = self._questions()
        can_ask = asked < self.max_questions
        ask = (f" Before answering, you may ask your partner up to {self.max_questions - asked} more question(s), "
               f"in {name}." if can_ask else "")
        told = ("Your partner, who sees them in the correct order, has told you the story."
                if self.allow_message else "Nobody can tell you the story.")
        return Observation(
            role, self.roles[role],
            f"You see {n} photos, numbered 1 to {n} in the order given, which are a shuffled picture story. {told} "
            f"Put the photos in the story's order.{ask} Your answer is only the {n} photo numbers in story order, "
            f"separated by commas (for example: 3, 1, 5, 2, 4).",
            images=list(self.item["shuffled"]), inbox=list(self.inbox[role]), can_message=can_ask, can_answer=True)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "arranger" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "arranger"]
        if not answers:
            return {"correct": None}
        n, gold = len(self.item["story"]), self.item["gold"]
        nums = [int(x) for x in re.findall(r"\d+", answers[-1])]
        guess = nums if sorted(nums) == list(range(1, n + 1)) else None  # must be a permutation
        msgs = [t.source for t in self.turns if t.kind == "message" and t.role == "narrator"]
        words = len(msgs[0].split()) if msgs else 0
        return {"guess": guess, "gold": gold, "correct": guess == gold,
                "positions_correct": sum(a == b for a, b in zip(guess, gold)) if guess else 0,
                "kendall_tau": kendall_tau(guess, gold) if guess else None, "valid_format": guess is not None,
                "narrator_words": words, "questions_asked": self._questions(),
                "over_word_limit": self.max_words is not None and words > self.max_words}
