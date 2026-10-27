"""Image reference game: the describer sees one image, the guesser picks it out of N.

Item fields: target (image path), candidates (list of image paths, target included),
gold (1-based index of the target among candidates). One describer message, then one
guess; the static variant scripts the describer with a native human caption.
"""

import re

from ..game import GameMaster, Observation
from ..languages import get as lang


class ImageRefGame(GameMaster):
    def __init__(self, item: dict, describer_lang: str, guesser_lang: str, allow_message: bool = True):
        super().__init__(item, {"describer": describer_lang, "guesser": guesser_lang})
        self.allow_message = allow_message  # False = no-communication prior (guesser answers blind)

    def next_role(self):
        return "describer" if (self.allow_message and not self.turns) else "guesser"

    def observe(self, role):
        if role == "describer":
            return Observation(
                role, self.roles[role],
                f"You see one photo. Describe it in {lang(self.roles[role]).name} so that your partner, who "
                f"sees this photo among {len(self.item['candidates'])} similar photos, can pick it out. "
                f"Write only in {lang(self.roles[role]).name}. You have one message.",
                images=[self.item["target"]], can_message=True, can_answer=False)
        n = len(self.item["candidates"])
        return Observation(
            role, self.roles[role],
            f"You see {n} photos, numbered 1 to {n} in the order given. Your partner has described one of "
            f"them. Pick the photo your partner described. Your answer is only the number (1-{n}).",
            images=list(self.item["candidates"]), inbox=list(self.inbox[role]),
            can_message=False, can_answer=True)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "guesser" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "guesser"]
        if not answers:
            return {"correct": None}
        n = len(self.item["candidates"])
        nums = [int(x) for x in re.findall(r"\d+", answers[-1]) if 1 <= int(x) <= n]
        guess = nums[0] if nums else None
        return {"guess": guess, "gold": self.item["gold"], "correct": guess == self.item["gold"]}
