"""Local-visitor culture game on BLEnD.

The visitor holds a question about the local's culture and four options, in the visitor's
language, and may ask the local through the channel. The local is grounded on its
culture's native survey answers, so the local's knowledge is the gold and every failure is
a communication or translation failure. The visitor must answer by the LOCAL culture.

Item fields: question, options (letter -> text, visitor language), answer_idx,
own_culture_idx, local_place (place name in English, for the local's instructions), and
local_knowledge: survey results for the target question AND several same-topic distractor
questions, each {"question": native local-language text, "answers": [{"answers": [...], "count": n}]},
in shuffled order. The distractors mean the local must understand which question the visitor
asked, so the visitor-to-local translation is tested too.
"""

import re

from ..game import GameMaster, Observation
from ..languages import get as lang


class BlendLocalVisitorGame(GameMaster):
    def __init__(self, item: dict, local_lang: str, visitor_lang: str, max_questions: int = 3):
        super().__init__(item, {"visitor": visitor_lang, "local": local_lang})
        self.max_questions = max_questions  # 0 = no-communication prior

    def _asked(self):
        return sum(1 for t in self.turns if t.role == "visitor" and t.kind == "message")

    def next_role(self):
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "visitor" and last.kind == "message":
            return "local"
        return "visitor"

    def observe(self, role):
        if role == "local":
            blocks = []
            for k in self.item["local_knowledge"]:
                answers = "; ".join(f"{', '.join(a['answers'])} ({a['count']})" for a in k["answers"])
                blocks.append(f"Q: {k['question']}\nA (number of local people giving each answer): {answers}")
            name = lang(self.roles[role]).name
            return Observation(
                role, self.roles[role],
                f"You live in {self.item['local_place']} and speak {name}. A visitor is asking you about local "
                f"customs. This is what local people answered in a survey:\n\n" + "\n\n".join(blocks) +
                f"\n\nAnswer the visitor's message helpfully from this knowledge, as a local would. "
                f"Write only in {name}.",
                inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        asked = self._asked()
        can_ask = asked < self.max_questions
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(self.item["options"].items()))
        name = lang(self.roles[role]).name
        budget = (f"You may ask the local up to {self.max_questions - asked} more message(s), in {name}, before "
                  f"answering." if can_ask else "You cannot ask anything more; answer now.")
        return Observation(
            role, self.roles[role],
            f"You are a visitor in {self.item['local_place']}. You must answer this question the way LOCAL people "
            f"there would, not the way people from your home would:\n{self.item['question']}\n{opts}\n{budget} "
            f"Your final answer is only the option letter.",
            inbox=list(self.inbox[role]), can_message=can_ask, can_answer=True)

    def is_done(self):
        return any(t.kind == "answer" and t.role == "visitor" for t in self.turns)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "visitor"]
        if not answers:
            return {"correct": None}
        letters = [m for m in re.findall(r"\b([A-Z])\b", answers[-1].upper()) if m in self.item["options"]]
        guess = letters[0] if letters else None
        return {"guess": guess, "gold": self.item["answer_idx"], "correct": guess == self.item["answer_idx"],
                "own_culture": guess == self.item["own_culture_idx"], "questions_asked": self._asked()}
