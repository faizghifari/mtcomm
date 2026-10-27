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


# Where the visitor is from, disclosed as part of the setting (BLEnD has no Javanese culture;
# Central Java is the visitor's home, Indonesia the proxy for the own-culture option).
VISITOR_HOME = {"ko": "South Korea", "id": "Jakarta, Indonesia", "su": "West Java, Indonesia",
                "jv": "Central Java, Indonesia", "en": "the United States"}


class BlendLocalVisitorGame(GameMaster):
    """prompt_style "neutral" (default) discloses the setting only: where each player is from and
    that they are talking. "explicit" is the first pilot's wording, which told the visitor to
    answer the way local people would and called the local's knowledge a survey; pilot
    transcripts showed both phrasings leaking into the dialogue ("according to the locals",
    "according to the survey here")."""

    def __init__(self, item: dict, local_lang: str, visitor_lang: str, max_questions: int = 3,
                 min_questions: int = 0, prompt_style: str = "neutral"):
        super().__init__(item, {"visitor": visitor_lang, "local": local_lang})
        if prompt_style not in ("neutral", "explicit"):
            raise ValueError(f"unknown prompt_style {prompt_style!r}")
        self.prompt_style = prompt_style
        self.max_questions = max_questions  # 0 = no-communication prior
        # Visitors that may answer at once often do (pilot: Flash-Lite models never asked), so the
        # protocol can require at least `min_questions` messages before an answer is allowed.
        self.min_questions = min(min_questions, max_questions)

    def _asked(self):
        return sum(1 for t in self.turns if t.role == "visitor" and t.kind == "message")

    def next_role(self):
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "visitor" and last.kind == "message":
            return "local"
        return "visitor"

    def observe(self, role):
        if self.prompt_style == "neutral":
            return self._observe_neutral(role)
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
        must_ask = asked < self.min_questions
        if must_ask:
            budget = (f"You must ask the local at least {self.min_questions - asked} more message(s), in {name}, "
                      f"before answering (at most {self.max_questions - asked}).")
        elif can_ask:
            budget = f"You may ask the local up to {self.max_questions - asked} more message(s), in {name}, before answering."
        else:
            budget = "You cannot ask anything more; answer now."
        return Observation(
            role, self.roles[role],
            f"You are a visitor in {self.item['local_place']}. You must answer this question the way LOCAL people "
            f"there would, not the way people from your home would:\n{self.item['question']}\n{opts}\n{budget} "
            f"Your final answer is only the option letter.",
            inbox=list(self.inbox[role]), can_message=can_ask, can_answer=not must_ask)

    def _observe_neutral(self, role):
        name = lang(self.roles[role]).name
        if role == "local":
            blocks = []
            for k in self.item["local_knowledge"]:
                ranked = [", ".join(a["answers"]) for a in k["answers"]]
                blocks.append(f"{k['question']}\n  {ranked[0]} (most common)" +
                              "".join(f"; {r}" for r in ranked[1:]))
            return Observation(
                role, self.roles[role],
                f"You live in {self.item['local_place']} and speak {name}. A visitor is talking with you. "
                f"From your own experience, this is what people around here commonly say about these topics, "
                f"most common first:\n\n" + "\n\n".join(blocks) +
                f"\n\nReply to the visitor's message naturally, as yourself. Write only in {name}.",
                inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        asked = self._asked()
        can_ask = asked < self.max_questions
        must_ask = asked < self.min_questions
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(self.item["options"].items()))
        if must_ask:
            budget = (f"Send the person at least {self.min_questions - asked} more message(s) in {name} before "
                      f"answering (at most {self.max_questions - asked}).")
        elif can_ask:
            budget = f"You may send the person up to {self.max_questions - asked} more message(s) in {name} before answering."
        else:
            budget = "You cannot send more messages; answer now."
        home = VISITOR_HOME.get(self.roles["visitor"], "abroad")
        return Observation(
            role, self.roles[role],
            f"You are from {home} and speak {name}. You are visiting {self.item['local_place']} and talking with "
            f"a person who lives there. Answer this question:\n{self.item['question']}\n{opts}\n{budget} "
            f"Your final answer is only the option letter.",
            inbox=list(self.inbox[role]), can_message=can_ask, can_answer=not must_ask)

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
