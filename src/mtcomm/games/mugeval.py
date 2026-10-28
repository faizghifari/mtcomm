"""MuG-Eval's two conversational tasks (Song et al., Findings of EMNLP 2025), made cross-lingual.

In MuG-Eval both instances of a model share one target language. Here the questioner works in
L_Q and the answerer in L_A, each with its inputs in its own language, and every message passes
through the channel. The questioner's answer is the one scored, so the same-language result in
L_Q (identity channel) is the baseline.

Prompts follow MuG-Eval's released prompts (github.com/seyoungsong/mugeval, commit 3a427fea3e),
with two changes: the answerer replies in L_A instead of English (so the reply is translated
too), and the questioner is not told about an intermediary language.

Easy Twenty Questions. The answerer holds a hidden entity (L_A); the questioner sees 100
candidate entities (L_Q), asks up to 20 yes/no questions and names one as [[entity]].
Item fields: target_q, target_a, candidates_q.

MCQ Conversation (Belebele). The answerer holds the passage (L_A); the questioner holds the
question and four options (L_Q), asks yes/no questions (MuG-Eval asks exactly 4, one per choice)
and answers [[X]]. Item fields: passage_a, question_q, options_q (list of 4), gold (1-4).
"""

import re
from typing import Optional

from ..game import GameMaster, Observation
from ..languages import get as lang

FINAL = re.compile(r"\[\[(.+?)\]\]", re.S)


def _final(text: str) -> Optional[str]:
    m = FINAL.findall(text or "")
    if m:
        return m[-1].strip()
    # The JSON action format already marks the final answer; accept a bare answer too.
    return text.strip() or None if text else None


# The answerer's three replies, given in the prompt when the game is built with given_words=True: small local
# players otherwise answer in English even after a rewrite. [su/jv words: native check pending]
ANSWER_WORDS = {"su": ("Muhun.", "Henteu.", "Panginten."), "jv": ("Iya.", "Ora.", "Mbokmenawa."),
                "id": ("Ya.", "Tidak.", "Mungkin."), "ko": ("네.", "아니요.", "아마도.")}


class _QAGame(GameMaster):
    """Questioner asks, answerer replies, until the questioner gives a [[final]] answer or runs out."""

    max_questions = 20

    def __init__(self, item: dict, questioner_lang: str, answerer_lang: str, allow_message: bool = True,
                 given_words: bool = False, original: bool = False):
        super().__init__(item, {"questioner": questioner_lang, "answerer": answerer_lang})
        self.allow_message = allow_message
        self.words = ANSWER_WORDS.get(answerer_lang) if given_words else None
        # original=True: MuG-Eval's released MCQ prompts verbatim (the answerer replies "Yes."/"No."/"Maybe." in
        # English), for screening whether a model can play the task at all in its own language (no translation).
        self.original = original

    def _answer_choice(self, name):
        if self.words:
            return f'exactly one of these three answers: "{self.words[0]}" (yes), "{self.words[1]}" (no), or "{self.words[2]}" (maybe).'
        return f'exactly one of these three answers: the {name} words for "Yes.", "No.", or "Maybe."'

    def _maybe(self, name):
        return f'"{self.words[2]}"' if self.words else '"Maybe."'

    def _asked(self):
        return sum(1 for t in self.turns if t.role == "questioner" and t.kind == "message")

    def validate(self, role, text):
        """An answerer must reply in its own language; small local players fall back to English "Maybe."
        (one-word replies are too short for GlotLID, so the English words are checked directly)."""
        code = self.roles[role]
        if role == "answerer" and code != "en" and not self.original and re.sub(r"[^a-z]", "", text.lower()) in ("yes", "no", "maybe"):
            return f"You replied in English. Reply in {lang(code).name} with {self._answer_choice(lang(code).name)}."
        return None

    def next_role(self):
        last = self.turns[-1] if self.turns else None
        if last is not None and last.role == "questioner" and last.kind == "message":
            return "answerer"
        return "questioner"

    def is_done(self):
        return any(t.kind == "answer" and t.role == "questioner" for t in self.turns)

    def _questioner_obs(self, intro: str, final: str) -> Observation:
        asked = self._asked()
        budget = self.max_questions if self.allow_message else 0
        can_ask = asked < budget
        tail = "" if can_ask else "\n\n" + final
        return Observation("questioner", self.roles["questioner"], intro + tail,
                           inbox=list(self.inbox["questioner"]), can_message=can_ask, can_answer=True)


class TwentyQuestionsGame(_QAGame):
    max_questions = 20

    def observe(self, role):
        if role == "answerer" and self.original:  # twenty_prompt_chat_easy_a.md; entity in the data language
            q = self.inbox[role][-1] if self.inbox[role] else ""
            entity = self.item["target_a"]
            return Observation(
                role, self.roles[role],
                f'Answer the following question about the hidden entity: {entity}. Respond only with "Yes," "No," or '
                f'"Maybe" (including the period) without providing explanations or additional words. Do not use '
                f'"Maybe" if the answer can be definitively "Yes" or "No." Your responses must be in English. Never '
                f'mention the entity in your response. If the query contains two or more questions asked '
                f'simultaneously, consider it invalid and respond with "Maybe."\n\n<|Entity|>\n{entity}\n\n'
                f"<|Language|>\n{lang(self.roles['questioner']).name}\n\n<|Question|>\n{q}\n\n"
                f"<|Options|>\n\n- Yes.\n- No.\n- Maybe.",
                inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        if role == "answerer":
            name = lang(self.roles[role]).name
            q = self.inbox[role][-1] if self.inbox[role] else ""
            return Observation(
                role, self.roles[role],
                f'Answer the following question about the hidden entity: {self.item["target_a"]}. Respond only with '
                f'the {name} words for "Yes," "No," or "Maybe" without providing explanations or additional words. '
                f'Do not use "Maybe" if the answer can be definitively "Yes" or "No." Your responses must be in {name}. '
                f"Never mention the entity in your response. If the query contains two or more questions asked "
                f'simultaneously, consider it invalid and respond with "Maybe."\n\n<|Entity|>\n{self.item["target_a"]}\n\n'
                f"<|Language|>\n{name}\n\n<|Question|>\n{q}",
                # MuG-Eval's 20Q answerer keeps its conversation (twenty_gen_answer.py appends to messages_a);
                # its MCQ answerer is reset for every question, so MCQConversationGame passes no inbox.
                inbox=list(self.inbox[role]), can_message=True, can_answer=False)
        name = lang(self.roles[role]).name
        entity_list = "\n".join(self.item["candidates_q"])
        intro = (f"You will be solving an entity deduction game by asking questions about a hidden item in {name}. "
                 f"Your goal is to identify exactly one correct entity from a list of {len(self.item['candidates_q'])} "
                 f"items through strategic questioning, using as few questions as possible. You can ask yes/no/maybe "
                 f"questions in {name}, one at a time. Each question must be concise and answerable only with \"Yes,\" "
                 f"\"No,\" or \"Maybe.\" Do not ask for hints or the answer directly.\n\nAmong the list, exactly one is "
                 f"correct. You have a maximum of {self.max_questions} questions before making your final choice, but "
                 f"fewer questions are better. When you have determined the correct entity, provide your final answer "
                 f"using double brackets: [[entity]]. The entity must be exactly as it appears in the Entity List (do "
                 f"not translate to another language).\n\n<|Entity List|>\n{entity_list}\n\n<|Language|>\n{name}"
                 + ("\n\nBegin with your first question." if self.original else ""))
        final = ("You have now used all available questions. Based on the responses you've received, provide your "
                 "final guess of the entity. Even if you cannot determine the entity with confidence, provide your best "
                 "guess based on the information gathered. Indicate your answer in double brackets: [[entity]]. The "
                 "entity must be exactly as it appears in the Entity List (do not translate to another language).")
        return self._questioner_obs(intro, final)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "questioner"]
        guess = _final(answers[-1]) if answers else None
        norm = (lambda s: re.sub(r"\s+", " ", s).strip().casefold()) if guess else None
        correct = guess is not None and norm(guess) == norm(self.item["target_q"])
        return {"guess": guess, "gold": self.item["target_q"], "correct": correct,
                "questions_asked": self._asked(), "valid_format": guess is not None}


class MCQConversationGame(_QAGame):
    max_questions = 4  # MuG-Eval's released prompt: exactly 4 questions, one per choice

    def observe(self, role):
        if role == "answerer" and self.original:  # mcqa_prompt_chat_eng_a.md
            q = self.inbox[role][-1] if self.inbox[role] else ""
            return Observation(
                role, self.roles[role],
                "Answer the following question based solely on the information contained in the passage. Do not use "
                "any external knowledge or make assumptions beyond what is explicitly stated in the text. Your response "
                'must be in English and limited to exactly one of these three answers: "Yes.", "No.", or "Maybe." '
                "(including the period). Do not provide any explanation or additional words. If the query contains two "
                'or more questions asked simultaneously, consider it invalid and respond with "Maybe."\n\n'
                f"<|Passage|>\n{self.item['passage_a']}\n\n<|Language|>\n{lang(self.roles['questioner']).name}\n\n"
                f"<|Question|>\n{q}\n\n<|Options|>\n\n- Yes.\n- No.\n- Maybe.",
                can_message=True, can_answer=False)
        if role == "answerer":
            name = lang(self.roles[role]).name
            q = self.inbox[role][-1] if self.inbox[role] else ""
            return Observation(
                role, self.roles[role],
                "Answer the following question based solely on the information contained in the passage. Do not use "
                "any external knowledge or make assumptions beyond what is explicitly stated in the text. Your response "
                f"must be in {name} and limited to {self._answer_choice(name)} Do not provide any explanation or "
                "additional words. If the query contains two or more questions asked simultaneously, consider it "
                f"invalid and respond with {self._maybe(name)}\n\n"
                f"<|Passage|>\n{self.item['passage_a']}\n\n<|Language|>\n{name}\n\n<|Question|>\n{q}",
                can_message=True, can_answer=False)
        name = lang(self.roles[role]).name
        choices = "\n".join(f"({i}) {c}" for i, c in enumerate(self.item["options_q"], 1))
        person = ("I am serving as an intermediary between you and a person who has this passage. You can ask me "
                  "questions about the passage content, and I will relay these to the person. They will respond with "
                  "only \"yes,\" \"no,\" or \"maybe,\" which I will then share with you." if self.original else
                  "A person has this passage; you can ask them questions about its content, and they will respond with "
                  "only \"yes,\" \"no,\" or \"maybe.\"")  # original: mcqa_prompt_chat_eng_q.md
        intro = (f"You will be solving a multiple-choice question by asking questions about a hidden passage. "
                 f"{person}\n\nYour questions must be in {name} and you can only ask one "
                 f"question at a time. Do not ask for hints or request the passage directly. Among the four answer "
                 f"choices provided, exactly one is correct. You must ask exactly {self.max_questions} questions (one "
                 f"corresponding to each answer choice) before making your final decision. After receiving all four "
                 f"responses, provide your final answer in {name}, indicating the correct number choice with double "
                 f"brackets: [[X]]\n\n<|Query|>\n{self.item['question_q']}\n\n<|Choices|>\n{choices}\n\n"
                 f"<|Language|>\n{name}")
        final = (f"You have now used all available questions. Based on the responses you've received, please provide "
                 f"your final answer in {name}. Indicate your choice by including the number of the correct answer in "
                 f"double brackets: [[X]]")
        return self._questioner_obs(intro, final)

    def score(self):
        answers = [t.source for t in self.turns if t.kind == "answer" and t.role == "questioner"]
        guess = _final(answers[-1]) if answers else None
        digits = re.findall(r"[1-4]", guess or "")
        g = int(digits[0]) if digits else None
        return {"guess": g, "gold": self.item["gold"], "correct": g == self.item["gold"],
                "questions_asked": self._asked(), "valid_format": g is not None}
