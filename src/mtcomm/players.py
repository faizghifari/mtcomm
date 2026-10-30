"""Players: scripted (fixed human-written text) and LLM-backed.

LLM players reply with one JSON object, {"action": "message" | "answer", "content": "..."},
so the runner never has to guess whether a reply is a message or a final answer.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .game import Move, Observation

ACTION_FORMAT = (
    'Reply with exactly one JSON object and nothing else: {"action": "message", "content": "..."} '
    'to send a message to your partner, or {"action": "answer", "content": "..."} to give your final answer.'
)


# What a monolingual player says when a message is not in its language. [su/jv: native check pending]
NOT_UNDERSTOOD = {
    "su": "Hapunten, abdi teu ngartos. Mangga nyarios nganggo basa Sunda.",
    "jv": "Nyuwun sewu, kula mboten mangertos. Mangga ngendika nganggo basa Jawa.",
    "ko": "죄송하지만 이해하지 못했어요. 한국어로 말해 주세요.",
    "id": "Maaf, saya tidak mengerti. Silakan bicara dalam bahasa Indonesia.",
    "en": "Sorry, I don't understand. Please speak English.",
}


def monolingual_system(name: str, code: str) -> str:
    """System prompt for a player that must not read its partner's language (strong models otherwise read
    an untranslated partner directly, so a translator can only hurt them). v2: judge the whole message, so
    words shared with related languages, names and loanwords do not trigger a refusal (v1 refused real
    Sundanese in 2/10 ceiling games, taking it for Indonesian)."""
    return (f"You are a native speaker of {name}, and {name} is the only language you understand. Apart from these "
            f"instructions, you cannot read any other language: not English, and not languages related to {name} or "
            f"spoken near it. Judge each message from your partner as a whole: a message written in {name} is "
            f"{name} even if it contains names, numbers, borrowed words, or words that {name} shares with related "
            f"languages, and even if it is a little unnatural. Only if a message is clearly written in another "
            f"language, you cannot understand any of it: do not use its content, do not guess, and send only this "
            f"message: \"{NOT_UNDERSTOOD[code]}\" Always write only in {name}.")


def is_not_understood(text: str, code: str) -> bool:
    norm = lambda s: re.sub(r"[^\w]+", " ", s).strip().casefold()  # noqa: E731
    return norm(NOT_UNDERSTOOD[code].split(".")[0]) in norm(text or "")


class Player(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> Move: ...


class ScriptedPlayer(Player):
    """Plays fixed moves in order, e.g. a native human caption as the describer's message."""

    def __init__(self, moves: List[Move]):
        self.moves = list(moves)

    def act(self, obs):
        if not self.moves:
            raise RuntimeError("scripted player ran out of moves")
        return self.moves.pop(0)


def parse_action(text: str, obs: Observation) -> Move:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            kind = obj.get("action")
            # Small local players bend the format: {"action": "ans", "content": ...}, or the reply itself
            # as the action ({"action": "panginten"}). Recover the content rather than send the raw JSON.
            if isinstance(kind, str) and kind not in ("message", "answer"):
                content = obj.get("content") or (kind if kind.lower() not in ("", "none") else "")
                obj, kind = {"content": content}, "answer" if kind.startswith("ans") else "message"
            if kind in ("message", "answer"):
                if kind == "message" and not obs.can_message:
                    kind = "answer"
                elif kind == "answer" and not obs.can_answer:
                    kind = "message"
                return Move(kind, str(obj.get("content", "")).strip())
        except json.JSONDecodeError:
            pass
    # Unparseable: treat as a message if allowed, else as an answer, and keep the raw text.
    return Move("message" if obs.can_message else "answer", text.strip())


class LLMPlayer(Player):
    """A player backed by a text or vision LLM.

    `generate(prompt, system_prompt, images)` is any callable returning the model's text, so
    mt-eval's text providers and mtcomm's vision provider plug in the same way.
    """

    def __init__(self, generate: Callable[[str, Optional[str], List[str]], str], label: str,
                 system: Optional[str] = None):
        self.generate, self.label, self.system = generate, label, system

    def prompt(self, obs: Observation) -> str:
        parts = [obs.instructions]
        if obs.dialogue:
            lines = "\n".join(f"{'You' if d['speaker'] == 'you' else 'Partner'}: "
                              f"{d['text'] if d['text'] is not None else '(nothing arrived)'}" for d in obs.dialogue)
            parts.append("The conversation so far, oldest first (your partner's messages as you received them):\n"
                         + lines)
        elif obs.inbox:
            msgs = "\n".join(f"- {m if m is not None else '(nothing arrived)'}" for m in obs.inbox)
            parts.append(f"Messages you have received from your partner, in order:\n{msgs}")
        if obs.feedback:
            parts.append(f"Your last message was NOT sent, because it broke the rules: {obs.feedback} "
                         "Write it again, following the rules.")
        allowed = []
        if obs.can_message:
            allowed.append("send a message")
        if obs.can_answer:
            allowed.append("give your final answer")
        parts.append("You may now " + " or ".join(allowed) + ". " + ACTION_FORMAT)
        return "\n\n".join(parts)

    def act(self, obs):
        raw = self.generate(self.prompt(obs), self.system, obs.images)
        return parse_action(raw, obs)


def text_provider_generate(provider) -> Callable:
    """Adapt an mt-eval LLMProvider (text only) to the LLMPlayer callable."""

    def generate(prompt, system_prompt, images):
        if images:
            raise ValueError(f"{provider.get_provider_name()} is text-only but the player was given images")
        return provider.generate(prompt, system_prompt=system_prompt)

    return generate
