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

    def __init__(self, generate: Callable[[str, Optional[str], List[str]], str], label: str):
        self.generate, self.label = generate, label

    def prompt(self, obs: Observation) -> str:
        parts = [obs.instructions]
        if obs.inbox:
            msgs = "\n".join(f"- {m if m is not None else '(nothing arrived)'}" for m in obs.inbox)
            parts.append(f"Messages you have received from your partner, in order:\n{msgs}")
        allowed = []
        if obs.can_message:
            allowed.append("send a message")
        if obs.can_answer:
            allowed.append("give your final answer")
        parts.append("You may now " + " or ".join(allowed) + ". " + ACTION_FORMAT)
        return "\n\n".join(parts)

    def act(self, obs):
        raw = self.generate(self.prompt(obs), None, obs.images)
        return parse_action(raw, obs)


def text_provider_generate(provider) -> Callable:
    """Adapt an mt-eval LLMProvider (text only) to the LLMPlayer callable."""

    def generate(prompt, system_prompt, images):
        if images:
            raise ValueError(f"{provider.get_provider_name()} is text-only but the player was given images")
        return provider.generate(prompt, system_prompt=system_prompt)

    return generate
