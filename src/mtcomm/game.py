"""Game master interface and the episode loop shared by every task.

Two players, each in their own language, exchange messages only through a Channel. The
game master owns the task state: what each player privately sees, whose turn it is, when
the game ends, and how it is scored. Games never call channels themselves; the runner
routes every message, so swapping the channel is the only change between conditions.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from .channels import Channel, ChannelOutput


@dataclass
class Observation:
    role: str
    lang: str
    instructions: str  # what this player is told, including private information
    images: List[str] = field(default_factory=list)  # image paths this player can see
    inbox: List[Optional[str]] = field(default_factory=list)  # messages received so far (already translated)
    can_message: bool = True
    can_answer: bool = False


@dataclass
class Move:
    kind: str  # "message" or "answer"
    content: str


@dataclass
class Turn:
    role: str
    kind: str
    source: str  # what the player wrote
    delivered: Optional[str] = None  # what the other player received (messages only)
    channel_meta: dict = field(default_factory=dict)
    lid: Optional[dict] = None  # language check on the player's own text


class GameMaster(ABC):
    """One episode of one task item."""

    def __init__(self, item: dict, roles: Dict[str, str]):
        self.item = item
        self.roles = roles  # role -> language code
        self.turns: List[Turn] = []
        self.inbox: Dict[str, List[Optional[str]]] = {r: [] for r in roles}

    def other(self, role: str) -> str:
        return next(r for r in self.roles if r != role)

    def history(self) -> List[dict]:
        return [{"role": t.role, "text": t.source} for t in self.turns if t.kind == "message"]

    @abstractmethod
    def next_role(self) -> str: ...

    @abstractmethod
    def observe(self, role: str) -> Observation: ...

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def score(self) -> dict: ...

    def apply(self, role: str, move: Move, delivered: Optional[ChannelOutput] = None, lid: Optional[dict] = None):
        """Record a move. Messages arrive with the channel's output; answers end the player's part."""
        if move.kind == "message":
            text = delivered.text if delivered else None
            self.inbox[self.other(role)].append(text)
            self.turns.append(Turn(role, "message", move.content, text, delivered.meta if delivered else {}, lid))
        elif move.kind == "answer":
            self.turns.append(Turn(role, "answer", move.content, lid=lid))
        else:
            raise ValueError(f"unknown move kind {move.kind!r}")

    def record(self) -> dict:
        return {"item": self.item, "roles": self.roles, "turns": [asdict(t) for t in self.turns], "score": self.score()}


def run_episode(gm: GameMaster, players: Dict[str, "Player"], channel: Channel,
                lid: Optional[Callable[[str, str], dict]] = None, max_moves: int = 20) -> dict:
    """Play one episode. `lid(text, lang)` optionally checks each player's text is in its language."""
    for _ in range(max_moves):
        if gm.is_done():
            break
        role = gm.next_role()
        move = players[role].act(gm.observe(role))
        check = lid(move.content, gm.roles[role]) if (lid and move.kind == "message") else None
        if move.kind == "message":
            out = channel.translate(move.content, gm.roles[role], gm.roles[gm.other(role)], history=gm.history())
            gm.apply(role, move, out, check)
        else:
            gm.apply(role, move, lid=check)
    rec = gm.record()
    rec["channel"] = channel.name
    rec["finished"] = gm.is_done()
    return rec
