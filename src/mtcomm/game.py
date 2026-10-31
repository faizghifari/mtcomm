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
from .languages import get as lang_info


@dataclass
class Observation:
    role: str
    lang: str
    instructions: str  # what this player is told, including private information
    images: List[str] = field(default_factory=list)  # image paths this player can see
    inbox: List[Optional[str]] = field(default_factory=list)  # messages received so far (already translated)
    # The whole conversation in this player's view (own text as written, partner's as delivered).
    # Filled by run_episode for players whose game shares a history (non-empty inbox).
    dialogue: List[dict] = field(default_factory=list)
    feedback: Optional[str] = None  # why this player's last attempt was rejected (games with validate())
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
    rejected: List[dict] = field(default_factory=list)  # attempts the game refused before this one
    violation: Optional[str] = None  # set when the sent message still broke the game's rules


UNREADABLE = "(a message in a language you cannot read)"


class GameMaster(ABC):
    """One episode of one task item."""

    def __init__(self, item: dict, roles: Dict[str, str]):
        self.item = item
        self.roles = roles  # role -> language code
        self.turns: List[Turn] = []
        self.inbox: Dict[str, List[Optional[str]]] = {r: [] for r in roles}
        self.unreadable: set = set()  # indices into turns of messages their receiver refused as unreadable

    def redact_received(self, role: str) -> int:
        """`role` just answered with "I don't understand": the partner messages it received since its own
        last message become unreadable in everything it sees from now on (a model told to ignore a message
        it can in fact read still uses it at answer time). Returns how many were redacted."""
        last_own = max((i for i, t in enumerate(self.turns) if t.role == role), default=-1)
        k = 0
        received = [i for i, t in enumerate(self.turns) if t.kind == "message" and t.role != role]
        for j, i in enumerate(received):
            if i > last_own and i not in self.unreadable:
                self.unreadable.add(i)
                self.inbox[role][j] = UNREADABLE
                k += 1
        return k

    def other(self, role: str) -> str:
        return next(r for r in self.roles if r != role)

    def dialogue(self, role: str) -> List[dict]:
        return [{"speaker": "you", "text": t.source} if t.role == role else
                {"speaker": "partner", "text": UNREADABLE if i in self.unreadable else t.delivered}
                for i, t in enumerate(self.turns) if t.kind == "message"]

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
        return {"item": self.item, "roles": self.roles, "turns": [asdict(t) for t in self.turns], "score": self.score(),
                "unreadable_turns": sorted(self.unreadable)}


MAX_REWRITES = 2


def run_episode(gm: GameMaster, players: Dict[str, "Player"], channel: Channel,
                lid: Optional[Callable[[str, str], dict]] = None, max_moves: int = 20,
                lang_reject: bool = False, refused: Optional[Callable[[str, str], bool]] = None) -> dict:
    """Play one episode. `lid(text, lang)` optionally checks each player's text is in its language.

    A game may define validate(role, text) -> reason or None: a rejected message never reaches the
    channel; the player is told why and asked to rewrite it, up to MAX_REWRITES times, after which the
    message is sent and flagged as a violation. With lang_reject, a message of 3+ words that the
    language check confidently places in another language is rejected the same way. With `refused(text,
    lang)`, a message that says the player could not understand makes the messages it refused unreadable
    to it from then on (GameMaster.redact_received)."""
    for _ in range(max_moves):
        if gm.is_done():
            break
        role = gm.next_role()
        obs = gm.observe(role)
        if obs.inbox:  # without its own earlier messages a player repeats itself (seen in 20Q dry run)
            obs.dialogue = gm.dialogue(role)
        move = players[role].act(obs)
        rejected, violation = [], None
        validate = getattr(gm, "validate", None)
        while move.kind == "message":
            # an empty message is always rejected (seen with small local players: {"action": "message"})
            violation = "Your message was empty." if not move.content.strip() else (
                validate(role, move.content) if validate is not None else None)
            if violation is None and lang_reject and lid and len(move.content.split()) >= 3:
                c = lid(move.content, gm.roles[role])
                if not c["ok"] and not c["needs_review"]:
                    name = lang_info(gm.roles[role]).name
                    violation = f"Your message is not in {name}. Write it only in {name}."
            if violation is None or len(rejected) == MAX_REWRITES:
                break
            rejected.append({"text": move.content, "reason": violation})
            obs.feedback = violation
            move = players[role].act(obs)
        if move.kind != "message":
            violation = None
        check = lid(move.content, gm.roles[role]) if (lid and move.kind == "message") else None
        if refused is not None and move.kind == "message" and refused(move.content, gm.roles[role]):
            gm.redact_received(role)
        if move.kind == "message":
            out = channel.translate(move.content, gm.roles[role], gm.roles[gm.other(role)], history=gm.history())
            gm.apply(role, move, out, check)
        else:
            gm.apply(role, move, lid=check)
        gm.turns[-1].rejected, gm.turns[-1].violation = rejected, violation
    rec = gm.record()
    rec["channel"] = channel.name
    rec["finished"] = gm.is_done()
    return rec
