"""Batch execution by deterministic replay.

Every wave, all episodes are replayed from the start. Player calls answered in earlier waves
come from an append-only cache; the first unanswered call of an episode raises Pending and its
request is collected. The collected requests are sent as one batch per model, and the next
wave replays again. Calls are keyed by (model, episode, role, call index), never by prompt,
so identical prompts in different episodes stay separate; a stored request hash catches any
replay that diverges from what was actually sent.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from .channels import Channel, ChannelOutput

MAX_ATTEMPTS = 2  # a failed call is resubmitted once, then the episode is marked errored


class Pending(Exception):
    pass


class CallFailed(Exception):
    pass


class ReplayCache:
    """Append-only JSONL of call records; the latest record per key wins."""

    def __init__(self, path: Path):
        self.path = path
        self.records: Dict[str, dict] = {}
        if path.exists():
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    self.records[r["key"]] = r

    def get(self, key: str) -> Optional[dict]:
        return self.records.get(key)

    def put(self, record: dict):
        self.records[record["key"]] = record
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_hash(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:16]


class BatchGenerate:
    """LLMPlayer callable for one (backend, episode, role); collects unanswered calls into `pending`."""

    def __init__(self, backend, cache: ReplayCache, episode_id: str, role: str, pending: Dict[str, dict]):
        self.backend, self.cache, self.episode_id, self.role, self.pending = backend, cache, episode_id, role, pending
        self.n = 0

    def __call__(self, prompt, system_prompt, images):
        key = f"{self.backend.model}|{self.episode_id}|{self.role}|{self.n}"
        self.n += 1
        req = self.backend.build(prompt, system_prompt, images)
        h = request_hash(req)
        rec = self.cache.get(key)
        attempts = 0
        if rec is not None:
            if rec["request_hash"] != h:
                raise RuntimeError(f"replay diverged at {key}: request differs from the one sent")
            if "text" in rec:
                return rec["text"]
            attempts = rec.get("attempts", 1)
            if attempts >= MAX_ATTEMPTS:
                raise CallFailed(f"{key}: {rec.get('error')}")
        self.pending[key] = {"request": req, "request_hash": h, "attempts": attempts, "model": self.backend.model}
        raise Pending(key)


class CachedChannel(Channel):
    """Caches channel outputs per (episode, call index) so non-deterministic channels replay identically."""

    def __init__(self, inner: Channel, cache: ReplayCache, episode_id: str):
        self.inner, self.cache, self.episode_id = inner, cache, episode_id
        self.name = inner.name
        self.n = 0

    def supports(self, src, tgt):
        return self.inner.supports(src, tgt)

    def translate(self, text, src, tgt, history=None):
        key = f"channel|{self.inner.name}|{self.episode_id}|{self.n}"
        self.n += 1
        rec = self.cache.get(key)
        if rec is not None and rec["source"] == text:
            return ChannelOutput(rec["text"], rec["meta"])
        out = self.inner.translate(text, src, tgt, history)
        self.cache.put({"key": key, "source": text, "text": out.text, "meta": out.meta})
        return out
