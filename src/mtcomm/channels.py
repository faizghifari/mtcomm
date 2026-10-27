"""Translation channels: everything a message passes through between the two players.

Every experimental control is a channel, so tasks never special-case them:
  IdentityChannel    same-language ceiling (both players share a language) or raw pass-through
  NoChannel          no communication at all (the visitor's / guesser's prior)
  NLLBChannel        local NLLB-200 on the GPU
  GoogleWebChannel   key-less Google web endpoints (PoC default)
  GoogleChannel      official Cloud Translation API (v2, API key)
  AmazonChannel      Amazon Translate via boto3
  LLMChannel         an LLM translator through mt-eval's InterpreterAgent
  PivotChannel       composition: src -> pivot -> tgt through one or two channels

Channels are stateless per message by default; `history` is passed so that history-aware
translators can be tested as an ablation.
"""

import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from . import languages


@dataclass
class ChannelOutput:
    text: Optional[str]  # None means the message was not delivered (NoChannel)
    meta: dict = field(default_factory=dict)


class Channel(ABC):
    name: str = "channel"
    dry_run: bool = False  # paid backends refuse to call out unless this is False

    def supports(self, src: str, tgt: str) -> bool:
        return True

    @abstractmethod
    def translate(self, text: str, src: str, tgt: str, history: Optional[List[dict]] = None) -> ChannelOutput:
        ...


class IdentityChannel(Channel):
    name = "identity"

    def translate(self, text, src, tgt, history=None):
        return ChannelOutput(text, {"channel": self.name})


class NoChannel(Channel):
    name = "none"

    def translate(self, text, src, tgt, history=None):
        return ChannelOutput(None, {"channel": self.name})


class NLLBChannel(Channel):
    """NLLB-200, loaded lazily in fp16. Prepends the FLORES source token manually because
    transformers 5.x no longer does it via tok.src_lang (same fix as mt-eval's local_nmt)."""

    def __init__(self, model_name="facebook/nllb-200-3.3B", device="cuda", num_beams=4, max_new_tokens=512):
        self.model_name, self.device = model_name, device
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens
        self.name = f"nllb:{model_name.split('/')[-1]}"
        self._tok = self._model = None
        self._lock = threading.Lock()  # one GPU model shared by episode threads

    def supports(self, src, tgt):
        return languages.get(src).nllb is not None and languages.get(tgt).nllb is not None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()

    def translate(self, text, src, tgt, history=None):
        if not self.supports(src, tgt):
            raise ValueError(f"{self.name}: unsupported pair {src}->{tgt}")
        with self._lock:
            return self._translate(text, src, tgt)

    def _translate(self, text, src, tgt):
        self._load()
        import torch

        s, t = languages.get(src).nllb, languages.get(tgt).nllb
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = torch.cat([torch.tensor([[self._tok.convert_tokens_to_ids(s)]]), enc.input_ids], dim=1).to(self.device)
        with torch.no_grad():
            gen = self._model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       forced_bos_token_id=self._tok.convert_tokens_to_ids(t),
                                       max_new_tokens=self.max_new_tokens, num_beams=self.num_beams)
        return ChannelOutput(self._tok.batch_decode(gen, skip_special_tokens=True)[0].strip(),
                             {"channel": self.name, "src_code": s, "tgt_code": t})


class GoogleChannel(Channel):
    """Official Cloud Translation API v2 with an API key (GOOGLE_TRANSLATE_API_KEY).
    mt-eval's GoogleWebTranslator uses the unofficial key-less web endpoint, which is not
    suitable for a published benchmark, so it is not reused here."""

    name = "google:v2"
    URL = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, dry_run=True):
        self.dry_run = dry_run

    def supports(self, src, tgt):
        return languages.get(src).google is not None and languages.get(tgt).google is not None

    def translate(self, text, src, tgt, history=None):
        s, t = languages.get(src).google, languages.get(tgt).google
        if s is None or t is None:
            raise ValueError(f"{self.name}: unsupported pair {src}->{tgt}")
        payload = {"q": text, "source": s, "target": t, "format": "text"}
        if self.dry_run:
            return ChannelOutput(None, {"channel": self.name, "dry_run": True, "request": payload})
        import requests

        r = requests.post(self.URL, params={"key": os.environ["GOOGLE_TRANSLATE_API_KEY"]}, json=payload, timeout=30)
        r.raise_for_status()
        return ChannelOutput(r.json()["data"]["translations"][0]["translatedText"], {"channel": self.name})


class GoogleWebChannel(Channel):
    """Key-less Google Translate via public web endpoints. Tries translate_a/single (the same
    endpoint as mt-eval's GoogleWebTranslator) and falls back to the Chrome-extension endpoint
    when that one is throttled (HTTP 429). Free but unofficial: fine for the PoC, to be replaced
    by GoogleChannel (official API) for the released benchmark."""

    name = "google:web"
    URL = "https://translate.googleapis.com/translate_a/single"
    FALLBACK_URL = "https://clients5.google.com/translate_a/t"
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    def __init__(self, timeout=20.0, max_concurrent=2, retries=5):
        self.timeout, self.retries = timeout, retries
        self._session = None
        self._sem = threading.Semaphore(max_concurrent)

    def supports(self, src, tgt):
        return languages.get(src).google is not None and languages.get(tgt).google is not None

    def translate(self, text, src, tgt, history=None):
        s, t = languages.get(src).google, languages.get(tgt).google
        if s is None or t is None:
            raise ValueError(f"{self.name}: unsupported pair {src}->{tgt}")
        if not text.strip():
            return ChannelOutput("", {"channel": self.name})
        with self._sem:
            for attempt in range(self.retries):
                try:
                    return self._translate(text, s, t)
                except Exception as e:  # throttled on both endpoints, or a network error
                    err = e
                    time.sleep(min(60, 3 * 2 ** attempt))
            raise err

    def _translate(self, text, s, t):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers["User-Agent"] = self.UA
        meta = {"channel": self.name, "src_code": s, "tgt_code": t}
        r = self._session.get(self.URL, params={"client": "gtx", "sl": s, "tl": t, "dt": "t", "q": text},
                              timeout=self.timeout)
        if r.status_code != 429:
            r.raise_for_status()
            segments = r.json()[0] or []
            return ChannelOutput("".join(seg[0] for seg in segments if seg and seg[0]).strip(),
                                 {**meta, "endpoint": "translate_a/single"})
        r = self._session.get(self.FALLBACK_URL, params={"client": "dict-chrome-ex", "sl": s, "tl": t, "q": text},
                              timeout=self.timeout)
        r.raise_for_status()  # both throttled: the caller retries later
        out = r.json()
        # A list with one string per query; with sl=auto each entry is [text, detected_lang].
        first = out[0] if out else ""
        return ChannelOutput((first[0] if isinstance(first, list) else first).strip(),
                             {**meta, "endpoint": "clients5/translate_a/t"})


class AmazonChannel(Channel):
    """Amazon Translate (uses the standard AWS credential chain)."""

    name = "amazon-translate"

    def __init__(self, region=None, dry_run=True):
        self.region, self.dry_run = region, dry_run
        self._client = None

    def supports(self, src, tgt):
        return languages.get(src).amazon is not None and languages.get(tgt).amazon is not None

    def translate(self, text, src, tgt, history=None):
        s, t = languages.get(src).amazon, languages.get(tgt).amazon
        if s is None or t is None:
            raise ValueError(f"{self.name}: unsupported pair {src}->{tgt}")
        request = {"Text": text, "SourceLanguageCode": s, "TargetLanguageCode": t}
        if self.dry_run:
            return ChannelOutput(None, {"channel": self.name, "dry_run": True, "request": request})
        if self._client is None:
            import boto3

            self._client = boto3.client("translate", region_name=self.region)
        out = self._client.translate_text(**request)
        return ChannelOutput(out["TranslatedText"], {"channel": self.name})


class LLMChannel(Channel):
    """An LLM translator through mt-eval's InterpreterAgent (same brief as the mt-eval paper)."""

    def __init__(self, provider, label: str, with_history=False, dry_run=True):
        self.provider, self.with_history, self.dry_run = provider, with_history, dry_run
        self.name = f"llm:{label}" + ("+history" if with_history else "")

    def translate(self, text, src, tgt, history=None):
        from interpreter_agent_eval.interpreter import InterpreterAgent

        s, t = languages.get(src), languages.get(tgt)
        context = None
        if self.with_history and history:
            context = "Earlier messages:\n" + "\n".join(f"{h['role']}: {h['text']}" for h in history[-6:])
        if self.dry_run:
            return ChannelOutput(None, {"channel": self.name, "dry_run": True, "context": context})
        agent = InterpreterAgent(self.provider, s.iso3, t.iso3)
        return ChannelOutput(agent.translate(text, context=context), {"channel": self.name})


class BackendProvider:
    """Adapts an mtcomm backend (synchronous call) to mt-eval's LLMProvider interface."""

    def __init__(self, backend):
        self.backend = backend

    def generate(self, prompt, system_prompt=None, **kwargs):
        out = self.backend.call(self.backend.build(prompt, system_prompt, []))
        if "text" not in out:
            raise RuntimeError(f"{self.backend.model}: {out.get('error')}")
        return out["text"].strip()

    def get_provider_name(self):
        return self.backend.model


def llm_translator(model: str, with_history: bool = False) -> "LLMChannel":
    """An LLM translator using mt-eval's InterpreterAgent brief; plain-text output, synchronous calls."""
    from .backends import make_backend

    kw = {"json_mode": False}
    if model.startswith("gpt"):
        kw["reasoning_effort"] = "low"
    return LLMChannel(BackendProvider(make_backend(model, **kw)), model, with_history=with_history, dry_run=False)


class PivotChannel(Channel):
    """src -> pivot -> tgt. The second leg defaults to the same channel as the first."""

    def __init__(self, first: Channel, pivot: str, second: Optional[Channel] = None):
        self.first, self.second, self.pivot = first, second or first, pivot
        self.name = f"pivot[{self.pivot}]:{first.name}" + (f"+{second.name}" if second else "")

    def supports(self, src, tgt):
        return self.first.supports(src, self.pivot) and self.second.supports(self.pivot, tgt)

    def translate(self, text, src, tgt, history=None):
        mid = self.first.translate(text, src, self.pivot, history)
        if mid.text is None:
            return ChannelOutput(None, {"channel": self.name, "leg1": mid.meta})
        out = self.second.translate(mid.text, self.pivot, tgt, history)
        return ChannelOutput(out.text, {"channel": self.name, "pivot_text": mid.text, "leg1": mid.meta, "leg2": out.meta})
