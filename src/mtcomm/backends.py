"""Provider backends for LLM players: build one request, parse one response, and run batches.

Each backend turns (prompt, system_prompt, images) into a provider request body, so the same
request can be sent synchronously or written into a batch file. Images are downscaled once to
a fixed copy (max side 768, JPEG) and that copy is used for every model.

Generation settings are explicit and recorded with every request (`config()`):
  Gemini: thinking_level where the model supports it, media_resolution, JSON response mime.
  OpenAI: Responses API, reasoning.effort, image detail, JSON object output. No temperature is
  sent: reasoning models may reject non-default values.
"""

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paths import DATA_ROOT, ENV_FILE

IMAGE_CACHE = DATA_ROOT / "derived" / "xm3600" / "img768"
MAX_SIDE = 768


def load_env():
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)


@lru_cache(maxsize=4096)
def image_b64(path: str) -> str:
    """Base64 of the fixed downscaled JPEG copy of an image (created on first use)."""
    src = Path(path)
    dst = IMAGE_CACHE / src.name
    if not dst.exists():
        from PIL import Image

        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        im = Image.open(src).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        dst.write_bytes(buf.getvalue())
    return base64.b64encode(dst.read_bytes()).decode()


class Backend:
    provider: str
    model: str

    def config(self) -> dict: ...

    def build(self, prompt: str, system_prompt: Optional[str], images: List[str]) -> dict: ...

    def parse(self, response: dict) -> str: ...

    def submit(self, requests: List[Tuple[str, dict]], workdir: Path) -> str: ...

    def status(self, job: str) -> str:  # "running" | "succeeded" | "failed"
        ...

    def results(self, job: str) -> Dict[str, dict]:  # key -> {"text": ...} or {"error": ...}
        ...


class GeminiBackend(Backend):
    provider = "gemini"
    THINKING_MODELS = {"gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview",
                       "gemini-3.1-pro-preview"}

    def __init__(self, model: str, thinking_level: Optional[str] = "low", media_resolution="MEDIA_RESOLUTION_HIGH"):
        self.model = model
        self.thinking_level = thinking_level if model in self.THINKING_MODELS else None
        self.media_resolution = media_resolution
        self._client = None

    def client(self):
        if self._client is None:
            from google import genai

            load_env()
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def config(self):
        return {"provider": self.provider, "model": self.model, "thinking_level": self.thinking_level,
                "media_resolution": self.media_resolution, "response_mime_type": "application/json",
                "image_max_side": MAX_SIDE}

    def build(self, prompt, system_prompt, images):
        parts = []
        for i, p in enumerate(images, 1):
            parts.append({"text": f"Photo {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64(p)}})
        parts.append({"text": prompt})
        gen = {"response_mime_type": "application/json", "media_resolution": self.media_resolution}
        if self.thinking_level:
            gen["thinking_config"] = {"thinking_level": self.thinking_level}
        req = {"contents": [{"role": "user", "parts": parts}], "generation_config": gen}
        if system_prompt:
            req["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return req

    def parse(self, response):
        cands = response.get("candidates") or []
        if not cands:
            raise ValueError(f"no candidates (prompt feedback: {response.get('promptFeedback') or response.get('prompt_feedback')})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise ValueError(f"empty response (finish reason: {cands[0].get('finishReason') or cands[0].get('finish_reason')})")
        return text

    def submit(self, requests, workdir):
        path = workdir / f"{self.model}.jsonl"
        with open(path, "w") as f:
            for key, req in requests:
                f.write(json.dumps({"key": key, "request": req}) + "\n")
        from google.genai import types

        up = self.client().files.upload(file=str(path), config=types.UploadFileConfig(mime_type="jsonl",
                                                                                       display_name=path.name))
        job = self.client().batches.create(model=self.model, src=up.name,
                                           config={"display_name": f"mtcomm-{workdir.name}-{self.model}"})
        return job.name

    def status(self, job):
        state = str(self.client().batches.get(name=job).state)
        if "SUCCEEDED" in state:
            return "succeeded"
        if any(s in state for s in ("FAILED", "CANCELLED", "EXPIRED")):
            return "failed"
        return "running"

    def results(self, job):
        b = self.client().batches.get(name=job)
        raw = self.client().files.download(file=b.dest.file_name).decode()
        out = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = r.get("key")
            if "error" in r and r["error"]:
                out[key] = {"error": json.dumps(r["error"])[:500]}
                continue
            try:
                out[key] = {"text": self.parse(r["response"]), "usage": r["response"].get("usageMetadata")}
            except Exception as e:  # blocked or empty: recorded as an outcome
                out[key] = {"error": str(e)[:500], "usage": r.get("response", {}).get("usageMetadata")}
        return out


class OpenAIBackend(Backend):
    provider = "openai"

    def __init__(self, model: str, reasoning_effort: str = "low", detail: str = "auto"):
        self.model, self.reasoning_effort, self.detail = model, reasoning_effort, detail
        self._client = None

    def client(self):
        if self._client is None:
            from openai import OpenAI

            load_env()
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client

    def config(self):
        return {"provider": self.provider, "model": self.model, "endpoint": "/v1/responses",
                "reasoning_effort": self.reasoning_effort, "image_detail": self.detail,
                "text_format": "json_object", "image_max_side": MAX_SIDE}

    def build(self, prompt, system_prompt, images):
        content = []
        for i, p in enumerate(images, 1):
            content.append({"type": "input_text", "text": f"Photo {i}:"})
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64(p)}",
                            "detail": self.detail})
        content.append({"type": "input_text", "text": prompt})
        body = {"model": self.model, "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": self.reasoning_effort}, "text": {"format": {"type": "json_object"}}}
        if system_prompt:
            body["instructions"] = system_prompt
        return body

    def parse(self, response):
        texts = []
        for item in response.get("output", []):
            if item.get("type") == "message":
                texts += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
        text = "".join(texts)
        if not text.strip():
            raise ValueError(f"empty response (status: {response.get('status')}, "
                             f"incomplete: {response.get('incomplete_details')})")
        return text

    def submit(self, requests, workdir):
        path = workdir / f"{self.model}.jsonl"
        with open(path, "w") as f:
            for key, body in requests:
                f.write(json.dumps({"custom_id": key, "method": "POST", "url": "/v1/responses", "body": body}) + "\n")
        size = path.stat().st_size
        if size > 190 * 1024 * 1024:
            raise RuntimeError(f"{path} is {size / 1e6:.0f} MB, above the OpenAI batch file limit; split the wave")
        up = self.client().files.create(file=open(path, "rb"), purpose="batch")
        job = self.client().batches.create(input_file_id=up.id, endpoint="/v1/responses", completion_window="24h",
                                           metadata={"name": f"mtcomm-{workdir.name}-{self.model}"})
        return job.id

    def status(self, job):
        s = self.client().batches.retrieve(job).status
        if s == "completed":
            return "succeeded"
        if s in ("failed", "expired", "cancelled"):
            return "failed"
        return "running"

    def results(self, job):
        b = self.client().batches.retrieve(job)
        out = {}
        for fid in (b.output_file_id, b.error_file_id):
            if not fid:
                continue
            for line in self.client().files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key, resp = r["custom_id"], r.get("response") or {}
                if r.get("error") or resp.get("status_code", 200) != 200:
                    out[key] = {"error": json.dumps(r.get("error") or resp.get("body"))[:500]}
                    continue
                try:
                    out[key] = {"text": self.parse(resp["body"]), "usage": resp["body"].get("usage")}
                except Exception as e:
                    out[key] = {"error": str(e)[:500], "usage": resp.get("body", {}).get("usage")}
        return out


def make_backend(model: str) -> Backend:
    if model.startswith("gemini"):
        return GeminiBackend(model)
    if model.startswith("gpt"):
        return OpenAIBackend(model)
    raise ValueError(f"no backend for {model}")
