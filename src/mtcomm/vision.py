"""Vision-capable player backend (Gemini via google-genai), matching LLMPlayer's callable.

mt-eval's providers are text-only, so image players get this separate backend. It is
dry-run by default: no paid call is made until a cost estimate has been approved.
"""

import os
from pathlib import Path
from typing import List, Optional


class GeminiVision:
    def __init__(self, model: str = "gemini-3.1-flash-lite-preview", temperature: float = 0.0, dry_run: bool = True):
        self.model, self.temperature, self.dry_run = model, temperature, dry_run
        self.calls = []  # dry-run log: prompts and image counts, for cost estimation
        self._client = None

    def __call__(self, prompt: str, system_prompt: Optional[str], images: List[str]) -> str:
        if self.dry_run:
            self.calls.append({"model": self.model, "prompt_chars": len(prompt), "n_images": len(images)})
            return '{"action": "answer", "content": "1"}' if images and "Pick" in prompt else \
                   '{"action": "message", "content": "(dry run)"}'
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        parts = []
        for i, p in enumerate(images, 1):
            parts.append(types.Part.from_text(text=f"Photo {i}:"))
            parts.append(types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=prompt))
        resp = self._client.models.generate_content(
            model=self.model, contents=parts,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=self.temperature))
        return resp.text or ""
