"""Language check on player and translator output, reusing mt-eval's GlotLID wrapper.

Records the detected label and confidence instead of failing hard: GlotLID can confuse
Sundanese/Javanese with Indonesian (mt-eval keeps a review list for exactly this), so the
pass threshold for the admission gate is decided after the pilot, not here.
"""

from functools import lru_cache

from interpreter_agent_eval.utils.language_verification import load_glotlid_model, verify_language_with_glotlid

from .languages import get as lang


class _NumpyTwoSafe:
    """fasttext's predict() calls np.array(..., copy=False), which raises under NumPy 2 (and
    mt-eval's wrapper then fails open, silently passing every check). Use the low-level call."""

    def __init__(self, model):
        self._m = model

    def predict(self, text, k=1, threshold=0.0):
        pairs = self._m.f.predict(text, k, threshold, "strict")
        return tuple(label for _, label in pairs), tuple(prob for prob, _ in pairs)


@lru_cache(maxsize=1)
def _model():
    m = load_glotlid_model()
    return _NumpyTwoSafe(m) if m is not None else None


def check(text: str, lang_code: str, min_confidence: float = 0.5) -> dict:
    res = verify_language_with_glotlid(_model(), text, lang(lang_code).iso3, min_confidence=min_confidence,
                                       context_name="player")
    return {"ok": res.is_correct, "detected": res.detected_language, "script": res.detected_script,
            "confidence": round(float(res.confidence), 3), "needs_review": res.needs_review}
