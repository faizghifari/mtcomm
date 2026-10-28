"""Language registry: one entry per language, with each backend's code (None = unsupported).

Coverage verified 2026-10-27 (docs/benchmark_lit/lit_i_*, lit_j_*): Amazon Translate lacks
su/jv/min/bug/ace; Google lacks bug; NLLB-200 covers all (CC-BY-NC model license).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Language:
    code: str  # short code used throughout mtcomm
    name: str
    iso3: str  # ISO 639-3, also what GlotLID and mt-eval use
    nllb: Optional[str]
    google: Optional[str]
    amazon: Optional[str]


LANGUAGES = {
    lang.code: lang
    for lang in [
        Language("en", "English", "eng", "eng_Latn", "en", "en"),
        Language("ko", "Korean", "kor", "kor_Hang", "ko", "ko"),
        Language("id", "Indonesian", "ind", "ind_Latn", "id", "id"),
        Language("su", "Sundanese", "sun", "sun_Latn", "su", None),
        Language("jv", "Javanese", "jav", "jav_Latn", "jw", None),
        Language("min", "Minangkabau", "min", "min_Latn", "min", None),
        Language("bug", "Buginese", "bug", "bug_Latn", None, None),
        Language("ace", "Acehnese", "ace", "ace_Latn", "ace", None),
        # reading languages for the player screen (neighbour / widely known languages)
        Language("ms", "Malay", "zsm", "zsm_Latn", "ms", "ms"),
        Language("zh", "Chinese", "cmn", "zho_Hans", "zh-CN", "zh"),
        Language("ja", "Japanese", "jpn", "jpn_Jpan", "ja", "ja"),
    ]
}


def get(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise KeyError(f"unknown language {code!r}; known: {sorted(LANGUAGES)}") from None
