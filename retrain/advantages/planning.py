"""Strategic planning-token detection."""

from __future__ import annotations

import re
from functools import lru_cache

DEFAULT_STRATEGIC_GRAMS = [
    "wait let me",
    "let me think",
    "on second thought",
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    "another way to",
    "or we could",
    "what if we",
    "notice that",
    "the key is",
    "the key insight",
]


@lru_cache(maxsize=8192)
def _clean_token_fragment(fragment: str) -> str:
    """Clean a tokenizer fragment: replace subword markers with space."""
    # sentencepiece: \u2581, GPT-2/BPE: \u0120
    return fragment.replace("\u2581", " ").replace("\u0120", " ").strip()


# Cache compiled regex patterns keyed by the tuple of grams
_pattern_cache: dict[tuple[str, ...], list[re.Pattern[str]]] = {}


def _get_gram_patterns(strategic_grams: list[str]) -> list[re.Pattern[str]]:
    """Return compiled regex patterns for strategic grams (cached)."""
    key = tuple(strategic_grams)
    if key not in _pattern_cache:
        _pattern_cache[key] = [
            re.compile(r"\b" + re.escape(gram) + r"\b", re.IGNORECASE)
            for gram in strategic_grams
        ]
    return _pattern_cache[key]


def identify_planning_tokens(
    token_strs: list[str],
    strategic_grams: list[str],
    max_window: int = 5,
) -> list[int]:
    """Identify planning tokens via strategic gram matching.

    Sliding window over token fragments, checking for word-boundary matches.
    """
    n_tokens = len(token_strs)
    if n_tokens == 0 or not strategic_grams:
        return [0] * n_tokens

    # Effective window covers longest gram by word count
    effective_window = max(max_window, max(len(g.split()) for g in strategic_grams))

    # Pre-clean all fragments
    cleaned = [_clean_token_fragment(t) for t in token_strs]

    patterns = _get_gram_patterns(strategic_grams)

    mask = [0] * n_tokens

    for start in range(n_tokens):
        window_end = min(start + effective_window, n_tokens)
        matched = False
        window_text = ""

        for end in range(start, window_end):
            if cleaned[end]:
                if window_text:
                    window_text += " " + cleaned[end]
                else:
                    window_text = cleaned[end]

            for pat in patterns:
                if pat.search(window_text):
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break

    return mask
