"""Regex/strategic-gram planning detector."""

from __future__ import annotations

import re

from retrain.planning.tokens import clean_fragment


class RegexPlanningDetector:
    """Backward-compatible regex/strategic-gram detector."""

    def __init__(self, strategic_grams: list[str]) -> None:
        self._grams = strategic_grams
        self._pattern = (
            re.compile(
                "(?:"
                + "|".join(
                    r"\b" + re.escape(gram) + r"\b"
                    for gram in strategic_grams
                )
                + ")",
                re.IGNORECASE,
            )
            if strategic_grams
            else None
        )
        self._effective_window = (
            max(5, max(len(gram.split()) for gram in strategic_grams))
            if strategic_grams
            else 5
        )
        self._clean_cache: dict[str, str] = {}

    def detect(self, token_strs: list[str]) -> list[int]:
        n_tokens = len(token_strs)
        if n_tokens == 0:
            return []
        if self._pattern is None:
            return [0] * n_tokens

        cleaned: list[str] = []
        clean_cache = self._clean_cache
        for token in token_strs:
            cleaned_token = clean_cache.get(token)
            if cleaned_token is None:
                cleaned_token = clean_fragment(token)
                if len(clean_cache) < 8192:
                    clean_cache[token] = cleaned_token
            cleaned.append(cleaned_token)

        mask = [0] * n_tokens
        pattern = self._pattern
        effective_window = self._effective_window
        for start in range(n_tokens):
            window_text = ""
            window_end = min(start + effective_window, n_tokens)
            for end in range(start, window_end):
                if cleaned[end]:
                    if window_text:
                        window_text += " " + cleaned[end]
                    else:
                        window_text = cleaned[end]
                if pattern.search(window_text):
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    break
        return mask
