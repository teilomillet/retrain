"""Reward functions for MATH training."""

from __future__ import annotations


def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from MATH solution text."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return ""

    start = idx + len(marker)
    if start >= len(text):
        return ""

    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1

    end = pos - 1 if depth == 0 else pos
    if end <= start:
        return ""
    return text[start:end].strip()


class BoxedMathReward:
    """Binary correctness reward: extract \\boxed{...} and string-match."""

    def score(self, response: str, reference: str) -> float:
        given = extract_boxed(response)
        if given.strip() == reference.strip():
            return 1.0
        return 0.0
