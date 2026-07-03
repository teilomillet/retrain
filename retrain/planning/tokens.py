"""Tokenizer-fragment cleaning shared by the planning detectors."""

from __future__ import annotations

import re

# Subword markers used by common tokenizers
_SUBWORD_RE = re.compile(r"[\u2581\u0120]")


def clean_fragment(fragment: str) -> str:
    """Clean a tokenizer fragment: replace subword markers with space."""
    return _SUBWORD_RE.sub(" ", fragment).strip()
