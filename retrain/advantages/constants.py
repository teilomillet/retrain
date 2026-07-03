"""Shared advantage constants."""

from __future__ import annotations

# Cap token surprisal values to prevent inf from poisoning downstream math.
# Real per-token surprisal (-logprob of sampled token) rarely exceeds ~15;
# 50 is a safe upper bound.
MAX_SURPRISAL = 50.0
MAX_ENTROPY = MAX_SURPRISAL  # backward-compat alias
