"""Compatibility exports for token id parsing."""

from __future__ import annotations

from retrain.tokens.ids import coerce_token_id_set, model_eos_token_ids

__all__ = ["coerce_token_id_set", "model_eos_token_ids"]
