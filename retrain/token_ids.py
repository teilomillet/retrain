"""Token-id helpers shared by model-specific runtime code."""

from __future__ import annotations

from collections.abc import Callable
from typing import SupportsInt, TypeAlias, cast

_TokenIdAtom: TypeAlias = int | str | bytes | bytearray | SupportsInt


def coerce_token_id_set(raw: object) -> set[int]:
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        token_ids: set[int] = set()
        for item in raw:
            if item is not None:
                token_ids.add(int(cast(_TokenIdAtom, item)))
        return token_ids
    return {int(cast(_TokenIdAtom, raw))}


def model_eos_token_ids(
    model: object,
    *,
    unwrap_model: Callable[[object], object] | None = None,
) -> set[int]:
    """Return generation-config EOS ids, falling back to model config."""
    generation_config = getattr(model, "generation_config", None)
    generation_ids = coerce_token_id_set(
        getattr(generation_config, "eos_token_id", None)
    )
    if generation_ids:
        return generation_ids

    config_owner = unwrap_model(model) if unwrap_model is not None else model
    config = getattr(config_owner, "config", None)
    return coerce_token_id_set(getattr(config, "eos_token_id", None))
