"""Gradient-checkpointing policy for the local training backend."""

from __future__ import annotations


CheckpointingMetrics = dict[str, int]


def _checkpointing_layers(model: object) -> list[object]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return []
    return [
        module
        for module in modules()
        if module is not model
        and hasattr(module, "gradient_checkpointing")
        and isinstance(getattr(module, "gradient_checkpointing"), bool)
    ]


def enable_gradient_checkpointing(model: object, use_reentrant: str) -> None:
    """Enable model checkpointing with the backend's compatibility mode."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        return
    mode = str(use_reentrant or "auto").lower()
    if mode in ("true", "false"):
        enable(gradient_checkpointing_kwargs={"use_reentrant": mode == "true"})
    else:
        enable()


def configure_gradient_checkpointing(
    model: object,
    *,
    enabled: bool,
    use_reentrant: str,
    skip_last_n: int = 0,
) -> CheckpointingMetrics:
    """Apply checkpointing policy and return layer-level runtime metrics."""
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing(model, use_reentrant)
        return _apply_layer_policy(model, skip_last_n)

    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    return {
        "total": len(_checkpointing_layers(model)),
        "enabled": 0,
        "skipped_last_n": 0,
    }


def _apply_layer_policy(model: object, skip_last_n: int) -> CheckpointingMetrics:
    layers = _checkpointing_layers(model)
    skipped = min(max(0, int(skip_last_n)), len(layers))
    if skipped:
        for module in layers[-skipped:]:
            setattr(module, "gradient_checkpointing", False)
    enabled = sum(
        int(bool(getattr(module, "gradient_checkpointing", False)))
        for module in layers
    )
    return {
        "total": len(layers),
        "enabled": enabled,
        "skipped_last_n": skipped,
    }
