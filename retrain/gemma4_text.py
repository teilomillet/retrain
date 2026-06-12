"""Utilities for Gemma4 multimodal checkpoints used as text-only CausalLMs."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def unwrap_peft_model(model):
    """Return the underlying Transformers model when wrapped by PEFT."""
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def is_gemma4_text_model(model) -> bool:
    """Gemma4 multimodal checkpoints need text-only train/inference paths."""
    unwrapped = unwrap_peft_model(model)
    config = getattr(unwrapped, "config", None)
    return (
        getattr(config, "model_type", None) == "gemma4"
        and hasattr(getattr(unwrapped, "model", None), "language_model")
        and hasattr(unwrapped, "lm_head")
    )


def resolve_lora_target_modules(model, suffixes: Iterable[str] = DEFAULT_LORA_TARGET_MODULES):
    """Use exact language-tower targets for Gemma4; suffixes for normal CausalLMs."""
    suffixes = tuple(suffixes)
    if not is_gemma4_text_model(model):
        return list(suffixes)

    unwrapped = unwrap_peft_model(model)
    targets = [
        name
        for name, _ in unwrapped.named_modules()
        if name.startswith("model.language_model.") and name.endswith(suffixes)
    ]
    if not targets:
        raise RuntimeError("Gemma4 text model found, but no language LoRA target modules matched")
    return targets


def forward_logits(model, input_ids, attention_mask):
    """Return logits while bypassing Gemma4's multimodal wrapper when needed."""
    if not is_gemma4_text_model(model):
        return model(input_ids, attention_mask=attention_mask).logits

    unwrapped = unwrap_peft_model(model)
    outputs = unwrapped.model.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return unwrapped.lm_head(outputs.last_hidden_state)


def eos_token_ids(model) -> set[int]:
    generation_config = getattr(model, "generation_config", None)
    token_ids = getattr(generation_config, "eos_token_id", None)
    if token_ids is None:
        token_ids = getattr(getattr(unwrap_peft_model(model), "config", None), "eos_token_id", None)
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}
