"""Utilities for Gemma4 multimodal checkpoints used as text-only CausalLMs."""

from __future__ import annotations

from collections.abc import Iterable

from retrain.tokens.ids import model_eos_token_ids


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_lora_target_module_suffixes(spec: str | None) -> tuple[str, ...]:
    """Return target module suffixes from a comma-separated backend option."""
    text = str(spec or "").strip()
    if not text or text.lower() in {"default", "defaults"}:
        return DEFAULT_LORA_TARGET_MODULES
    suffixes: list[str] = []
    seen: set[str] = set()
    for raw in text.split(","):
        suffix = raw.strip()
        if not suffix or suffix in seen:
            continue
        suffixes.append(suffix)
        seen.add(suffix)
    if not suffixes:
        return DEFAULT_LORA_TARGET_MODULES
    return tuple(suffixes)


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


def resolve_lora_target_modules(
    model, suffixes: Iterable[str] = DEFAULT_LORA_TARGET_MODULES
):
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
        raise RuntimeError(
            "Gemma4 text model found, but no language LoRA target modules matched"
        )
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


def forward_hidden_states_and_lm_head(model, input_ids, attention_mask):
    """Return LM hidden states and head for chunked logprob computation.

    Returns ``None`` when the wrapped model does not expose a standard
    text-backbone + lm_head structure.
    """
    if is_gemma4_text_model(model):
        unwrapped = unwrap_peft_model(model)
        outputs = unwrapped.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.last_hidden_state, unwrapped.lm_head

    unwrapped = unwrap_peft_model(model)
    body = getattr(unwrapped, "model", None)
    lm_head = getattr(unwrapped, "lm_head", None)
    if body is None or lm_head is None:
        return None
    outputs = body(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        return None
    return hidden_states, lm_head


def eos_token_ids(model) -> set[int]:
    return model_eos_token_ids(model, unwrap_model=unwrap_peft_model)
