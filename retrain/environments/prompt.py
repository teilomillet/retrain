"""Prompt rendering, tokenization, and observation masks."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import cast

from retrain.environments.coerce import integer as _coerce_int
from retrain.token_sequences import (
    common_prefix_length as _common_prefix_len,
    common_suffix_length as _common_suffix_len,
)
from retrain.types import PromptLike


_ECHO_OBSERVATION_ROLES = frozenset({"tool", "environment", "observation"})


def _coerce_int_ids(raw: object) -> list[int]:
    if not isinstance(raw, list):
        raise TypeError(f"Expected list of token ids, got {type(raw).__name__}")
    return [_coerce_int(tok) for tok in raw]


def preview(prompt: PromptLike, max_chars: int = 200) -> str:
    """Render a compact text preview for logs."""
    if isinstance(prompt, str):
        text = prompt
    else:
        parts: list[str] = []
        for msg in prompt:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            if content:
                parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    return text[:max_chars]


def _chat_template_kwargs(tokenizer: object) -> dict[str, object]:
    """Return optional chat-template kwargs supported by the tokenizer."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return {}
    try:
        sig = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return {}
    if "enable_thinking" in sig.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ):
        return {"enable_thinking": False}
    return {}


def _coerce_tokenizer_ids(raw: object) -> list[int]:
    if isinstance(raw, Mapping):
        input_ids = cast(Mapping[str, object], raw).get("input_ids")
        if input_ids is not None:
            return _coerce_int_ids(input_ids)
    if hasattr(raw, "input_ids"):
        return _coerce_int_ids(getattr(raw, "input_ids"))
    return _coerce_int_ids(raw)


def _encode_chat_template_ids(
    tokenizer: object,
    messages: list[dict[str, object]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if not messages:
        return []
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("Tokenizer must expose apply_chat_template() for chat prompts.")
    ids = apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **_chat_template_kwargs(tokenizer),
    )
    return _coerce_tokenizer_ids(ids)


def encode_for_sampling(tokenizer: object, prompt: PromptLike) -> list[int]:
    """Encode a prompt object (string or chat messages) for model sampling.

    Passes ``enable_thinking=False`` when the tokenizer supports it (e.g.
    Nemotron, Qwen3) to skip the thinking phase and produce direct output.
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)

    if isinstance(prompt, str):
        if callable(apply_chat_template):
            messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]
            return _encode_chat_template_ids(
                tokenizer,
                messages,
                add_generation_prompt=True,
            )
        if not callable(encode):
            raise TypeError("Tokenizer must expose encode() for string prompts.")
        ids = encode(prompt)
    else:
        if callable(apply_chat_template):
            messages = [
                dict(cast(Mapping[str, object], msg))
                if isinstance(msg, Mapping)
                else {"role": "", "content": str(msg)}
                for msg in prompt
            ]
            return _encode_chat_template_ids(
                tokenizer,
                messages,
                add_generation_prompt=True,
            )
        text = "\n".join(str(msg.get("content", "")) for msg in prompt)
        if not callable(encode):
            raise TypeError("Tokenizer must expose encode() for chat prompts.")
        ids = encode(text)

    return _coerce_tokenizer_ids(ids)


def _is_prefix(prefix: list[int], full: list[int]) -> bool:
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def observation_mask(
    tokenizer: object,
    prompt: PromptLike,
    prompt_ids: list[int],
) -> list[int] | None:
    """Build a prompt-aligned mask for environment/tool observation tokens.

    The mask is derived only when the tokenizer's chat template is prefix-stable
    between message prefixes and the sampled prompt. Returning ``None`` is safer
    than training on guessed positions.
    """

    if isinstance(prompt, str):
        return None
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None

    messages = [
        dict(cast(Mapping[str, object], msg))
        if isinstance(msg, Mapping)
        else {"role": "", "content": str(msg)}
        for msg in prompt
    ]
    if not messages:
        return None

    mask = [0] * len(prompt_ids)
    for idx, message in enumerate(messages):
        role = str(message.get("role", "")).lower()
        if role not in _ECHO_OBSERVATION_ROLES:
            continue
        try:
            before_ids = _encode_chat_template_ids(
                tokenizer,
                messages[:idx],
                add_generation_prompt=False,
            )
            empty_message = dict(message)
            empty_message["content"] = ""
            empty_ids = _encode_chat_template_ids(
                tokenizer,
                [*messages[:idx], empty_message],
                add_generation_prompt=False,
            )
            through_ids = _encode_chat_template_ids(
                tokenizer,
                messages[: idx + 1],
                add_generation_prompt=False,
            )
        except (TypeError, ValueError):
            return None
        if not _is_prefix(before_ids, empty_ids):
            return None
        if not _is_prefix(before_ids, through_ids):
            return None
        if not _is_prefix(through_ids, prompt_ids):
            return None
        empty_delta = empty_ids[len(before_ids) :]
        full_delta = through_ids[len(before_ids) :]
        prefix_len = _common_prefix_len(empty_delta, full_delta)
        suffix_len = _common_suffix_len(
            empty_delta,
            full_delta,
            prefix_len=prefix_len,
        )
        content_start = len(before_ids) + prefix_len
        content_end = len(through_ids) - suffix_len
        if content_end <= content_start:
            continue
        for token_idx in range(content_start, content_end):
            mask[token_idx] = 1

    return mask if any(mask) else None
