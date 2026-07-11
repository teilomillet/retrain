"""Exact token rows for ECHO environment-observation supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from renderers import (
    GLM5Renderer,
    Qwen35Renderer,
    Qwen36Renderer,
    Qwen3Renderer,
    create_renderer,
)

from retrain.types import PromptMessage


class EchoTokenRenderer(Protocol):
    """Renderer surface needed to extend one sampled action exactly."""

    def render_ids(
        self,
        messages: list[PromptMessage],
        *,
        tools: list[dict[str, object]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]: ...

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[PromptMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> object | None: ...


@dataclass(frozen=True)
class EchoTransitionTokens:
    """One sampled prompt/action extended through its new observation body."""

    token_ids: list[int]
    observation_mask: list[int]


class EchoTokenBridgeError(RuntimeError):
    """The token renderer could not prove an exact sampled-prefix bridge."""


def create_echo_token_renderer(tokenizer: object) -> EchoTokenRenderer:
    """Create the official Prime renderer matching retrain's no-thinking prompts."""

    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "qwen3.6" in name:
        return cast(EchoTokenRenderer, Qwen36Renderer(tokenizer, enable_thinking=False))
    if "qwen3.5" in name:
        return cast(EchoTokenRenderer, Qwen35Renderer(tokenizer, enable_thinking=False))
    if "qwen3" in name:
        return cast(EchoTokenRenderer, Qwen3Renderer(tokenizer, enable_thinking=False))
    if "glm-5" in name or "glm5" in name:
        return cast(EchoTokenRenderer, GLM5Renderer(tokenizer, enable_thinking=False))
    return cast(EchoTokenRenderer, create_renderer(tokenizer))


def bridge_observation_tokens(
    renderer: EchoTokenRenderer,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    observation_messages: list[PromptMessage],
) -> EchoTransitionTokens:
    """Extend an exact sampled action and mark only new observation-body tokens.

    Prime's bridge preserves the sampled token prefix. Counterfactual renders with
    an empty message body isolate content tokens without changing quaero's
    model-visible ``user`` role or guessing from a rerendered next prompt.
    """

    if not observation_messages:
        raise EchoTokenBridgeError("No observation messages were provided.")

    prefix = list(prompt_ids) + list(completion_ids)
    full_ids = _bridge_ids(
        renderer,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        observation_messages=observation_messages,
    )
    _require_exact_prefix(full_ids, prefix)

    mask = [0] * len(full_ids)
    for message_idx, message in enumerate(observation_messages):
        content = message.get("content")
        if not isinstance(content, str):
            raise EchoTokenBridgeError(
                "Observation message content must be text for exact ECHO masking."
            )
        empty_messages = [dict(item) for item in observation_messages]
        empty_messages[message_idx]["content"] = ""
        empty_ids = _bridge_ids(
            renderer,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            observation_messages=empty_messages,
        )
        _require_exact_prefix(empty_ids, prefix)
        start = _common_prefix_len(full_ids, empty_ids)
        suffix = _common_suffix_len(full_ids, empty_ids, prefix_len=start)
        stop = len(full_ids) - suffix
        for token_idx in range(max(start, len(prefix)), stop):
            mask[token_idx] = 1

    selected = [idx for idx, include in enumerate(mask) if include]
    if not selected:
        return EchoTransitionTokens(
            token_ids=prefix, observation_mask=[0] * len(prefix)
        )

    # Tokens after the final supervised observation cannot affect earlier causal
    # logits. Trimming them removes the next-assistant opener from the train row.
    stop = selected[-1] + 1
    return EchoTransitionTokens(
        token_ids=full_ids[:stop],
        observation_mask=mask[:stop],
    )


def _bridge_ids(
    renderer: EchoTokenRenderer,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    observation_messages: list[PromptMessage],
) -> list[int]:
    try:
        rendered = renderer.bridge_to_next_turn(
            list(prompt_ids),
            list(completion_ids),
            [dict(message) for message in observation_messages],
        )
    except Exception as exc:
        raise EchoTokenBridgeError(
            "The token renderer failed while bridging this sampled action."
        ) from exc
    if rendered is None:
        raise EchoTokenBridgeError(
            "The token renderer cannot safely bridge this sampled action."
        )
    raw_ids = getattr(rendered, "token_ids", rendered)
    if not isinstance(raw_ids, list) or not all(
        isinstance(token, int) for token in raw_ids
    ):
        raise EchoTokenBridgeError("The token renderer returned invalid token ids.")
    return [cast(int, token) for token in raw_ids]


def _require_exact_prefix(full_ids: list[int], prefix: list[int]) -> None:
    if full_ids[: len(prefix)] != prefix:
        raise EchoTokenBridgeError(
            "The token bridge changed or truncated model-sampled action tokens."
        )


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


def _common_suffix_len(
    left: list[int],
    right: list[int],
    *,
    prefix_len: int,
) -> int:
    limit = min(len(left), len(right)) - prefix_len
    count = 0
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    return count
