"""SFT tokenization and supervised-token batch construction."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from retrain.training.sft_data import SftExample


@dataclass(frozen=True)
class SftTokenizedBatch:
    """Tokenized SFT batch plus useful accounting."""

    tokens: list[list[int]]
    advantages: list[list[float]]
    total_tokens: int
    supervised_tokens: int


@dataclass(frozen=True)
class SftTokenizedExample:
    """One already-tokenized SFT datum."""

    tokens: list[int]
    advantages: list[float]

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def supervised_tokens(self) -> int:
        return _loss_bearing_supervised_tokens(self.advantages)


def _loss_bearing_supervised_tokens(advantages: list[float]) -> int:
    """Count supervised targets that survive the causal one-token shift."""

    return sum(1 for value in advantages[1:] if value > 0.0)


def build_sft_batch_metrics(
    batch: SftTokenizedBatch,
) -> dict[str, int | float]:
    """Describe logical-batch sequence shape without changing SFT tensors.

    These metrics are backend-independent. Local-backend telemetry separately
    reports the padding actually materialized after microbatching.
    """

    lengths = [len(row) for row in batch.tokens]
    if not lengths:
        minimum = median = p95 = maximum = padded_tokens = padding_tokens = 0
        mean = padding_fraction = supervised_fraction = 0.0
    else:
        sorted_lengths = sorted(lengths)
        count = len(sorted_lengths)

        def nearest_rank(percent: int) -> int:
            index = max(0, (percent * count + 99) // 100 - 1)
            return sorted_lengths[index]

        minimum = sorted_lengths[0]
        mean = batch.total_tokens / count
        median = nearest_rank(50)
        p95 = nearest_rank(95)
        maximum = sorted_lengths[-1]
        padded_tokens = count * maximum
        padding_tokens = max(0, padded_tokens - batch.total_tokens)
        padding_fraction = padding_tokens / padded_tokens if padded_tokens else 0.0
        supervised_fraction = (
            batch.supervised_tokens / batch.total_tokens if batch.total_tokens else 0.0
        )
    return {
        "sft_sequence_length_min": minimum,
        "sft_sequence_length_mean": mean,
        "sft_sequence_length_p50": median,
        "sft_sequence_length_p95": p95,
        "sft_sequence_length_max": maximum,
        "sft_logical_padded_tokens": padded_tokens,
        "sft_logical_padding_tokens": padding_tokens,
        "sft_logical_padding_fraction": padding_fraction,
        "sft_supervised_token_fraction": supervised_fraction,
    }


def _last_assistant_index(messages: list[dict[str, str]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            return idx
    raise ValueError("SFT messages must include an assistant target.")


type _ChatTemplate = Callable[..., object]


def _chat_template(tokenizer: object) -> _ChatTemplate | None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None
    return apply_chat_template


def _chat_template_kwargs(apply_chat_template: _ChatTemplate) -> dict[str, object]:
    try:
        sig = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return {}
    if "enable_thinking" in sig.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    ):
        return {"enable_thinking": False}
    return {}


def _render_sft_text(tokenizer: object, example: SftExample) -> tuple[str, str]:
    """Return (full_text, prompt_text) for one SFT example."""
    if example.messages is not None:
        apply_chat_template = _chat_template(tokenizer)
        if apply_chat_template is None:
            raise RuntimeError(
                "SFT rows with 'messages' require a tokenizer with apply_chat_template()."
            )
        chat_template_kwargs = _chat_template_kwargs(apply_chat_template)
        target_idx = _last_assistant_index(example.messages)
        full_messages = example.messages[: target_idx + 1]
        prompt_messages = example.messages[:target_idx]
        full_text = apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        prompt_text = apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        return str(full_text), str(prompt_text)

    if example.prompt or example.completion:
        return f"{example.prompt}{example.completion}", example.prompt

    return example.text, ""


def tokenize_sft_example(
    tokenizer: object,
    example: SftExample,
    *,
    max_tokens: int,
) -> tuple[list[int], list[float]]:
    """Tokenize one SFT datum and build the supervised-token mask."""
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("SFT tokenization requires a tokenizer with encode().")

    if example.messages is None and (example.prompt or example.completion):
        # Tokenize the two fields independently. BPE tokenization is not
        # prefix-stable: encoding ``prompt + completion`` can merge across the
        # boundary and make a completion token look like prompt context.
        prompt_tokens = list(encode(example.prompt, add_special_tokens=False))
        completion_tokens = list(encode(example.completion, add_special_tokens=False))
        full_tokens = prompt_tokens + completion_tokens
        n_prompt = len(prompt_tokens)
    else:
        full_text, prompt_text = _render_sft_text(tokenizer, example)
        full_tokens = list(encode(full_text, add_special_tokens=False))
        prompt_tokens = list(encode(prompt_text, add_special_tokens=False))
        n_prompt = min(len(prompt_tokens), len(full_tokens))
    if max_tokens > 0 and len(full_tokens) > max_tokens:
        crop_start = len(full_tokens) - max_tokens
        full_tokens = full_tokens[crop_start:]
        n_prompt = max(0, n_prompt - crop_start)
    advantages = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)
    if not full_tokens:
        raise ValueError("SFT example tokenized to zero tokens.")
    # Causal LM loss predicts token i from token i - 1, so position zero can
    # never contribute loss. A suffix crop that lands inside the target must
    # reserve its first retained token as causal context instead of reporting
    # it as supervised work that the backend silently shifts away.
    advantages[0] = 0.0
    if _loss_bearing_supervised_tokens(advantages) == 0:
        raise ValueError(
            "SFT example has no loss-bearing supervised target tokens after "
            "causal shifting/truncation; retain at least two tokens (for "
            "example, set sft_max_tokens >= 2)."
        )
    return full_tokens, advantages


def tokenize_sft_batch(
    tokenizer: object,
    examples: list[SftExample],
    *,
    max_tokens: int,
) -> SftTokenizedBatch:
    """Tokenize a batch of SFT examples."""
    return build_sft_tokenized_batch(
        tokenize_sft_dataset(
            tokenizer,
            examples,
            max_tokens=max_tokens,
        )
    )


def tokenize_sft_dataset(
    tokenizer: object,
    examples: list[SftExample],
    *,
    max_tokens: int,
) -> list[SftTokenizedExample]:
    """Tokenize an SFT dataset once so batching can use true lengths."""
    tokenized: list[SftTokenizedExample] = []
    for example in examples:
        row_tokens, row_advantages = tokenize_sft_example(
            tokenizer,
            example,
            max_tokens=max_tokens,
        )
        tokenized.append(
            SftTokenizedExample(tokens=row_tokens, advantages=row_advantages)
        )
    return tokenized


def build_sft_tokenized_batch(
    examples: list[SftTokenizedExample],
) -> SftTokenizedBatch:
    """Build a batch view from pre-tokenized examples."""
    zero_loss_rows = [
        index
        for index, example in enumerate(examples)
        if example.supervised_tokens == 0
    ]
    if zero_loss_rows:
        raise ValueError(
            "SFT tokenized batch contains rows with no loss-bearing supervised "
            f"tokens after causal shifting: {zero_loss_rows}."
        )
    tokens = [example.tokens for example in examples]
    advantages = [example.advantages for example in examples]
    return SftTokenizedBatch(
        tokens=tokens,
        advantages=advantages,
        total_tokens=sum(example.total_tokens for example in examples),
        supervised_tokens=sum(example.supervised_tokens for example in examples),
    )
