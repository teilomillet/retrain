"""SFT row shaping for the local backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


FloatRows = list[list[float]]
TokenRows = list[list[int]]


@dataclass(frozen=True)
class ContextCropResult:
    tokens: TokenRows
    logprobs: FloatRows | None
    advantages: FloatRows | None
    echo_advantages: FloatRows | None
    metrics: dict[str, int | float]


def microbatch_ranges(
    lengths: Sequence[int],
    microbatch_size: int,
    token_budget: int = 0,
) -> list[tuple[int, int]]:
    """Return microbatch [start, stop) ranges under count/token limits."""
    batch_size = len(lengths)
    if batch_size <= 0:
        return []
    max_count = int(microbatch_size or batch_size)
    max_count = max(1, min(max_count, batch_size))
    token_budget = max(0, int(token_budget or 0))

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < batch_size:
        stop = start
        max_len = 0
        while stop < batch_size and stop - start < max_count:
            candidate_max = max(max_len, int(lengths[stop]))
            candidate_count = stop - start + 1
            candidate_padded = candidate_count * candidate_max
            if token_budget > 0 and stop > start and candidate_padded > token_budget:
                break
            max_len = candidate_max
            stop += 1
        if stop == start:
            stop += 1
        ranges.append((start, stop))
        start = stop
    return ranges


def padding_stats(
    lengths: Sequence[int],
    microbatch_size: int,
    token_budget: int = 0,
) -> tuple[int, int]:
    """Return global and microbatch-local padded token counts."""
    if not lengths:
        return 0, 0
    batch_size = len(lengths)
    global_padded = batch_size * max(lengths)
    microbatch_padded = 0
    for start, stop in microbatch_ranges(lengths, microbatch_size, token_budget):
        chunk = lengths[start:stop]
        if chunk:
            microbatch_padded += len(chunk) * max(chunk)
    return global_padded, microbatch_padded


def pad_microbatch(
    all_tokens: Sequence[Sequence[int]],
    all_advantages: Sequence[Sequence[float]],
    start: int,
    stop: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_tensors = [
        torch.tensor(tokens, dtype=torch.long) for tokens in all_tokens[start:stop]
    ]
    advantage_tensors = [
        torch.tensor(row, dtype=torch.float32) for row in all_advantages[start:stop]
    ]

    input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0).to(
        device
    )
    advantages = pad_sequence(
        advantage_tensors,
        batch_first=True,
        padding_value=0.0,
    ).to(device)

    lengths = torch.tensor(
        [len(tokens) for tokens in all_tokens[start:stop]],
        device=device,
    )
    max_len = input_ids.shape[1]
    attention_mask = torch.arange(max_len, device=device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    return input_ids, advantages, attention_mask


def first_supervised_token_index(*rows: Sequence[float] | None) -> int | None:
    earliest: int | None = None
    for row in rows:
        if row is None:
            continue
        for idx, value in enumerate(row[1:], start=1):
            if float(value) != 0.0:
                earliest = idx if earliest is None else min(earliest, idx)
                break
    return earliest


def crop_supervised_context(
    all_tokens: TokenRows,
    *,
    all_logprobs: FloatRows | None = None,
    all_advantages: FloatRows | None = None,
    echo_advantages: FloatRows | None = None,
    context_tokens: int,
    enabled: bool,
) -> ContextCropResult:
    original_lengths = [len(row) for row in all_tokens]
    if not enabled or not original_lengths:
        max_len = max(original_lengths, default=0)
        return ContextCropResult(
            tokens=all_tokens,
            logprobs=all_logprobs,
            advantages=all_advantages,
            echo_advantages=echo_advantages,
            metrics={
                "local_train_context_rows_cropped": 0,
                "local_train_context_tokens_removed": 0,
                "local_train_context_original_max_tokens": max_len,
                "local_train_context_cropped_max_tokens": max_len,
            },
        )

    cropped_tokens: TokenRows = []
    cropped_logprobs: FloatRows | None = [] if all_logprobs is not None else None
    cropped_advantages: FloatRows | None = [] if all_advantages is not None else None
    cropped_echo: FloatRows | None = [] if echo_advantages is not None else None
    rows_cropped = 0
    tokens_removed = 0

    for idx, tokens in enumerate(all_tokens):
        logprobs = all_logprobs[idx] if all_logprobs is not None else None
        advantages = all_advantages[idx] if all_advantages is not None else None
        echo = echo_advantages[idx] if echo_advantages is not None else None
        earliest = first_supervised_token_index(advantages, echo)
        start = 0
        if earliest is not None:
            start = max(0, int(earliest) - context_tokens)
            if start >= earliest:
                start = max(0, int(earliest) - 1)
        if start > 0:
            rows_cropped += 1
            tokens_removed += start
        cropped_tokens.append(tokens[start:])
        if cropped_logprobs is not None:
            assert logprobs is not None
            cropped_logprobs.append(logprobs[start:])
        if cropped_advantages is not None:
            assert advantages is not None
            cropped_advantages.append(advantages[start:])
        if cropped_echo is not None:
            assert echo is not None
            cropped_echo.append(echo[start:])

    cropped_lengths = [len(row) for row in cropped_tokens]
    return ContextCropResult(
        tokens=cropped_tokens,
        logprobs=cropped_logprobs,
        advantages=cropped_advantages,
        echo_advantages=cropped_echo,
        metrics={
            "local_train_context_rows_cropped": rows_cropped,
            "local_train_context_tokens_removed": tokens_removed,
            "local_train_context_original_max_tokens": max(
                original_lengths,
                default=0,
            ),
            "local_train_context_cropped_max_tokens": max(
                cropped_lengths,
                default=0,
            ),
        },
    )
