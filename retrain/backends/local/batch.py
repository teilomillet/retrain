"""Tensor batch builders for the local backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PolicyBatch:
    input_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class SftBatch:
    input_ids: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class EchoBatch:
    input_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    echo_advantages: torch.Tensor
    echo_counts: torch.Tensor
    attention_mask: torch.Tensor


def policy(
    tokens: Sequence[Sequence[int]],
    logprobs: Sequence[Sequence[float]],
    advantages: Sequence[Sequence[float]],
    *,
    device: str | torch.device,
) -> PolicyBatch:
    input_ids = _pad_long(tokens, device=device)
    return PolicyBatch(
        input_ids=input_ids,
        old_logprobs=_pad_float(logprobs, device=device),
        advantages=_pad_float(advantages, device=device),
        attention_mask=mask_for_lengths(
            [len(row) for row in tokens],
            max_len=int(input_ids.shape[1]),
            device=device,
        ),
    )


def sft(
    tokens: Sequence[Sequence[int]],
    advantages: Sequence[Sequence[float]],
    *,
    device: str | torch.device,
) -> SftBatch:
    input_ids = _pad_long(tokens, device=device)
    return SftBatch(
        input_ids=input_ids,
        advantages=_pad_float(advantages, device=device),
        attention_mask=mask_for_lengths(
            [len(row) for row in tokens],
            max_len=int(input_ids.shape[1]),
            device=device,
        ),
    )


def echo(
    tokens: Sequence[Sequence[int]],
    logprobs: Sequence[Sequence[float]],
    advantages: Sequence[Sequence[float]],
    echo_advantages: Sequence[Sequence[float]],
    echo_counts: Sequence[int],
    *,
    device: str | torch.device,
) -> EchoBatch:
    input_ids = _pad_long(tokens, device=device)
    return EchoBatch(
        input_ids=input_ids,
        old_logprobs=_pad_float(logprobs, device=device),
        advantages=_pad_float(advantages, device=device),
        echo_advantages=_pad_float(echo_advantages, device=device),
        echo_counts=torch.tensor(
            [float(count) for count in echo_counts],
            dtype=torch.float32,
            device=device,
        ),
        attention_mask=mask_for_lengths(
            [len(row) for row in tokens],
            max_len=int(input_ids.shape[1]),
            device=device,
        ),
    )


def mask_for_lengths(
    lengths: Sequence[int],
    *,
    max_len: int,
    device: str | torch.device,
) -> torch.Tensor:
    row_lengths = torch.tensor(
        [int(length) for length in lengths],
        device=device,
    )
    columns = torch.arange(max_len, device=device).unsqueeze(0)
    return columns < row_lengths.unsqueeze(1)


def _pad_long(
    rows: Sequence[Sequence[int]],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
    return pad_sequence(tensors, batch_first=True, padding_value=0).to(device)


def _pad_float(
    rows: Sequence[Sequence[float]],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    tensors = [torch.tensor(row, dtype=torch.float32) for row in rows]
    return pad_sequence(tensors, batch_first=True, padding_value=0.0).to(device)
