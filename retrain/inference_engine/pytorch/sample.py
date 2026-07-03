"""Token sampling math for the PyTorch inference engine."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shannon_entropy_from_probs_logprobs(
    probs: torch.Tensor,
    log_probs: torch.Tensor,
) -> torch.Tensor:
    safe_log_probs = log_probs.masked_fill(probs == 0, 0.0)
    return -(probs * safe_log_probs).sum(dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    compute_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    scaled = logits / max(float(temperature), 1e-7)
    probs = F.softmax(scaled.float(), dim=-1)
    entropy = None
    if compute_entropy:
        entropy = shannon_entropy_from_probs_logprobs(
            probs,
            probs.clamp_min(1e-12).log(),
        )

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > top_p
        filtered = sorted_probs.masked_fill(remove, 0.0)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(filtered, num_samples=1)
        next_token = sorted_indices.gather(1, sampled_sorted)
        next_prob = filtered.gather(1, sampled_sorted).squeeze(1)
    else:
        next_token = torch.multinomial(probs, num_samples=1)
        next_prob = probs.gather(1, next_token).squeeze(1)

    return next_token, next_prob.clamp_min(1e-12).log(), entropy
