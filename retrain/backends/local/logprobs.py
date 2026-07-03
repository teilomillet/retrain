"""Token log-probability paths for the local backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import torch
import torch.nn.functional as F

from retrain.backends.local import metrics as local_metrics
from retrain.kernels.logprobs import (
    packed_quantized_linear_target_logprobs as _packed_quantized_linear_target_logprobs,
)
from retrain.models.gemma4 import (
    forward_hidden_states_and_lm_head as _forward_hidden_states_and_lm_head,
    forward_logits as _forward_logits,
)


HiddenAndHead = tuple[torch.Tensor, torch.nn.Module]
HiddenAndHeadFn = Callable[
    [object, torch.Tensor, torch.Tensor | None],
    HiddenAndHead | None,
]
ForwardLogitsFn = Callable[[object, torch.Tensor, torch.Tensor | None], torch.Tensor]


class LogprobOwner(Protocol):
    train_model: torch.nn.Module
    train_logprob_chunk_size: int
    train_selective_suffix_logits: bool
    _train_logits_to_keep_supported: bool | None

    def _supports_train_logits_to_keep(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> bool: ...

    def _selective_suffix_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ): ...

    def _selective_hidden_token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ): ...

    def _liger_fused_linear_ce_loss(self): ...

    def _compiled_selective_ce_logprobs(
        self,
        selected_hidden: torch.Tensor,
        lm_head: torch.nn.Module,
        target_ids: torch.Tensor,
    ): ...


def supports_train_logits_to_keep(
    owner: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> bool:
    state = cast(LogprobOwner, owner)
    cached = getattr(state, "_train_logits_to_keep_supported", None)
    if cached is not None:
        return bool(cached)

    seq_len = int(input_ids.shape[1])
    probe_len = min(seq_len, 4)
    if probe_len < 2:
        state._train_logits_to_keep_supported = False
        return False

    logits_to_keep = min(2, probe_len - 1)
    probe_ids = input_ids[:1, -probe_len:]
    probe_mask = attention_mask[:1, -probe_len:] if attention_mask is not None else None
    train_model = state.train_model
    was_training = bool(getattr(train_model, "training", False))
    try:
        train_model.eval()
        with torch.no_grad():
            outputs = train_model(
                input_ids=probe_ids,
                attention_mask=probe_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
    except TypeError:
        supported = False
    except Exception:  # Fall back to the hidden-state path.
        supported = False
    else:
        logits = getattr(outputs, "logits", None)
        supported = (
            logits is not None
            and len(getattr(logits, "shape", ())) >= 2
            and int(logits.shape[1]) == logits_to_keep
        )
    finally:
        train_model.train(was_training)

    state._train_logits_to_keep_supported = supported
    return supported


def selective_suffix_token_logprobs(
    owner: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target_mask: torch.Tensor | None,
):
    state = cast(LogprobOwner, owner)
    if not getattr(state, "train_selective_suffix_logits", False):
        return None
    if target_mask is None or not bool(target_mask.any().item()):
        return None
    if int(getattr(state, "train_logprob_chunk_size", 0)) > 0:
        return None

    selected = torch.nonzero(target_mask, as_tuple=False)
    if selected.numel() == 0:
        return None

    seq_len = int(input_ids.shape[1])
    min_target_pos = int(selected[:, 1].min().item())
    logits_to_keep = seq_len - min_target_pos
    if logits_to_keep <= 1 or logits_to_keep >= seq_len:
        return None
    selected_tokens = int(selected.shape[0])
    suffix_target_slots = max(1, logits_to_keep - 1)
    # Long ECHO/tool traces often supervise a small number of tokens inside
    # a large suffix. In that shape, logits_to_keep would still materialize
    # [long_suffix, vocab]; the hidden path keeps the same transformer
    # forward and computes the LM head only for selected target positions.
    if suffix_target_slots > 2048 and selected_tokens / suffix_target_slots < 0.25:
        local_metrics.record_selective_logprob_path(state, "sparse_suffix_skip")
        return None
    if not state._supports_train_logits_to_keep(input_ids, attention_mask):
        return None

    try:
        outputs = state.train_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
    except TypeError:
        return None

    logits = getattr(outputs, "logits", None)
    if logits is None:
        return None
    if int(logits.shape[1]) != logits_to_keep:
        state._train_logits_to_keep_supported = False
        return None

    logits_start = seq_len - logits_to_keep
    target_suffix = input_ids[:, logits_start + 1 :]
    usable_logits = logits[:, : target_suffix.shape[1], :]
    suffix_logprobs = F.log_softmax(usable_logits.float(), dim=-1)
    suffix_logprobs = suffix_logprobs.gather(
        2,
        target_suffix.unsqueeze(2),
    ).squeeze(2)
    full = suffix_logprobs.new_zeros((input_ids.shape[0], seq_len - 1))
    full[:, logits_start : logits_start + suffix_logprobs.shape[1]] = suffix_logprobs
    local_metrics.record_selective_logprob_path(state, "suffix")
    return full


def selective_hidden_token_logprobs(
    owner: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    *,
    hidden_and_head_fn: HiddenAndHeadFn = _forward_hidden_states_and_lm_head,
):
    state = cast(LogprobOwner, owner)
    if target_mask is None or not bool(target_mask.any().item()):
        return None

    selected = torch.nonzero(target_mask, as_tuple=False)
    if selected.numel() == 0:
        return None

    hidden_and_head = hidden_and_head_fn(
        state.train_model,
        input_ids,
        attention_mask,
    )
    if hidden_and_head is None:
        return None

    hidden_states, lm_head = hidden_and_head
    shifted_len = int(input_ids.shape[1]) - 1
    if int(hidden_states.shape[1]) < shifted_len:
        return None

    target_ids = input_ids[selected[:, 0], selected[:, 1] + 1]
    selected_hidden = hidden_states[selected[:, 0], selected[:, 1], :]
    chunk_size = int(getattr(state, "train_logprob_chunk_size", 0))
    if chunk_size <= 0:
        chunk_size = 256

    selected_logprobs = state._compiled_selective_ce_logprobs(
        selected_hidden,
        lm_head,
        target_ids,
    )
    used_compiled_ce = selected_logprobs is not None
    used_packed_quantized = False
    if selected_logprobs is None:
        selected_logprobs = _packed_quantized_linear_target_logprobs(
            selected_hidden,
            lm_head,
            target_ids,
        )
        used_packed_quantized = selected_logprobs is not None
        if selected_logprobs is None:
            chunks = []
            for start in range(0, selected_hidden.shape[0], chunk_size):
                stop = min(start + chunk_size, selected_hidden.shape[0])
                logits = lm_head(selected_hidden[start:stop])
                # cross_entropy is exactly LogSoftmax + NLLLoss for class-index
                # targets, and avoids materializing the full selected-token
                # log-probability matrix before gathering one class per row.
                chunks.append(
                    -F.cross_entropy(
                        logits.float(),
                        target_ids[start:stop],
                        reduction="none",
                    )
                )
            if not chunks:
                return None
            selected_logprobs = torch.cat(chunks, dim=0)
    if selected_logprobs.numel() == 0:
        return None
    full = selected_logprobs.new_zeros((input_ids.shape[0], shifted_len))
    full[selected[:, 0], selected[:, 1]] = selected_logprobs
    local_metrics.record_selective_logprob_path(state, "hidden")
    if used_compiled_ce:
        local_metrics.record_selective_logprob_path(state, "compiled_ce")
    if used_packed_quantized:
        local_metrics.record_selective_logprob_path(
            state,
            "packed_quantized_lm_head",
        )
    return full


def shifted_token_logprobs(
    owner: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target_mask: torch.Tensor | None = None,
    *,
    hidden_and_head_fn: HiddenAndHeadFn = _forward_hidden_states_and_lm_head,
    forward_logits_fn: ForwardLogitsFn = _forward_logits,
):
    """Compute next-token logprobs, optionally chunking the LM head."""

    state = cast(LogprobOwner, owner)
    selective = state._selective_suffix_token_logprobs(
        input_ids,
        attention_mask,
        target_mask,
    )
    if selective is not None:
        return selective

    if getattr(state, "train_selective_suffix_logits", False) and target_mask is not None:
        selective = state._selective_hidden_token_logprobs(
            input_ids,
            attention_mask,
            target_mask,
        )
        if selective is not None:
            return selective
        local_metrics.record_selective_logprob_path(state, "fallback")

    liger_loss = state._liger_fused_linear_ce_loss()
    if liger_loss is not None:
        hidden_and_head = hidden_and_head_fn(
            state.train_model,
            input_ids,
            attention_mask,
        )
        if hidden_and_head is not None:
            hidden_states, lm_head = hidden_and_head
            shifted_hidden = hidden_states[:, :-1, :]
            target_ids = input_ids[:, 1:]
            flat_hidden = shifted_hidden.reshape(-1, shifted_hidden.shape[-1])
            flat_target_ids = target_ids.reshape(-1)
            packed_logprobs = _packed_quantized_linear_target_logprobs(
                flat_hidden,
                lm_head,
                flat_target_ids,
            )
            if packed_logprobs is not None:
                local_metrics.record_loss_path(
                    state,
                    "packed_quantized_lm_head",
                )
                return packed_logprobs.reshape_as(target_ids)
            weight = getattr(lm_head, "weight")
            try:
                nll = liger_loss(weight, flat_hidden, flat_target_ids)
            except TypeError:
                nll = liger_loss(flat_hidden, weight, flat_target_ids)
            local_metrics.record_loss_path(state, "liger_fused_ce")
            return -nll.reshape_as(target_ids)

    chunk_size = int(getattr(state, "train_logprob_chunk_size", 0))
    if chunk_size <= 0:
        logits = forward_logits_fn(state.train_model, input_ids, attention_mask)[:, :-1]
        new_logprobs = F.log_softmax(logits.float(), dim=-1)
        target_ids = input_ids[:, 1:]
        local_metrics.record_loss_path(state, "dense_logprob")
        return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

    hidden_and_head = hidden_and_head_fn(
        state.train_model,
        input_ids,
        attention_mask,
    )
    if hidden_and_head is None:
        logits = forward_logits_fn(state.train_model, input_ids, attention_mask)[:, :-1]
        new_logprobs = F.log_softmax(logits.float(), dim=-1)
        target_ids = input_ids[:, 1:]
        local_metrics.record_loss_path(state, "dense_logprob")
        return new_logprobs.gather(2, target_ids.unsqueeze(2)).squeeze(2)

    hidden_states, lm_head = hidden_and_head
    shifted_hidden = hidden_states[:, :-1, :]
    target_ids = input_ids[:, 1:]
    packed_logprobs = _packed_quantized_linear_target_logprobs(
        shifted_hidden,
        lm_head,
        target_ids,
    )
    if packed_logprobs is not None:
        local_metrics.record_loss_path(state, "packed_quantized_lm_head")
        return packed_logprobs

    chunks = []
    for start in range(0, shifted_hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, shifted_hidden.shape[1])
        logits = lm_head(shifted_hidden[:, start:stop, :])
        logprobs = F.log_softmax(logits.float(), dim=-1)
        chunks.append(
            logprobs.gather(
                2,
                target_ids[:, start:stop].unsqueeze(2),
            ).squeeze(2)
        )
    if not chunks:
        return shifted_hidden.new_empty((shifted_hidden.shape[0], 0))
    local_metrics.record_loss_path(state, "chunked_logprob")
    return torch.cat(chunks, dim=1)
