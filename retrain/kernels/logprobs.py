"""Selected-token logprob kernels shared by local training paths."""

import torch
import torch.nn.functional as F


def selected_linear_ce_logprobs_no_bias(hidden, weight, target_ids):
    logits = hidden @ weight.T
    return -F.cross_entropy(logits.float(), target_ids, reduction="none")


def selected_linear_ce_logprobs_with_bias(hidden, weight, bias, target_ids):
    logits = (hidden @ weight.T) + bias
    return -F.cross_entropy(logits.float(), target_ids, reduction="none")


def apply_static_range_quantization(x, scale, bits=8):
    scale = scale.to(x.dtype)
    max_value = 2 ** (bits - 1) - 1
    min_value = -max_value - 1
    calibrated = scale != 0
    safe_scale = torch.where(calibrated, scale, torch.ones_like(scale))
    x_q = (
        torch.clamp(
            torch.round(x / safe_scale),
            float(min_value),
            float(max_value),
        )
        * safe_scale
    )
    return torch.where(calibrated, x_q, x)


def unpack_quantized_weight(weight, num_bits, original_width):
    if num_bits == 2:
        packed = weight.to(torch.uint8)
        v0 = (packed & 0x03).to(torch.int8) - 2
        v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
        v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
        v3 = (packed >> 6).to(torch.int8) - 2
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
            *packed.shape[:-1],
            -1,
        )[..., :original_width]
    if num_bits == 4:
        packed = weight.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int8) - 8
        high = (packed >> 4).to(torch.int8) - 8
        return torch.stack([low, high], dim=-1).reshape(
            *packed.shape[:-1],
            -1,
        )[..., :original_width]
    if num_bits == 8:
        return weight
    return None


def packed_quantized_linear_target_logprobs(
    hidden,
    lm_head,
    target_ids,
    *,
    vocab_chunk_size=8192,
):
    """Exact target logprobs for packed Gemma QAT-style QuantizedLinear heads."""
    num_bits = getattr(lm_head, "num_bits", None)
    if num_bits not in (2, 4, 8):
        return None

    weight = getattr(lm_head, "weight", None)
    weight_scale = getattr(lm_head, "weight_scale", None)
    in_features = getattr(lm_head, "in_features", None)
    out_features = getattr(lm_head, "out_features", None)
    if (
        weight is None
        or weight_scale is None
        or in_features is None
        or out_features is None
        or int(hidden.shape[-1]) != int(in_features)
    ):
        return None

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_target_ids = target_ids.reshape(-1).long()
    if flat_hidden.shape[0] != flat_target_ids.numel():
        return None
    if flat_target_ids.numel() == 0:
        return flat_hidden.new_empty(target_ids.shape)

    invalid_targets = (flat_target_ids < 0) | (flat_target_ids >= int(out_features))
    if bool(invalid_targets.any().item()):
        raise ValueError("target id is outside the quantized LM head vocabulary.")

    input_scale = getattr(lm_head, "input_activation_scale", None)
    if input_scale is not None:
        flat_hidden = apply_static_range_quantization(flat_hidden, input_scale)

    bias = getattr(lm_head, "bias", None)
    output_scale = getattr(lm_head, "output_activation_scale", None)
    vocab_chunk_size = max(1, int(vocab_chunk_size))
    target_logits = flat_hidden.new_empty(flat_target_ids.shape, dtype=torch.float32)
    log_denominator = None

    for start in range(0, int(out_features), vocab_chunk_size):
        stop = min(start + vocab_chunk_size, int(out_features))
        int_weight = unpack_quantized_weight(
            weight[start:stop],
            int(num_bits),
            int(in_features),
        )
        if int_weight is None:
            return None
        dequant_weight = int_weight.to(flat_hidden.dtype) * weight_scale[start:stop].to(
            flat_hidden.dtype
        )
        chunk_bias = None if bias is None else bias[start:stop].to(flat_hidden.dtype)
        logits = F.linear(flat_hidden, dequant_weight, chunk_bias)
        if output_scale is not None:
            logits = apply_static_range_quantization(logits, output_scale)
        logits = logits.float()

        chunk_denominator = torch.logsumexp(logits, dim=-1)
        if log_denominator is None:
            log_denominator = chunk_denominator
        else:
            log_denominator = torch.logaddexp(log_denominator, chunk_denominator)

        in_chunk = (flat_target_ids >= start) & (flat_target_ids < stop)
        if bool(in_chunk.any().item()):
            local_target_ids = flat_target_ids[in_chunk] - start
            selected_logits = (
                logits[in_chunk]
                .gather(
                    1,
                    local_target_ids.unsqueeze(1),
                )
                .squeeze(1)
            )
            target_logits[in_chunk] = selected_logits

    assert log_denominator is not None
    return (target_logits - log_denominator).reshape_as(target_ids)
