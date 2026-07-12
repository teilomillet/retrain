"""Policy-gradient loss variants for local RL training."""

import torch


def _masked_mean(values, mask_float, denom):
    return (values * mask_float).sum() / denom


def compute_policy_loss(
    old_logprobs,
    new_logprobs,
    adv,
    mask,
    clip_eps,
    clip_eps_high,
    policy_loss_mode="standard",
    kl_cov_percent=0.2,
    kl_cov_coef=1.0,
    clip_cov_ratio=0.0002,
    clip_cov_min=1.0,
    clip_cov_max=5.0,
):
    """Compute policy loss, optionally with covariance-aware entropy control."""
    logprob_delta = new_logprobs - old_logprobs
    ratio = torch.exp(logprob_delta)
    mask_float = mask.float()
    mask_sum = mask_float.sum()
    denom = mask_sum.clamp(min=1)
    abs_kl = _masked_mean(logprob_delta.abs(), mask_float, denom)

    if policy_loss_mode == "kl_cov":
        per_token_loss = -(ratio * adv)
        valid = mask > 0
        valid_count = int(valid.sum().item())
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        if valid_count > 0 and kl_cov_percent > 0:
            with torch.no_grad():
                valid_adv = adv[valid]
                valid_logp = new_logprobs[valid]
                cov_values = (valid_adv - valid_adv.mean()) * (
                    valid_logp - valid_logp.mean()
                )
                k_tokens = max(
                    1,
                    int(valid_count * min(kl_cov_percent, 100.0) / 100.0),
                )
                selected_flat = torch.topk(cov_values, k_tokens, largest=True).indices
                valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=True)[0]
                selected_indices = valid_indices[selected_flat]
                cov_selected = cov_selected.reshape(-1)
                cov_selected[selected_indices] = True
                cov_selected = cov_selected.reshape_as(mask)
        if cov_selected.any():
            per_token_loss = torch.where(
                cov_selected,
                per_token_loss + kl_cov_coef * logprob_delta.abs(),
                per_token_loss,
            )
        cov_fraction = (cov_selected.float() * mask_float).sum() / denom
        masked_loss = (per_token_loss * mask_float).sum() / denom
        return (
            masked_loss,
            0.0,
            float(cov_fraction.detach().item()),
            float(abs_kl.detach().item()),
        )

    if policy_loss_mode == "clip_cov":
        cov_selected = torch.zeros_like(mask, dtype=torch.bool)
        eps_low = clip_eps if clip_eps > 0 else 1.0
        eps_high = clip_eps_high if clip_eps_high > 0 else eps_low
        pg_loss_unclipped = -(ratio * adv)
        pg_loss_clipped = -(torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv)
        clip_by_origin = (pg_loss_clipped > pg_loss_unclipped) & (mask > 0)
        per_token_loss = torch.maximum(pg_loss_unclipped, pg_loss_clipped)
        with torch.no_grad():
            cov_all = (adv - _masked_mean(adv, mask_float, denom)) * (
                new_logprobs - _masked_mean(new_logprobs, mask_float, denom)
            )
            eligible = (
                (mask > 0)
                & ~clip_by_origin
                & (cov_all > clip_cov_min)
                & (cov_all < clip_cov_max)
            )
            eligible_idx = torch.nonzero(eligible)
            clip_num = max(int(float(clip_cov_ratio) * mask_sum.item()), 1)
            if len(eligible_idx) > 0:
                perm = torch.randperm(len(eligible_idx), device=eligible_idx.device)
                selected = eligible_idx[perm[: min(clip_num, len(eligible_idx))]]
                cov_selected[selected[:, 0], selected[:, 1]] = True
        per_token_loss = torch.where(
            cov_selected,
            torch.zeros_like(per_token_loss),
            per_token_loss,
        )
        cov_fraction = (cov_selected.float() * mask_float).sum() / denom
        masked_loss = (per_token_loss * mask_float).sum() / denom
        cov_fraction_value = float(cov_fraction.detach().item())
        return (
            masked_loss,
            cov_fraction_value,
            cov_fraction_value,
            float(abs_kl.detach().item()),
        )

    if policy_loss_mode != "standard":
        raise ValueError(
            "policy_loss_mode must be 'standard', 'kl_cov', or 'clip_cov'."
        )

    if clip_eps > 0:
        eps_high = clip_eps_high if clip_eps_high > 0 else clip_eps
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + eps_high)
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        per_token_loss = -torch.min(surr1, surr2)
        with torch.no_grad():
            clipped = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + eps_high)).float()
            frac = (clipped * mask_float).sum().item() / denom.item()
    else:
        per_token_loss = -(ratio * adv)
        frac = 0.0

    masked_loss = (per_token_loss * mask_float).sum() / denom
    return masked_loss, frac, 0.0, float(abs_kl.detach().item())
