from __future__ import annotations

import pytest
import torch

from retrain.training.loss import compute_policy_loss


def test_standard_policy_loss_matches_importance_sampling() -> None:
    old = torch.zeros((1, 3), dtype=torch.float32)
    new = torch.tensor([[-0.2, -0.4, -0.6]], dtype=torch.float32)
    adv = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)

    loss, clip_frac, cov_frac, abs_kl = compute_policy_loss(
        old,
        new,
        adv,
        mask,
        clip_eps=0.0,
        clip_eps_high=0.0,
    )

    expected = -(torch.exp(new - old) * adv).mean()
    assert loss.item() == pytest.approx(expected.item())
    assert clip_frac == pytest.approx(0.0)
    assert cov_frac == pytest.approx(0.0)
    assert abs_kl == pytest.approx(new.abs().mean().item())


def test_kl_cov_penalizes_high_covariance_tokens() -> None:
    old = torch.zeros((1, 4), dtype=torch.float32)
    new = torch.tensor([[-5.0, -1.0, -0.1, -0.2]], dtype=torch.float32)
    adv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)

    standard_loss, *_ = compute_policy_loss(
        old,
        new,
        adv,
        mask,
        clip_eps=0.0,
        clip_eps_high=0.0,
    )
    kl_cov_loss, clip_frac, cov_frac, _ = compute_policy_loss(
        old,
        new,
        adv,
        mask,
        clip_eps=0.0,
        clip_eps_high=0.0,
        policy_loss_mode="kl_cov",
        kl_cov_percent=25.0,
        kl_cov_coef=1.0,
    )

    assert kl_cov_loss.item() > standard_loss.item()
    assert clip_frac == pytest.approx(0.0)
    assert cov_frac == pytest.approx(0.25)
    assert kl_cov_loss.item() == pytest.approx(
        standard_loss.item() + 5.0 / 4.0,
    )


def test_clip_cov_detaches_selected_covariance_tokens() -> None:
    old = torch.zeros((1, 4), dtype=torch.float32)
    new = torch.tensor([[-5.0, -1.0, -0.1, -0.2]], dtype=torch.float32)
    adv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)

    standard_loss, *_ = compute_policy_loss(
        old,
        new,
        adv,
        mask,
        clip_eps=1.0,
        clip_eps_high=1.0,
    )
    clip_cov_loss, clip_frac, cov_frac, _ = compute_policy_loss(
        old,
        new,
        adv,
        mask,
        clip_eps=1.0,
        clip_eps_high=1.0,
        policy_loss_mode="clip_cov",
        clip_cov_ratio=1.0,
        clip_cov_min=0.0,
        clip_cov_max=10.0,
    )

    assert clip_cov_loss.item() > standard_loss.item()
    assert clip_frac == pytest.approx(cov_frac)
    assert cov_frac > 0.0


def test_unknown_policy_loss_mode_raises() -> None:
    values = torch.zeros((1, 1), dtype=torch.float32)
    mask = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="policy_loss_mode"):
        compute_policy_loss(
            values,
            values,
            values,
            mask,
            clip_eps=0.0,
            clip_eps_high=0.0,
            policy_loss_mode="bad",
        )
