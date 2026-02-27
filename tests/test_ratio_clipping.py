"""Tests for _compute_policy_loss ratio clipping in local_train_helper."""

import pytest
import torch

from retrain.local_train_helper import _compute_policy_loss


class TestComputePolicyLoss:
    def test_disabled_matches_unclipped(self):
        """clip_eps=0 gives same result as raw -(ratio * adv) formula."""
        ratio = torch.tensor([1.0, 1.1, 0.9, 1.5])
        adv = torch.tensor([1.0, -0.5, 0.3, -0.2])
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0])

        loss, frac = _compute_policy_loss(ratio, adv, mask, clip_eps=0.0, clip_eps_high=0.0)

        expected = -(ratio * adv)
        expected_loss = expected.sum() / mask.sum()
        assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-6)
        assert frac == 0.0

    def test_symmetric_clipping(self):
        """clip_eps=0.2 clips ratio to [0.8, 1.2]."""
        # ratio=1.5 should be clipped to 1.2, ratio=0.5 clipped to 0.8
        ratio = torch.tensor([1.5, 0.5, 1.0])
        adv = torch.tensor([1.0, 1.0, 1.0])  # positive advantage
        mask = torch.tensor([1.0, 1.0, 1.0])

        loss, frac = _compute_policy_loss(ratio, adv, mask, clip_eps=0.2, clip_eps_high=0.0)

        # For positive advantage:
        # surr1 = [1.5, 0.5, 1.0], surr2 = [1.2, 0.8, 1.0]
        # min(surr1, surr2) = [1.2, 0.5, 1.0]
        # loss = -mean([1.2, 0.5, 1.0]) = -0.9
        assert loss.item() == pytest.approx(-0.9, abs=1e-6)
        # 2 of 3 tokens are clipped
        assert frac == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_asymmetric_clip_higher(self):
        """clip_eps=0.2, clip_eps_high=0.28 clips to [0.8, 1.28]."""
        ratio = torch.tensor([1.3, 0.7, 1.0])
        adv = torch.tensor([1.0, 1.0, 1.0])  # positive advantage
        mask = torch.tensor([1.0, 1.0, 1.0])

        loss, frac = _compute_policy_loss(ratio, adv, mask, clip_eps=0.2, clip_eps_high=0.28)

        # Bounds: [0.8, 1.28]
        # ratio=1.3 > 1.28 -> clipped_ratio=1.28
        # ratio=0.7 < 0.8 -> clipped_ratio=0.8
        # surr1 = [1.3, 0.7, 1.0], surr2 = [1.28, 0.8, 1.0]
        # min(surr1, surr2) = [1.28, 0.7, 1.0]
        expected_loss = -(1.28 + 0.7 + 1.0) / 3.0
        assert loss.item() == pytest.approx(expected_loss, abs=1e-6)
        # 2 of 3 tokens clipped
        assert frac == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_clip_fraction_correct(self):
        """Verify fraction matches manual count of clipped tokens."""
        ratio = torch.tensor([0.7, 0.9, 1.0, 1.1, 1.3])
        adv = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

        _, frac = _compute_policy_loss(ratio, adv, mask, clip_eps=0.2, clip_eps_high=0.0)

        # Bounds [0.8, 1.2]: 0.7 and 1.3 are outside -> 2/5
        assert frac == pytest.approx(2.0 / 5.0, abs=1e-6)

    def test_padding_excluded_from_fraction(self):
        """Padding tokens should not count toward clip fraction."""
        ratio = torch.tensor([0.5, 1.0, 1.5, 1.0])
        adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])  # last 2 are padding

        _, frac = _compute_policy_loss(ratio, adv, mask, clip_eps=0.2, clip_eps_high=0.0)

        # Only 2 real tokens: 0.5 is clipped, 1.0 is not -> 1/2
        assert frac == pytest.approx(0.5, abs=1e-6)

    def test_negative_advantage_clipping(self):
        """With negative advantage, min() selects the higher surrogate."""
        ratio = torch.tensor([1.5])
        adv = torch.tensor([-1.0])  # negative advantage
        mask = torch.tensor([1.0])

        loss, _ = _compute_policy_loss(ratio, adv, mask, clip_eps=0.2, clip_eps_high=0.0)

        # surr1 = 1.5 * (-1) = -1.5, surr2 = 1.2 * (-1) = -1.2
        # min(-1.5, -1.2) = -1.5
        # loss = -(-1.5) = 1.5
        assert loss.item() == pytest.approx(1.5, abs=1e-6)
