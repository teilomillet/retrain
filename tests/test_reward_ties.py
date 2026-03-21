"""Tests for reward tie diagnostics in retrain.trainer."""

from retrain.flow import _UNIFORMITY_EPS
from retrain.trainer import _summarize_reward_ties


class TestSummarizeRewardTies:
    def test_empty_group_is_ineligible(self):
        stats = _summarize_reward_ties([])
        assert stats == {
            "eligible": False,
            "has_tie": False,
            "is_uniform": False,
            "unique_count": 0,
            "tied_pairs": 0,
            "total_pairs": 0,
        }

    def test_singleton_group_is_ineligible(self):
        stats = _summarize_reward_ties([0.5])
        assert stats == {
            "eligible": False,
            "has_tie": False,
            "is_uniform": False,
            "unique_count": 1,
            "tied_pairs": 0,
            "total_pairs": 0,
        }

    def test_distinct_rewards_have_no_ties(self):
        stats = _summarize_reward_ties([0.1, 0.4, 0.8, 1.2])
        assert stats["eligible"] is True
        assert stats["has_tie"] is False
        assert stats["is_uniform"] is False
        assert stats["unique_count"] == 4
        assert stats["tied_pairs"] == 0
        assert stats["total_pairs"] == 6

    def test_epsilon_close_rewards_count_as_ties(self):
        stats = _summarize_reward_ties([0.2, 0.2 + _UNIFORMITY_EPS / 2, 0.9])
        assert stats["eligible"] is True
        assert stats["has_tie"] is True
        assert stats["is_uniform"] is False
        assert stats["unique_count"] == 2
        assert stats["tied_pairs"] == 1
        assert stats["total_pairs"] == 3

    def test_uniform_group_counts_all_pairs_as_tied(self):
        stats = _summarize_reward_ties(
            [0.5, 0.5 + _UNIFORMITY_EPS / 3, 0.5 - _UNIFORMITY_EPS / 3]
        )
        assert stats["eligible"] is True
        assert stats["has_tie"] is True
        assert stats["is_uniform"] is True
        assert stats["unique_count"] == 1
        assert stats["tied_pairs"] == 3
        assert stats["total_pairs"] == 3
