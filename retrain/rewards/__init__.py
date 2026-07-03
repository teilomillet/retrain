"""Reward functions for RLVR training."""

from __future__ import annotations

from retrain.rewards.boxed import BoxedMathReward, extract_boxed
from retrain.rewards.create import create_reward
from retrain.rewards.custom import CustomReward
from retrain.rewards.types import RewardFunction
from retrain.rewards.verifiers import VerifiersJudgeReward, VerifiersMathReward

__all__ = [
    "BoxedMathReward",
    "CustomReward",
    "RewardFunction",
    "VerifiersJudgeReward",
    "VerifiersMathReward",
    "create_reward",
    "extract_boxed",
]
