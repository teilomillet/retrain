"""Reward factory dispatched on ``[reward] type``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from retrain.rewards.boxed import BoxedMathReward
from retrain.rewards.custom import CustomReward
from retrain.rewards.types import RewardFunction
from retrain.rewards.verifiers import VerifiersJudgeReward, VerifiersMathReward

if TYPE_CHECKING:
    from retrain.config import TrainConfig


_VERIFIERS_TYPES = {"math", "judge"}


def create_reward(config: TrainConfig) -> RewardFunction:
    """Create the reward function specified by *config.reward_type*."""
    rtype = config.reward_type

    if rtype == "match":
        return BoxedMathReward()

    if rtype in _VERIFIERS_TYPES:
        try:
            importlib.import_module("verifiers")
        except ModuleNotFoundError:
            raise ImportError(
                f"Reward type '{rtype}' requires the verifiers library. "
                "Install it with:  pip install verifiers"
            ) from None

        if rtype == "math":
            return VerifiersMathReward()
        if rtype == "judge":
            model = config.reward_judge_model or "gpt-4o-mini"
            return VerifiersJudgeReward(model=model)

    if rtype == "custom":
        if not config.reward_custom_module:
            raise ValueError(
                "Reward type 'custom' requires [reward] custom_module to be set."
            )
        return CustomReward(
            config.reward_custom_module,
            config.reward_custom_function or "score",
        )

    raise ValueError(
        f"Unknown reward type '{rtype}'. "
        "Choose from: match, math, judge, custom"
    )
