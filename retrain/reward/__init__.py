# This file ensures that reward function modules are imported, which triggers
# the @reward_function decorator and populates the registry in base.py.

from . import exact_match
from . import arithmetic

# Import the core reward components and the new batch reward function creator
from .reward import (
    REWARD_REGISTRY, 
    get_reward_function, 
    reward, 
    calculate_total_reward, 
    create_grpo_batch_reward_func,
    BatchRewardFunction # Also expose the type alias for clarity if used externally
)

print("Reward functions registered.")

# Explicitly define what is exported from this package
__all__ = [
    "REWARD_REGISTRY",
    "get_reward_function",
    "reward",
    "calculate_total_reward",
    "create_grpo_batch_reward_func",
    "BatchRewardFunction"
]

# You can optionally expose the registry or lookup function here if desired
# from .base import REWARD_REGISTRY, get_reward_function
