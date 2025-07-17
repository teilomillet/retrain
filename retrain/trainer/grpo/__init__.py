"""
GRPO Module - Clean, Unified Implementation

Exports:
- GRPO: Single class with auto hardware detection
- DRGRPO: Dr. GRPO (GRPO Done Right) - removes length and std normalization biases

Use GRPO.remote(config) and DRGRPO.remote(config) directly.
"""

from .grpo import GRPO
from .drgrpo import DRGRPO

__all__ = [
    'GRPO',
    'DRGRPO',
]
