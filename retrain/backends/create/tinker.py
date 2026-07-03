"""Tinker backend constructor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from retrain.training.sft import effective_sft_loss_fn

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


def create_tinker(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.tinker import TinkerTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'tinker' requires the tinker SDK.\n"
            "Install it with: pip install retrain[tinker]"
        ) from None

    tinker_url = config.inference_url or config.base_url
    helper = TinkerTrainHelper(
        config.model,
        tinker_url,
        config.lora_rank,
        optim_beta1=config.optim_beta1,
        optim_beta2=config.optim_beta2,
        optim_eps=config.optim_eps,
        throttle_dir=config.tinker_throttle_dir,
        max_concurrent=config.tinker_max_concurrent,
        clip_eps=config.clip_eps,
        clip_eps_high=config.clip_eps_high,
        grad_clip_norm=config.grad_clip_norm,
        clip_ratio_c=config.clip_ratio_c,
        sample_log_dir=str(Path(config.log_dir).resolve()),
    )
    setattr(helper, "sft_loss_fn", effective_sft_loss_fn(config))
    return helper
