"""Environment-backed config defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.config.env import _first_non_empty_env

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def apply_env_defaults(config: TrainConfig) -> None:
    if not config.wandb_project:
        config.wandb_project = _first_non_empty_env(
            "SOMA_WANDB_PROJECT",
            "RETRAIN_WANDB_PROJECT",
            "WANDB_PROJECT",
        )
    if not config.wandb_entity:
        config.wandb_entity = _first_non_empty_env(
            "SOMA_WANDB_ENTITY",
            "RETRAIN_WANDB_ENTITY",
            "WANDB_ENTITY",
        )
    if not config.wandb_group:
        config.wandb_group = _first_non_empty_env(
            "SOMA_WANDB_GROUP",
            "RETRAIN_WANDB_GROUP",
        )
    if not config.wandb_tags:
        config.wandb_tags = _first_non_empty_env(
            "SOMA_WANDB_TAGS",
            "RETRAIN_WANDB_TAGS",
        )
