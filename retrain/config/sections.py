"""TOML and CLI field maps for TrainConfig."""

from __future__ import annotations

import typing
from dataclasses import fields

from retrain.config.schema import TrainConfig


# TOML section -> config field mapping
_TOML_MAP: dict[str, dict[str, str]] = {
    "algorithm": {
        "algorithm_mode": "algorithm_mode",
        "advantage_mode": "advantage_mode",
        "transform_mode": "transform_mode",
        "uncertainty_kind": "uncertainty_kind",
        "surprisal_mask_rho": "surprisal_mask_rho",
        "entropy_mask_rho": "surprisal_mask_rho",
    },
    "backend": {
        "backend": "backend",
        "devices": "devices",
        "adapter_path": "adapter_path",
        "throttle_dir": "tinker_throttle_dir",
        "max_concurrent": "tinker_max_concurrent",
    },
    "model": {
        "model": "model",
        "base_url": "base_url",
        "lora_rank": "lora_rank",
    },
    "training": {
        "seed": "seed",
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "group_size": "group_size",
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "clip_eps": "clip_eps",
        "clip_eps_high": "clip_eps_high",
        "policy_loss_mode": "policy_loss_mode",
        "kl_cov_percent": "kl_cov_percent",
        "kl_cov_coef": "kl_cov_coef",
        "clip_cov_ratio": "clip_cov_ratio",
        "clip_cov_min": "clip_cov_min",
        "clip_cov_max": "clip_cov_max",
        "clip_ratio_c": "clip_ratio_c",
        "adv_clip_max": "adv_clip_max",
        "batch_advantage_norm": "batch_advantage_norm",
        "max_examples": "max_examples",
        "save_every": "save_every",
        "trainer": "trainer",
        "trainer_command": "trainer_command",
        "tl_grpo": "tl_grpo",
        "tl_grpo_branch_mode": "tl_grpo_branch_mode",
        "tl_grpo_branch_size": "tl_grpo_branch_size",
        "tl_grpo_lookahead_steps": "tl_grpo_lookahead_steps",
        "tl_grpo_ema_decay": "tl_grpo_ema_decay",
        "tl_grpo_ema_init": "tl_grpo_ema_init",
        "sft_warmup_steps": "sft_warmup_steps",
        "sft_data_path": "sft_data_path",
        "sft_batch_size": "sft_batch_size",
        "sft_max_tokens": "sft_max_tokens",
        "sft_lr": "sft_lr",
        "sft_loss_fn": "sft_loss_fn",
        "sft_batch_order": "sft_batch_order",
        "sft_length_bucket_size": "sft_length_bucket_size",
    },
    "echo": {
        "enabled": "echo_enabled",
        "weight": "echo_weight",
        "loss_fn": "echo_loss_fn",
        "max_tokens_per_step": "echo_max_tokens_per_step",
        "max_token_ratio": "echo_max_token_ratio",
        "entropy_floor": "echo_entropy_floor",
        "min_prompt_overlap": "echo_min_prompt_overlap",
    },
    "optimizer": {
        "beta1": "optim_beta1",
        "beta2": "optim_beta2",
        "eps": "optim_eps",
        "grad_clip_norm": "grad_clip_norm",
    },
    "lora": {
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
    },
    "gtpo": {"beta": "gtpo_beta"},
    "hicra": {"alpha": "hicra_alpha"},
    "sepa": {
        "steps": "sepa_steps",
        "schedule": "sepa_schedule",
        "delay_steps": "sepa_delay_steps",
        "correct_rate_gate": "sepa_correct_rate_gate",
    },
    "inference": {
        "engine": "inference_engine",
        "url": "inference_url",
        "attention_kernel": "attention_kernel",
        "dtype": "inference_dtype",
        "kv_cache_dtype": "kv_cache_dtype",
        "prefix_caching": "prefix_caching",
    },
    "backpressure": {
        "enabled": "bp_enabled",
        "warmup_steps": "bp_warmup_steps",
        "ema_decay": "bp_ema_decay",
        "throttle_margin": "bp_throttle_margin",
        "increase_margin": "bp_increase_margin",
        "min_batch_size": "bp_min_batch_size",
        "max_batch_size": "bp_max_batch_size",
        "peak_gflops": "bp_peak_gflops",
        "peak_bw_gb_s": "bp_peak_bw_gb_s",
    },
    "planning": {
        "detector": "planning_detector",
        "model": "planning_model",
        "threshold": "planning_threshold",
    },
    "data": {
        "source": "data_source",
    },
    "environment": {
        "provider": "environment_provider",
        "id": "environment_id",
        "args": "environment_args",
        "max_turns": "environment_max_turns",
        "auto_install": "environment_auto_install",
        "rollout_env_workers": "environment_rollout_env_workers",
        "rollout_buffer_size": "environment_rollout_buffer_size",
    },
    "reward": {
        "type": "reward_type",
        "judge_model": "reward_judge_model",
        "custom_module": "reward_custom_module",
        "custom_function": "reward_custom_function",
    },
    "resume": {
        "from": "resume_from",
    },
    "logging": {
        "log_dir": "log_dir",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "wandb_entity": "wandb_entity",
        "wandb_group": "wandb_group",
        "wandb_tags": "wandb_tags",
        "log_generations": "log_generations",
        "generation_log_samples_per_prompt": "generation_log_samples_per_prompt",
        "generation_top_surprisal_limit": "generation_top_surprisal_limit",
        "strategic_grams": "strategic_grams",
    },
    "plugins": {
        "strict": "plugins_strict",
    },
}

# Resolve annotations to actual types (works with `from __future__ import annotations`)
_FIELD_TYPES: dict[str, type] = typing.get_type_hints(TrainConfig)
_MAPPING_OVERRIDE_FIELDS = frozenset(
    {
        "backend_options",
        "algorithm_params",
        "advantage_params",
        "transform_params",
    }
)

# Build CLI flag map: --kebab-case → snake_case field name
_CLI_FLAG_MAP: dict[str, str] = {}
for _f in fields(TrainConfig):
    if _f.name in _MAPPING_OVERRIDE_FIELDS:
        continue
    _CLI_FLAG_MAP["--" + _f.name.replace("_", "-")] = _f.name
# Explicit alias
_CLI_FLAG_MAP["--resume"] = "resume_from"
