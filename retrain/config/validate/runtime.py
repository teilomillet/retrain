"""Runtime, backend, and environment consistency checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from retrain.backends.catalog import normalize_backend_options
from retrain.config.constants import _VALID_ENVIRONMENT_PROVIDERS

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def collect_runtime_errors(config: TrainConfig, errors: list[str]) -> None:
    if not isinstance(config.plugins_search_paths, list) or not all(
        isinstance(p, str) and p.strip() for p in config.plugins_search_paths
    ):
        errors.append(
            "plugins_search_paths must be a non-empty list of module prefixes."
        )
    if config.environment_provider not in _VALID_ENVIRONMENT_PROVIDERS:
        errors.append(
            f"Invalid environment_provider '{config.environment_provider}'. "
            f"Must be one of: {sorted(_VALID_ENVIRONMENT_PROVIDERS)}"
        )
    if config.environment_provider and not config.environment_id:
        errors.append(
            "environment_id is required when environment_provider is set."
        )
    if (
        config.environment_provider == "openenv"
        and config.environment_id
        and not config.environment_id.startswith(
            ("http://", "https://", "ws://", "wss://")
        )
    ):
        errors.append(
            "environment_id must be a server URL (http(s):// or ws(s)://) "
            "when environment_provider is 'openenv'. "
            "Try: id = \"http://localhost:8765\""
        )
    if config.environment_provider and config.environment_args:
        try:
            parsed_args = json.loads(config.environment_args)
        except json.JSONDecodeError:
            errors.append(
                "environment_args must be valid JSON when "
                "environment_provider is set. "
                "For TOML, prefer: [environment] args = { ... }"
            )
        else:
            if not isinstance(parsed_args, dict):
                errors.append(
                    "environment_args must decode to a JSON object "
                    "when environment_provider is set."
                )
    if config.environment_rollout_env_workers < 1:
        errors.append("environment_rollout_env_workers must be >= 1.")
    if config.environment_rollout_buffer_size < 0:
        errors.append("environment_rollout_buffer_size must be >= 0.")

    if config.tl_grpo and config.tl_grpo_branch_mode not in ("action_space", "llm"):
        errors.append(
            f"tl_grpo_branch_mode must be 'action_space' or 'llm', "
            f"got '{config.tl_grpo_branch_mode}'. Try: tl_grpo_branch_mode = 'action_space'"
        )
    if config.tl_grpo and config.tl_grpo_ema_decay <= 0.0:
        errors.append(
            "tl_grpo_ema_decay must be > 0. Try: tl_grpo_ema_decay = 0.9"
        )
    if config.tl_grpo and config.tl_grpo_ema_decay >= 1.0:
        errors.append(
            "tl_grpo_ema_decay must be < 1. Try: tl_grpo_ema_decay = 0.9"
        )
    if config.tl_grpo and config.tl_grpo_branch_size < 2:
        errors.append(
            "tl_grpo_branch_size must be >= 2 when tl_grpo is enabled. "
            "Try: tl_grpo_branch_size = 4"
        )

    if config.trainer == "command" and not config.trainer_command:
        errors.append(
            "trainer='command' requires [training] trainer_command to be set."
        )

    if not isinstance(config.backend_options, dict):
        errors.append(
            "backend_options must be a mapping. "
            "Use [backend.options] in TOML or --backend-opt key=value."
        )
    else:
        try:
            config.backend_options = normalize_backend_options(
                config.backend,
                config.backend_options,
            )
        except ValueError as exc:
            errors.append(str(exc))

    if (
        config.backend == "local"
        and isinstance(config.backend_options, dict)
        and bool(config.backend_options.get("strict_deterministic", False))
        and config.seed < 0
    ):
        errors.append(
            "backend.options.strict_deterministic=true requires "
            "[training] seed >= 0."
        )

    if (
        config.backend == "prime_rl"
        and isinstance(config.backend_options, dict)
        and config.backend_options.get("strict_advantages") is False
    ):
        errors.append(
            "backend.options.strict_advantages=false is disallowed for backend='prime_rl' "
            "to prevent silent token-advantage aggregation. "
            "Use strict_advantages=true."
        )

    if config.backend == "unsloth" and isinstance(config.backend_options, dict):
        active_unsloth_modes = sum(
            int(bool(config.backend_options.get(key, False)))
            for key in ("load_in_4bit", "load_in_8bit", "load_in_16bit")
        )
        if active_unsloth_modes > 1:
            errors.append(
                "backend='unsloth' accepts only one precision mode. "
                "Set only one of load_in_4bit, load_in_8bit, or load_in_16bit."
            )

    if (
        config.uncertainty_kind == "shannon_entropy"
        and config.inference_engine != "pytorch"
    ):
        errors.append(
            f"uncertainty_kind='shannon_entropy' requires "
            f"inference_engine='pytorch' (got '{config.inference_engine}'). "
            f"Shannon entropy needs GPU-side logit access to compute "
            f"per-token entropy."
        )
    if (
        config.uncertainty_kind == "shannon_entropy"
        and config.backend == "tinker"
    ):
        errors.append(
            "uncertainty_kind='shannon_entropy' is not supported with "
            "backend='tinker'. The Tinker API returns only scalar "
            "logprobs; use backend='local' with inference_engine='pytorch'."
        )
