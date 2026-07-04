"""Terminal summaries for training runs."""

from __future__ import annotations

from retrain.config import TrainConfig
from retrain.training.flow import _condition_label
from retrain.training.signals import CORRECT_THRESHOLD


def print_config_summary(config: TrainConfig) -> None:
    """Print a bordered summary of key config values at train start."""
    lines = [
        f"  model         : {config.model}",
        f"  backend       : {config.backend}",
        f"  algorithm     : {_condition_label(config)}",
        f"  batch_size    : {config.batch_size}",
        f"  group_size    : {config.group_size}",
        f"  max_steps     : {config.max_steps}",
        f"  lr            : {config.lr}"
        + (f"  (sft_lr: {config.sft_lr})" if config.sft_lr > 0 else ""),
        f"  lora_rank     : {config.lora_rank}",
        f"  max_tokens    : {config.max_tokens}",
        f"  temperature   : {config.temperature}",
        f"  seed          : {config.seed}",
        f"  adapter_path  : {config.adapter_path}",
    ]
    if config.wandb_project:
        lines.append(f"  wandb         : {config.wandb_project}")
    lines.append(f"  ckpt_artifacts: {config.checkpoint_artifacts}")
    if config.resume_from:
        lines.append(f"  resume_from   : {config.resume_from}")
    if config.echo_enabled:
        lines.append(
            "  echo          : "
            f"on weight={config.echo_weight} "
            f"cap={config.echo_max_tokens_per_step} "
            f"ratio={config.echo_max_token_ratio}"
        )
    width = max(len(line) for line in lines) + 2
    sep = "-" * width
    print(sep)
    for line in lines:
        print(line)
    print(sep)


def print_backend_capability_summary(
    backend_name: str,
    source: str,
    reports_sync_loss: bool,
    preserves_token_advantages: bool,
    supports_checkpoint_resume: bool,
    resume_runtime_dependent: bool,
) -> None:
    """Print backend capability metadata for run-time diagnostics."""
    print(
        "Backend capabilities: "
        f"backend={backend_name}, "
        f"source={source}, "
        f"reports_sync_loss={reports_sync_loss}, "
        f"preserves_token_advantages={preserves_token_advantages}, "
        f"supports_checkpoint_resume={supports_checkpoint_resume}, "
        f"resume_runtime_dependent={resume_runtime_dependent}"
    )
    if not reports_sync_loss:
        print("Backend note: loss is reported as placeholder by backend design.")


def print_flow_warnings(trace_result: object) -> None:
    """Print non-fatal training-flow warnings before setup starts."""
    for issue in getattr(trace_result, "issues", []):
        if getattr(issue, "severity", "") != "warning":
            continue
        category = getattr(issue, "category", "config")
        message = getattr(issue, "message", str(issue))
        print(f"Training flow warning [{category}]: {message}")


def print_group_summary(rewards: list[float], answer: str) -> None:
    correct = sum(1 for reward in rewards if reward > CORRECT_THRESHOLD)
    print(f"  group: {correct}/{len(rewards)} correct | answer={answer[:40]}")


def keep_uniform_group(
    rewards: list[float],
    *,
    batch_advantage_norm: bool,
    keep_for_echo: bool,
) -> bool:
    """Print the disposition of an all-same-reward group; False means skip it.

    Batch advantage normalization can recover cross-group signal from uniform
    groups. ECHO can still use their observation-token datums.
    """
    if batch_advantage_norm:
        print(f"    -> uniform (reward={rewards[0]:.3f}, kept for batch norm)")
        return True
    if keep_for_echo:
        print(f"    -> uniform (reward={rewards[0]:.3f}, kept for ECHO)")
        return True
    if rewards[0] > CORRECT_THRESHOLD:
        print("    -> skipped (all correct)")
    else:
        print(f"    -> skipped (all same, reward={rewards[0]:.3f})")
    return False
