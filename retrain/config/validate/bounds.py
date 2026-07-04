"""Numeric and enum bound checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrain.config.schema import TrainConfig


def collect_bounds_errors(config: TrainConfig, errors: list[str]) -> None:
    if config.batch_size <= 0:
        errors.append("batch_size must be > 0. Try: batch_size = 4")
    if config.group_size <= 0:
        errors.append("group_size must be > 0. Try: group_size = 16")
    if config.max_steps <= 0:
        errors.append("max_steps must be > 0. Try: max_steps = 100")
    if config.max_tokens <= 0:
        errors.append("max_tokens must be > 0. Try: max_tokens = 2048")
    if config.lora_rank <= 0:
        errors.append("lora_rank must be > 0. Try: lora_rank = 32")
    if config.lr <= 0:
        errors.append("lr must be > 0. Try: lr = 4e-5")
    if config.temperature < 0:
        errors.append("temperature must be >= 0. Try: temperature = 0.7")
    if config.top_p <= 0 or config.top_p > 1:
        errors.append("top_p must be in (0, 1]. Try: top_p = 0.95")
    if config.surprisal_mask_rho < 0.0 or config.surprisal_mask_rho > 1.0:
        errors.append(
            "surprisal_mask_rho must be in [0.0, 1.0]. Try: surprisal_mask_rho = 0.2"
        )
    if config.clip_eps < 0:
        errors.append(
            "clip_eps must be >= 0. Try: clip_eps = 0.2"
        )
    if config.clip_eps_high < 0:
        errors.append(
            "clip_eps_high must be >= 0. Try: clip_eps_high = 0.28"
        )
    if config.clip_eps_high > 0 and config.clip_eps <= 0:
        errors.append(
            "clip_eps_high > 0 requires clip_eps > 0. "
            "Set clip_eps first. Try: clip_eps = 0.2"
        )
    if config.policy_loss_mode not in ("standard", "kl_cov", "clip_cov"):
        errors.append(
            "policy_loss_mode must be 'standard', 'kl_cov', or 'clip_cov'."
        )
    if config.policy_loss_mode != "standard" and config.backend != "local":
        errors.append(
            f"policy_loss_mode='{config.policy_loss_mode}' currently requires "
            "backend='local' so covariance-aware entropy control is not "
            "silently dropped by another backend."
        )
    if config.kl_cov_percent < 0.0 or config.kl_cov_percent > 100.0:
        errors.append(
            "kl_cov_percent must be in [0.0, 100.0]. Try: kl_cov_percent = 0.2"
        )
    if config.kl_cov_coef < 0.0:
        errors.append("kl_cov_coef must be >= 0. Try: kl_cov_coef = 1.0")
    if config.clip_cov_ratio < 0.0 or config.clip_cov_ratio > 1.0:
        errors.append(
            "clip_cov_ratio must be in [0.0, 1.0]. Try: clip_cov_ratio = 0.0002"
        )
    if config.clip_cov_min >= config.clip_cov_max:
        errors.append(
            "clip_cov_min must be < clip_cov_max. Try: clip_cov_min = 1.0, "
            "clip_cov_max = 5.0"
        )
    if config.adv_clip_max < 0:
        errors.append(
            "adv_clip_max must be >= 0. Try: adv_clip_max = 5.0"
        )
    if config.sft_batch_size < 0:
        errors.append(
            "sft_batch_size must be >= 0. Try: sft_batch_size = 4"
        )
    if config.sft_max_tokens < 0:
        errors.append(
            "sft_max_tokens must be >= 0. Try: sft_max_tokens = 2048"
        )
    if config.sft_lr < 0:
        errors.append(
            "sft_lr must be >= 0. Try: sft_lr = 2e-5"
        )
    if config.sft_data_sha256:
        checksum = config.sft_data_sha256.strip().lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            errors.append(
                "sft_data_sha256 must be a 64-character hexadecimal SHA256 digest."
            )
    if config.sft_data_rows < 0:
        errors.append(
            "sft_data_rows must be >= 0. Use 0 to leave the row count unpinned."
        )
    if config.sft_loss_fn not in ("auto", "importance_sampling", "cross_entropy"):
        errors.append(
            "sft_loss_fn must be 'auto', 'importance_sampling', or 'cross_entropy'."
        )
    if config.sft_batch_order not in (
        "shuffle",
        "length",
        "length_asc",
        "length_desc",
        "length_bucket",
    ):
        errors.append(
            "sft_batch_order must be 'shuffle', 'length', 'length_asc', "
            "'length_desc', or 'length_bucket'."
        )
    if config.sft_length_bucket_size < 0:
        errors.append(
            "sft_length_bucket_size must be >= 0. Try: sft_length_bucket_size = 64"
        )
    if config.trainer == "sft" and not config.sft_data_path:
        errors.append(
            "trainer='sft' requires [training] sft_data_path to point at a JSONL dataset."
        )
    if config.echo_weight < 0.0 or config.echo_weight > 1.0:
        errors.append(
            "echo_weight must be in [0.0, 1.0]. Try: echo_weight = 0.05"
        )
    if config.echo_enabled and config.echo_weight <= 0.0:
        errors.append(
            "echo_enabled requires echo_weight > 0. Try: echo_weight = 0.05"
        )
    if config.echo_loss_fn != "cross_entropy":
        errors.append(
            "echo_loss_fn must be 'cross_entropy' for paper-faithful ECHO."
        )
    if config.echo_max_tokens_per_step <= 0:
        errors.append(
            "echo_max_tokens_per_step must be > 0. Try: echo_max_tokens_per_step = 2048"
        )
    if config.echo_max_token_ratio <= 0.0:
        errors.append(
            "echo_max_token_ratio must be > 0. Try: echo_max_token_ratio = 0.5"
        )
    if config.echo_entropy_floor < 0.0:
        errors.append(
            "echo_entropy_floor must be >= 0. Try: echo_entropy_floor = 0.01"
        )
    if config.echo_min_prompt_overlap < 0.0 or config.echo_min_prompt_overlap > 1.0:
        errors.append(
            "echo_min_prompt_overlap must be in [0.0, 1.0]. "
            "Try: echo_min_prompt_overlap = 0.5"
        )
    if config.generation_log_samples_per_prompt < 0:
        errors.append(
            "generation_log_samples_per_prompt must be >= 0. "
            "Try: generation_log_samples_per_prompt = 2"
        )
    if config.generation_top_surprisal_limit < 0:
        errors.append(
            "generation_top_surprisal_limit must be >= 0. "
            "Try: generation_top_surprisal_limit = 5"
        )
