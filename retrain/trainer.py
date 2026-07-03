"""Main training loop: sample rollouts, score, compute advantages, train.

Backend-agnostic — drives any TrainHelper (local, Unsloth, Tinker, ...).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from heapq import nlargest
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

from transformers import AutoTokenizer

from retrain.advantages import (
    AdvantageResult,
    EntropyStats,
    apply_batch_advantage_normalization,
    compute_algorithm_advantages,
    compute_composable_advantages,
)
from retrain.backends import (
    EntropySamplingHelper,
    TrainHelper,
    collect_runtime_metrics,
    run_sft_train_step,
)
from retrain.backpressure import (
    StepObservation,
)
from retrain.config import TrainConfig
from retrain.data import Example
from retrain.echo import (
    EchoBuildStats,
    EchoLimitStats,
    build_rollout_echo_datum,
    limit_echo_masks,
)
from retrain.flow import (
    TrainingFlow,
    _UNIFORMITY_EPS,
    _condition_label,
    build_flow,
)
from retrain.logging_utils import JsonlLogger
from retrain.process_metrics import process_max_rss_mb
from retrain.registry import PlanningDetector, RewardFunction, get_registry
from retrain.runtime_support import (
    ExamplePromptCache,
    RuntimeCounters,
    TokenTextLookup,
    decode_sequence_groups,
    top_surprisal_entries,
)
from retrain.sepa import SEPAStateDict
from retrain.sft import (
    SftExample,
    SftTokenizedExample,
    build_sft_example_order,
    build_sft_tokenized_batch,
    load_sft_jsonl,
    select_sft_batch_indices,
    tokenize_sft_dataset,
)
from retrain.type_defs import ExampleInfoLike, PromptLike
from retrain.verifiers_bridge import (
    encode_prompt_for_sampling,
    is_multiturn_environment,
    load_examples_from_environment,
    load_verifiers_environment,
    prompt_preview,
    run_multiturn_group,
    score_singleturn_group,
)


_TRAINER_STATE_FILE = "trainer_state.json"
_CORRECT_THRESHOLD = 0.5
_PROMPT_PAD_EPS = 1e-9


def _accumulate_metric_totals(
    totals: dict[str, float],
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


class WandbRunLike(Protocol):
    def log(
        self,
        data: Mapping[str, object],
        *,
        step: int | None = None,
    ) -> object: ...

    def finish(self) -> object: ...


class WandbModuleLike(Protocol):
    def init(
        self,
        *,
        project: str,
        name: str,
        config: Mapping[str, str | int | float],
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> WandbRunLike: ...


def _generation_log_indices(
    sample_count: int,
    *,
    samples_per_prompt: int,
    rewards: list[float] | None = None,
) -> list[int]:
    """Select the generation indices to log for one prompt.

    When a cap is active, prefer the highest-reward completions because they
    are the most representative of the learning signal. Ties are broken by the
    earlier sample index to keep the result deterministic.
    """
    if sample_count <= 0:
        return []
    if samples_per_prompt <= 0 or samples_per_prompt >= sample_count:
        return list(range(sample_count))
    if rewards is None or len(rewards) != sample_count:
        return list(range(samples_per_prompt))
    if samples_per_prompt == 1:
        return [
            max(
                range(sample_count),
                key=lambda idx: (rewards[idx], -idx),
            )
        ]

    ranked = nlargest(
        samples_per_prompt,
        range(sample_count),
        key=lambda idx: (rewards[idx], -idx),
    )
    return ranked


def _top_surprisal_payload(
    logprobs: list[float],
    token_ids: list[int],
    token_lookup: TokenTextLookup,
    *,
    limit: int,
) -> list[dict[str, int | float | str]]:
    """Build optional top-surprisal diagnostics for one sampled completion."""
    if limit <= 0 or not logprobs or not token_ids:
        return []
    return top_surprisal_entries(
        logprobs,
        token_ids,
        token_lookup,
        limit=limit,
    )


def _apply_advantage_cap(
    all_advantages: list[list[float]],
    cap: float,
) -> tuple[list[list[float]], float, float]:
    """Cap per-token advantages to [-cap, +cap].

    This is NOT ratio clipping (PPO). It bounds the advantage magnitude sent
    to the backend, limiting how hard any single token can push the gradient.
    The mechanism is upstream of the loss function — it modifies the signal,
    not the optimization dynamics.

    Returns:
        (capped_advantages, cap_fraction, mean_cap_magnitude)
        cap_fraction: fraction of non-zero tokens that were capped.
        mean_cap_magnitude: mean absolute value of tokens that were capped
            (before capping). 0.0 if nothing was capped.
    """
    total = 0
    capped_count = 0
    capped_magnitude_sum = 0.0
    result: list[list[float]] = []
    for seq in all_advantages:
        capped_seq: list[float] = []
        for a in seq:
            if a == 0.0:
                capped_seq.append(a)
                continue
            total += 1
            if a > cap:
                capped_magnitude_sum += abs(a)
                capped_count += 1
                capped_seq.append(cap)
            elif a < -cap:
                capped_magnitude_sum += abs(a)
                capped_count += 1
                capped_seq.append(-cap)
            else:
                capped_seq.append(a)
        result.append(capped_seq)
    frac = capped_count / max(total, 1)
    mean_mag = capped_magnitude_sum / max(capped_count, 1)
    return result, frac, mean_mag


def _has_nonzero_advantage(rows: list[list[float]]) -> bool:
    return any(abs(value) > 0.0 for row in rows for value in row)


def _run_rl_echo_train_step(
    helper: object,
    all_tokens: list[list[int]],
    all_logprobs: list[list[float]],
    all_advantages: list[list[float]],
    echo_advantages: list[list[float]],
    echo_full_observation_counts: list[int],
    *,
    echo_loss_fn: str,
    lr: float,
    weight_decay: float,
) -> tuple[float, float, bool]:
    """Run one RL update, optionally with ECHO in the same optimizer step.

    ECHO is independent of the chosen RL algorithm: algorithms produce the
    sampled-token advantages above, while ECHO adds a same-rollout
    environment-token mask. Paper-faithful RL+ECHO requires a backend
    ``train_step_with_echo_masks`` implementation that computes both losses
    from the same actor forward/backward pass over those rollout rows.
    """

    if not all_tokens:
        return 0.0, 0.0, False

    if echo_advantages:
        train_step_with_echo_masks = getattr(helper, "train_step_with_echo_masks", None)
        if callable(train_step_with_echo_masks):
            rl_loss, echo_loss = train_step_with_echo_masks(
                all_tokens,
                all_logprobs,
                all_advantages,
                echo_advantages,
                echo_full_observation_counts,
                echo_loss_fn,
                lr,
                weight_decay,
            )
            return float(rl_loss), float(echo_loss), True
        raise RuntimeError(
            "ECHO requires a backend train_step_with_echo_masks implementation "
            "so RL and environment-token losses are computed from the same "
            "rollout rows in one actor forward/backward pass."
        )

    train_step = getattr(helper, "train_step")
    rl_loss = float(
        train_step(
            all_tokens,
            all_logprobs,
            all_advantages,
            lr,
            weight_decay,
        )
    )
    return rl_loss, 0.0, False


def _echo_allowed_tokens(
    *,
    rl_completion_tokens: int,
    max_tokens_per_step: int,
    max_token_ratio: float,
) -> int:
    """Compute the active ECHO cap for this step."""

    ratio_cap = int(rl_completion_tokens * max_token_ratio)
    return max(0, min(max_tokens_per_step, ratio_cap))


@dataclass(frozen=True)
class _EchoStepPlan:
    limit: EchoLimitStats
    allowed_tokens: int
    reference_completion_tokens: int
    skipped_entropy_floor: bool
    rl_completion_surprisal_mean: float
    echo_completion_surprisal_mean: float

    @property
    def token_ratio(self) -> float:
        if self.reference_completion_tokens <= 0:
            return 0.0
        return self.limit.kept_tokens / self.reference_completion_tokens


def _add_echo_build_stats(
    left: EchoBuildStats,
    right: EchoBuildStats,
) -> EchoBuildStats:
    return EchoBuildStats(
        candidate_datums=left.candidate_datums + right.candidate_datums,
        candidate_tokens=left.candidate_tokens + right.candidate_tokens,
        observation_mask_datums=(
            left.observation_mask_datums + right.observation_mask_datums
        ),
        skipped_first_turns=left.skipped_first_turns + right.skipped_first_turns,
        skipped_no_suffix=left.skipped_no_suffix + right.skipped_no_suffix,
        skipped_low_overlap=left.skipped_low_overlap + right.skipped_low_overlap,
        skipped_bad_observation_mask=(
            left.skipped_bad_observation_mask + right.skipped_bad_observation_mask
        ),
    )


class TrainerState(TypedDict):
    """Serialized trainer state stored in checkpoint directories."""

    step: int
    example_idx: int
    total_correct: int
    total_completions: int
    current_batch_size: int
    current_group_size: int
    checkpoint_name: str
    checkpoint_path: NotRequired[str]
    sepa: SEPAStateDict
    tl_grpo_ema: NotRequired[float]
    delight_eta_ema: NotRequired[float]


class RewardTieStats(TypedDict):
    """Approximate within-group reward tie summary."""

    eligible: bool
    has_tie: bool
    is_uniform: bool
    unique_count: int
    tied_pairs: int
    total_pairs: int


def _require_int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Trainer state field '{key}' must be an integer.")
    return value


def _optional_str_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"Trainer state field '{key}' must be a string.")
    return value


def _optional_sepa_state(payload: Mapping[str, object]) -> SEPAStateDict:
    value = payload.get("sepa", {})
    if not isinstance(value, dict):
        raise ValueError("Trainer state field 'sepa' must be an object.")
    return cast(SEPAStateDict, value)


def _optional_float_field(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Trainer state field '{key}' must be a number.")
    return float(value)


def _uses_adaptive_delight(params: Mapping[str, object] | None) -> bool:
    if params is None:
        return False
    raw_mode = params.get("delight_eta_mode")
    if isinstance(raw_mode, str):
        return raw_mode.strip().lower() == "adaptive"
    return params.get("delight_eta_adaptive") is True


def _prepare_transform_params_for_step(
    params: Mapping[str, object] | None,
    *,
    delight_eta_prev: float | None,
) -> dict[str, object]:
    prepared = dict(params) if params is not None else {}
    if delight_eta_prev is not None and _uses_adaptive_delight(prepared):
        prepared["delight_eta_prev"] = delight_eta_prev
    return prepared


def _prepare_algorithm_params_for_step(
    params: Mapping[str, object],
    *,
    delight_eta_prev: float | None,
) -> dict[str, object]:
    prepared = dict(params)
    raw_transform_params = prepared.get("transform_params")
    transform_params = (
        dict(raw_transform_params)
        if isinstance(raw_transform_params, Mapping)
        else {}
    )
    prepared["transform_params"] = _prepare_transform_params_for_step(
        transform_params,
        delight_eta_prev=delight_eta_prev,
    )
    return prepared


def _compute_group_advantages(
    config: TrainConfig,
    rewards_G: list[float],
    logprobs_G: list[list[float]],
    planning_masks_G: list[list[int]],
    *,
    step: int,
    sepa_lambda: float,
    algorithm_params: Mapping[str, object],
    transform_params: Mapping[str, object],
    precomputed_entropies_G: list[list[float]] | None = None,
) -> AdvantageResult:
    """Dispatch one group to the full-algorithm or composable advantage path."""
    if config.algorithm_mode:
        return compute_algorithm_advantages(
            rewards_G,
            logprobs_G,
            planning_masks_G,
            algorithm_mode=config.algorithm_mode,
            params=algorithm_params,
            gtpo_beta=config.gtpo_beta,
            hicra_alpha=config.hicra_alpha,
            sepa_lambda=sepa_lambda,
            step=step,
            token_distributions_G=None,
            precomputed_entropies_G=precomputed_entropies_G,
        )
    return compute_composable_advantages(
        rewards_G,
        logprobs_G,
        planning_masks_G,
        advantage_mode=config.advantage_mode,
        transform_mode=config.transform_mode,
        gtpo_beta=config.gtpo_beta,
        hicra_alpha=config.hicra_alpha,
        sepa_lambda=sepa_lambda,
        advantage_params=config.effective_advantage_params,
        transform_params=transform_params,
        step=step,
        post_process_params=config.post_process_params,
        token_distributions_G=None,
        precomputed_entropies_G=precomputed_entropies_G,
    )


def _print_config_summary(config: TrainConfig) -> None:
    """Print a bordered summary of key config values at train start."""
    lines = [
        f"  model         : {config.model}",
        f"  backend       : {config.backend}",
        f"  algorithm     : {_condition_label(config)}",
        f"  batch_size    : {config.batch_size}",
        f"  group_size    : {config.group_size}",
        f"  max_steps     : {config.max_steps}",
        f"  lr            : {config.lr}" + (f"  (sft_lr: {config.sft_lr})" if config.sft_lr > 0 else ""),
        f"  lora_rank     : {config.lora_rank}",
        f"  max_tokens    : {config.max_tokens}",
        f"  temperature   : {config.temperature}",
        f"  seed          : {config.seed}",
        f"  adapter_path  : {config.adapter_path}",
    ]
    if config.wandb_project:
        lines.append(f"  wandb         : {config.wandb_project}")
    if config.resume_from:
        lines.append(f"  resume_from   : {config.resume_from}")
    if config.echo_enabled:
        lines.append(
            "  echo          : "
            f"on weight={config.echo_weight} "
            f"cap={config.echo_max_tokens_per_step} "
            f"ratio={config.echo_max_token_ratio}"
        )
    width = max(len(l) for l in lines) + 2
    sep = "-" * width
    print(sep)
    for l in lines:
        print(l)
    print(sep)


def _print_backend_capability_summary(
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


def _format_loss_for_display(loss_value: float, reports_sync_loss: bool) -> str:
    """Format loss consistently, including async placeholder semantics."""
    formatted = f"{loss_value:.4f}"
    if reports_sync_loss:
        return formatted
    return f"{formatted} (placeholder)"


def _print_flow_warnings(trace_result: object) -> None:
    """Print non-fatal training-flow warnings before setup starts."""
    for issue in getattr(trace_result, "issues", []):
        if getattr(issue, "severity", "") != "warning":
            continue
        category = getattr(issue, "category", "config")
        message = getattr(issue, "message", str(issue))
        print(f"Training flow warning [{category}]: {message}")


def _assert_uniform_completion_advantages_for_non_preserving_backend(
    all_logprobs: list[list[float]],
    all_advantages: list[list[float]],
    *,
    backend_name: str,
    eps: float = _UNIFORMITY_EPS,
) -> None:
    """Runtime guard: non-preserving backends must receive scalar completion advantages."""
    for sample_idx, (logprobs, advantages) in enumerate(zip(all_logprobs, all_advantages)):
        n = min(len(logprobs), len(advantages))
        if n <= 1:
            continue
        prompt_len = 0
        for lp, adv in zip(logprobs[:n], advantages[:n]):
            if abs(lp) <= _PROMPT_PAD_EPS and abs(adv) <= _PROMPT_PAD_EPS:
                prompt_len += 1
            else:
                break
        completion = advantages[prompt_len:n]
        if len(completion) <= 1:
            continue
        lo = min(completion)
        hi = max(completion)
        if (hi - lo) > eps:
            raise RuntimeError(
                f"backend='{backend_name}' does not preserve token-level advantages, "
                f"but sample {sample_idx} contains non-uniform completion advantages."
            )


def _summarize_reward_ties(
    rewards: list[float],
    *,
    eps: float = _UNIFORMITY_EPS,
) -> RewardTieStats:
    """Summarize approximate reward ties inside one prompt group."""
    n = len(rewards)
    if n < 2:
        return {
            "eligible": False,
            "has_tie": False,
            "is_uniform": False,
            "unique_count": n,
            "tied_pairs": 0,
            "total_pairs": 0,
        }

    bucket_sizes: list[int] = []
    bucket_anchor = 0.0
    for reward in sorted(rewards):
        if not bucket_sizes:
            bucket_sizes.append(1)
            bucket_anchor = reward
            continue
        if abs(reward - bucket_anchor) <= eps:
            bucket_sizes[-1] += 1
        else:
            bucket_sizes.append(1)
            bucket_anchor = reward

    unique_count = len(bucket_sizes)
    total_pairs = n * (n - 1) // 2
    tied_pairs = sum(size * (size - 1) // 2 for size in bucket_sizes)
    return {
        "eligible": True,
        "has_tie": unique_count < n,
        "is_uniform": unique_count == 1,
        "unique_count": unique_count,
        "tied_pairs": tied_pairs,
        "total_pairs": total_pairs,
    }


@dataclass
class _RewardTieAccumulator:
    """Per-step aggregate of group-level reward-tie diagnostics."""

    eligible_groups: int = 0
    tie_groups: int = 0
    uniform_groups: int = 0
    tied_pairs: int = 0
    total_pairs: int = 0
    unique_fraction_sum: float = 0.0

    def observe(self, rewards: list[float]) -> RewardTieStats:
        stats = _summarize_reward_ties(rewards)
        if stats["eligible"]:
            self.eligible_groups += 1
            self.tie_groups += int(stats["has_tie"])
            self.uniform_groups += int(stats["is_uniform"])
            self.tied_pairs += stats["tied_pairs"]
            self.total_pairs += stats["total_pairs"]
            self.unique_fraction_sum += stats["unique_count"] / len(rewards)
        return stats

    @property
    def tie_group_rate(self) -> float:
        return self.tie_groups / self.eligible_groups if self.eligible_groups else 0.0

    @property
    def uniform_group_rate(self) -> float:
        return (
            self.uniform_groups / self.eligible_groups if self.eligible_groups else 0.0
        )

    @property
    def tie_pair_rate(self) -> float:
        return self.tied_pairs / self.total_pairs if self.total_pairs else 0.0

    @property
    def unique_fraction_mean(self) -> float:
        return (
            self.unique_fraction_sum / self.eligible_groups
            if self.eligible_groups
            else 0.0
        )


def _print_group_summary(rewards: list[float], answer: str) -> None:
    correct = sum(1 for r in rewards if r > _CORRECT_THRESHOLD)
    print(f"  group: {correct}/{len(rewards)} correct | answer={answer[:40]}")


def _keep_uniform_group(
    rewards: list[float],
    *,
    batch_advantage_norm: bool,
    keep_for_echo: bool,
) -> bool:
    """Print the disposition of an all-same-reward group; False → skip it.

    With batch_advantage_norm, uniform groups are kept — cross-group
    reward differences still provide signal after batch normalization.
    ECHO keeps them too: its datums come from observation tokens, not
    reward contrast.
    """
    if batch_advantage_norm:
        print(f"    -> uniform (reward={rewards[0]:.3f}, kept for batch norm)")
        return True
    if keep_for_echo:
        print(f"    -> uniform (reward={rewards[0]:.3f}, kept for ECHO)")
        return True
    if rewards[0] > _CORRECT_THRESHOLD:
        print("    -> skipped (all correct)")
    else:
        print(f"    -> skipped (all same, reward={rewards[0]:.3f})")
    return False


@dataclass
class _SftWarmupData:
    """SFT warmup dataset, loaded and tokenized once before the step loop."""

    examples: list[SftExample]
    tokenized: list[SftTokenizedExample]
    order: list[int]


def _load_sft_warmup_data(
    config: TrainConfig,
    tokenizer: object,
) -> _SftWarmupData | None:
    """Load the SFT warmup dataset, or None when unconfigured or missing."""
    if config.sft_warmup_steps <= 0 or not config.sft_data_path:
        return None
    sft_path = Path(config.sft_data_path)
    if not sft_path.exists():
        print(f"WARNING: SFT data path {sft_path} not found, skipping warmup")
        return None
    examples = load_sft_jsonl(sft_path)
    token_limit = (
        config.sft_max_tokens
        if config.sft_max_tokens > 0
        else config.max_tokens + 512
    )
    tokenized = tokenize_sft_dataset(tokenizer, examples, max_tokens=token_limit)
    order = build_sft_example_order(
        len(tokenized),
        config.seed,
        lengths=[example.total_tokens for example in tokenized],
        batch_order=config.sft_batch_order,
        length_bucket_size=config.sft_length_bucket_size,
    )
    print(
        f"Loaded {len(examples)} SFT warmup examples from {sft_path} "
        f"(order={config.sft_batch_order})"
    )
    return _SftWarmupData(examples=examples, tokenized=tokenized, order=order)


def _run_sft_warmup_step(
    helper: TrainHelper,
    config: TrainConfig,
    sft_data: _SftWarmupData,
    step: int,
    *,
    metrics_logger: JsonlLogger,
    steps_logger: JsonlLogger,
    wandb_run: WandbRunLike | None,
) -> None:
    """Run one supervised warmup step from oracle demonstrations."""
    step_start = time.perf_counter()
    helper.checkpoint(f"step_{step}")

    batch_size = (
        config.sft_batch_size
        if config.sft_batch_size > 0
        else min(16, len(sft_data.examples))
    )
    batch_indices = select_sft_batch_indices(
        sft_data.order,
        batch_size=batch_size,
        step=step,
    )
    batch = [sft_data.tokenized[idx] for idx in batch_indices]
    tokenized = build_sft_tokenized_batch(batch)

    # Use sft_lr if set, otherwise fall back to main lr.
    effective_lr = config.sft_lr if config.sft_lr > 0 else config.lr
    print(
        f"Step {step} [SFT warmup]: {len(batch)} examples "
        f"(lr={effective_lr:.1e})...",
        flush=True,
    )
    loss = run_sft_train_step(
        helper,
        tokenized.tokens,
        tokenized.advantages,
        effective_lr,
        config.weight_decay,
    )
    elapsed = time.perf_counter() - step_start
    # Tinker's importance_sampling loss with logprobs=0 produces negative
    # values that get MORE negative as the model learns
    # (-exp(logprob) * advantage). Report both the raw IS loss and the
    # flipped "sft_signal" (= -loss, higher = better) so the learning
    # curve is intuitive.
    sft_signal = -loss
    print(
        f"Step {step} [SFT warmup] | is_loss={loss:.4f} | "
        f"sft_signal={sft_signal:.4f} | "
        f"datums={len(batch)} | time={elapsed:.1f}s",
        flush=True,
    )

    metrics: dict[str, int | float | str] = {
        "step": step,
        "loss": loss,
        "sft_signal": sft_signal,
        "phase": "sft",
        "datums": len(batch),
        "tokens": tokenized.total_tokens,
        "supervised_tokens": tokenized.supervised_tokens,
        "sft_batch_order": config.sft_batch_order,
        "sft_length_bucket_size": int(config.sft_length_bucket_size),
        "sft_unique_examples_seen": min(
            len(sft_data.examples),
            (step + 1) * batch_size,
        ),
        "sft_dataset_coverage": min(
            1.0,
            ((step + 1) * batch_size) / max(len(sft_data.examples), 1),
        ),
        "time_s": round(elapsed, 2),
        "advantage_mode": config.advantage_mode,
        "lr": effective_lr,
    }
    rss_mb = process_max_rss_mb()
    if rss_mb is not None:
        metrics["process_max_rss_mb"] = round(rss_mb, 3)
    metrics_logger.log(metrics)
    steps_logger.log(metrics)

    if wandb_run is not None:
        wandb_run.log(
            {
                "train/loss": loss,
                "train/sft_signal": sft_signal,
                "train/sft_warmup": 1,
                "train/step": step,
                "train/lr": effective_lr,
            },
            step=step,
        )

    if config.save_every > 0 and (step + 1) % config.save_every == 0:
        ckpt_name = f"checkpoint_step_{step + 1}"
        helper.save_adapter(config.adapter_path, ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")


def _init_wandb(config: TrainConfig) -> WandbRunLike | None:
    """Start a wandb run mirroring the config, or None when unconfigured."""
    if not config.wandb_project:
        return None
    import wandb as wandb_module

    wandb = cast(WandbModuleLike, wandb_module)
    condition_label = _condition_label(config)
    run_name = config.wandb_run_name or Path(config.log_dir).name or condition_label
    wandb_tags = (
        [t.strip() for t in config.wandb_tags.split(",") if t.strip()]
        if config.wandb_tags
        else None
    )
    wandb_config: dict[str, str | int | float] = {
        "algorithm_mode": config.algorithm_mode,
        "advantage_mode": config.advantage_mode,
        "transform_mode": config.transform_mode,
        "uncertainty_kind": config.uncertainty_kind,
        "condition": condition_label,
        "model": config.model,
        "lora_rank": config.lora_rank,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "group_size": config.group_size,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "gtpo_beta": config.gtpo_beta,
        "hicra_alpha": config.hicra_alpha,
        "sepa_steps": config.sepa_steps,
        "sepa_delay_steps": config.sepa_delay_steps,
        "sepa_correct_rate_gate": config.sepa_correct_rate_gate,
        "max_steps": config.max_steps,
        "backend": config.backend,
        "seed": config.seed,
        "batch_advantage_norm": int(config.batch_advantage_norm),
        "clip_eps": config.clip_eps,
        "clip_eps_high": config.clip_eps_high,
        "policy_loss_mode": config.policy_loss_mode,
        "kl_cov_percent": config.kl_cov_percent,
        "kl_cov_coef": config.kl_cov_coef,
        "clip_cov_ratio": config.clip_cov_ratio,
        "clip_cov_min": config.clip_cov_min,
        "clip_cov_max": config.clip_cov_max,
        "adv_clip_max": config.adv_clip_max,
        "sft_warmup_steps": config.sft_warmup_steps,
        "tl_grpo": int(config.tl_grpo),
        "echo_enabled": int(config.echo_enabled),
        "echo_weight": config.echo_weight,
        "echo_max_tokens_per_step": config.echo_max_tokens_per_step,
        "echo_max_token_ratio": config.echo_max_token_ratio,
        "echo_entropy_floor": config.echo_entropy_floor,
    }
    run = wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=wandb_config,
        entity=config.wandb_entity or None,
        group=config.wandb_group or None,
        tags=wandb_tags,
    )
    print(f"Wandb initialized: {config.wandb_project}/{run_name}")
    return run


def _save_trainer_state(
    path: Path,
    *,
    step: int,
    example_idx: int,
    total_correct: int,
    total_completions: int,
    current_batch_size: int,
    current_group_size: int,
    checkpoint_name: str,
    checkpoint_path: str | None = None,
    sepa_state: SEPAStateDict,
    tl_grpo_ema: float | None = None,
    delight_eta_ema: float | None = None,
) -> None:
    """Write trainer-side state to JSON for checkpoint resume."""
    path.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "step": step,
        "example_idx": example_idx,
        "total_correct": total_correct,
        "total_completions": total_completions,
        "current_batch_size": current_batch_size,
        "current_group_size": current_group_size,
        "checkpoint_name": checkpoint_name,
        "sepa": sepa_state,
    }
    if checkpoint_path:
        state["checkpoint_path"] = checkpoint_path
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
    tmp = path / f"{_TRAINER_STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path / _TRAINER_STATE_FILE)
    if checkpoint_path:
        latest_tmp = path / "latest_sampler_path.txt.tmp"
        latest_tmp.write_text(f"{checkpoint_path}\n")
        latest_tmp.replace(path / "latest_sampler_path.txt")


def _load_trainer_state(resume_dir: str) -> TrainerState:
    """Load trainer state from a checkpoint directory."""
    p = Path(resume_dir)
    state_file = p / _TRAINER_STATE_FILE
    if not state_file.is_file():
        raise FileNotFoundError(
            f"No {_TRAINER_STATE_FILE} found in {resume_dir}. "
            f"Cannot resume without trainer state."
        )
    payload = json.loads(state_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid trainer state file {state_file}: expected JSON object."
        )
    payload_map = cast(Mapping[str, object], payload)
    state: TrainerState = {
        "step": _require_int_field(payload_map, "step"),
        "example_idx": _require_int_field(payload_map, "example_idx"),
        "total_correct": _require_int_field(payload_map, "total_correct"),
        "total_completions": _require_int_field(payload_map, "total_completions"),
        "current_batch_size": _require_int_field(payload_map, "current_batch_size"),
        "current_group_size": _require_int_field(payload_map, "current_group_size"),
        "checkpoint_name": _optional_str_field(payload_map, "checkpoint_name"),
        "sepa": _optional_sepa_state(payload_map),
    }
    checkpoint_path = _optional_str_field(payload_map, "checkpoint_path")
    if checkpoint_path:
        state["checkpoint_path"] = checkpoint_path
    else:
        latest_sampler_path = p / "latest_sampler_path.txt"
        if latest_sampler_path.is_file():
            fallback_path = latest_sampler_path.read_text().strip()
            if fallback_path:
                state["checkpoint_path"] = fallback_path
    tl_grpo_ema = _optional_float_field(payload_map, "tl_grpo_ema")
    if tl_grpo_ema is not None:
        state["tl_grpo_ema"] = tl_grpo_ema
    delight_eta_ema = _optional_float_field(payload_map, "delight_eta_ema")
    if delight_eta_ema is not None:
        state["delight_eta_ema"] = delight_eta_ema
    return state


@dataclass
class _PromptBatch:
    """Parallel per-prompt arrays for one training batch."""

    objs: list[PromptLike] = field(default_factory=list)
    previews: list[str] = field(default_factory=list)
    ids: list[list[int]] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    infos: list[ExampleInfoLike] = field(default_factory=list)


def _select_prompt_batch(
    examples: list[Example],
    prompt_cache: ExamplePromptCache,
    start_index: int,
    batch_size: int,
) -> tuple[_PromptBatch, int]:
    """Take the next batch_size examples round-robin; returns new cursor."""
    prompts = _PromptBatch()
    example_idx = start_index
    for _ in range(batch_size):
        ex_idx = example_idx % len(examples)
        example_idx += 1
        ex = examples[ex_idx]
        prompts.objs.append(ex.prompt)
        prompts.previews.append(prompt_cache.preview(ex_idx))
        prompts.ids.append(list(prompt_cache.prompt_ids(ex_idx)))
        prompts.answers.append(ex.reference)
        prompts.tasks.append(ex.task)
        prompts.infos.append(ex.info)
    return prompts, example_idx


@dataclass
class _RolloutAccumulator:
    """Everything one RL step's rollout phase produces for training/logging."""

    rewards: list[float] = field(default_factory=list)
    correct: int = 0
    max_token_hits: int = 0
    total_completions: int = 0
    ties: _RewardTieAccumulator = field(default_factory=_RewardTieAccumulator)
    surprisal_stats: list[EntropyStats] = field(default_factory=list)
    adv_results: list[AdvantageResult] = field(default_factory=list)
    logprobs_sepa: list[list[float]] = field(default_factory=list)
    planning_masks_sepa: list[list[int]] = field(default_factory=list)
    datum_tokens: list[list[int]] = field(default_factory=list)
    datum_logprobs: list[list[float]] = field(default_factory=list)
    datum_advantages: list[list[float]] = field(default_factory=list)
    datum_echo_advantages: list[list[float]] = field(default_factory=list)
    datum_echo_full_observation_counts: list[int] = field(default_factory=list)
    echo_build: EchoBuildStats = field(default_factory=EchoBuildStats)
    rl_completion_token_count: int = 0
    rl_completion_surprisal_sum: float = 0.0
    sampled_completion_token_count: int = 0
    sampled_completion_surprisal_sum: float = 0.0
    behavior_turns: int = 0
    behavior_invalid: int = 0
    behavior_actions: dict[str, int] = field(default_factory=dict)
    behavior_resp_lens: list[int] = field(default_factory=list)
    rollout_timing_metrics: dict[str, float] = field(default_factory=dict)
    sample_time_s: float = 0.0
    tl_grpo_ema: float | None = None


def _prepare_echo_step_plan(
    config: TrainConfig,
    acc: _RolloutAccumulator,
) -> _EchoStepPlan:
    """Apply ECHO token limits and return the values logged for this step."""
    rl_completion_surprisal_mean = (
        acc.rl_completion_surprisal_sum / acc.rl_completion_token_count
        if acc.rl_completion_token_count > 0
        else 0.0
    )
    echo_completion_surprisal_mean = (
        acc.sampled_completion_surprisal_sum / acc.sampled_completion_token_count
        if acc.sampled_completion_token_count > 0
        else rl_completion_surprisal_mean
    )
    if not config.echo_enabled:
        return _EchoStepPlan(
            limit=EchoLimitStats(),
            allowed_tokens=0,
            reference_completion_tokens=0,
            skipped_entropy_floor=False,
            rl_completion_surprisal_mean=rl_completion_surprisal_mean,
            echo_completion_surprisal_mean=echo_completion_surprisal_mean,
        )

    reference_completion_tokens = acc.sampled_completion_token_count
    allowed_tokens = _echo_allowed_tokens(
        rl_completion_tokens=reference_completion_tokens,
        max_tokens_per_step=config.echo_max_tokens_per_step,
        max_token_ratio=config.echo_max_token_ratio,
    )
    if echo_completion_surprisal_mean < config.echo_entropy_floor:
        acc.datum_echo_advantages = [
            [0.0] * len(row) for row in acc.datum_echo_advantages
        ]
        return _EchoStepPlan(
            limit=EchoLimitStats(
                kept_datums=0,
                kept_tokens=0,
                truncated_tokens=acc.echo_build.candidate_tokens,
            ),
            allowed_tokens=allowed_tokens,
            reference_completion_tokens=reference_completion_tokens,
            skipped_entropy_floor=True,
            rl_completion_surprisal_mean=rl_completion_surprisal_mean,
            echo_completion_surprisal_mean=echo_completion_surprisal_mean,
        )

    acc.datum_echo_advantages, limit = limit_echo_masks(
        acc.datum_echo_advantages,
        max_positive_tokens=allowed_tokens,
    )
    return _EchoStepPlan(
        limit=limit,
        allowed_tokens=allowed_tokens,
        reference_completion_tokens=reference_completion_tokens,
        skipped_entropy_floor=False,
        rl_completion_surprisal_mean=rl_completion_surprisal_mean,
        echo_completion_surprisal_mean=echo_completion_surprisal_mean,
    )


def _run_multiturn_rollouts(
    config: TrainConfig,
    helper: TrainHelper,
    tokenizer: object,
    verifiers_env: object,
    prompts: _PromptBatch,
    acc: _RolloutAccumulator,
    *,
    step: int,
    group_size: int,
    sepa_lambda: float,
    algorithm_params: Mapping[str, object],
    transform_params: Mapping[str, object],
    needs_planning: bool,
    detector: PlanningDetector | None,
    token_lookup: TokenTextLookup,
    generations_logger: JsonlLogger,
) -> None:
    """Roll out, score, and datum-ize multiturn environment groups."""
    sample_start = time.perf_counter()
    for f_idx in range(len(prompts.ids)):
        prompt_obj = prompts.objs[f_idx]
        answer = prompts.answers[f_idx]
        task = prompts.tasks[f_idx]
        info = prompts.infos[f_idx]

        (
            rewards_G,
            turns_G,
            completion_texts_G,
            turn_rewards_G,
            turn_advantages_G,
            turn_logs_G,
            branch_rewards_G,
            rollout_timing,
        ) = run_multiturn_group(
            verifiers_env,
            helper=helper,
            tokenizer=tokenizer,
            model_name=config.model,
            prompt=prompt_obj,
            answer=answer,
            task=task,
            info=info,
            num_rollouts=group_size,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            max_turns_override=config.environment_max_turns,
            tl_grpo=config.tl_grpo,
            tl_grpo_branch_mode=config.tl_grpo_branch_mode,
            tl_grpo_branch_size=config.tl_grpo_branch_size,
            tl_grpo_lookahead_steps=config.tl_grpo_lookahead_steps,
            tl_grpo_outcome_baseline=acc.tl_grpo_ema,
            rollout_env_workers=config.environment_rollout_env_workers,
            rollout_buffer_size=config.environment_rollout_buffer_size,
        )
        _accumulate_metric_totals(
            acc.rollout_timing_metrics,
            rollout_timing.as_metrics(),
        )

        logprobs_G: list[list[float]] = []
        planning_masks_G: list[list[int]] = []
        turns_logprobs_G: list[list[list[float]]] = []
        turns_token_ids_G: list[list[list[int]]] = []
        turns_prompt_ids_G: list[list[list[int]]] = []

        for turns in turns_G:
            seq_logprobs: list[float] = []
            seq_token_ids: list[int] = []
            turn_logprobs: list[list[float]] = []
            turn_token_ids: list[list[int]] = []
            turn_prompt_ids: list[list[int]] = []
            for turn in turns:
                turn_prompt_ids.append(list(turn.prompt_ids))
                turn_token_ids.append(list(turn.completion_ids))
                turn_logprobs.append(list(turn.completion_logprobs))
                seq_logprobs.extend(turn.completion_logprobs)
                seq_token_ids.extend(turn.completion_ids)
            logprobs_G.append(seq_logprobs)
            turns_logprobs_G.append(turn_logprobs)
            turns_token_ids_G.append(turn_token_ids)
            turns_prompt_ids_G.append(turn_prompt_ids)
            if needs_planning:
                assert detector is not None
                planning_masks_G.append(
                    detector.detect(token_lookup.get_many(seq_token_ids))
                )
            else:
                planning_masks_G.append([0] * len(seq_logprobs))

            acc.total_completions += 1
            acc.sampled_completion_token_count += len(seq_token_ids)
            acc.sampled_completion_surprisal_sum += sum(
                -lp for lp in seq_logprobs
            )
            if seq_token_ids and len(seq_token_ids) >= config.max_tokens:
                acc.max_token_hits += 1

        acc.logprobs_sepa.extend(logprobs_G)
        acc.planning_masks_sepa.extend(planning_masks_G)

        for r in rewards_G:
            acc.rewards.append(r)
            if r > _CORRECT_THRESHOLD:
                acc.correct += 1
            if acc.tl_grpo_ema is not None:
                acc.tl_grpo_ema = (
                    config.tl_grpo_ema_decay * acc.tl_grpo_ema
                    + (1 - config.tl_grpo_ema_decay) * r
                )

        _print_group_summary(rewards_G, answer)
        reward_tie_stats = acc.ties.observe(rewards_G)
        if reward_tie_stats["is_uniform"] and not config.tl_grpo:
            if not _keep_uniform_group(
                rewards_G,
                batch_advantage_norm=config.batch_advantage_norm,
                keep_for_echo=config.echo_enabled,
            ):
                continue

        adv_result = _compute_group_advantages(
            config,
            rewards_G,
            logprobs_G,
            planning_masks_G,
            step=step,
            sepa_lambda=sepa_lambda,
            algorithm_params=algorithm_params,
            transform_params=transform_params,
        )
        all_token_advs_G = adv_result.token_advs
        if adv_result.has_stats:
            acc.surprisal_stats.append(adv_result.stats)
        if adv_result.extra_metrics:
            acc.adv_results.append(adv_result)

        for s_idx in range(len(rewards_G)):
            turn_token_ids = turns_token_ids_G[s_idx]
            turn_logprobs = turns_logprobs_G[s_idx]
            token_advs = all_token_advs_G[s_idx]

            # MT-GRPO: when per-turn advantages are provided by the
            # environment rubric (e.g. soma's _compute_turn_advantages),
            # use them directly as the advantage for each turn's tokens
            # instead of the uniform episode-level expansion.
            # All-or-nothing per rollout: if turn_advantages covers all
            # turns, use it; otherwise fall back entirely to episode-level
            # to avoid offset drift between the two modes.
            s_turn_advs: list[float] | None = None
            if s_idx < len(turn_advantages_G) and turn_advantages_G[s_idx]:
                candidate = turn_advantages_G[s_idx]
                if len(candidate) >= len(turn_token_ids):
                    s_turn_advs = candidate

            offset = 0
            seq_advs_by_turn: list[list[float]] = []
            for t_idx in range(len(turn_token_ids)):
                seq_tokens = turn_token_ids[t_idx]

                if s_turn_advs is not None:
                    # Per-turn advantage: broadcast the turn's advantage
                    # to all tokens in this turn's completion.
                    seq_advs = [s_turn_advs[t_idx]] * len(seq_tokens)
                else:
                    # Fallback: use episode-level token advantages.
                    seq_advs = token_advs[offset : offset + len(seq_tokens)]

                offset += len(seq_tokens)
                seq_advs_by_turn.append(seq_advs)

            if config.echo_enabled:
                rollout_datum, rollout_echo_build = build_rollout_echo_datum(
                    turns_G[s_idx],
                    completion_advantages=seq_advs_by_turn,
                    weight=config.echo_weight,
                    min_prompt_overlap=config.echo_min_prompt_overlap,
                )
                acc.echo_build = _add_echo_build_stats(
                    acc.echo_build,
                    rollout_echo_build,
                )
                if rollout_datum is not None:
                    acc.datum_tokens.append(rollout_datum.tokens)
                    acc.datum_logprobs.append(rollout_datum.logprobs)
                    acc.datum_advantages.append(rollout_datum.advantages)
                    acc.datum_echo_advantages.append(
                        rollout_datum.echo_advantages
                    )
                    acc.datum_echo_full_observation_counts.append(
                        rollout_datum.full_observation_count
                    )
                    acc.rl_completion_token_count += sum(
                        len(tokens) for tokens in turn_token_ids
                    )
                    acc.rl_completion_surprisal_sum += sum(
                        -lp for turn_lps in turn_logprobs for lp in turn_lps
                    )
                    continue

            turn_prompt_ids = turns_prompt_ids_G[s_idx]
            for t_idx in range(len(turn_token_ids)):
                seq_tokens = turn_token_ids[t_idx]
                seq_logprobs = turn_logprobs[t_idx]
                prompt_ids = turn_prompt_ids[t_idx]
                seq_advs = seq_advs_by_turn[t_idx]

                full_tokens = list(prompt_ids) + list(seq_tokens)
                padded_logprobs = [0.0] * len(prompt_ids) + list(seq_logprobs)
                padded_advantages = [0.0] * len(prompt_ids) + list(seq_advs)
                acc.datum_tokens.append(full_tokens)
                acc.datum_logprobs.append(padded_logprobs)
                acc.datum_advantages.append(padded_advantages)
                acc.datum_echo_advantages.append(
                    [0.0] * len(full_tokens)
                )
                acc.datum_echo_full_observation_counts.append(0)
                acc.rl_completion_token_count += len(seq_tokens)
                acc.rl_completion_surprisal_sum += sum(
                    -lp for lp in seq_logprobs
                )

        generation_entries: list[dict[str, object]] = []
        selected_generation_indices = (
            _generation_log_indices(
                len(completion_texts_G),
                samples_per_prompt=config.generation_log_samples_per_prompt,
                rewards=rewards_G,
            )
            if generations_logger.enabled
            else []
        )
        for s_idx in selected_generation_indices:
            comp_text = completion_texts_G[s_idx]
            gen_entry: dict[str, object] = {
                "step": step,
                "prompt": prompts.previews[f_idx],
                "completion": comp_text[:500],
                "reward": rewards_G[s_idx],
                "num_tokens": len(logprobs_G[s_idx]),
            }
            if s_idx < len(turn_logs_G) and turn_logs_G[s_idx]:
                turn_summary = []
                for tl in turn_logs_G[s_idx]:
                    obs_raw = tl.get("observation", {})
                    obs = (
                        cast(Mapping[str, object], obs_raw)
                        if isinstance(obs_raw, Mapping)
                        else {}
                    )
                    entry: dict[str, object] = {
                        "turn": tl.get("turn"),
                        "tick": obs.get("tick", 0),
                        "customer_waiting": obs.get(
                            "customer_waiting",
                            False,
                        ),
                        "inventory": obs.get("inventory", 0),
                        "operation": tl.get("operation"),
                        "reward_delta": tl.get("reward_delta", 0.0),
                        "valid": tl.get("valid", True),
                    }
                    if not tl.get("valid"):
                        entry["error"] = tl.get("error", "")
                    turn_summary.append(entry)
                    # ── Behavior accumulation ──
                    acc.behavior_turns += 1
                    if not tl.get("valid", True):
                        acc.behavior_invalid += 1
                    _op = str(tl.get("operation", "unknown"))
                    acc.behavior_actions[_op] = (
                        acc.behavior_actions.get(_op, 0) + 1
                    )
                gen_entry["turn_log"] = turn_summary
                acc.behavior_resp_lens.append(
                    len(str(gen_entry.get("completion", "")))
                )
            if s_idx < len(turn_advantages_G) and turn_advantages_G[s_idx]:
                gen_entry["turn_advantages"] = turn_advantages_G[s_idx]
            if s_idx < len(branch_rewards_G) and branch_rewards_G[s_idx]:
                gen_entry["branch_rewards"] = branch_rewards_G[s_idx]
            if s_idx < len(turns_logprobs_G) and turns_logprobs_G[s_idx]:
                turn_lps = turns_logprobs_G[s_idx]
                gen_entry["turn_mean_logprobs"] = [
                    sum(lps) / len(lps) if lps else 0.0
                    for lps in turn_lps
                ]
                gen_entry["turn_logprob_var"] = [
                    (sum((x - sum(lps) / len(lps)) ** 2 for x in lps) / len(lps))
                    if len(lps) > 1 else 0.0
                    for lps in turn_lps
                ]
            # Log top-K highest surprisal tokens with decoded text.
            # Useful for debugging DG gating and blog post analysis:
            # shows which tokens the gate considers "fork-points".
            if s_idx < len(logprobs_G):
                # Flatten token IDs for this sample
                s_tids: list[int] = []
                for t_idx2 in range(len(turns_token_ids_G[s_idx])):
                    s_tids.extend(turns_token_ids_G[s_idx][t_idx2])
                top_entries = _top_surprisal_payload(
                    logprobs_G[s_idx],
                    s_tids,
                    token_lookup,
                    limit=config.generation_top_surprisal_limit,
                )
                if top_entries:
                    gen_entry["top_surprisal_tokens"] = top_entries
            generation_entries.append(gen_entry)
        if generation_entries:
            generations_logger.log_many(generation_entries)
    acc.sample_time_s = time.perf_counter() - sample_start


def _run_singleturn_rollouts(
    config: TrainConfig,
    helper: TrainHelper,
    tokenizer: object,
    verifiers_env: object | None,
    reward_fn: RewardFunction | None,
    prompts: _PromptBatch,
    acc: _RolloutAccumulator,
    *,
    step: int,
    group_size: int,
    sepa_lambda: float,
    algorithm_params: Mapping[str, object],
    transform_params: Mapping[str, object],
    needs_planning: bool,
    detector: PlanningDetector | None,
    token_lookup: TokenTextLookup,
    runtime_counters: RuntimeCounters,
    generations_logger: JsonlLogger,
) -> None:
    """Sample, score, and datum-ize single-turn completion groups."""
    # 10c. Sample completions
    sample_start = time.perf_counter()
    entropy_helper = (
        helper
        if config.uncertainty_kind == "shannon_entropy"
        and isinstance(helper, EntropySamplingHelper)
        else None
    )
    precomputed_entropies_batch: list[list[list[float]]] | None = None
    if entropy_helper is not None:
        enriched_sequences = entropy_helper.sample_with_entropy(
            prompts.ids,
            group_size,
            config.max_tokens,
            config.temperature,
            config.top_p,
        )
        # Separate into standard 2-tuples + entropy side channel
        all_group_sequences = [
            [(ids, lps) for ids, lps, _ent in group]
            for group in enriched_sequences
        ]
        precomputed_entropies_batch = [
            [ent if ent is not None else [] for _ids, _lps, ent in group]
            for group in enriched_sequences
        ]
    else:
        all_group_sequences = helper.sample(
            prompts.ids,
            group_size,
            config.max_tokens,
            config.temperature,
            config.top_p,
        )
    acc.sample_time_s = time.perf_counter() - sample_start

    decoded_groups = decode_sequence_groups(
        tokenizer,
        all_group_sequences,
        needs_planning=needs_planning,
        token_lookup=token_lookup if needs_planning else None,
        detector=detector if needs_planning else None,
        counters=runtime_counters,
    )

    for f_idx, decoded_group in enumerate(decoded_groups):
        prompt_ids = prompts.ids[f_idx]
        answer = prompts.answers[f_idx]
        task = prompts.tasks[f_idx]
        info = prompts.infos[f_idx]
        prompt_obj = prompts.objs[f_idx]

        rewards_G: list[float] = []
        logprobs_G: list[list[float]] = []
        planning_masks_G: list[list[int]] = []
        completion_texts_G: list[str] = []
        for sample in decoded_group:
            completion_texts_G.append(sample.text)
            logprobs_G.append(sample.logprobs)
            planning_masks_G.append(sample.planning_mask)

        if verifiers_env is None:
            assert reward_fn is not None
            for text in completion_texts_G:
                rewards_G.append(reward_fn.score(text, answer))
        else:
            rewards_G = score_singleturn_group(
                verifiers_env,
                prompt=prompt_obj,
                answer=answer,
                task=task,
                info=info,
                completion_texts=completion_texts_G,
            )

        acc.logprobs_sepa.extend(logprobs_G)
        acc.planning_masks_sepa.extend(planning_masks_G)

        for r in rewards_G:
            acc.rewards.append(r)
            if r > _CORRECT_THRESHOLD:
                acc.correct += 1

        for sample in decoded_group:
            acc.total_completions += 1
            if len(sample.token_ids) >= config.max_tokens:
                acc.max_token_hits += 1

        _print_group_summary(rewards_G, answer)
        reward_tie_stats = acc.ties.observe(rewards_G)
        if reward_tie_stats["is_uniform"] and not config.tl_grpo:
            if not _keep_uniform_group(
                rewards_G,
                batch_advantage_norm=config.batch_advantage_norm,
                keep_for_echo=False,
            ):
                continue

        # Resolve per-group precomputed entropies
        group_entropies_G: list[list[float]] | None = None
        if precomputed_entropies_batch is not None:
            group_entropies_G = precomputed_entropies_batch[f_idx]

        adv_result = _compute_group_advantages(
            config,
            rewards_G,
            logprobs_G,
            planning_masks_G,
            step=step,
            sepa_lambda=sepa_lambda,
            algorithm_params=algorithm_params,
            transform_params=transform_params,
            precomputed_entropies_G=group_entropies_G,
        )
        all_token_advs_G = adv_result.token_advs
        if adv_result.has_stats:
            acc.surprisal_stats.append(adv_result.stats)
        if adv_result.extra_metrics:
            acc.adv_results.append(adv_result)

        for sample, token_advs in zip(decoded_group, all_token_advs_G):
            full_tokens = list(prompt_ids) + list(sample.token_ids)
            padded_logprobs = [0.0] * len(prompt_ids) + list(sample.logprobs)
            padded_advantages = [0.0] * len(prompt_ids) + list(token_advs)
            acc.datum_tokens.append(full_tokens)
            acc.datum_logprobs.append(padded_logprobs)
            acc.datum_advantages.append(padded_advantages)
            acc.datum_echo_advantages.append([0.0] * len(full_tokens))
            acc.datum_echo_full_observation_counts.append(0)
            acc.rl_completion_token_count += len(sample.token_ids)
            acc.rl_completion_surprisal_sum += sum(
                -lp for lp in sample.logprobs
            )

        generation_entries: list[dict[str, object]] = []
        selected_generation_indices = (
            _generation_log_indices(
                len(decoded_group),
                samples_per_prompt=config.generation_log_samples_per_prompt,
                rewards=rewards_G,
            )
            if generations_logger.enabled
            else []
        )
        for s_idx in selected_generation_indices:
            sample = decoded_group[s_idx]
            gen_entry: dict[str, object] = {
                "step": step,
                "prompt": prompts.previews[f_idx],
                "completion": sample.text[:500],
                "reward": rewards_G[s_idx],
                "num_tokens": len(sample.logprobs),
            }
            # Top-K highest surprisal tokens with decoded text
            top_entries = _top_surprisal_payload(
                sample.logprobs,
                sample.token_ids,
                token_lookup,
                limit=config.generation_top_surprisal_limit,
            )
            if top_entries:
                gen_entry["top_surprisal_tokens"] = top_entries
            generation_entries.append(gen_entry)
        if generation_entries:
            generations_logger.log_many(generation_entries)


def train(config: TrainConfig, flow: TrainingFlow | None = None) -> str | None:
    """Main training loop -- fully self-contained. Returns final adapter path."""

    _print_config_summary(config)

    # -----------------------------------------------------------------------
    # 0a. Build and validate flow
    # -----------------------------------------------------------------------
    if flow is None:
        flow = build_flow(config, gpu=True)
        trace_result = flow.trace()
        if not trace_result.ok:
            shutdown = getattr(flow.backend, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            msgs = [i.message for i in trace_result.issues if i.severity == "error"]
            raise ValueError("Training flow validation failed:\n" + "\n".join(msgs))
        _print_flow_warnings(trace_result)

    # -----------------------------------------------------------------------
    # 0. Setup directories + loggers
    # -----------------------------------------------------------------------
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    emergence_dir = log_path / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlLogger(str(log_path / "metrics.jsonl"))
    steps_logger = JsonlLogger(str(emergence_dir / "steps.jsonl"))
    generations_logger = JsonlLogger(
        str(emergence_dir / "generations.jsonl"),
        flush_every=32,
        flush_interval_s=1.0,
        enabled=config.log_generations,
    )

    wandb_run: WandbRunLike | None = None
    backend_for_cleanup = getattr(flow, "backend", None)

    try:
        # -----------------------------------------------------------------------
        # 1. Seed for reproducibility
        # -----------------------------------------------------------------------
        if config.seed >= 0:
            import random

            import numpy as np
            import torch

            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            print(f"Seeded RNGs with {config.seed}")

        # -----------------------------------------------------------------------
        # 2. Load dataset (fail fast before backend/tokenizer setup)
        # -----------------------------------------------------------------------
        print("Loading dataset...")
        verifiers_env = None
        if config.environment_provider == "verifiers":
            verifiers_env = load_verifiers_environment(config)
            examples = load_examples_from_environment(verifiers_env, config)
            print(
                f"Loaded {len(examples)} examples from verifiers env "
                f"'{config.environment_id}'"
            )
            if config.data_source != "math":
                print(
                    "NOTE: [data].source is ignored when [environment].provider is set."
                )
            if config.reward_type != "match":
                print(
                    "NOTE: [reward] settings are ignored with [environment].provider="
                    "'verifiers'; the environment rubric is used."
                )
        else:
            examples = get_registry("data_source").create(config.data_source, config).load()
        if not examples:
            raise RuntimeError("Dataset is empty — cannot train with zero examples.")
        if verifiers_env is None:
            print(f"Loaded {len(examples)} examples")

        # -----------------------------------------------------------------------
        # 3. Use flow-resolved backend + capabilities
        # -----------------------------------------------------------------------
        helper = flow.backend
        assert helper is not None
        backend_caps = flow.backend_capabilities
        _print_backend_capability_summary(
            config.backend,
            flow.backend_capability_source,
            backend_caps.reports_sync_loss,
            backend_caps.preserves_token_advantages,
            backend_caps.supports_checkpoint_resume,
            backend_caps.resume_runtime_dependent,
        )

        # -----------------------------------------------------------------------
        # 4. Load tokenizer + lazy token/prompt caches
        # -----------------------------------------------------------------------
        print(f"Loading tokenizer for {config.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        runtime_counters = RuntimeCounters()
        token_lookup = TokenTextLookup(tokenizer, counters=runtime_counters)
        prompt_cache = ExamplePromptCache(
            tokenizer,
            [ex.prompt for ex in examples],
            encoder=encode_prompt_for_sampling,
            preview_renderer=prompt_preview,
            counters=runtime_counters,
        )
        print("Token lookup: lazy")
        print("Prompt encoding cache: lazy")

        # -----------------------------------------------------------------------
        # 4b. Planning detector
        # -----------------------------------------------------------------------
        detector = flow.planning_detector
        print(f"Planning detector: {config.planning_detector}")

        # -----------------------------------------------------------------------
        # 5. SEPA controller
        # -----------------------------------------------------------------------
        sepa_controller = flow.sepa_controller
        assert sepa_controller is not None

        # -----------------------------------------------------------------------
        # 6. Back pressure
        # -----------------------------------------------------------------------
        backpressure = flow.backpressure
        assert backpressure is not None

        # -----------------------------------------------------------------------
        # 8. Optional wandb
        # -----------------------------------------------------------------------
        wandb_run = _init_wandb(config)

        # -----------------------------------------------------------------------
        # 9. Training loop
        # -----------------------------------------------------------------------
        reward_fn = None
        if verifiers_env is None:
            reward_fn = get_registry("reward").create(config.reward_type, config)
        verifiers_multiturn = (
            verifiers_env is not None and is_multiturn_environment(verifiers_env)
        )
        example_idx = 0
        total_correct = 0
        total_completions = 0
        sepa_lambda_val = 0.0
        current_batch_size = config.batch_size
        current_group_size = config.group_size
        needs_planning = flow.needs_planning
        uses_sepa_controller = flow.uses_sepa_controller
        generation_log_samples_per_prompt = config.generation_log_samples_per_prompt
        generation_top_surprisal_limit = config.generation_top_surprisal_limit
        if config.algorithm_mode:
            print(
                "Algorithm mode active: "
                f"{config.algorithm_mode}. "
                "Ignoring advantage_mode/transform_mode composition."
            )
        start_step = 0
        tl_grpo_ema: float | None = config.tl_grpo_ema_init if config.tl_grpo else None
        delight_eta_ema: float | None = None

        # -----------------------------------------------------------------------
        # 10b. Resume from checkpoint (if requested)
        # -----------------------------------------------------------------------
        if config.resume_from:
            saved = _load_trainer_state(config.resume_from)
            start_step = saved["step"] + 1
            example_idx = saved["example_idx"]
            total_correct = saved["total_correct"]
            total_completions = saved["total_completions"]
            current_batch_size = saved["current_batch_size"]
            current_group_size = saved["current_group_size"]

            # Restore SEPA controller state
            if "sepa" in saved:
                sepa_controller.load_state_dict(saved["sepa"])

            # Restore TL-GRPO EMA baseline
            if "tl_grpo_ema" in saved and tl_grpo_ema is not None:
                tl_grpo_ema = float(saved["tl_grpo_ema"])
            if "delight_eta_ema" in saved:
                delight_eta_ema = float(saved["delight_eta_ema"])

            # Restore backend model state
            ckpt_name = saved.get("checkpoint_name", "")
            checkpoint_ref = saved.get("checkpoint_path", "") or ckpt_name
            if checkpoint_ref:
                helper.load_state(checkpoint_ref)

            print(
                f"Resumed from step {saved['step']} "
                f"(checkpoint: {checkpoint_ref or ckpt_name}), continuing from step {start_step}"
            )

        # Warmup sweep schedule: geometric [1,2,4,...] clamped to [min, max]
        warmup_batch_sizes: list[int] = []
        if config.bp_enabled:
            bs = max(1, config.bp_min_batch_size)
            while bs <= config.bp_max_batch_size:
                warmup_batch_sizes.append(bs)
                bs *= 2
            if warmup_batch_sizes and warmup_batch_sizes[-1] != config.bp_max_batch_size:
                warmup_batch_sizes.append(config.bp_max_batch_size)

        # -----------------------------------------------------------------------
        # SFT warmup data (load once if configured)
        # -----------------------------------------------------------------------
        sft_data = _load_sft_warmup_data(config, tokenizer)

        for batch_idx in range(start_step, config.max_steps):
            step_start = time.perf_counter()

            # =================================================================
            # SFT warmup: supervised training from oracle demonstrations
            # =================================================================
            if batch_idx < config.sft_warmup_steps and sft_data and sft_data.examples:
                _run_sft_warmup_step(
                    helper,
                    config,
                    sft_data,
                    batch_idx,
                    metrics_logger=metrics_logger,
                    steps_logger=steps_logger,
                    wandb_run=wandb_run,
                )
                continue  # Skip the RL pipeline for this step

            # Back pressure warmup sweep
            bp_warmup = False
            if config.bp_enabled and warmup_batch_sizes and batch_idx < config.bp_warmup_steps:
                bp_warmup = True
                current_batch_size = warmup_batch_sizes[batch_idx % len(warmup_batch_sizes)]

            # 10a. Checkpoint for sampling
            helper.checkpoint(f"step_{batch_idx}")

            # 10b. Select prompts
            prompts, example_idx = _select_prompt_batch(
                examples, prompt_cache, example_idx, current_batch_size
            )

            # 10d. Rollout accumulation + post-rollout ECHO state
            acc = _RolloutAccumulator(tl_grpo_ema=tl_grpo_ema)
            echo_loss = 0.0
            echo_train_time = 0.0
            echo_joint_optimizer_step = False
            step_transform_params = _prepare_transform_params_for_step(
                config.transform_params,
                delight_eta_prev=delight_eta_ema,
            )
            step_algorithm_params = _prepare_algorithm_params_for_step(
                config.effective_algorithm_params,
                delight_eta_prev=delight_eta_ema,
            )

            # Resolve SEPA lambda once per step (before group loop)
            if uses_sepa_controller:
                sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

            if verifiers_multiturn:
                _run_multiturn_rollouts(
                    config,
                    helper,
                    tokenizer,
                    verifiers_env,
                    prompts,
                    acc,
                    step=batch_idx,
                    group_size=current_group_size,
                    sepa_lambda=sepa_lambda_val,
                    algorithm_params=step_algorithm_params,
                    transform_params=step_transform_params,
                    needs_planning=needs_planning,
                    detector=detector,
                    token_lookup=token_lookup,
                    generations_logger=generations_logger,
                )
            else:
                _run_singleturn_rollouts(
                    config,
                    helper,
                    tokenizer,
                    verifiers_env,
                    reward_fn,
                    prompts,
                    acc,
                    step=batch_idx,
                    group_size=current_group_size,
                    sepa_lambda=sepa_lambda_val,
                    algorithm_params=step_algorithm_params,
                    transform_params=step_transform_params,
                    needs_planning=needs_planning,
                    detector=detector,
                    token_lookup=token_lookup,
                    runtime_counters=runtime_counters,
                    generations_logger=generations_logger,
                )
            tl_grpo_ema = acc.tl_grpo_ema

            # 10e. SEPA state updates
            total_completions += len(acc.rewards)
            total_correct += acc.correct
            correct_rate = (
                acc.correct / len(acc.rewards) if acc.rewards else 0.0
            )

            if uses_sepa_controller:
                sepa_controller.observe_correct_rate(correct_rate)

                if sepa_controller.enabled() and sepa_controller.sepa_schedule == "auto":
                    for t_idx in range(len(acc.logprobs_sepa)):
                        logprobs = acc.logprobs_sepa[t_idx]
                        pmask = acc.planning_masks_sepa[t_idx]
                        exec_ent = [
                            -logprobs[j]
                            for j in range(len(logprobs))
                            if pmask[j] == 0
                        ]
                        sepa_controller.update_auto_state(exec_ent)

            # 10f. Train
            num_datums = len(acc.datum_tokens)
            if num_datums > 0 and not backend_caps.preserves_token_advantages:
                _assert_uniform_completion_advantages_for_non_preserving_backend(
                    acc.datum_logprobs,
                    acc.datum_advantages,
                    backend_name=config.backend,
                )

            # REINFORCE++ batch normalization (before capping)
            batch_norm_metrics: dict[str, float] = {}
            if num_datums > 0 and config.batch_advantage_norm:
                acc.datum_advantages, batch_norm_metrics = (
                    apply_batch_advantage_normalization(acc.datum_advantages)
                )

            # Advantage capping (pre-training, any backend)
            adv_cap_fraction = 0.0
            adv_cap_magnitude = 0.0
            if num_datums > 0 and config.adv_clip_max > 0:
                acc.datum_advantages, adv_cap_fraction, adv_cap_magnitude = (
                    _apply_advantage_cap(acc.datum_advantages, config.adv_clip_max)
                )

            echo_plan = _prepare_echo_step_plan(config, acc)

            rl_has_signal = _has_nonzero_advantage(acc.datum_advantages)
            echo_has_datums = bool(
                config.echo_enabled and _has_nonzero_advantage(
                    acc.datum_echo_advantages
                )
            )
            if not rl_has_signal and not echo_has_datums:
                print(f"Step {batch_idx}: no informative datums, skipping.")
                obs = StepObservation(
                    step_time_s=time.perf_counter() - step_start,
                    sample_time_s=acc.sample_time_s,
                    batch_size=current_batch_size,
                    group_size=current_group_size,
                    skipped=True,
                )
                backpressure.observe(obs)
                continue

            print(
                f"Step {batch_idx}: submitting {num_datums} RL datums "
                f"and {echo_plan.limit.kept_datums if echo_has_datums else 0} ECHO datums..."
            )
            train_start = time.perf_counter()
            loss_value, echo_loss, echo_joint_optimizer_step = _run_rl_echo_train_step(
                helper,
                acc.datum_tokens,
                acc.datum_logprobs,
                acc.datum_advantages,
                acc.datum_echo_advantages if echo_has_datums else [],
                acc.datum_echo_full_observation_counts if echo_has_datums else [],
                echo_loss_fn=config.echo_loss_fn,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
            train_time = time.perf_counter() - train_start
            rl_train_time = train_time if num_datums > 0 else 0.0
            echo_train_time = train_time if echo_has_datums else 0.0
            clip_fraction = getattr(helper, '_clip_fraction', 0.0)
            policy_cov_fraction = getattr(helper, "_policy_cov_fraction", 0.0)
            policy_abs_kl = getattr(helper, "_policy_abs_kl", 0.0)

            step_time = time.perf_counter() - step_start

            # Back pressure
            bp_total_tokens = sum(len(t) for t in acc.datum_tokens)
            obs = StepObservation(
                step_time_s=step_time,
                sample_time_s=acc.sample_time_s,
                train_time_s=train_time,
                num_datums=num_datums,
                batch_size=current_batch_size,
                group_size=current_group_size,
                total_tokens=bp_total_tokens,
                loss=loss_value,
                skipped=False,
            )
            backpressure.observe(obs)
            bp_decision = backpressure.recommend()

            if config.bp_enabled and not bp_warmup:
                if bp_decision.action in ("throttle", "increase"):
                    new_bs = bp_decision.recommended_batch_size
                    new_bs = max(config.bp_min_batch_size, min(config.bp_max_batch_size, new_bs))
                    if new_bs > 0:
                        current_batch_size = new_bs

            # 10g. Logging
            mean_reward = (
                sum(acc.rewards) / len(acc.rewards) if acc.rewards else 0.0
            )
            running_correct_rate = (
                total_correct / total_completions if total_completions > 0 else 0.0
            )
            max_token_hit_rate = (
                acc.max_token_hits / acc.total_completions
                if acc.total_completions > 0
                else 0.0
            )

            # Aggregate entropy stats
            step_exec_mean = step_exec_var = step_plan_mean = step_plan_var = 0.0
            step_post_exec_mean = step_post_exec_var = 0.0
            step_post_plan_mean = step_post_plan_var = 0.0
            if acc.surprisal_stats:
                n_stats = len(acc.surprisal_stats)
                step_exec_mean = sum(s.exec_mean for s in acc.surprisal_stats) / n_stats
                step_exec_var = sum(s.exec_var for s in acc.surprisal_stats) / n_stats
                step_plan_mean = sum(s.plan_mean for s in acc.surprisal_stats) / n_stats
                step_plan_var = sum(s.plan_var for s in acc.surprisal_stats) / n_stats
                step_post_exec_mean = sum(s.post_exec_mean for s in acc.surprisal_stats) / n_stats
                step_post_exec_var = sum(s.post_exec_var for s in acc.surprisal_stats) / n_stats
                step_post_plan_mean = sum(s.post_plan_mean for s in acc.surprisal_stats) / n_stats
                step_post_plan_var = sum(s.post_plan_var for s in acc.surprisal_stats) / n_stats

            condition_label = _condition_label(config)
            sepa_gate = (
                sepa_controller.gate_open()
                if uses_sepa_controller
                else False
            )

            metrics: dict = {
                "step": batch_idx,
                "algorithm_mode": config.algorithm_mode,
                "advantage_mode": config.advantage_mode,
                "transform_mode": config.transform_mode,
                "uncertainty_kind": config.uncertainty_kind,
                "condition": condition_label,
                "backend_reports_sync_loss": backend_caps.reports_sync_loss,
                "backend_preserves_token_advantages": backend_caps.preserves_token_advantages,
                "loss_is_placeholder": not backend_caps.reports_sync_loss,
                "reported_loss": loss_value,
                "loss": loss_value,
                "mean_reward": mean_reward,
                "correct_rate": correct_rate,
                "running_correct_rate": running_correct_rate,
                "reward_tie_eligible_groups": acc.ties.eligible_groups,
                "reward_tie_groups": acc.ties.tie_groups,
                "reward_uniform_groups": acc.ties.uniform_groups,
                "reward_tie_group_rate": acc.ties.tie_group_rate,
                "reward_uniform_group_rate": acc.ties.uniform_group_rate,
                "reward_tie_pair_rate": acc.ties.tie_pair_rate,
                "reward_unique_fraction_mean": acc.ties.unique_fraction_mean,
                "sepa_lambda": sepa_lambda_val,
                "sepa_gate_open": sepa_gate,
                "num_datums": num_datums,
                "max_token_hit_rate": max_token_hit_rate,
                "step_time_s": step_time,
                "sample_time_s": acc.sample_time_s,
                "train_time_s": train_time,
                "batch_size": current_batch_size,
                "group_size": current_group_size,
                "bp_warmup": bp_warmup,
                "bp_action": bp_decision.action,
                "bp_regime": bp_decision.regime,
                "bp_p_star": bp_decision.p_star,
                "bp_sigma": bp_decision.sigma,
                "bp_kappa": bp_decision.kappa,
                "bp_utilization": bp_decision.utilization,
                "bp_throughput": bp_decision.throughput,
            }
            metrics["tokens_per_step"] = bp_total_tokens
            metrics["tokens_per_second"] = (
                bp_total_tokens / step_time if step_time > _PROMPT_PAD_EPS else 0.0
            )
            metrics["sample_share"] = (
                acc.sample_time_s / step_time if step_time > _PROMPT_PAD_EPS else 0.0
            )
            metrics["train_share"] = (
                train_time / step_time if step_time > _PROMPT_PAD_EPS else 0.0
            )
            echo_token_ratio = echo_plan.token_ratio
            metrics["rl/train_time_s"] = rl_train_time
            metrics["rl/completion_tokens"] = acc.rl_completion_token_count
            metrics["rl/completion_surprisal_mean"] = (
                echo_plan.rl_completion_surprisal_mean
            )
            metrics["echo/enabled"] = int(config.echo_enabled)
            metrics["echo/loss"] = echo_loss
            metrics["echo/train_time_s"] = echo_train_time
            metrics["echo/weight"] = config.echo_weight
            metrics["echo/allowed_tokens"] = echo_plan.allowed_tokens
            metrics["echo/reference_completion_tokens"] = (
                echo_plan.reference_completion_tokens
            )
            metrics["echo/completion_surprisal_mean"] = (
                echo_plan.echo_completion_surprisal_mean
            )
            metrics["echo/candidate_datums"] = acc.echo_build.candidate_datums
            metrics["echo/candidate_tokens"] = acc.echo_build.candidate_tokens
            metrics["echo/observation_mask_datums"] = (
                acc.echo_build.observation_mask_datums
            )
            metrics["echo/kept_datums"] = echo_plan.limit.kept_datums
            metrics["echo/kept_tokens"] = echo_plan.limit.kept_tokens
            metrics["echo/truncated_tokens"] = echo_plan.limit.truncated_tokens
            metrics["echo/token_ratio"] = echo_token_ratio
            metrics["echo/skipped_first_turns"] = acc.echo_build.skipped_first_turns
            metrics["echo/skipped_no_suffix"] = acc.echo_build.skipped_no_suffix
            metrics["echo/skipped_low_overlap"] = acc.echo_build.skipped_low_overlap
            metrics["echo/skipped_bad_observation_mask"] = (
                acc.echo_build.skipped_bad_observation_mask
            )
            metrics["echo/skipped_entropy_floor"] = int(echo_plan.skipped_entropy_floor)
            metrics["echo/entropy_floor"] = config.echo_entropy_floor
            metrics["echo/mode_collapse_guard"] = int(echo_plan.skipped_entropy_floor)
            metrics["echo/joint_optimizer_step"] = int(echo_joint_optimizer_step)
            rss_mb = process_max_rss_mb()
            if rss_mb is not None:
                metrics["process_max_rss_mb"] = round(rss_mb, 3)
            metrics.update(runtime_counters.metrics())
            metrics.update(collect_runtime_metrics(helper))
            if acc.rollout_timing_metrics:
                metrics.update(acc.rollout_timing_metrics)
                rollout_total = acc.rollout_timing_metrics.get("rollout/total_s", 0.0)
                rollout_share = (
                    rollout_total / acc.sample_time_s if acc.sample_time_s > _PROMPT_PAD_EPS else 0.0
                )
                metrics["rollout/accounted_share_of_sample"] = min(
                    max(rollout_share, 0.0),
                    1.0,
                )
            if acc.surprisal_stats:
                metrics["exec_entropy_mean"] = step_exec_mean
                metrics["exec_entropy_var"] = step_exec_var
                metrics["plan_entropy_mean"] = step_plan_mean
                metrics["plan_entropy_var"] = step_plan_var
                # Preferred names: these values are sampled-token surprisal.
                metrics["exec_surprisal_mean"] = step_exec_mean
                metrics["exec_surprisal_var"] = step_exec_var
                metrics["plan_surprisal_mean"] = step_plan_mean
                metrics["plan_surprisal_var"] = step_plan_var
                metrics["post_exec_surprisal_mean"] = step_post_exec_mean
                metrics["post_exec_surprisal_var"] = step_post_exec_var
                metrics["post_plan_surprisal_mean"] = step_post_plan_mean
                metrics["post_plan_surprisal_var"] = step_post_plan_var
            metrics["clip_fraction"] = clip_fraction
            metrics["policy_loss_mode"] = config.policy_loss_mode
            metrics["policy/cov_fraction"] = policy_cov_fraction
            metrics["policy/abs_kl"] = policy_abs_kl
            metrics["adv_cap_fraction"] = adv_cap_fraction
            metrics["adv_cap_magnitude"] = adv_cap_magnitude
            if batch_norm_metrics:
                metrics.update(batch_norm_metrics)
            if acc.adv_results:
                all_extra_keys = {k for r in acc.adv_results for k in r.extra_metrics}
                for k in all_extra_keys:
                    vals = [r.extra_metrics[k] for r in acc.adv_results if k in r.extra_metrics]
                    metrics[k] = sum(vals) / len(vals)
            if "dg_eta" in metrics:
                delight_eta_ema = float(metrics["dg_eta"])

            # ── Behavior monitoring ─────────────────────────────────────
            # Aggregate turn-level behavior from this step's generations.
            # Tracks model behavior drift that loss alone cannot detect:
            # action collapse, invalid action rate, response length.
            if acc.behavior_turns > 0:
                metrics["behavior/invalid_action_rate"] = (
                    acc.behavior_invalid / acc.behavior_turns
                )
                metrics["behavior/action_type_count"] = len(acc.behavior_actions)
                if acc.behavior_actions:
                    _act_total = sum(acc.behavior_actions.values())
                    _max_frac = max(acc.behavior_actions.values()) / _act_total
                    metrics["behavior/action_dominance"] = _max_frac
            if acc.behavior_resp_lens:
                metrics["behavior/avg_response_chars"] = (
                    sum(acc.behavior_resp_lens) / len(acc.behavior_resp_lens)
                )
            # ────────────────────────────────────────────────────────────

            metrics_logger.log(metrics)

            loss_display = _format_loss_for_display(
                loss_value,
                backend_caps.reports_sync_loss,
            )
            print(
                f"Step {batch_idx} [{condition_label}] | loss={loss_display}"
                f" | reward={mean_reward:.3f}"
                f" | correct={correct_rate * 100:.1f}%"
                f" | datums={num_datums}"
                f" | bs={current_batch_size}"
                f" | gs={current_group_size}"
                f" | tie_g={acc.ties.tie_group_rate * 100:.1f}%"
                f" | sepa_l={sepa_lambda_val:.4f}"
                f" | time={step_time:.1f}s"
            )

            # Wandb
            if wandb_run is not None:
                wandb_metrics: dict[str, int | float | str] = {
                    "train/loss": loss_value,
                    "train/reported_loss": loss_value,
                    "train/uncertainty_kind": config.uncertainty_kind,
                    "train/loss_is_placeholder": int(not backend_caps.reports_sync_loss),
                    "train/rewards/mean_reward": mean_reward,
                    "train/rewards/correct_rate": correct_rate,
                    "train/rewards/running_correct_rate": running_correct_rate,
                    "train/rewards/tie_eligible_groups": acc.ties.eligible_groups,
                    "train/rewards/tie_groups": acc.ties.tie_groups,
                    "train/rewards/uniform_groups": acc.ties.uniform_groups,
                    "train/rewards/tie_group_rate": acc.ties.tie_group_rate,
                    "train/rewards/uniform_group_rate": acc.ties.uniform_group_rate,
                    "train/rewards/tie_pair_rate": acc.ties.tie_pair_rate,
                    "train/rewards/unique_fraction_mean": acc.ties.unique_fraction_mean,
                    "train/backend/reports_sync_loss": int(backend_caps.reports_sync_loss),
                    "train/backend/preserves_token_advantages": int(
                        backend_caps.preserves_token_advantages
                    ),
                    "train/sepa_lambda": sepa_lambda_val,
                    "train/sepa_gate_open": int(sepa_gate),
                    "train/max_token_hit_rate": max_token_hit_rate,
                    "train/num_datums": num_datums,
                    "train/step_time_s": step_time,
                    "train/batch_size": current_batch_size,
                    "train/group_size": current_group_size,
                    "train/entropy/exec_mean": step_exec_mean,
                    "train/entropy/exec_var": step_exec_var,
                    "train/entropy/plan_mean": step_plan_mean,
                    "train/entropy/plan_var": step_plan_var,
                    "train/surprisal/exec_mean": step_exec_mean,
                    "train/surprisal/exec_var": step_exec_var,
                    "train/surprisal/plan_mean": step_plan_mean,
                    "train/surprisal/plan_var": step_plan_var,
                    "train/surprisal/post_exec_mean": step_post_exec_mean,
                    "train/surprisal/post_exec_var": step_post_exec_var,
                    "train/surprisal/post_plan_mean": step_post_plan_mean,
                    "train/surprisal/post_plan_var": step_post_plan_var,
                    "train/clip_fraction": clip_fraction,
                    "train/policy_cov_fraction": policy_cov_fraction,
                    "train/policy_abs_kl": policy_abs_kl,
                    "train/adv_cap_fraction": adv_cap_fraction,
                    "train/adv_cap_magnitude": adv_cap_magnitude,
                    **{f"train/{k}": v for k, v in batch_norm_metrics.items()},
                    "train/backpressure/action": bp_decision.action,
                    "train/backpressure/regime": bp_decision.regime,
                    "train/backpressure/p_star": bp_decision.p_star,
                    "train/backpressure/sigma": bp_decision.sigma,
                    "train/backpressure/kappa": bp_decision.kappa,
                    "train/backpressure/utilization": bp_decision.utilization,
                    "train/backpressure/throughput": bp_decision.throughput,
                    "train/backpressure/warmup": int(bp_warmup),
                }
                if acc.adv_results:
                    for k in {k for r in acc.adv_results for k in r.extra_metrics}:
                        wandb_metrics[f"train/{k}"] = metrics.get(k, 0.0)
                for _ek in (
                    "rl/train_time_s",
                    "rl/completion_tokens",
                    "rl/completion_surprisal_mean",
                    "echo/enabled",
                    "echo/loss",
                    "echo/train_time_s",
                    "echo/weight",
                    "echo/allowed_tokens",
                    "echo/reference_completion_tokens",
                    "echo/completion_surprisal_mean",
                    "echo/candidate_datums",
                    "echo/candidate_tokens",
                    "echo/observation_mask_datums",
                    "echo/kept_datums",
                    "echo/kept_tokens",
                    "echo/truncated_tokens",
                    "echo/token_ratio",
                    "echo/skipped_low_overlap",
                    "echo/skipped_bad_observation_mask",
                    "echo/skipped_entropy_floor",
                    "echo/entropy_floor",
                    "echo/mode_collapse_guard",
                ):
                    wandb_metrics[f"train/{_ek}"] = metrics[_ek]
                if tl_grpo_ema is not None:
                    wandb_metrics["train/tl_grpo_ema_baseline"] = tl_grpo_ema
                # Behavior monitoring → W&B
                for _bk in ("behavior/invalid_action_rate", "behavior/action_type_count",
                            "behavior/action_dominance", "behavior/avg_response_chars"):
                    if _bk in metrics:
                        wandb_metrics[f"train/{_bk}"] = metrics[_bk]
                wandb_run.log(wandb_metrics, step=batch_idx)

            # Step record for emergence analysis
            step_entry: dict = {
                "step": batch_idx,
                "mean_reward": mean_reward,
                "correct_count": acc.correct,
                "total_count": len(acc.rewards),
                "condition": condition_label,
                "uncertainty_kind": config.uncertainty_kind,
            }
            if acc.surprisal_stats:
                step_entry["exec_entropy_mean"] = step_exec_mean
                step_entry["exec_entropy_var"] = step_exec_var
                step_entry["plan_entropy_mean"] = step_plan_mean
                step_entry["plan_entropy_var"] = step_plan_var
                step_entry["exec_surprisal_mean"] = step_exec_mean
                step_entry["exec_surprisal_var"] = step_exec_var
                step_entry["plan_surprisal_mean"] = step_plan_mean
                step_entry["plan_surprisal_var"] = step_plan_var
                step_entry["post_exec_surprisal_mean"] = step_post_exec_mean
                step_entry["post_exec_surprisal_var"] = step_post_exec_var
                step_entry["post_plan_surprisal_mean"] = step_post_plan_mean
                step_entry["post_plan_surprisal_var"] = step_post_plan_var
            step_entry["clip_fraction"] = clip_fraction
            step_entry["policy_loss_mode"] = config.policy_loss_mode
            step_entry["policy/cov_fraction"] = policy_cov_fraction
            step_entry["policy/abs_kl"] = policy_abs_kl
            step_entry["adv_cap_fraction"] = adv_cap_fraction
            step_entry["adv_cap_magnitude"] = adv_cap_magnitude
            step_entry["echo/enabled"] = int(config.echo_enabled)
            step_entry["echo/kept_tokens"] = echo_plan.limit.kept_tokens
            step_entry["echo/token_ratio"] = echo_token_ratio
            step_entry["echo/skipped_entropy_floor"] = int(echo_plan.skipped_entropy_floor)
            if acc.adv_results:
                for k in {k for r in acc.adv_results for k in r.extra_metrics}:
                    step_entry[k] = metrics.get(k, 0.0)
            steps_logger.log(step_entry)

            # Periodic checkpoint
            if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
                ckpt_name = f"checkpoint_step_{batch_idx + 1}"
                checkpoint_path = helper.save_adapter(
                    config.adapter_path,
                    ckpt_name,
                )
                _save_trainer_state(
                    log_path,
                    step=batch_idx,
                    example_idx=example_idx,
                    total_correct=total_correct,
                    total_completions=total_completions,
                    current_batch_size=current_batch_size,
                    current_group_size=current_group_size,
                    checkpoint_name=ckpt_name,
                    checkpoint_path=checkpoint_path,
                    sepa_state=sepa_controller.state_dict(),
                    tl_grpo_ema=tl_grpo_ema,
                    delight_eta_ema=delight_eta_ema,
                )
                print(f"Saved checkpoint: {ckpt_name}")

        # -----------------------------------------------------------------------
        # Final
        # -----------------------------------------------------------------------
        final_path = helper.save_adapter(config.adapter_path, "final")
        _save_trainer_state(
            log_path,
            step=config.max_steps - 1,
            example_idx=example_idx,
            total_correct=total_correct,
            total_completions=total_completions,
            current_batch_size=current_batch_size,
            current_group_size=current_group_size,
            checkpoint_name="final",
            checkpoint_path=final_path,
            sepa_state=sepa_controller.state_dict(),
            tl_grpo_ema=tl_grpo_ema,
            delight_eta_ema=delight_eta_ema,
        )
        final_rate = (
            100.0 * total_correct / total_completions if total_completions > 0 else 0.0
        )
        print(
            f"Training complete. {_condition_label(config)}, "
            f"{config.max_steps} steps, running correct rate: {final_rate:.1f}%"
        )
        metrics_path = log_path / "metrics.jsonl"
        if metrics_path.is_file():
            print(f"Metrics saved to {metrics_path}")
        else:
            print("No metrics file written (all steps skipped / no informative datums).")

        return final_path
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        shutdown = getattr(backend_for_cleanup, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        for logger in (metrics_logger, steps_logger, generations_logger):
            try:
                logger.close()
            except Exception:
                pass
