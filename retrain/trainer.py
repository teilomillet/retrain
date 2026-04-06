"""Main training loop -- calls LocalTrainHelper directly, no Mojo.

Ports the training loop from src/main.mojo into pure Python.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from transformers import AutoTokenizer

from retrain.advantages import (
    EntropyStats,
    apply_batch_advantage_normalization,
    compute_algorithm_advantages,
    compute_composable_advantages,
)
from retrain.backpressure import (
    StepObservation,
)
from retrain.config import TrainConfig
from retrain.flow import (
    TrainingFlow,
    _UNIFORMITY_EPS,
    _condition_label,
    build_flow,
)
from retrain.logging_utils import JsonlLogger
from retrain.registry import get_registry
from retrain.sepa import SEPAStateDict
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
    tmp.rename(path / _TRAINER_STATE_FILE)
    if checkpoint_path:
        latest_tmp = path / "latest_sampler_path.txt.tmp"
        latest_tmp.write_text(f"{checkpoint_path}\n")
        latest_tmp.rename(path / "latest_sampler_path.txt")


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
            msgs = [i.message for i in trace_result.issues if i.severity == "error"]
            raise ValueError("Training flow validation failed:\n" + "\n".join(msgs))

    # -----------------------------------------------------------------------
    # 0. Setup directories + loggers
    # -----------------------------------------------------------------------
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    emergence_dir = log_path / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlLogger(str(log_path / "metrics.jsonl"))
    steps_logger = JsonlLogger(str(emergence_dir / "steps.jsonl"))
    generations_logger = JsonlLogger(str(emergence_dir / "generations.jsonl"))

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
    # 4. Load tokenizer + vocab table
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer for {config.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    print("Loading vocabulary table...")
    vocab_size = tokenizer.vocab_size  # type: ignore[unresolved-attribute]
    if hasattr(tokenizer, "added_tokens_encoder"):
        vocab_size += len(tokenizer.added_tokens_encoder)
    all_ids = list(range(vocab_size))
    vocab_table: list[str] = []
    py_tokens = tokenizer.convert_ids_to_tokens(all_ids)  # type: ignore[unresolved-attribute]
    for tok in py_tokens:
        vocab_table.append(str(tok) if tok is not None else "")
    print(f"Vocabulary table: {len(vocab_table)} entries")

    # -----------------------------------------------------------------------
    # 4b. Planning detector
    # -----------------------------------------------------------------------
    detector = flow.planning_detector
    print(f"Planning detector: {config.planning_detector}")

    # -----------------------------------------------------------------------
    # 5. Pre-encode all prompts
    # -----------------------------------------------------------------------
    print("Pre-encoding prompts...")
    pre_encoded_prompts: list[list[int]] = []
    for ex in examples:
        pre_encoded_prompts.append(encode_prompt_for_sampling(tokenizer, ex.prompt))
    print(f"Pre-encoded {len(pre_encoded_prompts)} prompts")

    # -----------------------------------------------------------------------
    # 5. SEPA controller
    # -----------------------------------------------------------------------
    sepa_controller = flow.sepa_controller
    assert sepa_controller is not None

    # -----------------------------------------------------------------------
    # 6. Back pressure
    # -----------------------------------------------------------------------
    backpressure = flow.backpressure

    # -----------------------------------------------------------------------
    # 8. Optional wandb
    # -----------------------------------------------------------------------
    wandb_run = None
    wandb_enabled = bool(config.wandb_project)
    if wandb_enabled:
        import wandb  # type: ignore[unresolved-import]

        condition_label = _condition_label(config)
        default_run_name = Path(config.log_dir).name or condition_label
        run_name = config.wandb_run_name or default_run_name
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
            "adv_clip_max": config.adv_clip_max,
            "sft_warmup_steps": config.sft_warmup_steps,
            "tl_grpo": int(config.tl_grpo),
        }
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=wandb_config,
            entity=config.wandb_entity or None,
            group=config.wandb_group or None,
            tags=wandb_tags,
        )
        print(f"Wandb initialized: {config.wandb_project}/{run_name}")

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
            helper.load_state(checkpoint_ref)  # type: ignore[unresolved-attribute]

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
    sft_examples: list[list[dict[str, str]]] = []
    if config.sft_warmup_steps > 0 and config.sft_data_path:
        sft_path = Path(config.sft_data_path)
        if sft_path.exists():
            import json as _json
            with open(sft_path) as _f:
                for _line in _f:
                    row = _json.loads(_line)
                    sft_examples.append(row["messages"])
            print(f"Loaded {len(sft_examples)} SFT warmup examples from {sft_path}")
        else:
            print(f"WARNING: SFT data path {sft_path} not found, skipping warmup")

    for batch_idx in range(start_step, config.max_steps):
        step_start = time.perf_counter()

        # =================================================================
        # SFT warmup: supervised training from oracle demonstrations
        # =================================================================
        if batch_idx < config.sft_warmup_steps and sft_examples:
            helper.checkpoint(f"step_{batch_idx}")  # type: ignore[unresolved-attribute]

            # Sample a batch of SFT examples
            sft_batch_size = min(16, len(sft_examples))
            sft_start = (batch_idx * sft_batch_size) % len(sft_examples)
            sft_batch = sft_examples[sft_start : sft_start + sft_batch_size]

            # Tokenize: full conversation (system + user + assistant)
            sft_tokens_list: list[list[int]] = []
            sft_logprobs_list: list[list[float]] = []
            sft_advantages_list: list[list[float]] = []

            for msgs in sft_batch:
                # Tokenize prompt (system + user) and response (assistant) separately
                # to create a mask: advantages=0 for prompt, advantages=1 for response
                prompt_msgs = msgs[:2]  # system + user
                full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)  # type: ignore[unresolved-attribute]
                prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)  # type: ignore[unresolved-attribute]
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore[unresolved-attribute]
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)  # type: ignore[unresolved-attribute]
                if len(full_tokens) > config.max_tokens + 512:
                    full_tokens = full_tokens[: config.max_tokens + 512]
                # Mask: 0 for prompt tokens, 1 for response tokens
                n_prompt = min(len(prompt_tokens), len(full_tokens))
                advantages = [0.0] * n_prompt + [1.0] * (len(full_tokens) - n_prompt)
                sft_tokens_list.append(full_tokens)
                sft_logprobs_list.append([0.0] * len(full_tokens))
                sft_advantages_list.append(advantages)

            # Train with cross-entropy loss (actual SFT, not importance sampling)
            # Use sft_lr if set, otherwise fall back to main lr
            effective_sft_lr = config.sft_lr if config.sft_lr > 0 else config.lr
            print(
                f"Step {batch_idx} [SFT warmup]: {len(sft_batch)} examples "
                f"(lr={effective_sft_lr:.1e})...",
                flush=True,
            )
            if hasattr(helper, "sft_train_step"):
                loss = helper.sft_train_step(  # type: ignore[call-non-callable]
                    sft_tokens_list,
                    sft_advantages_list,
                    effective_sft_lr,
                    config.weight_decay,
                )
            else:
                # Fallback: use standard train_step with importance_sampling
                loss = helper.train_step(  # type: ignore[unresolved-attribute]
                    sft_tokens_list,
                    sft_logprobs_list,
                    sft_advantages_list,
                    effective_sft_lr,
                    config.weight_decay,
                )
            elapsed = time.perf_counter() - step_start
            # Note: tinker's importance_sampling loss with logprobs=0 produces
            # negative values that get MORE negative as the model learns
            # (-exp(logprob) * advantage).  We report both the raw IS loss
            # and the flipped "sft_signal" (= -loss, higher = better) so
            # the learning curve is intuitive.
            sft_signal = -loss
            print(
                f"Step {batch_idx} [SFT warmup] | is_loss={loss:.4f} | "
                f"sft_signal={sft_signal:.4f} | "
                f"datums={len(sft_batch)} | time={elapsed:.1f}s",
                flush=True,
            )

            # Log to metrics.jsonl (trace every SFT step)
            sft_metrics: dict[str, int | float | str] = {
                "step": batch_idx,
                "loss": loss,
                "sft_signal": sft_signal,
                "phase": "sft",
                "datums": len(sft_batch),
                "time_s": round(elapsed, 2),
                "advantage_mode": config.advantage_mode,
                "lr": effective_sft_lr,
            }
            metrics_logger.log(sft_metrics)
            steps_logger.log(sft_metrics)

            # Wandb
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss,
                        "train/sft_signal": sft_signal,
                        "train/sft_warmup": 1,
                        "train/step": batch_idx,
                        "train/lr": effective_sft_lr,
                    },
                    step=batch_idx,
                )

            # Save checkpoint
            if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
                ckpt_name = f"checkpoint_step_{batch_idx + 1}"
                helper.save_adapter(config.adapter_path, ckpt_name)  # type: ignore[unresolved-attribute]
                print(f"Saved checkpoint: {ckpt_name}")

            # Note: SFT→GRPO transition eval removed (caused context overflow).
            # The first GRPO step (step 10) serves as the eval — if reward > 0,
            # the SFT produced valid actions.

            continue  # Skip the RL pipeline for this step

        # Back pressure warmup sweep
        bp_warmup = False
        if config.bp_enabled and warmup_batch_sizes and batch_idx < config.bp_warmup_steps:
            bp_warmup = True
            current_batch_size = warmup_batch_sizes[batch_idx % len(warmup_batch_sizes)]

        # 10a. Checkpoint for sampling
        helper.checkpoint(f"step_{batch_idx}")  # type: ignore[unresolved-attribute]

        # 10b. Select prompts
        batch_prompt_objs: list[PromptLike] = []
        batch_prompt_previews: list[str] = []
        batch_prompt_ids: list[list[int]] = []
        batch_answers: list[str] = []
        batch_tasks: list[str] = []
        batch_infos: list[ExampleInfoLike] = []

        for _ in range(current_batch_size):
            ex_idx = example_idx % len(examples)
            example_idx += 1
            ex = examples[ex_idx]
            batch_prompt_objs.append(ex.prompt)
            batch_prompt_previews.append(prompt_preview(ex.prompt))
            batch_prompt_ids.append(list(pre_encoded_prompts[ex_idx]))
            batch_answers.append(ex.reference)
            batch_tasks.append(ex.task)
            batch_infos.append(ex.info)

        # 10d. Process groups, compute advantages
        batch_rewards: list[float] = []
        batch_correct = 0
        batch_max_token_hits = 0
        batch_total_completions = 0
        batch_reward_tie_eligible_groups = 0
        batch_reward_tie_groups = 0
        batch_reward_uniform_groups = 0
        batch_reward_tied_pairs = 0
        batch_reward_total_pairs = 0
        batch_reward_unique_fraction_sum = 0.0
        batch_surprisal_stats: list[EntropyStats] = []
        batch_adv_results: list = []
        all_logprobs_sepa: list[list[float]] = []
        all_planning_masks_sepa: list[list[int]] = []
        all_datum_tokens: list[list[int]] = []
        all_datum_logprobs: list[list[float]] = []
        all_datum_advantages: list[list[float]] = []
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

        # Behavior monitoring accumulators for this step.
        _step_behavior_turns = 0
        _step_behavior_invalid = 0
        _step_behavior_actions: dict[str, int] = {}
        _step_behavior_resp_lens: list[int] = []

        if verifiers_multiturn:
            all_group_sequences: list[list[tuple[list[int], list[float]]]] = []
            sample_start = time.perf_counter()
            for f_idx in range(len(batch_prompt_ids)):
                prompt_obj = batch_prompt_objs[f_idx]
                answer = batch_answers[f_idx]
                task = batch_tasks[f_idx]
                info = batch_infos[f_idx]

                rewards_G, turns_G, completion_texts_G, turn_rewards_G, turn_advantages_G, turn_logs_G, branch_rewards_G = run_multiturn_group(
                    verifiers_env,
                    helper=helper,  # type: ignore[invalid-argument-type]
                    tokenizer=tokenizer,
                    model_name=config.model,
                    prompt=prompt_obj,
                    answer=answer,
                    task=task,
                    info=info,
                    num_rollouts=current_group_size,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    max_turns_override=config.environment_max_turns,
                    tl_grpo=config.tl_grpo,
                    tl_grpo_branch_mode=config.tl_grpo_branch_mode,
                    tl_grpo_branch_size=config.tl_grpo_branch_size,
                    tl_grpo_lookahead_steps=config.tl_grpo_lookahead_steps,
                    tl_grpo_outcome_baseline=tl_grpo_ema,
                )

                logprobs_G: list[list[float]] = []
                planning_masks_G: list[list[int]] = []
                turns_logprobs_G: list[list[list[float]]] = []
                turns_token_ids_G: list[list[list[int]]] = []
                turns_prompt_ids_G: list[list[list[int]]] = []

                for turns in turns_G:
                    seq_logprobs: list[float] = []
                    seq_token_ids: list[int] = []
                    seq_token_strs: list[str] = []
                    turn_logprobs: list[list[float]] = []
                    turn_token_ids: list[list[int]] = []
                    turn_prompt_ids: list[list[int]] = []
                    for turn in turns:
                        turn_prompt_ids.append(list(turn.prompt_ids))
                        turn_token_ids.append(list(turn.completion_ids))
                        turn_logprobs.append(list(turn.completion_logprobs))
                        seq_logprobs.extend(turn.completion_logprobs)
                        seq_token_ids.extend(turn.completion_ids)
                        for tid in turn.completion_ids:
                            seq_token_strs.append(
                                vocab_table[tid] if 0 <= tid < len(vocab_table) else ""
                            )
                    logprobs_G.append(seq_logprobs)
                    turns_logprobs_G.append(turn_logprobs)
                    turns_token_ids_G.append(turn_token_ids)
                    turns_prompt_ids_G.append(turn_prompt_ids)
                    if needs_planning:
                        planning_masks_G.append(detector.detect(seq_token_strs))  # type: ignore[unresolved-attribute]
                    else:
                        planning_masks_G.append([0] * len(seq_logprobs))

                    batch_total_completions += 1
                    if seq_token_ids and len(seq_token_ids) >= config.max_tokens:
                        batch_max_token_hits += 1

                all_logprobs_sepa.extend(logprobs_G)
                all_planning_masks_sepa.extend(planning_masks_G)

                for r in rewards_G:
                    batch_rewards.append(r)
                    if r > _CORRECT_THRESHOLD:
                        batch_correct += 1
                    if tl_grpo_ema is not None:
                        tl_grpo_ema = (
                            config.tl_grpo_ema_decay * tl_grpo_ema
                            + (1 - config.tl_grpo_ema_decay) * r
                        )

                group_correct = sum(1 for r in rewards_G if r > _CORRECT_THRESHOLD)
                answer_preview = answer[:40] if len(answer) > 40 else answer
                print(
                    f"  group: {group_correct}/{len(rewards_G)} correct "
                    f"| answer={answer_preview}"
                )

                reward_tie_stats = _summarize_reward_ties(rewards_G)
                if reward_tie_stats["eligible"]:
                    batch_reward_tie_eligible_groups += 1
                    batch_reward_tie_groups += int(reward_tie_stats["has_tie"])
                    batch_reward_uniform_groups += int(reward_tie_stats["is_uniform"])
                    batch_reward_tied_pairs += reward_tie_stats["tied_pairs"]
                    batch_reward_total_pairs += reward_tie_stats["total_pairs"]
                    batch_reward_unique_fraction_sum += (
                        reward_tie_stats["unique_count"] / len(rewards_G)
                    )

                if reward_tie_stats["is_uniform"] and not config.tl_grpo:
                    # With batch_advantage_norm, keep uniform groups — cross-group
                    # reward differences provide signal after batch normalization.
                    if config.batch_advantage_norm:
                        print(f"    -> uniform (reward={rewards_G[0]:.3f}, kept for batch norm)")
                    elif rewards_G[0] > _CORRECT_THRESHOLD:
                        print("    -> skipped (all correct)")
                        continue
                    else:
                        print(f"    -> skipped (all same, reward={rewards_G[0]:.3f})")
                        continue

                if config.algorithm_mode:
                    adv_result = compute_algorithm_advantages(
                        rewards_G,
                        logprobs_G,
                        planning_masks_G,
                        algorithm_mode=config.algorithm_mode,
                        params=step_algorithm_params,
                        gtpo_beta=config.gtpo_beta,
                        hicra_alpha=config.hicra_alpha,
                        sepa_lambda=sepa_lambda_val,
                        step=batch_idx,
                        token_distributions_G=None,
                    )
                else:
                    adv_result = compute_composable_advantages(
                        rewards_G,
                        logprobs_G,
                        planning_masks_G,
                        advantage_mode=config.advantage_mode,
                        transform_mode=config.transform_mode,
                        gtpo_beta=config.gtpo_beta,
                        hicra_alpha=config.hicra_alpha,
                        sepa_lambda=sepa_lambda_val,
                        advantage_params=config.effective_advantage_params,
                        transform_params=step_transform_params,
                        step=batch_idx,
                        post_process_params=config.post_process_params,
                        token_distributions_G=None,
                    )
                all_token_advs_G = adv_result.token_advs
                if adv_result.has_stats:
                    batch_surprisal_stats.append(adv_result.stats)
                if adv_result.extra_metrics:
                    batch_adv_results.append(adv_result)

                for s_idx in range(len(rewards_G)):
                    turn_prompt_ids = turns_prompt_ids_G[s_idx]
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
                    for t_idx in range(len(turn_token_ids)):
                        seq_tokens = turn_token_ids[t_idx]
                        seq_logprobs = turn_logprobs[t_idx]
                        prompt_ids = turn_prompt_ids[t_idx]

                        if s_turn_advs is not None:
                            # Per-turn advantage: broadcast the turn's advantage
                            # to all tokens in this turn's completion.
                            seq_advs = [s_turn_advs[t_idx]] * len(seq_tokens)
                        else:
                            # Fallback: use episode-level token advantages.
                            seq_advs = token_advs[offset : offset + len(seq_tokens)]

                        offset += len(seq_tokens)
                        full_tokens = list(prompt_ids) + list(seq_tokens)
                        padded_logprobs = [0.0] * len(prompt_ids) + list(seq_logprobs)
                        padded_advantages = [0.0] * len(prompt_ids) + list(seq_advs)
                        all_datum_tokens.append(full_tokens)
                        all_datum_logprobs.append(padded_logprobs)
                        all_datum_advantages.append(padded_advantages)

                for s_idx, comp_text in enumerate(completion_texts_G):
                    gen_entry: dict[str, object] = {
                        "step": batch_idx,
                        "prompt": batch_prompt_previews[f_idx],
                        "completion": comp_text[:500],
                        "reward": rewards_G[s_idx],
                        "num_tokens": len(logprobs_G[s_idx]),
                    }
                    if s_idx < len(turn_logs_G) and turn_logs_G[s_idx]:
                        turn_summary = []
                        for tl in turn_logs_G[s_idx]:
                            obs = tl.get("observation", {})
                            entry: dict[str, object] = {
                                "turn": tl.get("turn"),
                                "tick": obs.get("tick", 0) if isinstance(obs, dict) else 0,  # type: ignore[no-matching-overload]
                                "customer_waiting": obs.get("customer_waiting") if isinstance(obs, dict) else False,  # type: ignore[invalid-argument-type]
                                "inventory": obs.get("inventory") if isinstance(obs, dict) else 0,  # type: ignore[invalid-argument-type]
                                "operation": tl.get("operation"),
                                "reward_delta": tl.get("reward_delta", 0.0),
                                "valid": tl.get("valid", True),
                            }
                            if not tl.get("valid"):
                                entry["error"] = tl.get("error", "")
                            turn_summary.append(entry)
                            # ── Behavior accumulation ──
                            _step_behavior_turns += 1
                            if not tl.get("valid", True):
                                _step_behavior_invalid += 1
                            _op = str(tl.get("operation", "unknown"))
                            _step_behavior_actions[_op] = (
                                _step_behavior_actions.get(_op, 0) + 1
                            )
                        gen_entry["turn_log"] = turn_summary
                        _step_behavior_resp_lens.append(
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
                    if s_idx < len(logprobs_G) and logprobs_G[s_idx]:
                        lps = logprobs_G[s_idx]
                        # Flatten token IDs for this sample
                        s_tids: list[int] = []
                        for t_idx2 in range(len(turns_token_ids_G[s_idx])):
                            s_tids.extend(turns_token_ids_G[s_idx][t_idx2])
                        n_tok = min(len(lps), len(s_tids))
                        if n_tok > 0:
                            indexed = [
                                (i, -lps[i], s_tids[i])
                                for i in range(n_tok)
                            ]
                            indexed.sort(key=lambda x: x[1], reverse=True)
                            top_k = indexed[:10]
                            gen_entry["top_surprisal_tokens"] = [
                                {
                                    "pos": pos,
                                    "surprisal": round(surp, 4),
                                    "token": vocab_table[tid]
                                    if 0 <= tid < len(vocab_table)
                                    else f"<{tid}>",
                                }
                                for pos, surp, tid in top_k
                            ]
                    generations_logger.log(gen_entry)
            sample_time = time.perf_counter() - sample_start
        else:
            # 10c. Sample completions
            sample_start = time.perf_counter()
            use_entropy_sampling = (
                config.uncertainty_kind == "shannon_entropy"
                and hasattr(helper, "sample_with_entropy")
            )
            precomputed_entropies_batch: list[list[list[float]]] | None = None
            if use_entropy_sampling:
                enriched_sequences = helper.sample_with_entropy(  # type: ignore[unresolved-attribute]
                    batch_prompt_ids,
                    current_group_size,
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
                all_group_sequences = helper.sample(  # type: ignore[unresolved-attribute]
                    batch_prompt_ids,
                    current_group_size,
                    config.max_tokens,
                    config.temperature,
                    config.top_p,
                )
            sample_time = time.perf_counter() - sample_start

            # Build flat token sequences for batch decode
            all_token_seqs_flat: list[list[int]] = []
            group_flat_offsets: list[int] = []

            for group in all_group_sequences:
                group_flat_offsets.append(len(all_token_seqs_flat))
                for token_ids, _logprobs in group:
                    all_token_seqs_flat.append(list(token_ids))

            all_decoded_texts = tokenizer.batch_decode(  # type: ignore[unresolved-attribute]
                all_token_seqs_flat, skip_special_tokens=True
            )

            for f_idx, group in enumerate(all_group_sequences):
                prompt_ids = batch_prompt_ids[f_idx]
                answer = batch_answers[f_idx]
                task = batch_tasks[f_idx]
                info = batch_infos[f_idx]
                prompt_obj = batch_prompt_objs[f_idx]
                ob_len = len(prompt_ids)
                flat_offset = group_flat_offsets[f_idx]

                rewards_G: list[float] = []
                logprobs_G: list[list[float]] = []
                planning_masks_G: list[list[int]] = []
                completion_texts_G: list[str] = []
                turns_prompt_ids_G: list[list[list[int]]] = []
                turns_token_ids_G: list[list[list[int]]] = []
                turns_logprobs_G: list[list[list[float]]] = []

                for s_idx, (seq_tokens, seq_logprobs) in enumerate(group):
                    text = all_decoded_texts[flat_offset + s_idx]
                    completion_texts_G.append(text)
                    logprobs = list(seq_logprobs)
                    logprobs_G.append(logprobs)
                    turns_prompt_ids_G.append([list(prompt_ids)])
                    turns_token_ids_G.append([list(seq_tokens)])
                    turns_logprobs_G.append([logprobs])

                    if needs_planning:
                        token_strs = [
                            vocab_table[tid] if 0 <= tid < len(vocab_table) else ""
                            for tid in seq_tokens
                        ]
                        planning_masks_G.append(detector.detect(token_strs))  # type: ignore[unresolved-attribute]
                    else:
                        planning_masks_G.append([0] * len(logprobs))

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

                all_logprobs_sepa.extend(logprobs_G)
                all_planning_masks_sepa.extend(planning_masks_G)

                for r in rewards_G:
                    batch_rewards.append(r)
                    if r > _CORRECT_THRESHOLD:
                        batch_correct += 1

                for seq_tokens, _ in group:
                    batch_total_completions += 1
                    if len(seq_tokens) >= config.max_tokens:
                        batch_max_token_hits += 1

                group_correct = sum(1 for r in rewards_G if r > _CORRECT_THRESHOLD)
                answer_preview = answer[:40] if len(answer) > 40 else answer
                print(
                    f"  group: {group_correct}/{len(rewards_G)} correct "
                    f"| answer={answer_preview}"
                )

                reward_tie_stats = _summarize_reward_ties(rewards_G)
                if reward_tie_stats["eligible"]:
                    batch_reward_tie_eligible_groups += 1
                    batch_reward_tie_groups += int(reward_tie_stats["has_tie"])
                    batch_reward_uniform_groups += int(reward_tie_stats["is_uniform"])
                    batch_reward_tied_pairs += reward_tie_stats["tied_pairs"]
                    batch_reward_total_pairs += reward_tie_stats["total_pairs"]
                    batch_reward_unique_fraction_sum += (
                        reward_tie_stats["unique_count"] / len(rewards_G)
                    )

                if reward_tie_stats["is_uniform"] and not config.tl_grpo:
                    if config.batch_advantage_norm:
                        print(f"    -> uniform (reward={rewards_G[0]:.3f}, kept for batch norm)")
                    elif rewards_G[0] > _CORRECT_THRESHOLD:
                        print("    -> skipped (all correct)")
                        continue
                    else:
                        print(f"    -> skipped (all same, reward={rewards_G[0]:.3f})")
                        continue

                # Resolve per-group precomputed entropies
                group_entropies_G: list[list[float]] | None = None
                if precomputed_entropies_batch is not None:
                    group_entropies_G = precomputed_entropies_batch[f_idx]

                if config.algorithm_mode:
                    adv_result = compute_algorithm_advantages(
                        rewards_G,
                        logprobs_G,
                        planning_masks_G,
                        algorithm_mode=config.algorithm_mode,
                        params=step_algorithm_params,
                        gtpo_beta=config.gtpo_beta,
                        hicra_alpha=config.hicra_alpha,
                        sepa_lambda=sepa_lambda_val,
                        step=batch_idx,
                        token_distributions_G=None,
                        precomputed_entropies_G=group_entropies_G,
                    )
                else:
                    adv_result = compute_composable_advantages(
                        rewards_G,
                        logprobs_G,
                        planning_masks_G,
                        advantage_mode=config.advantage_mode,
                        transform_mode=config.transform_mode,
                        gtpo_beta=config.gtpo_beta,
                        hicra_alpha=config.hicra_alpha,
                        sepa_lambda=sepa_lambda_val,
                        advantage_params=config.effective_advantage_params,
                        transform_params=step_transform_params,
                        step=batch_idx,
                        post_process_params=config.post_process_params,
                        token_distributions_G=None,
                        precomputed_entropies_G=group_entropies_G,
                    )
                all_token_advs_G = adv_result.token_advs
                if adv_result.has_stats:
                    batch_surprisal_stats.append(adv_result.stats)
                if adv_result.extra_metrics:
                    batch_adv_results.append(adv_result)

                for s_idx in range(len(rewards_G)):
                    token_advs = all_token_advs_G[s_idx]
                    offset = 0
                    for t_idx in range(len(turns_token_ids_G[s_idx])):
                        seq_tokens = turns_token_ids_G[s_idx][t_idx]
                        seq_logprobs = turns_logprobs_G[s_idx][t_idx]
                        turn_prompt_ids = turns_prompt_ids_G[s_idx][t_idx]
                        seq_advs = token_advs[offset : offset + len(seq_tokens)]
                        offset += len(seq_tokens)
                        full_tokens = list(turn_prompt_ids) + list(seq_tokens)
                        padded_logprobs = [0.0] * len(turn_prompt_ids) + list(seq_logprobs)
                        padded_advantages = [0.0] * len(turn_prompt_ids) + list(seq_advs)
                        all_datum_tokens.append(full_tokens)
                        all_datum_logprobs.append(padded_logprobs)
                        all_datum_advantages.append(padded_advantages)

                for s_idx, comp_text in enumerate(completion_texts_G):
                    gen_entry: dict[str, object] = {
                        "step": batch_idx,
                        "prompt": batch_prompt_previews[f_idx],
                        "completion": comp_text[:500],
                        "reward": rewards_G[s_idx],
                        "num_tokens": len(logprobs_G[s_idx]),
                    }
                    # Top-K highest surprisal tokens with decoded text
                    if s_idx < len(logprobs_G) and logprobs_G[s_idx]:
                        lps = logprobs_G[s_idx]
                        tids = turns_token_ids_G[s_idx][0] if turns_token_ids_G[s_idx] else []
                        n_tok = min(len(lps), len(tids))
                        if n_tok > 0:
                            indexed = sorted(
                                [(i, -lps[i], tids[i]) for i in range(n_tok)],
                                key=lambda x: x[1],
                                reverse=True,
                            )
                            gen_entry["top_surprisal_tokens"] = [
                                {
                                    "pos": p,
                                    "surprisal": round(s, 4),
                                    "token": vocab_table[t]
                                    if 0 <= t < len(vocab_table)
                                    else f"<{t}>",
                                }
                                for p, s, t in indexed[:10]
                            ]
                    generations_logger.log(gen_entry)

        # 10e. SEPA state updates
        total_completions += len(batch_rewards)
        total_correct += batch_correct
        correct_rate = (
            batch_correct / len(batch_rewards) if batch_rewards else 0.0
        )

        if uses_sepa_controller:
            sepa_controller.observe_correct_rate(correct_rate)

            if sepa_controller.enabled() and sepa_controller.sepa_schedule == "auto":
                for t_idx in range(len(all_logprobs_sepa)):
                    logprobs = all_logprobs_sepa[t_idx]
                    pmask = all_planning_masks_sepa[t_idx]
                    exec_ent = [
                        -logprobs[j]
                        for j in range(len(logprobs))
                        if pmask[j] == 0
                    ]
                    sepa_controller.update_auto_state(exec_ent)

        # 10f. Train
        num_datums = len(all_datum_tokens)
        if num_datums == 0:
            print(f"Step {batch_idx}: no informative datums, skipping.")
            obs = StepObservation(
                step_time_s=time.perf_counter() - step_start,
                sample_time_s=sample_time,
                batch_size=current_batch_size,
                group_size=current_group_size,
                skipped=True,
            )
            backpressure.observe(obs)  # type: ignore[unresolved-attribute]
            continue

        print(f"Step {batch_idx}: submitting {num_datums} datums for training...")

        if not backend_caps.preserves_token_advantages:
            _assert_uniform_completion_advantages_for_non_preserving_backend(
                all_datum_logprobs,
                all_datum_advantages,
                backend_name=config.backend,
            )

        # REINFORCE++ batch normalization (before capping)
        batch_norm_metrics: dict[str, float] = {}
        if config.batch_advantage_norm:
            all_datum_advantages, batch_norm_metrics = (
                apply_batch_advantage_normalization(all_datum_advantages)
            )

        # Advantage capping (pre-training, any backend)
        adv_cap_fraction = 0.0
        adv_cap_magnitude = 0.0
        if config.adv_clip_max > 0:
            all_datum_advantages, adv_cap_fraction, adv_cap_magnitude = (
                _apply_advantage_cap(all_datum_advantages, config.adv_clip_max)
            )

        train_start = time.perf_counter()
        loss_value = helper.train_step(  # type: ignore[unresolved-attribute]
            all_datum_tokens,
            all_datum_logprobs,
            all_datum_advantages,
            config.lr,
            config.weight_decay,
        )
        train_time = time.perf_counter() - train_start
        clip_fraction = getattr(helper, '_clip_fraction', 0.0)

        step_time = time.perf_counter() - step_start

        # Back pressure
        bp_total_tokens = sum(len(t) for t in all_datum_tokens)
        obs = StepObservation(
            step_time_s=step_time,
            sample_time_s=sample_time,
            train_time_s=train_time,
            num_datums=num_datums,
            batch_size=current_batch_size,
            group_size=current_group_size,
            total_tokens=bp_total_tokens,
            loss=loss_value,
            skipped=False,
        )
        backpressure.observe(obs)  # type: ignore[unresolved-attribute]
        bp_decision = backpressure.recommend()  # type: ignore[unresolved-attribute]

        if config.bp_enabled and not bp_warmup:
            if bp_decision.action in ("throttle", "increase"):
                new_bs = bp_decision.recommended_batch_size
                new_bs = max(config.bp_min_batch_size, min(config.bp_max_batch_size, new_bs))
                if new_bs > 0:
                    current_batch_size = new_bs

        # 10g. Logging
        mean_reward = (
            sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        )
        running_correct_rate = (
            total_correct / total_completions if total_completions > 0 else 0.0
        )
        max_token_hit_rate = (
            batch_max_token_hits / batch_total_completions
            if batch_total_completions > 0
            else 0.0
        )
        reward_tie_group_rate = (
            batch_reward_tie_groups / batch_reward_tie_eligible_groups
            if batch_reward_tie_eligible_groups > 0
            else 0.0
        )
        reward_uniform_group_rate = (
            batch_reward_uniform_groups / batch_reward_tie_eligible_groups
            if batch_reward_tie_eligible_groups > 0
            else 0.0
        )
        reward_tie_pair_rate = (
            batch_reward_tied_pairs / batch_reward_total_pairs
            if batch_reward_total_pairs > 0
            else 0.0
        )
        reward_unique_fraction_mean = (
            batch_reward_unique_fraction_sum / batch_reward_tie_eligible_groups
            if batch_reward_tie_eligible_groups > 0
            else 0.0
        )

        # Aggregate entropy stats
        step_exec_mean = step_exec_var = step_plan_mean = step_plan_var = 0.0
        step_post_exec_mean = step_post_exec_var = 0.0
        step_post_plan_mean = step_post_plan_var = 0.0
        if batch_surprisal_stats:
            n_stats = len(batch_surprisal_stats)
            step_exec_mean = sum(s.exec_mean for s in batch_surprisal_stats) / n_stats
            step_exec_var = sum(s.exec_var for s in batch_surprisal_stats) / n_stats
            step_plan_mean = sum(s.plan_mean for s in batch_surprisal_stats) / n_stats
            step_plan_var = sum(s.plan_var for s in batch_surprisal_stats) / n_stats
            step_post_exec_mean = sum(s.post_exec_mean for s in batch_surprisal_stats) / n_stats
            step_post_exec_var = sum(s.post_exec_var for s in batch_surprisal_stats) / n_stats
            step_post_plan_mean = sum(s.post_plan_mean for s in batch_surprisal_stats) / n_stats
            step_post_plan_var = sum(s.post_plan_var for s in batch_surprisal_stats) / n_stats

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
            "reward_tie_eligible_groups": batch_reward_tie_eligible_groups,
            "reward_tie_groups": batch_reward_tie_groups,
            "reward_uniform_groups": batch_reward_uniform_groups,
            "reward_tie_group_rate": reward_tie_group_rate,
            "reward_uniform_group_rate": reward_uniform_group_rate,
            "reward_tie_pair_rate": reward_tie_pair_rate,
            "reward_unique_fraction_mean": reward_unique_fraction_mean,
            "sepa_lambda": sepa_lambda_val,
            "sepa_gate_open": sepa_gate,
            "num_datums": num_datums,
            "max_token_hit_rate": max_token_hit_rate,
            "step_time_s": step_time,
            "sample_time_s": sample_time,
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
        if batch_surprisal_stats:
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
        metrics["adv_cap_fraction"] = adv_cap_fraction
        metrics["adv_cap_magnitude"] = adv_cap_magnitude
        if batch_norm_metrics:
            metrics.update(batch_norm_metrics)
        if batch_adv_results:
            all_extra_keys = {k for r in batch_adv_results for k in r.extra_metrics}
            for k in all_extra_keys:
                vals = [r.extra_metrics[k] for r in batch_adv_results if k in r.extra_metrics]
                metrics[k] = sum(vals) / len(vals)
        if "dg_eta" in metrics:
            delight_eta_ema = float(metrics["dg_eta"])

        # ── Behavior monitoring ─────────────────────────────────────
        # Aggregate turn-level behavior from this step's generations.
        # Tracks model behavior drift that loss alone cannot detect:
        # action collapse, invalid action rate, response length.
        if _step_behavior_turns > 0:
            metrics["behavior/invalid_action_rate"] = (
                _step_behavior_invalid / _step_behavior_turns
            )
            metrics["behavior/action_type_count"] = len(_step_behavior_actions)
            if _step_behavior_actions:
                _act_total = sum(_step_behavior_actions.values())
                _max_frac = max(_step_behavior_actions.values()) / _act_total
                metrics["behavior/action_dominance"] = _max_frac
        if _step_behavior_resp_lens:
            metrics["behavior/avg_response_chars"] = (
                sum(_step_behavior_resp_lens) / len(_step_behavior_resp_lens)
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
            f" | tie_g={reward_tie_group_rate * 100:.1f}%"
            f" | sepa_l={sepa_lambda_val:.4f}"
            f" | time={step_time:.1f}s"
        )

        # Wandb
        if wandb_enabled and wandb_run is not None:
            wandb_metrics: dict[str, int | float | str] = {
                "train/loss": loss_value,
                "train/reported_loss": loss_value,
                "train/uncertainty_kind": config.uncertainty_kind,
                "train/loss_is_placeholder": int(not backend_caps.reports_sync_loss),
                "train/rewards/mean_reward": mean_reward,
                "train/rewards/correct_rate": correct_rate,
                "train/rewards/running_correct_rate": running_correct_rate,
                "train/rewards/tie_eligible_groups": batch_reward_tie_eligible_groups,
                "train/rewards/tie_groups": batch_reward_tie_groups,
                "train/rewards/uniform_groups": batch_reward_uniform_groups,
                "train/rewards/tie_group_rate": reward_tie_group_rate,
                "train/rewards/uniform_group_rate": reward_uniform_group_rate,
                "train/rewards/tie_pair_rate": reward_tie_pair_rate,
                "train/rewards/unique_fraction_mean": reward_unique_fraction_mean,
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
            if batch_adv_results:
                for k in {k for r in batch_adv_results for k in r.extra_metrics}:
                    wandb_metrics[f"train/{k}"] = metrics.get(k, 0.0)
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
            "correct_count": batch_correct,
            "total_count": len(batch_rewards),
            "condition": condition_label,
            "uncertainty_kind": config.uncertainty_kind,
        }
        if batch_surprisal_stats:
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
        step_entry["adv_cap_fraction"] = adv_cap_fraction
        step_entry["adv_cap_magnitude"] = adv_cap_magnitude
        if batch_adv_results:
            for k in {k for r in batch_adv_results for k in r.extra_metrics}:
                step_entry[k] = metrics.get(k, 0.0)
        steps_logger.log(step_entry)

        # Periodic checkpoint
        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            ckpt_name = f"checkpoint_step_{batch_idx + 1}"
            checkpoint_path = helper.save_adapter(  # type: ignore[unresolved-attribute]
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
    final_path = helper.save_adapter(config.adapter_path, "final")  # type: ignore[unresolved-attribute]
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

    if wandb_enabled and wandb_run is not None:
        wandb_run.finish()

    return final_path
