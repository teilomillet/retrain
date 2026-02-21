"""Main training loop -- calls LocalTrainHelper directly, no Mojo.

Ports the training loop from src/main.mojo into pure Python.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from retrain.advantages import (
    DEFAULT_STRATEGIC_GRAMS,
    EntropyStats,
    compute_composable_advantages,
    identify_planning_tokens,
)
from retrain.backpressure import (
    BackPressureDecision,
    NoOpBackPressure,
    StepObservation,
    USLBackPressure,
)
from retrain.config import TrainConfig
from retrain.data import MathDataSource
from retrain.logging_utils import JsonlLogger
from retrain.rewards import create_reward
from retrain.sepa import SEPAController


_TRAINER_STATE_FILE = "trainer_state.json"
_CORRECT_THRESHOLD = 0.5


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
    sepa_state: dict[str, Any],
) -> None:
    """Write trainer-side state to JSON for checkpoint resume."""
    state = {
        "step": step,
        "example_idx": example_idx,
        "total_correct": total_correct,
        "total_completions": total_completions,
        "current_batch_size": current_batch_size,
        "current_group_size": current_group_size,
        "checkpoint_name": checkpoint_name,
        "sepa": sepa_state,
    }
    tmp = path / f"{_TRAINER_STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.rename(path / _TRAINER_STATE_FILE)


def _load_trainer_state(resume_dir: str) -> dict[str, Any]:
    """Load trainer state from a checkpoint directory."""
    p = Path(resume_dir)
    state_file = p / _TRAINER_STATE_FILE
    if not state_file.is_file():
        raise FileNotFoundError(
            f"No {_TRAINER_STATE_FILE} found in {resume_dir}. "
            f"Cannot resume without trainer state."
        )
    return json.loads(state_file.read_text())


def train(config: TrainConfig) -> None:
    """Main training loop -- fully self-contained."""

    # -----------------------------------------------------------------------
    # 1. Setup directories + loggers
    # -----------------------------------------------------------------------
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    emergence_dir = log_path / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlLogger(str(log_path / "metrics.jsonl"))
    steps_logger = JsonlLogger(str(emergence_dir / "steps.jsonl"))
    generations_logger = JsonlLogger(str(emergence_dir / "generations.jsonl"))

    # -----------------------------------------------------------------------
    # 1b. Seed for reproducibility
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
    # 2. Load tokenizer + vocab table
    # -----------------------------------------------------------------------
    print(f"Loading tokenizer for {config.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Loading vocabulary table...")
    vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, "added_tokens_encoder"):
        vocab_size += len(tokenizer.added_tokens_encoder)
    all_ids = list(range(vocab_size))
    vocab_table: list[str] = []
    py_tokens = tokenizer.convert_ids_to_tokens(all_ids)
    for tok in py_tokens:
        vocab_table.append(str(tok) if tok is not None else "")
    print(f"Vocabulary table: {len(vocab_table)} entries")

    # -----------------------------------------------------------------------
    # 3. Load dataset
    # -----------------------------------------------------------------------
    print("Loading dataset...")
    examples = MathDataSource(config.max_examples).load()
    if not examples:
        raise RuntimeError("Dataset is empty — cannot train with zero examples.")
    print(f"Loaded {len(examples)} examples")

    # -----------------------------------------------------------------------
    # 4. Pre-encode all prompts
    # -----------------------------------------------------------------------
    print("Pre-encoding prompts...")
    pre_encoded_prompts: list[list[int]] = []
    for ex in examples:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": ex.prompt}]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        else:
            ids = tokenizer.encode(ex.prompt)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        pre_encoded_prompts.append(list(ids))
    print(f"Pre-encoded {len(pre_encoded_prompts)} prompts")

    # -----------------------------------------------------------------------
    # 5. SEPA controller
    # -----------------------------------------------------------------------
    sepa_controller = SEPAController(
        sepa_steps=config.sepa_steps,
        sepa_schedule=config.sepa_schedule,
        sepa_delay_steps=config.sepa_delay_steps,
        sepa_correct_rate_gate=config.sepa_correct_rate_gate,
    )

    # -----------------------------------------------------------------------
    # 6. Strategic grams
    # -----------------------------------------------------------------------
    strategic_grams: list[str]
    if config.strategic_grams:
        raw = config.strategic_grams
        if raw.startswith("["):
            strategic_grams = [g.strip() for g in json.loads(raw) if g.strip()]
        else:
            strategic_grams = [g.strip() for g in raw.split(",") if g.strip()]
    else:
        strategic_grams = list(DEFAULT_STRATEGIC_GRAMS)

    # -----------------------------------------------------------------------
    # 7. Init backend (local or tinker)
    # -----------------------------------------------------------------------
    if config.backend == "tinker":
        from retrain.tinker_backend import TinkerTrainHelper

        helper = TinkerTrainHelper(
            config.model,
            config.inference_url,
            config.lora_rank,
            optim_beta1=config.optim_beta1,
            optim_beta2=config.optim_beta2,
            optim_eps=config.optim_eps,
        )
    elif config.backend == "local":
        from retrain.local_train_helper import LocalTrainHelper

        helper = LocalTrainHelper(
            config.model,
            config.adapter_path,
            config.devices,
            config.lora_rank,
            config.inference_engine,
            config.inference_url,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            optim_beta1=config.optim_beta1,
            optim_beta2=config.optim_beta2,
            optim_eps=config.optim_eps,
        )
    else:
        raise ValueError(
            f"Unknown backend '{config.backend}'. Use 'local' or 'tinker'."
        )

    # -----------------------------------------------------------------------
    # 8. Back pressure
    # -----------------------------------------------------------------------
    if config.bp_enabled:
        backpressure: NoOpBackPressure | USLBackPressure = USLBackPressure(
            warmup_steps=config.bp_warmup_steps,
            ema_decay=config.bp_ema_decay,
            throttle_margin=config.bp_throttle_margin,
            increase_margin=config.bp_increase_margin,
            min_batch_size=config.bp_min_batch_size,
            max_batch_size=config.bp_max_batch_size,
            peak_gflops=config.bp_peak_gflops,
            peak_bw_gb_s=config.bp_peak_bw_gb_s,
        )
    else:
        backpressure = NoOpBackPressure()

    # -----------------------------------------------------------------------
    # 9. Optional wandb
    # -----------------------------------------------------------------------
    wandb_run = None
    wandb_enabled = bool(config.wandb_project)
    if wandb_enabled:
        import wandb

        condition_label = f"{config.advantage_mode}+{config.transform_mode}"
        run_name = config.wandb_run_name or condition_label
        wandb_tags = (
            [t.strip() for t in config.wandb_tags.split(",") if t.strip()]
            if config.wandb_tags
            else None
        )
        wandb_kwargs: dict[str, Any] = {
            "project": config.wandb_project,
            "name": run_name,
            "config": {
                "advantage_mode": config.advantage_mode,
                "transform_mode": config.transform_mode,
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
            },
        }
        if config.wandb_entity:
            wandb_kwargs["entity"] = config.wandb_entity
        if config.wandb_group:
            wandb_kwargs["group"] = config.wandb_group
        if wandb_tags:
            wandb_kwargs["tags"] = wandb_tags
        wandb_run = wandb.init(**wandb_kwargs)
        print(f"Wandb initialized: {config.wandb_project}/{run_name}")

    # -----------------------------------------------------------------------
    # 10. Training loop
    # -----------------------------------------------------------------------
    reward_fn = create_reward(config)
    example_idx = 0
    total_correct = 0
    total_completions = 0
    sepa_lambda_val = 0.0
    current_batch_size = config.batch_size
    current_group_size = config.group_size
    needs_planning = config.transform_mode in ("gtpo_hicra", "gtpo_sepa")
    start_step = 0

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

        # Restore backend model state
        ckpt_name = saved.get("checkpoint_name", "")
        if ckpt_name:
            helper.load_state(ckpt_name)

        print(
            f"Resumed from step {saved['step']} "
            f"(checkpoint: {ckpt_name}), continuing from step {start_step}"
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

    for batch_idx in range(start_step, config.max_steps):
        step_start = time.perf_counter()

        # Back pressure warmup sweep
        bp_warmup = False
        if config.bp_enabled and warmup_batch_sizes and batch_idx < config.bp_warmup_steps:
            bp_warmup = True
            current_batch_size = warmup_batch_sizes[batch_idx % len(warmup_batch_sizes)]

        # 10a. Checkpoint for sampling
        helper.checkpoint(f"step_{batch_idx}")

        # 10b. Select prompts
        batch_prompts: list[str] = []
        batch_prompt_ids: list[list[int]] = []
        batch_answers: list[str] = []

        for _ in range(current_batch_size):
            ex_idx = example_idx % len(examples)
            example_idx += 1
            batch_prompts.append(examples[ex_idx].prompt)
            batch_prompt_ids.append(list(pre_encoded_prompts[ex_idx]))
            batch_answers.append(examples[ex_idx].reference)

        # 10c. Sample completions
        sample_start = time.perf_counter()
        all_group_sequences = helper.sample(
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

        all_decoded_texts = tokenizer.batch_decode(
            all_token_seqs_flat, skip_special_tokens=True
        )

        # 10d. Process groups, compute advantages
        batch_rewards: list[float] = []
        batch_correct = 0
        batch_max_token_hits = 0
        batch_total_completions = 0
        batch_entropy_stats: list[EntropyStats] = []
        all_logprobs_sepa: list[list[float]] = []
        all_planning_masks_sepa: list[list[int]] = []
        all_datum_tokens: list[list[int]] = []
        all_datum_logprobs: list[list[float]] = []
        all_datum_advantages: list[list[float]] = []

        # Resolve SEPA lambda once per step (before group loop)
        if config.transform_mode == "gtpo_sepa":
            sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

        for f_idx, group in enumerate(all_group_sequences):
            prompt_ids = batch_prompt_ids[f_idx]
            answer = batch_answers[f_idx]
            ob_len = len(prompt_ids)
            flat_offset = group_flat_offsets[f_idx]
            n_seqs = len(group)

            # Compute rewards + planning masks
            rewards_G: list[float] = []
            logprobs_G: list[list[float]] = []
            planning_masks_G: list[list[int]] = []

            for s_idx, (seq_tokens, seq_logprobs) in enumerate(group):
                text = all_decoded_texts[flat_offset + s_idx]
                reward = reward_fn.score(text, answer)
                rewards_G.append(reward)
                logprobs_G.append(list(seq_logprobs))

                if needs_planning:
                    token_strs = [
                        vocab_table[tid] if 0 <= tid < len(vocab_table) else ""
                        for tid in seq_tokens
                    ]
                    planning_masks_G.append(
                        identify_planning_tokens(token_strs, strategic_grams)
                    )
                else:
                    planning_masks_G.append([0] * len(seq_tokens))

            # Accumulate for SEPA state update
            all_logprobs_sepa.extend(logprobs_G)
            all_planning_masks_sepa.extend(planning_masks_G)

            for r in rewards_G:
                batch_rewards.append(r)
                if r > _CORRECT_THRESHOLD:
                    batch_correct += 1

            # Track max-token truncations
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

            # Skip uninformative groups
            if rewards_G and all(r == rewards_G[0] for r in rewards_G):
                if rewards_G[0] > _CORRECT_THRESHOLD:
                    print("    -> skipped (all correct)")
                else:
                    print("    -> skipped (all wrong)")
                continue

            # Composable advantage pipeline
            adv_result = compute_composable_advantages(
                rewards_G,
                logprobs_G,
                planning_masks_G,
                advantage_mode=config.advantage_mode,
                transform_mode=config.transform_mode,
                gtpo_beta=config.gtpo_beta,
                hicra_alpha=config.hicra_alpha,
                sepa_lambda=sepa_lambda_val,
            )
            all_token_advs_G = adv_result.token_advs
            if adv_result.has_stats:
                batch_entropy_stats.append(adv_result.stats)

            # Build datums
            for s_idx, (seq_tokens, seq_logprobs) in enumerate(group):
                full_tokens = list(prompt_ids) + list(seq_tokens)
                padded_logprobs = [0.0] * ob_len + list(seq_logprobs)
                padded_advantages = [0.0] * ob_len + all_token_advs_G[s_idx]
                all_datum_tokens.append(full_tokens)
                all_datum_logprobs.append(padded_logprobs)
                all_datum_advantages.append(padded_advantages)

            # Per-generation emergence logging
            for s_idx, (seq_tokens, _) in enumerate(group):
                comp_text = all_decoded_texts[flat_offset + s_idx]
                generations_logger.log({
                    "step": batch_idx,
                    "prompt": batch_prompts[f_idx][:200],
                    "completion": comp_text[:500],
                    "reward": rewards_G[s_idx],
                    "num_tokens": len(seq_tokens),
                })

        # 10e. SEPA state updates
        total_completions += len(batch_rewards)
        total_correct += batch_correct
        correct_rate = (
            batch_correct / len(batch_rewards) if batch_rewards else 0.0
        )

        if config.transform_mode == "gtpo_sepa":
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
            backpressure.observe(obs)
            continue

        print(f"Step {batch_idx}: submitting {num_datums} datums for training...")

        train_start = time.perf_counter()
        loss_value = helper.train_step(
            all_datum_tokens,
            all_datum_logprobs,
            all_datum_advantages,
            config.lr,
            config.weight_decay,
        )
        train_time = time.perf_counter() - train_start

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

        # Aggregate entropy stats
        step_exec_mean = step_exec_var = step_plan_mean = step_plan_var = 0.0
        if batch_entropy_stats:
            n_stats = len(batch_entropy_stats)
            step_exec_mean = sum(s.exec_mean for s in batch_entropy_stats) / n_stats
            step_exec_var = sum(s.exec_var for s in batch_entropy_stats) / n_stats
            step_plan_mean = sum(s.plan_mean for s in batch_entropy_stats) / n_stats
            step_plan_var = sum(s.plan_var for s in batch_entropy_stats) / n_stats

        condition_label = f"{config.advantage_mode}+{config.transform_mode}"
        sepa_gate = (
            sepa_controller.gate_open()
            if config.transform_mode == "gtpo_sepa"
            else False
        )

        metrics: dict = {
            "step": batch_idx,
            "advantage_mode": config.advantage_mode,
            "transform_mode": config.transform_mode,
            "condition": condition_label,
            "loss": loss_value,
            "mean_reward": mean_reward,
            "correct_rate": correct_rate,
            "running_correct_rate": running_correct_rate,
            "sepa_lambda": sepa_lambda_val,
            "sepa_gate_open": sepa_gate,
            "num_datums": num_datums,
            "max_token_hit_rate": max_token_hit_rate,
            "step_time_s": step_time,
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
        if batch_entropy_stats:
            metrics["exec_entropy_mean"] = step_exec_mean
            metrics["exec_entropy_var"] = step_exec_var
            metrics["plan_entropy_mean"] = step_plan_mean
            metrics["plan_entropy_var"] = step_plan_var
        metrics_logger.log(metrics)

        print(
            f"Step {batch_idx} [{condition_label}] | loss={loss_value:.4f}"
            f" | reward={mean_reward:.3f}"
            f" | correct={correct_rate * 100:.1f}%"
            f" | datums={num_datums}"
            f" | bs={current_batch_size}"
            f" | gs={current_group_size}"
            f" | sepa_l={sepa_lambda_val:.4f}"
            f" | time={step_time:.1f}s"
        )

        # Wandb
        if wandb_enabled and wandb_run is not None:
            wandb_metrics: dict[str, Any] = {
                "train/loss": loss_value,
                "train/rewards/mean_reward": mean_reward,
                "train/rewards/correct_rate": correct_rate,
                "train/rewards/running_correct_rate": running_correct_rate,
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
                "train/backpressure/action": bp_decision.action,
                "train/backpressure/regime": bp_decision.regime,
                "train/backpressure/p_star": bp_decision.p_star,
                "train/backpressure/sigma": bp_decision.sigma,
                "train/backpressure/kappa": bp_decision.kappa,
                "train/backpressure/utilization": bp_decision.utilization,
                "train/backpressure/throughput": bp_decision.throughput,
                "train/backpressure/warmup": int(bp_warmup),
            }
            wandb_run.log(wandb_metrics, step=batch_idx)

        # Step record for emergence analysis
        step_entry: dict = {
            "step": batch_idx,
            "mean_reward": mean_reward,
            "correct_count": batch_correct,
            "total_count": len(batch_rewards),
            "condition": condition_label,
        }
        if batch_entropy_stats:
            step_entry["exec_entropy_mean"] = step_exec_mean
            step_entry["exec_entropy_var"] = step_exec_var
            step_entry["plan_entropy_mean"] = step_plan_mean
            step_entry["plan_entropy_var"] = step_plan_var
        steps_logger.log(step_entry)

        # Periodic checkpoint
        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            ckpt_name = f"checkpoint_step_{batch_idx + 1}"
            helper.save_adapter(config.adapter_path, ckpt_name)
            _save_trainer_state(
                log_path,
                step=batch_idx,
                example_idx=example_idx,
                total_correct=total_correct,
                total_completions=total_completions,
                current_batch_size=current_batch_size,
                current_group_size=current_group_size,
                checkpoint_name=ckpt_name,
                sepa_state=sepa_controller.state_dict(),
            )
            print(f"Saved checkpoint: {ckpt_name}")

    # -----------------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------------
    helper.save_adapter(config.adapter_path, "final")
    _save_trainer_state(
        log_path,
        step=config.max_steps - 1,
        example_idx=example_idx,
        total_correct=total_correct,
        total_completions=total_completions,
        current_batch_size=current_batch_size,
        current_group_size=current_group_size,
        checkpoint_name="final",
        sepa_state=sepa_controller.state_dict(),
    )
    final_rate = (
        100.0 * total_correct / total_completions if total_completions > 0 else 0.0
    )
    print(
        f"Training complete. {config.advantage_mode}+{config.transform_mode}, "
        f"{config.max_steps} steps, running correct rate: {final_rate:.1f}%"
    )
    print(f"Metrics saved to {log_path / 'metrics.jsonl'}")

    if wandb_enabled and wandb_run is not None:
        wandb_run.finish()
