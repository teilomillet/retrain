"""Main training loop -- calls LocalTrainHelper directly, no Mojo.

Ports the training loop from src/main.mojo into pure Python.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from retrain.advantages import (
    DEFAULT_STRATEGIC_GRAMS,
    EntropyStats,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_entropy_stats,
    compute_grpo_advantages,
    compute_maxrl_advantages,
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
from retrain.local_train_helper import LocalTrainHelper
from retrain.logging_utils import JsonlLogger
from retrain.rewards import BoxedMathReward
from retrain.sepa import SEPAController


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
    # 7. Init LocalTrainHelper
    # -----------------------------------------------------------------------
    if config.backend != "local":
        raise NotImplementedError(
            f"Backend '{config.backend}' not implemented in Python CLI. "
            "Use the Mojo binary for tinker backend."
        )

    helper = LocalTrainHelper(
        config.model,
        config.adapter_path,
        config.devices,
        config.lora_rank,
        config.inference_engine,
        config.inference_url,
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
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
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
                "max_steps": config.max_steps,
                "backend": config.backend,
            },
        )
        print(f"Wandb initialized: {config.wandb_project}/{run_name}")

    # -----------------------------------------------------------------------
    # 10. Training loop
    # -----------------------------------------------------------------------
    reward_fn = BoxedMathReward()
    example_idx = 0
    total_correct = 0
    total_completions = 0
    sepa_lambda_val = 0.0
    current_batch_size = config.batch_size
    current_group_size = config.group_size
    needs_planning = config.transform_mode in ("gtpo_hicra", "gtpo_sepa")

    # Warmup sweep schedule: geometric [1,2,4,...] clamped to [min, max]
    warmup_batch_sizes: list[int] = []
    if config.bp_enabled:
        bs = max(1, config.bp_min_batch_size)
        while bs <= config.bp_max_batch_size:
            warmup_batch_sizes.append(bs)
            bs *= 2
        if warmup_batch_sizes and warmup_batch_sizes[-1] != config.bp_max_batch_size:
            warmup_batch_sizes.append(config.bp_max_batch_size)

    for batch_idx in range(config.max_steps):
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
            0.95,
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
                if r > 0.5:
                    batch_correct += 1

            # Track max-token truncations
            for seq_tokens, _ in group:
                batch_total_completions += 1
                if len(seq_tokens) >= config.max_tokens:
                    batch_max_token_hits += 1

            group_correct = sum(1 for r in rewards_G if r > 0.5)
            answer_preview = answer[:40] if len(answer) > 40 else answer
            print(
                f"  group: {group_correct}/{len(rewards_G)} correct "
                f"| answer={answer_preview}"
            )

            # Skip uninformative groups
            if rewards_G and all(r == rewards_G[0] for r in rewards_G):
                if rewards_G[0] > 0.5:
                    print("    -> skipped (all correct)")
                else:
                    print("    -> skipped (all wrong)")
                continue

            # Update SEPA lambda
            if config.transform_mode == "gtpo_sepa":
                sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

            # Episode-level advantages
            if config.advantage_mode == "maxrl":
                advantages_G = compute_maxrl_advantages(rewards_G)
            else:
                advantages_G = compute_grpo_advantages(rewards_G)

            # Token-level expansion
            all_token_advs_G: list[list[float]] = []
            all_exec_ent: list[float] = []
            all_plan_ent: list[float] = []

            if config.transform_mode == "none":
                for i in range(n_seqs):
                    all_token_advs_G.append(
                        [advantages_G[i]] * len(logprobs_G[i])
                    )
            else:
                for i in range(n_seqs):
                    entropies = [-lp for lp in logprobs_G[i]]
                    pmask = planning_masks_G[i]

                    for j, e in enumerate(entropies):
                        if pmask[j]:
                            all_plan_ent.append(e)
                        else:
                            all_exec_ent.append(e)

                    if config.transform_mode == "gtpo_sepa" and sepa_lambda_val > 0.0:
                        entropies = apply_sepa_pooling(entropies, pmask, sepa_lambda_val)

                    token_advs = apply_gtpo_weighting(
                        advantages_G[i], entropies, beta=config.gtpo_beta
                    )

                    if config.transform_mode == "gtpo_hicra":
                        token_advs = apply_hicra(
                            token_advs, pmask, alpha=config.hicra_alpha
                        )

                    all_token_advs_G.append(token_advs)

                if all_exec_ent or all_plan_ent:
                    batch_entropy_stats.append(
                        compute_entropy_stats(all_exec_ent, all_plan_ent)
                    )

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

        if config.transform_mode == "gtpo_sepa":
            sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

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
            wandb_run.log(
                {
                    "step": batch_idx,
                    "loss": loss_value,
                    "mean_reward": mean_reward,
                    "correct_rate": correct_rate,
                    "running_correct_rate": running_correct_rate,
                    "sepa_lambda": sepa_lambda_val,
                    "num_datums": num_datums,
                    "step_time_s": step_time,
                    "batch_size": current_batch_size,
                    "group_size": current_group_size,
                    "bp_warmup": bp_warmup,
                    "bp_p_star": bp_decision.p_star,
                    "bp_sigma": bp_decision.sigma,
                    "bp_kappa": bp_decision.kappa,
                    "bp_utilization": bp_decision.utilization,
                    "bp_throughput": bp_decision.throughput,
                },
                step=batch_idx,
            )

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
            print(f"Saved checkpoint: {ckpt_name}")

    # -----------------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------------
    helper.save_adapter(config.adapter_path, "final")
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
