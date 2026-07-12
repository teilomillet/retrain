"""Main training loop: sample rollouts, score, compute advantages, train.

Backend-agnostic — drives any TrainHelper (local, Unsloth, Tinker, ...).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

from transformers import AutoTokenizer

from retrain.advantages import apply_batch_advantage_normalization
from retrain.training.backpressure import (
    StepObservation,
)
from retrain.training.batch_digest import logical_optimizer_batch_sha256
from retrain.training.optimizer_batch import (
    OptimizerBatch,
    preflight_optimizer_batch_capture,
    save_optimizer_batch_capture,
)
from retrain.config import TrainConfig
from retrain.config.readiness import assert_readiness_runtime_matches_file
from retrain.training.echo import (
    assert_echo_live_observation_contract,
    run_rl_echo_train_step,
)
from retrain.training.flow import (
    TrainingFlow,
    _condition_label,
    build_flow,
)
from retrain.training.examples import load_training_examples
from retrain.training.prompts import select_prompt_batch
from retrain.training import console
from retrain.io.log import JsonlLogger
from retrain.registry.builtin import get_registry
from retrain.training.rollouts import (
    ExamplePromptCache,
    RuntimeCounters,
    TokenTextLookup,
)
from retrain.training.rollout import (
    RolloutAccumulator,
    has_nonzero_advantage,
    prepare_echo_step_plan,
    run_multiturn,
    run_singleturn,
)
from retrain.training.log import (
    StepLoggingContext,
    WandbRunLike,
    init_wandb,
    record_training_step,
)
from retrain.training.recoverability import (
    announce_checkpoint_recoverability,
    upload_checkpoint_artifact,
)
from retrain.training.resume import contract_for_capabilities
from retrain.training.telemetry import RolloutTelemetry
from retrain.training.signals import (
    apply_advantage_cap,
    assert_uniform_completion_advantages_for_non_preserving_backend,
    prepare_algorithm_params_for_step,
    prepare_transform_params_for_step,
)
from retrain.training.state import load_trainer_state, save_trainer_state
from retrain.training.warmup import (
    load_sft_warmup_data,
    run_sft_warmup_step,
    verify_sft_warmup_resume_schedule,
)
from retrain.environments.verifiers import (
    encode_prompt_for_sampling,
    is_multiturn_environment,
    load_examples_from_environment,
    load_verifiers_environment,
    prompt_preview,
    run_multiturn_group,
    score_singleturn_group,
)


def train(config: TrainConfig, flow: TrainingFlow | None = None) -> str | None:
    """Main training loop -- fully self-contained. Returns final adapter path."""

    assert_readiness_runtime_matches_file(config)

    capture_initial_adapter = (
        preflight_optimizer_batch_capture(config)
        if config.optimizer_batch_capture
        else None
    )

    console.print_config_summary(config)

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
        console.print_flow_warnings(trace_result)

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
        loaded_examples = load_training_examples(
            config,
            data_source_factory=lambda source, cfg: get_registry("data_source").create(
                source, cfg
            ),
            verifiers_environment_loader=load_verifiers_environment,
            verifiers_examples_loader=load_examples_from_environment,
        )
        examples = loaded_examples.examples
        verifiers_env = loaded_examples.environment
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
        console.print_backend_capability_summary(
            config.backend,
            flow.backend_capability_source,
            backend_caps.reports_sync_loss,
            backend_caps.preserves_token_advantages,
            backend_caps.supports_checkpoint_resume,
            backend_caps.resume_runtime_dependent,
            backend_caps.checkpoint_resume_mode,
        )
        resume_contract = contract_for_capabilities(backend_caps)
        print(f"Checkpoint resume mode: {resume_contract.mode}")
        if resume_contract.warning:
            print(f"Checkpoint resume warning: {resume_contract.warning}")

        # -----------------------------------------------------------------------
        # 4. Load tokenizer + lazy token/prompt caches
        # -----------------------------------------------------------------------
        print(f"Loading tokenizer for {config.model} ...")
        from retrain.training.sft_data import sft_tokenizer_load_kwargs

        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            **sft_tokenizer_load_kwargs(config),
        )
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
        condition_label = _condition_label(config)
        wandb_run = init_wandb(config, condition_label=condition_label)
        announce_checkpoint_recoverability(config, wandb_run)

        # -----------------------------------------------------------------------
        # 9. Training loop
        # -----------------------------------------------------------------------
        reward_fn = None
        if verifiers_env is None:
            reward_fn = get_registry("reward").create(config.reward_type, config)
        verifiers_multiturn = verifiers_env is not None and is_multiturn_environment(
            verifiers_env
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
        optimizer_batch_captured = False

        # Load and bind warmup data before restoring a checkpoint. A resume
        # that would re-enter warmup must prove the same tokenized traversal
        # before the backend mutates or loads model state.
        sft_data = load_sft_warmup_data(config, tokenizer)

        # -----------------------------------------------------------------------
        # 10b. Resume from checkpoint (if requested)
        # -----------------------------------------------------------------------
        if config.resume_from:
            saved = load_trainer_state(config.resume_from)
            start_step = saved["step"] + 1
            example_idx = saved["example_idx"]
            total_correct = saved["total_correct"]
            total_completions = saved["total_completions"]
            current_batch_size = saved["current_batch_size"]
            current_group_size = saved["current_group_size"]
            verify_sft_warmup_resume_schedule(
                saved.get("sft_schedule"),
                sft_data,
                start_step=start_step,
                warmup_steps=config.sft_warmup_steps,
            )

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
            if (
                warmup_batch_sizes
                and warmup_batch_sizes[-1] != config.bp_max_batch_size
            ):
                warmup_batch_sizes.append(config.bp_max_batch_size)

        for batch_idx in range(start_step, config.max_steps):
            step_start = time.perf_counter()

            # =================================================================
            # SFT warmup: supervised training from oracle demonstrations
            # =================================================================
            if batch_idx < config.sft_warmup_steps and sft_data and sft_data.examples:
                warmup_checkpoint = run_sft_warmup_step(
                    helper,
                    config,
                    sft_data,
                    batch_idx,
                    metrics_logger=metrics_logger,
                    steps_logger=steps_logger,
                    wandb_run=wandb_run,
                )
                if warmup_checkpoint is not None:
                    checkpoint_name, checkpoint_path = warmup_checkpoint
                    save_trainer_state(
                        log_path,
                        step=batch_idx,
                        example_idx=example_idx,
                        total_correct=total_correct,
                        total_completions=total_completions,
                        current_batch_size=current_batch_size,
                        current_group_size=current_group_size,
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=checkpoint_path,
                        resume_mode=resume_contract.mode,
                        resume_warning=resume_contract.warning,
                        sepa_state=sepa_controller.state_dict(),
                        tl_grpo_ema=tl_grpo_ema,
                        delight_eta_ema=delight_eta_ema,
                        sft_schedule=sft_data.schedule_contract,
                    )
                    upload_checkpoint_artifact(
                        config,
                        wandb_run,
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=checkpoint_path,
                        step=batch_idx,
                    )
                continue  # Skip the RL pipeline for this step

            # Back pressure warmup sweep
            bp_warmup = False
            if (
                config.bp_enabled
                and warmup_batch_sizes
                and batch_idx < config.bp_warmup_steps
            ):
                bp_warmup = True
                current_batch_size = warmup_batch_sizes[
                    batch_idx % len(warmup_batch_sizes)
                ]

            # 10a. Checkpoint for sampling
            helper.checkpoint(f"step_{batch_idx}")

            # 10b. Select prompts
            prompts, example_idx = select_prompt_batch(
                examples, prompt_cache, example_idx, current_batch_size
            )

            # 10d. Rollout accumulation + post-rollout ECHO state
            acc = RolloutAccumulator(tl_grpo_ema=tl_grpo_ema)
            echo_loss = 0.0
            echo_train_time = 0.0
            echo_joint_optimizer_step = False
            step_transform_params = prepare_transform_params_for_step(
                config.transform_params,
                delight_eta_prev=delight_eta_ema,
            )
            step_algorithm_params = prepare_algorithm_params_for_step(
                config.effective_algorithm_params,
                delight_eta_prev=delight_eta_ema,
            )

            # Resolve SEPA lambda once per step (before group loop)
            if uses_sepa_controller:
                sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

            if verifiers_multiturn:
                run_multiturn(
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
                    group_runner=run_multiturn_group,
                )
            else:
                run_singleturn(
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
                    singleturn_scorer=score_singleturn_group,
                )
            tl_grpo_ema = acc.tl_grpo_ema

            # 10e. SEPA state updates
            total_completions += len(acc.rewards)
            total_correct += acc.correct
            correct_rate = acc.correct / len(acc.rewards) if acc.rewards else 0.0

            if uses_sepa_controller:
                sepa_controller.observe_correct_rate(correct_rate)

                if (
                    sepa_controller.enabled()
                    and sepa_controller.sepa_schedule == "auto"
                ):
                    for t_idx in range(len(acc.logprobs_sepa)):
                        logprobs = acc.logprobs_sepa[t_idx]
                        pmask = acc.planning_masks_sepa[t_idx]
                        exec_ent = [
                            -logprobs[j] for j in range(len(logprobs)) if pmask[j] == 0
                        ]
                        sepa_controller.update_auto_state(exec_ent)

            # 10f. Train
            num_datums = len(acc.datum_tokens)
            if num_datums > 0 and not backend_caps.preserves_token_advantages:
                assert_uniform_completion_advantages_for_non_preserving_backend(
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
                    apply_advantage_cap(acc.datum_advantages, config.adv_clip_max)
                )

            # Rollout telemetry is captured before trainer-side batch
            # normalization/capping. Recount here so the optimizer-signal
            # metric describes the exact advantages submitted to the backend.
            acc.refresh_optimizer_advantage_token_count()

            echo_plan = prepare_echo_step_plan(config, acc)
            assert_echo_live_observation_contract(
                required=config.echo_require_live_observation_bridge,
                build=acc.echo_build,
                limit=echo_plan.limit,
                final_masks=acc.datum_echo_advantages,
                eligible_rollouts=acc.echo_eligible_rollout_count,
                skipped_entropy_floor=echo_plan.skipped_entropy_floor,
                target_retention=config.echo_target_retention,
            )

            rl_has_signal = has_nonzero_advantage(acc.datum_advantages)
            echo_has_datums = bool(
                config.echo_enabled and has_nonzero_advantage(acc.datum_echo_advantages)
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

            acc.optimizer_logical_batch_sha256 = logical_optimizer_batch_sha256(
                acc.datum_tokens,
                acc.datum_logprobs,
                acc.datum_advantages,
                echo_observation_masks=(
                    acc.datum_echo_advantages if echo_has_datums else None
                ),
                echo_full_observation_counts=(
                    acc.datum_echo_full_observation_counts if echo_has_datums else None
                ),
                echo_rollout_denominator=(
                    acc.echo_eligible_rollout_count if echo_has_datums else None
                ),
            )
            if config.optimizer_batch_capture:
                assert capture_initial_adapter is not None
                captured = save_optimizer_batch_capture(
                    log_path,
                    step=batch_idx,
                    batch=OptimizerBatch(
                        tokens=acc.datum_tokens,
                        old_logprobs=acc.datum_logprobs,
                        advantages=acc.datum_advantages,
                        echo_advantages=(
                            acc.datum_echo_advantages if echo_has_datums else None
                        ),
                        echo_full_observation_counts=(
                            acc.datum_echo_full_observation_counts
                            if echo_has_datums
                            else None
                        ),
                        echo_rollout_denominator=(
                            acc.echo_eligible_rollout_count if echo_has_datums else None
                        ),
                    ),
                    config=config,
                    initial_adapter=capture_initial_adapter,
                )
                if captured.logical_batch_sha256 != acc.optimizer_logical_batch_sha256:
                    raise RuntimeError(
                        "optimizer-batch capture changed the logical batch digest."
                    )
                acc.optimizer_batch_capture_manifest = str(
                    captured.manifest_path.resolve()
                )
                acc.optimizer_batch_payload_sha256 = captured.payload_sha256
                acc.optimizer_batch_manifest_sha256 = captured.manifest_sha256
                acc.optimizer_batch_config_sha256 = captured.config_sha256
                acc.optimizer_batch_contract_sha256 = captured.optimizer_contract_sha256
                acc.optimizer_batch_initial_adapter_sha256 = (
                    captured.initial_adapter_sha256
                )
                optimizer_batch_captured = True

            print(
                f"Step {batch_idx}: submitting {num_datums} RL datums "
                f"and {echo_plan.limit.kept_datums if echo_has_datums else 0} ECHO datums..."
            )
            train_start = time.perf_counter()
            loss_value, echo_loss, echo_joint_optimizer_step = run_rl_echo_train_step(
                helper,
                acc.datum_tokens,
                acc.datum_logprobs,
                acc.datum_advantages,
                acc.datum_echo_advantages if echo_has_datums else [],
                acc.datum_echo_full_observation_counts if echo_has_datums else [],
                echo_loss_fn=config.echo_loss_fn,
                lr=config.lr,
                weight_decay=config.weight_decay,
                echo_rollout_denominator=acc.echo_eligible_rollout_count,
            )
            train_time = time.perf_counter() - train_start
            rl_train_time = train_time if num_datums > 0 else 0.0
            echo_train_time = train_time if echo_has_datums else 0.0
            clip_fraction = getattr(helper, "_clip_fraction", 0.0)
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
                    new_bs = max(
                        config.bp_min_batch_size, min(config.bp_max_batch_size, new_bs)
                    )
                    if new_bs > 0:
                        current_batch_size = new_bs

            # 10g. Logging
            step_logging = record_training_step(
                StepLoggingContext(
                    step=batch_idx,
                    condition_label=condition_label,
                    loss_value=loss_value,
                    echo_loss=echo_loss,
                    echo_joint_optimizer_step=echo_joint_optimizer_step,
                    num_datums=num_datums,
                    total_correct=total_correct,
                    total_completions=total_completions,
                    step_time=step_time,
                    train_time=train_time,
                    rl_train_time=rl_train_time,
                    echo_train_time=echo_train_time,
                    bp_total_tokens=bp_total_tokens,
                    batch_size=current_batch_size,
                    group_size=current_group_size,
                    bp_warmup=bp_warmup,
                    sepa_lambda=sepa_lambda_val,
                    sepa_gate=(
                        sepa_controller.gate_open() if uses_sepa_controller else False
                    ),
                    clip_fraction=clip_fraction,
                    policy_cov_fraction=policy_cov_fraction,
                    policy_abs_kl=policy_abs_kl,
                    adv_cap_fraction=adv_cap_fraction,
                    adv_cap_magnitude=adv_cap_magnitude,
                    tl_grpo_ema=tl_grpo_ema,
                    surprisal_stats=acc.surprisal_stats,
                ),
                config=config,
                backend_caps=backend_caps,
                rollout=cast(RolloutTelemetry, acc),
                echo_plan=echo_plan,
                bp_decision=bp_decision,
                batch_norm_metrics=batch_norm_metrics,
                runtime_counters=runtime_counters,
                helper=helper,
                metrics_logger=metrics_logger,
                steps_logger=steps_logger,
                wandb_run=wandb_run,
            )
            if step_logging.delight_eta_ema is not None:
                delight_eta_ema = step_logging.delight_eta_ema

            # Periodic checkpoint
            if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
                ckpt_name = f"checkpoint_step_{batch_idx + 1}"
                checkpoint_path = helper.save_adapter(
                    config.adapter_path,
                    ckpt_name,
                )
                save_trainer_state(
                    log_path,
                    step=batch_idx,
                    example_idx=example_idx,
                    total_correct=total_correct,
                    total_completions=total_completions,
                    current_batch_size=current_batch_size,
                    current_group_size=current_group_size,
                    checkpoint_name=ckpt_name,
                    checkpoint_path=checkpoint_path,
                    resume_mode=resume_contract.mode,
                    resume_warning=resume_contract.warning,
                    sepa_state=sepa_controller.state_dict(),
                    tl_grpo_ema=tl_grpo_ema,
                    delight_eta_ema=delight_eta_ema,
                    sft_schedule=(
                        sft_data.schedule_contract if sft_data is not None else None
                    ),
                )
                print(f"Saved checkpoint: {ckpt_name}")
                upload_checkpoint_artifact(
                    config,
                    wandb_run,
                    checkpoint_name=ckpt_name,
                    checkpoint_path=checkpoint_path,
                    step=batch_idx,
                )

        if config.optimizer_batch_capture and not optimizer_batch_captured:
            raise RuntimeError(
                "optimizer-batch capture completed no optimizer update; the "
                "single source batch had no informative RL or ECHO signal."
            )

        # -----------------------------------------------------------------------
        # Final
        # -----------------------------------------------------------------------
        final_path = helper.save_adapter(config.adapter_path, "final")
        save_trainer_state(
            log_path,
            step=config.max_steps - 1,
            example_idx=example_idx,
            total_correct=total_correct,
            total_completions=total_completions,
            current_batch_size=current_batch_size,
            current_group_size=current_group_size,
            checkpoint_name="final",
            checkpoint_path=final_path,
            resume_mode=resume_contract.mode,
            resume_warning=resume_contract.warning,
            sepa_state=sepa_controller.state_dict(),
            tl_grpo_ema=tl_grpo_ema,
            delight_eta_ema=delight_eta_ema,
            sft_schedule=(sft_data.schedule_contract if sft_data is not None else None),
        )
        upload_checkpoint_artifact(
            config,
            wandb_run,
            checkpoint_name="final",
            checkpoint_path=final_path,
            step=config.max_steps - 1,
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
            print(
                "No metrics file written (all steps skipped / no informative datums)."
            )

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
