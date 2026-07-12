"""Standalone supervised fine-tuning runner."""

from __future__ import annotations

import time
from pathlib import Path

from retrain.backends import collect_runtime_metrics, run_sft_train_step
from retrain.config import TrainConfig
from retrain.process.metrics import max_rss_mb
from retrain.training.runner.result import (
    TrainingRunResult,
    build_run_result,
    failed_run_result,
)


def _sft_runtime_metric_key(key: str) -> str:
    """Keep optimizer evidence canonical; namespace ordinary backend metrics."""

    return key if key.startswith("optimizer/") else f"backend/{key}"


_SFT_BATCH_WANDB_KEYS = (
    "sft_sequence_length_min",
    "sft_sequence_length_mean",
    "sft_sequence_length_p50",
    "sft_sequence_length_p95",
    "sft_sequence_length_max",
    "sft_logical_padded_tokens",
    "sft_logical_padding_tokens",
    "sft_logical_padding_fraction",
    "sft_supervised_token_fraction",
)


class SftRunner:
    """Standalone supervised fine-tuning runner using retrain backends."""

    def run(self, config: TrainConfig) -> TrainingRunResult:
        try:
            return self._run(config)
        except Exception as exc:
            return failed_run_result(
                config,
                failure_status=f"exception:{type(exc).__name__}",
                error_message=str(exc),
            )

    def _run(self, config: TrainConfig) -> TrainingRunResult:
        if config.backend == "local" and bool(
            config.backend_options.get("strict_deterministic", False)
        ):
            # SFT seeds CUDA before constructing its backend. Establish strict
            # controls first so manual_seed_all cannot initialize CUDA ahead
            # of the deterministic cuBLAS/PyTorch contract.
            from retrain.backends.determinism import establish_strict_determinism

            establish_strict_determinism(enabled=True)

        from transformers import AutoTokenizer

        from retrain.backends.catalog import (
            backend_capability_source,
            resolve_backend_capabilities,
        )
        from retrain.io.log import JsonlLogger
        from retrain.registry.builtin import get_registry
        from retrain.training.sft import (
            build_sft_batch_metrics,
            build_sft_tokenized_batch,
            build_sft_artifact_manifest,
            build_sft_example_order,
            build_sft_resume_schedule_contract,
            build_sft_schedule_metrics,
            effective_sft_loss_fn,
            load_sft_dataset,
            select_sft_batch_indices,
            sft_tokenizer_load_kwargs,
            tokenize_sft_dataset,
            verify_sft_data_contract,
            verify_sft_resume_schedule_contract,
            write_sft_artifact_manifest,
            write_sft_run_snapshot_artifacts,
        )
        from retrain.training.telemetry import build_runtime_wandb_metrics
        from retrain.training.state import (
            TRAINER_STATE_FILE,
            load_trainer_state,
            save_trainer_state,
        )
        from retrain.training.log import WandbRunLike, init_wandb
        from retrain.training.recoverability import (
            announce_checkpoint_recoverability,
            checkpoint_recoverability_wandb_metrics,
            upload_checkpoint_artifact,
        )
        from retrain.training.resume import contract_for_capabilities

        if not config.sft_data_path:
            raise ValueError(
                "trainer='sft' requires [training] sft_data_path to point at a JSONL dataset."
            )

        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        emergence_dir = log_dir / "emergence"
        emergence_dir.mkdir(parents=True, exist_ok=True)
        metrics_logger = JsonlLogger(str(log_dir / "metrics.jsonl"))
        steps_logger = JsonlLogger(str(emergence_dir / "steps.jsonl"))

        dataset = load_sft_dataset(config.sft_data_path)
        verify_sft_data_contract(config, dataset.provenance)
        snapshot_artifacts = write_sft_run_snapshot_artifacts(
            log_dir,
            config,
            dataset.provenance,
        )
        examples = dataset.examples
        if not examples:
            raise RuntimeError("SFT dataset is empty — cannot fine-tune with zero examples.")
        for warning in dataset.provenance.data_warnings:
            print(f"WARNING: {warning}")

        if config.seed >= 0:
            import random

            random.seed(config.seed)
            try:
                import numpy as np

                np.random.seed(config.seed)
            except ImportError:
                pass
            try:
                import torch

                torch.manual_seed(config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(config.seed)
            except ImportError:
                pass

        print("retrain SFT")
        print(f"  model         : {config.model}")
        print(f"  backend       : {config.backend}")
        print(f"  examples      : {len(examples)}")
        print(f"  data_path     : {dataset.provenance.data_path}")
        print(f"  data_sha256   : {dataset.provenance.data_sha256}")
        print(f"  max_steps     : {config.max_steps}")
        print(f"  adapter_path  : {config.adapter_path}")

        resume_state = None
        resume_checkpoint_ref = ""
        start_step = 0
        resume_batch_size: int | None = None
        if config.resume_from:
            resume_state_path = Path(config.resume_from) / TRAINER_STATE_FILE
            if resume_state_path.is_file():
                resume_state = load_trainer_state(config.resume_from)
                if "sft_schedule" not in resume_state:
                    verify_sft_resume_schedule_contract(None, {})
                start_step = resume_state["step"] + 1
                if start_step > config.max_steps:
                    raise ValueError(
                        "SFT resume checkpoint is beyond configured max_steps: "
                        f"checkpoint step {resume_state['step']}, "
                        f"max_steps {config.max_steps}."
                    )
                resume_batch_size = resume_state["current_batch_size"]
                resume_checkpoint_ref = (
                    resume_state.get("checkpoint_path", "")
                    or resume_state.get("checkpoint_name", "")
                )

        helper = get_registry("backend").create(config.backend, config)
        if config.resume_from and not callable(getattr(helper, "load_state", None)):
            raise RuntimeError(
                "trainer='sft' resume_from requires a backend with load_state()."
            )
        loss_fn = effective_sft_loss_fn(config)
        setattr(helper, "sft_loss_fn", loss_fn)

        backend_caps = resolve_backend_capabilities(
            config.backend,
            config.backend_options,
        )
        print(
            "Backend capabilities: "
            f"backend={config.backend}, "
            f"source={backend_capability_source(config.backend, config.backend_options)}, "
            f"reports_sync_loss={backend_caps.reports_sync_loss}, "
            f"preserves_token_advantages={backend_caps.preserves_token_advantages}, "
            f"supports_checkpoint_resume={backend_caps.supports_checkpoint_resume}, "
            f"resume_runtime_dependent={backend_caps.resume_runtime_dependent}, "
            f"checkpoint_resume_mode={backend_caps.checkpoint_resume_mode}"
        )
        resume_contract = contract_for_capabilities(backend_caps)
        print(f"Checkpoint resume mode: {resume_contract.mode}")
        if resume_contract.warning:
            print(f"Checkpoint resume warning: {resume_contract.warning}")
        print(f"SFT loss: {loss_fn}")

        print(f"Loading tokenizer for {config.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            **sft_tokenizer_load_kwargs(config, dataset.provenance),
        )

        batch_size = config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
        batch_size = min(max(1, batch_size), len(examples))
        if resume_batch_size is not None and resume_batch_size > 0:
            batch_size = min(resume_batch_size, len(examples))
        max_tokens = config.sft_max_tokens if config.sft_max_tokens > 0 else config.max_tokens
        lr = config.sft_lr if config.sft_lr > 0 else config.lr
        print("Tokenizing SFT dataset ...")
        tokenized_examples = tokenize_sft_dataset(
            tokenizer,
            examples,
            max_tokens=max_tokens,
        )
        token_lengths = [example.total_tokens for example in tokenized_examples]
        order = build_sft_example_order(
            len(tokenized_examples),
            config.seed,
            lengths=token_lengths,
            batch_order=config.sft_batch_order,
            length_bucket_size=config.sft_length_bucket_size,
        )
        epoch_order_cache = {0: order}
        epoch_order_sha256_cache: dict[int, str] = {}
        schedule_contract = build_sft_resume_schedule_contract(
            config,
            dataset.provenance,
            batch_size=batch_size,
            max_tokens=max_tokens,
            example_order=order,
        )
        if resume_state is not None:
            verify_sft_resume_schedule_contract(
                resume_state.get("sft_schedule"),
                schedule_contract,
            )
        print(
            "SFT batching: "
            f"order={config.sft_batch_order}, "
            f"length_bucket_size={config.sft_length_bucket_size or len(tokenized_examples)}, "
            f"reshuffle_each_epoch={config.sft_reshuffle_each_epoch}"
        )

        if config.resume_from:
            load_state = getattr(helper, "load_state", None)
            if not callable(load_state):
                raise RuntimeError(
                    "trainer='sft' resume_from requires a backend with load_state()."
                )
            if resume_state is not None:
                if not resume_checkpoint_ref:
                    raise RuntimeError(
                        "trainer='sft' resume_from log dir did not provide "
                        "a checkpoint path or name."
                    )
                print(
                    "Resuming SFT from "
                    f"{config.resume_from} at step {resume_state['step']} "
                    f"(checkpoint: {resume_checkpoint_ref}), "
                    f"continuing from step {start_step} ..."
                )
                load_state(resume_checkpoint_ref)
            else:
                print(f"Loading SFT initial adapter from {config.resume_from} ...")
                load_state(config.resume_from)

        policy_ref = ""
        last_metrics: dict[str, int | float | str] = {}
        wandb_run: WandbRunLike | None = None
        try:
            wandb_run = init_wandb(config, condition_label="sft")
            announce_checkpoint_recoverability(config, wandb_run)
            for step in range(start_step, config.max_steps):
                step_start = time.perf_counter()
                indices = select_sft_batch_indices(
                    order,
                    batch_size=batch_size,
                    step=step,
                    seed=config.seed,
                    lengths=token_lengths,
                    batch_order=config.sft_batch_order,
                    length_bucket_size=config.sft_length_bucket_size,
                    reshuffle_each_epoch=config.sft_reshuffle_each_epoch,
                    epoch_order_cache=epoch_order_cache,
                )
                batch = [tokenized_examples[idx] for idx in indices]
                tokenized = build_sft_tokenized_batch(batch)
                batch_metrics = build_sft_batch_metrics(tokenized)

                train_start = time.perf_counter()
                loss = run_sft_train_step(
                    helper,
                    tokenized.tokens,
                    tokenized.advantages,
                    lr,
                    config.weight_decay,
                )
                train_elapsed = time.perf_counter() - train_start

                elapsed = time.perf_counter() - step_start
                train_time_semantics = (
                    "submit_enqueue_latency"
                    if config.backend == "prime_rl"
                    else "synchronous_optimizer_update"
                )
                metrics: dict[str, int | float | str] = {
                    "step": step,
                    "phase": "sft",
                    "trainer": "sft",
                    "backend": config.backend,
                    "loss": loss,
                    "sft_loss_fn": loss_fn,
                    "sft_batch_order": config.sft_batch_order,
                    "sft_length_bucket_size": int(config.sft_length_bucket_size),
                    "sft_reshuffle_each_epoch": int(
                        config.sft_reshuffle_each_epoch
                    ),
                    "lr": lr,
                    "datums": len(batch),
                    "tokens": tokenized.total_tokens,
                    "supervised_tokens": tokenized.supervised_tokens,
                    "sft_unique_examples_seen": min(
                        len(examples),
                        (step + 1) * batch_size,
                    ),
                    "sft_dataset_coverage": min(
                        1.0,
                        ((step + 1) * batch_size) / max(len(examples), 1),
                    ),
                    "time_s": round(elapsed, 2),
                    "step_time_s": elapsed,
                    "train_time_s": train_elapsed,
                    "train_time_semantics": train_time_semantics,
                }
                metrics.update(
                    build_sft_schedule_metrics(
                        order,
                        indices,
                        batch_size=batch_size,
                        step=step,
                        seed=config.seed,
                        lengths=token_lengths,
                        batch_order=config.sft_batch_order,
                        length_bucket_size=config.sft_length_bucket_size,
                        reshuffle_each_epoch=config.sft_reshuffle_each_epoch,
                        epoch_order_cache=epoch_order_cache,
                        epoch_order_sha256_cache=epoch_order_sha256_cache,
                    )
                )
                if config.backend == "prime_rl":
                    metrics["train_submit_enqueue_time_s"] = train_elapsed
                    metrics["train_submit_enqueue_share"] = (
                        train_elapsed / elapsed if elapsed > 0.0 else 0.0
                    )
                else:
                    metrics["train_share"] = (
                        train_elapsed / elapsed if elapsed > 0.0 else 0.0
                    )
                metrics.update(batch_metrics)
                rss_mb = max_rss_mb()
                if rss_mb is not None:
                    metrics["process_max_rss_mb"] = round(rss_mb, 3)
                runtime_metrics = collect_runtime_metrics(helper)
                for key, value in runtime_metrics.items():
                    metrics[_sft_runtime_metric_key(key)] = value

                metrics_logger.log(metrics)
                steps_logger.log(metrics)
                if wandb_run is not None:
                    wandb_metrics: dict[str, object] = {
                        "train/loss": loss,
                        "train/sft": 1,
                        "train/step": step,
                        "train/lr": lr,
                        "train/datums": len(batch),
                        "train/tokens": tokenized.total_tokens,
                        "train/supervised_tokens": tokenized.supervised_tokens,
                        "train/sft_dataset_coverage": metrics[
                            "sft_dataset_coverage"
                        ],
                        "train/sft/epoch": metrics["sft_epoch"],
                        "train/sft/epoch_end": metrics["sft_epoch_end"],
                        "train/sft/epoch_seed": metrics["sft_epoch_seed"],
                        "train/sft/epoch_end_seed": metrics[
                            "sft_epoch_end_seed"
                        ],
                        "train/sft/epoch_sample_offset": metrics[
                            "sft_epoch_sample_offset"
                        ],
                        "train/sft/reshuffle_each_epoch": metrics[
                            "sft_reshuffle_each_epoch"
                        ],
                        "train/sft/batch_indices_sha256": metrics[
                            "sft_batch_indices_sha256"
                        ],
                        "train/sft/epoch_start_order_sha256": metrics[
                            "sft_epoch_start_order_sha256"
                        ],
                        "train/step_time_s": elapsed,
                        "train/train_time_s": train_elapsed,
                        "train/train_time_semantics": train_time_semantics,
                    }
                    if "sft_epoch_end_order_sha256" in metrics:
                        wandb_metrics["train/sft/epoch_end_order_sha256"] = metrics[
                            "sft_epoch_end_order_sha256"
                        ]
                    if config.backend == "prime_rl":
                        wandb_metrics["train/train_submit_enqueue_time_s"] = (
                            train_elapsed
                        )
                        wandb_metrics["train/train_submit_enqueue_share"] = metrics[
                            "train_submit_enqueue_share"
                        ]
                    else:
                        wandb_metrics["train/train_share"] = metrics["train_share"]
                    for key in _SFT_BATCH_WANDB_KEYS:
                        wandb_metrics[f"train/sft/{key.removeprefix('sft_')}"] = (
                            metrics[key]
                        )
                    if "process_max_rss_mb" in metrics:
                        wandb_metrics["train/process_max_rss_mb"] = metrics[
                            "process_max_rss_mb"
                        ]
                    wandb_metrics.update(
                        build_runtime_wandb_metrics(runtime_metrics)
                    )
                    wandb_metrics.update(
                        checkpoint_recoverability_wandb_metrics(config, wandb_run)
                    )
                    wandb_metrics.update(
                        {
                            f"train/{key}": value
                            for key, value in metrics.items()
                            if key.startswith("optimizer/")
                        }
                    )
                    wandb_run.log(wandb_metrics, step=step)
                last_metrics = dict(metrics)
                print(
                    f"Step {step} [SFT] | loss={loss:.4f} | "
                    f"datums={len(batch)} | tokens={tokenized.total_tokens} | "
                    f"supervised={tokenized.supervised_tokens} | time={elapsed:.1f}s",
                    flush=True,
                )

                if config.save_every > 0 and (step + 1) % config.save_every == 0:
                    checkpoint_name = f"checkpoint_step_{step + 1}"
                    policy_ref = helper.save_adapter(
                        config.adapter_path,
                        checkpoint_name,
                    )
                    save_trainer_state(
                        log_dir,
                        step=step,
                        example_idx=(step + 1) * batch_size,
                        total_correct=0,
                        total_completions=0,
                        current_batch_size=batch_size,
                        current_group_size=1,
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=policy_ref,
                        resume_mode=resume_contract.mode,
                        resume_warning=resume_contract.warning,
                        sepa_state={},
                        sft_schedule=schedule_contract,
                    )
                    print(f"Saved checkpoint: {checkpoint_name}")
                    upload_checkpoint_artifact(
                        config,
                        wandb_run,
                        checkpoint_name=checkpoint_name,
                        checkpoint_path=policy_ref,
                        step=step,
                    )

            policy_ref = helper.save_adapter(
                config.adapter_path,
                "final",
            )
            save_trainer_state(
                log_dir,
                step=config.max_steps - 1,
                example_idx=config.max_steps * batch_size,
                total_correct=0,
                total_completions=0,
                current_batch_size=batch_size,
                current_group_size=1,
                checkpoint_name="final",
                checkpoint_path=policy_ref,
                resume_mode=resume_contract.mode,
                resume_warning=resume_contract.warning,
                sepa_state={},
                sft_schedule=schedule_contract,
            )
            manifest = build_sft_artifact_manifest(
                config,
                policy_ref=policy_ref,
                examples_count=len(examples),
                batch_size=batch_size,
                max_tokens=max_tokens,
                loss_fn=loss_fn,
                data_provenance=dataset.provenance,
                snapshot_artifacts=snapshot_artifacts,
                latest_metrics=last_metrics,
            )
            manifest_paths = write_sft_artifact_manifest(
                log_dir,
                policy_ref,
                manifest,
            )
            upload_checkpoint_artifact(
                config,
                wandb_run,
                checkpoint_name="final",
                checkpoint_path=policy_ref,
                step=config.max_steps - 1,
            )
            print(f"SFT manifest: {manifest_paths['log_manifest']}")
        finally:
            if wandb_run is not None:
                try:
                    wandb_run.finish()
                except Exception:
                    pass
            shutdown = getattr(helper, "shutdown", None)
            if callable(shutdown):
                shutdown()

        print(f"SFT complete. Adapter: {policy_ref}")
        return build_run_result(config, policy_ref=policy_ref)
