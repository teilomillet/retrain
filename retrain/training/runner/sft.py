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
        from transformers import AutoTokenizer

        from retrain.backends.catalog import (
            backend_capability_source,
            resolve_backend_capabilities,
        )
        from retrain.io.log import JsonlLogger
        from retrain.registry.builtin import get_registry
        from retrain.training.sft import (
            build_sft_tokenized_batch,
            build_sft_artifact_manifest,
            build_sft_example_order,
            effective_sft_loss_fn,
            load_sft_dataset,
            select_sft_batch_indices,
            tokenize_sft_dataset,
            write_sft_artifact_manifest,
        )
        from retrain.training.state import save_trainer_state

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

        helper = get_registry("backend").create(config.backend, config)
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
            f"resume_runtime_dependent={backend_caps.resume_runtime_dependent}"
        )
        print(f"SFT loss: {loss_fn}")

        if config.resume_from:
            load_state = getattr(helper, "load_state", None)
            if not callable(load_state):
                raise RuntimeError(
                    "trainer='sft' resume_from requires a backend with load_state()."
                )
            print(f"Loading SFT initial adapter from {config.resume_from} ...")
            load_state(config.resume_from)

        print(f"Loading tokenizer for {config.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)

        batch_size = config.sft_batch_size if config.sft_batch_size > 0 else config.batch_size
        batch_size = min(max(1, batch_size), len(examples))
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
        print(
            "SFT batching: "
            f"order={config.sft_batch_order}, "
            f"length_bucket_size={config.sft_length_bucket_size or len(tokenized_examples)}"
        )

        policy_ref = ""
        last_metrics: dict[str, int | float | str] = {}
        try:
            for step in range(config.max_steps):
                step_start = time.perf_counter()
                indices = select_sft_batch_indices(
                    order,
                    batch_size=batch_size,
                    step=step,
                )
                batch = [tokenized_examples[idx] for idx in indices]
                tokenized = build_sft_tokenized_batch(batch)

                loss = run_sft_train_step(
                    helper,
                    tokenized.tokens,
                    tokenized.advantages,
                    lr,
                    config.weight_decay,
                )

                elapsed = time.perf_counter() - step_start
                metrics: dict[str, int | float | str] = {
                    "step": step,
                    "phase": "sft",
                    "trainer": "sft",
                    "backend": config.backend,
                    "loss": loss,
                    "sft_loss_fn": loss_fn,
                    "sft_batch_order": config.sft_batch_order,
                    "sft_length_bucket_size": int(config.sft_length_bucket_size),
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
                }
                rss_mb = max_rss_mb()
                if rss_mb is not None:
                    metrics["process_max_rss_mb"] = round(rss_mb, 3)
                for key, value in collect_runtime_metrics(helper).items():
                    metrics[f"backend/{key}"] = value

                metrics_logger.log(metrics)
                steps_logger.log(metrics)
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
                        sepa_state={},
                    )
                    print(f"Saved checkpoint: {checkpoint_name}")

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
                sepa_state={},
            )
            manifest = build_sft_artifact_manifest(
                config,
                policy_ref=policy_ref,
                examples_count=len(examples),
                batch_size=batch_size,
                max_tokens=max_tokens,
                loss_fn=loss_fn,
                data_provenance=dataset.provenance,
                latest_metrics=last_metrics,
            )
            manifest_paths = write_sft_artifact_manifest(
                log_dir,
                policy_ref,
                manifest,
            )
            print(f"SFT manifest: {manifest_paths['log_manifest']}")
        finally:
            shutdown = getattr(helper, "shutdown", None)
            if callable(shutdown):
                shutdown()

        print(f"SFT complete. Adapter: {policy_ref}")
        return build_run_result(config, policy_ref=policy_ref)
