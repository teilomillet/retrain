"""Main training loop -- calls LocalTrainHelper directly, no Mojo.

Ports the training loop from src/main.mojo into pure Python.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from retrain.advantages import (
    EntropyStats,
    compute_composable_advantages,
    get_transform_spec,
)
from retrain.backpressure import (
    BackPressureDecision,
    StepObservation,
)
from retrain.backend_definitions import (
    backend_capability_source,
    resolve_backend_capabilities,
)
from retrain.config import TrainConfig
from retrain.logging_utils import JsonlLogger
from retrain.registry import get_registry
from retrain.sepa import SEPAController
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


def _print_config_summary(config: TrainConfig) -> None:
    """Print a bordered summary of key config values at train start."""
    lines = [
        f"  model         : {config.model}",
        f"  backend       : {config.backend}",
        f"  algorithm     : {config.advantage_mode}+{config.transform_mode}",
        f"  batch_size    : {config.batch_size}",
        f"  group_size    : {config.group_size}",
        f"  max_steps     : {config.max_steps}",
        f"  lr            : {config.lr}",
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
    sepa_state: dict[str, object],
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


def _load_trainer_state(resume_dir: str) -> dict[str, object]:
    """Load trainer state from a checkpoint directory."""
    p = Path(resume_dir)
    state_file = p / _TRAINER_STATE_FILE
    if not state_file.is_file():
        raise FileNotFoundError(
            f"No {_TRAINER_STATE_FILE} found in {resume_dir}. "
            f"Cannot resume without trainer state."
        )
    return json.loads(state_file.read_text())


def train(config: TrainConfig) -> str | None:
    """Main training loop -- fully self-contained. Returns final adapter path."""

    _print_config_summary(config)

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
    # 3. Init backend (after data preflight)
    # -----------------------------------------------------------------------
    helper = get_registry("backend").create(config.backend, config)
    backend_caps = resolve_backend_capabilities(config.backend, config.backend_options)
    _print_backend_capability_summary(
        config.backend,
        backend_capability_source(config.backend, config.backend_options),
        backend_caps.reports_sync_loss,
        backend_caps.preserves_token_advantages,
        backend_caps.supports_checkpoint_resume,
        backend_caps.resume_runtime_dependent,
    )

    # -----------------------------------------------------------------------
    # 4. Load tokenizer + vocab table
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
    # 4b. Planning detector
    # -----------------------------------------------------------------------
    detector = get_registry("planning_detector").create(config.planning_detector, config)
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
    sepa_controller = SEPAController(
        sepa_steps=config.sepa_steps,
        sepa_schedule=config.sepa_schedule,
        sepa_delay_steps=config.sepa_delay_steps,
        sepa_correct_rate_gate=config.sepa_correct_rate_gate,
    )

    # -----------------------------------------------------------------------
    # 6. Back pressure
    # -----------------------------------------------------------------------
    bp_name = "usl" if config.bp_enabled else "noop"
    backpressure = get_registry("backpressure").create(bp_name, config)

    # -----------------------------------------------------------------------
    # 8. Optional wandb
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
        wandb_kwargs: dict[str, object] = {
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
    transform_spec = get_transform_spec(config.transform_mode)
    needs_planning = transform_spec.needs_planning
    uses_sepa_controller = transform_spec.uses_sepa_controller
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
        batch_prompt_objs: list[str | list[dict[str, object]]] = []
        batch_prompt_previews: list[str] = []
        batch_prompt_ids: list[list[int]] = []
        batch_answers: list[str] = []
        batch_tasks: list[str] = []
        batch_infos: list[dict[str, object] | str | None] = []

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
        batch_entropy_stats: list[EntropyStats] = []
        batch_adv_results: list = []
        all_logprobs_sepa: list[list[float]] = []
        all_planning_masks_sepa: list[list[int]] = []
        all_datum_tokens: list[list[int]] = []
        all_datum_logprobs: list[list[float]] = []
        all_datum_advantages: list[list[float]] = []

        # Resolve SEPA lambda once per step (before group loop)
        if uses_sepa_controller:
            sepa_lambda_val = sepa_controller.resolve_lambda(step=float(batch_idx))

        if verifiers_multiturn:
            all_group_sequences: list[list[tuple[list[int], list[float]]]] = []
            sample_start = time.perf_counter()
            for f_idx in range(len(batch_prompt_ids)):
                prompt_obj = batch_prompt_objs[f_idx]
                answer = batch_answers[f_idx]
                task = batch_tasks[f_idx]
                info = batch_infos[f_idx]

                rewards_G, turns_G, completion_texts_G = run_multiturn_group(
                    verifiers_env,
                    helper=helper,
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
                        planning_masks_G.append(detector.detect(seq_token_strs))
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

                group_correct = sum(1 for r in rewards_G if r > _CORRECT_THRESHOLD)
                answer_preview = answer[:40] if len(answer) > 40 else answer
                print(
                    f"  group: {group_correct}/{len(rewards_G)} correct "
                    f"| answer={answer_preview}"
                )

                if rewards_G and all(r == rewards_G[0] for r in rewards_G):
                    if rewards_G[0] > _CORRECT_THRESHOLD:
                        print("    -> skipped (all correct)")
                    else:
                        print("    -> skipped (all wrong)")
                    continue

                adv_result = compute_composable_advantages(
                    rewards_G,
                    logprobs_G,
                    planning_masks_G,
                    advantage_mode=config.advantage_mode,
                    transform_mode=config.transform_mode,
                    gtpo_beta=config.gtpo_beta,
                    hicra_alpha=config.hicra_alpha,
                    sepa_lambda=sepa_lambda_val,
                    post_process_params=config.post_process_params,
                )
                all_token_advs_G = adv_result.token_advs
                if adv_result.has_stats:
                    batch_entropy_stats.append(adv_result.stats)
                if adv_result.extra_metrics:
                    batch_adv_results.append(adv_result)

                for s_idx in range(len(rewards_G)):
                    turn_prompt_ids = turns_prompt_ids_G[s_idx]
                    turn_token_ids = turns_token_ids_G[s_idx]
                    turn_logprobs = turns_logprobs_G[s_idx]
                    token_advs = all_token_advs_G[s_idx]
                    offset = 0
                    for t_idx in range(len(turn_token_ids)):
                        seq_tokens = turn_token_ids[t_idx]
                        seq_logprobs = turn_logprobs[t_idx]
                        prompt_ids = turn_prompt_ids[t_idx]
                        seq_advs = token_advs[offset : offset + len(seq_tokens)]
                        offset += len(seq_tokens)
                        full_tokens = list(prompt_ids) + list(seq_tokens)
                        padded_logprobs = [0.0] * len(prompt_ids) + list(seq_logprobs)
                        padded_advantages = [0.0] * len(prompt_ids) + list(seq_advs)
                        all_datum_tokens.append(full_tokens)
                        all_datum_logprobs.append(padded_logprobs)
                        all_datum_advantages.append(padded_advantages)

                for s_idx, comp_text in enumerate(completion_texts_G):
                    generations_logger.log({
                        "step": batch_idx,
                        "prompt": batch_prompt_previews[f_idx],
                        "completion": comp_text[:500],
                        "reward": rewards_G[s_idx],
                        "num_tokens": len(logprobs_G[s_idx]),
                    })
            sample_time = time.perf_counter() - sample_start
        else:
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
                        planning_masks_G.append(detector.detect(token_strs))
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

                if rewards_G and all(r == rewards_G[0] for r in rewards_G):
                    if rewards_G[0] > _CORRECT_THRESHOLD:
                        print("    -> skipped (all correct)")
                    else:
                        print("    -> skipped (all wrong)")
                    continue

                adv_result = compute_composable_advantages(
                    rewards_G,
                    logprobs_G,
                    planning_masks_G,
                    advantage_mode=config.advantage_mode,
                    transform_mode=config.transform_mode,
                    gtpo_beta=config.gtpo_beta,
                    hicra_alpha=config.hicra_alpha,
                    sepa_lambda=sepa_lambda_val,
                    post_process_params=config.post_process_params,
                )
                all_token_advs_G = adv_result.token_advs
                if adv_result.has_stats:
                    batch_entropy_stats.append(adv_result.stats)
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
                    generations_logger.log({
                        "step": batch_idx,
                        "prompt": batch_prompt_previews[f_idx],
                        "completion": comp_text[:500],
                        "reward": rewards_G[s_idx],
                        "num_tokens": len(logprobs_G[s_idx]),
                    })

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
            if uses_sepa_controller
            else False
        )

        metrics: dict = {
            "step": batch_idx,
            "advantage_mode": config.advantage_mode,
            "transform_mode": config.transform_mode,
            "condition": condition_label,
            "backend_reports_sync_loss": backend_caps.reports_sync_loss,
            "backend_preserves_token_advantages": backend_caps.preserves_token_advantages,
            "loss_is_placeholder": not backend_caps.reports_sync_loss,
            "reported_loss": loss_value,
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
        if batch_adv_results:
            all_extra_keys = {k for r in batch_adv_results for k in r.extra_metrics}
            for k in all_extra_keys:
                vals = [r.extra_metrics[k] for r in batch_adv_results if k in r.extra_metrics]
                metrics[k] = sum(vals) / len(vals)
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
            f" | sepa_l={sepa_lambda_val:.4f}"
            f" | time={step_time:.1f}s"
        )

        # Wandb
        if wandb_enabled and wandb_run is not None:
            wandb_metrics: dict[str, object] = {
                "train/loss": loss_value,
                "train/reported_loss": loss_value,
                "train/loss_is_placeholder": int(not backend_caps.reports_sync_loss),
                "train/rewards/mean_reward": mean_reward,
                "train/rewards/correct_rate": correct_rate,
                "train/rewards/running_correct_rate": running_correct_rate,
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
        sepa_state=sepa_controller.state_dict(),
    )
    final_rate = (
        100.0 * total_correct / total_completions if total_completions > 0 else 0.0
    )
    print(
        f"Training complete. {config.advantage_mode}+{config.transform_mode}, "
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
