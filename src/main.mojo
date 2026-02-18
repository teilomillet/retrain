"""Composable training loop — Mojo entry point.

The training loop is generic over five traits: TrainingBackend, RewardFn,
DataSource, EpisodeAdvantageFn, and TokenTransformFn. Users compose
a pipeline in main() and get full monomorphization — zero dispatch overhead.

Built-in defaults (BoxedMathReward, MathDataSource, MaxRL + GTPO-SEPA)
reproduce the original hardcoded behavior. External users bring their own
reward function and dataset without forking.
"""

from python import Python, PythonObject
from collections import Optional
from time import perf_counter_ns

from src.config import TrainConfig, parse_args
from src.advantages import (
    compute_grpo_advantages,
    compute_maxrl_advantages,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_entropy_stats,
    EntropyStats,
    identify_planning_tokens_native,
)
from src.sepa import SEPAController
from src.logging import JsonlLogger, KeyValue, json_float, json_int, json_string, json_bool, build_json_line
from src.backend import TrainingBackend, SampleSequence
from src.tinker_backend import TinkerBackend
from src.max_backend import MAXBackend
from src.reward import RewardFn, BoxedMathReward, extract_boxed, grade_answer, get_reward
from src.data import DataSource, Example, MathDataSource
from src.advantage_fns import (
    EpisodeAdvantageFn,
    TokenTransformFn,
    AdvantageResult,
    GRPOAdvantage,
    MaxRLAdvantage,
    UniformExpand,
    GTPOTransform,
    GTPOHicraTransform,
    GTPOSepaTransform,
)
from src.pybridge import (
    load_tokenizer,
    encode_prompt,
    batch_decode,
    load_math_dataset,
    MathExample,
    load_vocab_table,
    vocab_lookup,
    init_wandb,
    log_wandb,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Default strategic grams (mirrored from advantages.py)
# ---------------------------------------------------------------------------

fn default_strategic_grams() -> List[String]:
    var grams = List[String]()
    grams.append("wait let me")
    grams.append("let me think")
    grams.append("on second thought")
    grams.append("let me check")
    grams.append("let me verify")
    grams.append("is this right")
    grams.append("double check")
    grams.append("try another approach")
    grams.append("go back and")
    grams.append("start over")
    grams.append("that's not right")
    grams.append("that doesn't work")
    grams.append("another way to")
    grams.append("or we could")
    grams.append("what if we")
    grams.append("notice that")
    grams.append("the key is")
    grams.append("the key insight")
    return grams^


# ---------------------------------------------------------------------------
# Composable advantage pipeline (backward-compatible convenience wrapper)
# ---------------------------------------------------------------------------


fn compute_composable_advantages(
    rewards_G: List[Float64],
    logprobs_G: List[List[Float64]],
    planning_masks_G: List[List[Int]],
    *,
    advantage_mode: String = "grpo",
    transform_mode: String = "none",
    gtpo_beta: Float64 = 0.1,
    hicra_alpha: Float64 = 0.2,
    sepa_lambda: Float64 = 0.0,
) raises -> AdvantageResult:
    """Compute token-level advantages with composable transforms.

    Pure Mojo — no Python dependency. Planning masks are pre-computed
    by the caller via identify_planning_tokens_native.
    """
    # Step 1: Episode-level advantages
    var advantages_G: List[Float64]
    if advantage_mode == "maxrl":
        advantages_G = compute_maxrl_advantages(rewards_G)
    else:
        advantages_G = compute_grpo_advantages(rewards_G)

    # Step 2: Token-level expansion
    if transform_mode == "none":
        var all_token_advs = List[List[Float64]]()
        for i in range(len(logprobs_G)):
            var n_tokens = len(logprobs_G[i])
            all_token_advs.append(List[Float64](length=n_tokens, fill=advantages_G[i]))
        return AdvantageResult(all_token_advs^, False, EntropyStats())

    # For all GTPO-based transforms, we need entropies
    var all_token_advs = List[List[Float64]]()
    var all_exec_entropies = List[Float64]()
    var all_plan_entropies = List[Float64]()

    for idx in range(len(logprobs_G)):
        var logprobs = logprobs_G[idx].copy()
        var advantage = advantages_G[idx]
        var planning_mask = planning_masks_G[idx].copy()

        # Entropy proxy: -logprob
        var entropies = List[Float64](capacity=len(logprobs))
        for j in range(len(logprobs)):
            entropies.append(-logprobs[j])

        # Collect entropy stats
        for j in range(len(entropies)):
            if planning_mask[j] != 0:
                all_plan_entropies.append(entropies[j])
            else:
                all_exec_entropies.append(entropies[j])

        # SEPA: pool execution entropy
        if transform_mode == "gtpo_sepa" and sepa_lambda > 0.0:
            entropies = apply_sepa_pooling(entropies, planning_mask, sepa_lambda)

        # GTPO: entropy-weighted credit assignment
        var token_advs = apply_gtpo_weighting(advantage, entropies, beta=gtpo_beta)

        # HICRA: amplify planning tokens
        if transform_mode == "gtpo_hicra":
            token_advs = apply_hicra(token_advs, planning_mask, alpha=hicra_alpha)

        all_token_advs.append(token_advs^)

    var stats = compute_entropy_stats(all_exec_entropies, all_plan_entropies)
    return AdvantageResult(all_token_advs^, True, stats^)


# ---------------------------------------------------------------------------
# Training loop (generic over backend)
# ---------------------------------------------------------------------------


fn train[B: TrainingBackend, R: RewardFn, D: DataSource, E: EpisodeAdvantageFn, T: TokenTransformFn](
    mut backend: B, reward_fn: R, mut data_source: D, episode_adv: E, mut token_transform: T, config: TrainConfig,
) raises:
    """Main training loop — fully composable.

    Generic over backend, reward, dataset, episode advantage, and token
    transform. All monomorphized at compile time — zero dispatch overhead.
    """
    var os = Python.import_module("os")
    var pathlib = Python.import_module("pathlib")

    # Setup directories
    var log_path = pathlib.Path(config.log_dir)
    _ = log_path.mkdir(parents=True, exist_ok=True)

    var emergence_dir = log_path / "emergence"
    _ = emergence_dir.mkdir(parents=True, exist_ok=True)

    var metrics_path = String(log_path / "metrics.jsonl")
    var steps_path = String(emergence_dir / "steps.jsonl")
    var generations_path = String(emergence_dir / "generations.jsonl")

    var metrics_logger = JsonlLogger(metrics_path)
    var steps_logger = JsonlLogger(steps_path)
    var generations_logger = JsonlLogger(generations_path)

    # Load tokenizer + vocabulary table
    print("Loading tokenizer for " + config.model + " ...")
    var tokenizer = load_tokenizer(config.model)
    print("Loading vocabulary table...")
    var vocab_table = load_vocab_table(tokenizer)
    print("Vocabulary table: " + String(len(vocab_table)) + " entries")

    # Load dataset
    print("Loading dataset...")
    var examples = data_source.load()
    print("Loaded " + String(len(examples)) + " examples")

    # Pre-encode all prompts (one-time cost at startup)
    print("Pre-encoding prompts...")
    var pre_encoded_prompts = List[List[Int]]()
    for i in range(len(examples)):
        pre_encoded_prompts.append(encode_prompt(tokenizer, examples[i].prompt))
    print("Pre-encoded " + String(len(pre_encoded_prompts)) + " prompts")

    # SEPA controller
    var sepa_controller = SEPAController(
        sepa_steps=config.sepa_steps,
        sepa_schedule=config.sepa_schedule,
        sepa_delay_steps=config.sepa_delay_steps,
        sepa_correct_rate_gate=config.sepa_correct_rate_gate,
    )

    # Strategic grams: parse from config or use defaults
    var strategic_grams: List[String]
    if len(config.strategic_grams) > 0:
        # Parse comma-separated list (or JSON array via Python)
        var raw = config.strategic_grams
        if raw.startswith("["):
            # JSON array — parse via Python json
            var json_mod = Python.import_module("json")
            var parsed = json_mod.loads(raw)
            strategic_grams = List[String]()
            for i in range(len(parsed)):
                var gram = String(String(parsed[i]).strip())
                if len(gram) > 0:
                    strategic_grams.append(gram)
        else:
            # Comma-separated
            strategic_grams = List[String]()
            var remaining = raw
            while True:
                var comma_pos = remaining.find(",")
                if comma_pos == -1:
                    var gram = String(remaining.strip())
                    if len(gram) > 0:
                        strategic_grams.append(gram)
                    break
                var gram = String(String(remaining[:comma_pos]).strip())
                if len(gram) > 0:
                    strategic_grams.append(gram)
                remaining = String(remaining[comma_pos + 1 :])
    else:
        strategic_grams = default_strategic_grams()

    # Wandb
    var wandb_run = Python.none()
    var wandb_enabled = len(config.wandb_project) > 0
    if wandb_enabled:
        var condition_label = config.advantage_mode + "+" + config.transform_mode
        var run_name = config.wandb_run_name
        if len(run_name) == 0:
            run_name = condition_label
        var wandb_config = Python.dict()
        wandb_config["advantage_mode"] = PythonObject(config.advantage_mode)
        wandb_config["transform_mode"] = PythonObject(config.transform_mode)
        wandb_config["condition"] = PythonObject(condition_label)
        wandb_config["model"] = PythonObject(config.model)
        wandb_config["lora_rank"] = PythonObject(config.lora_rank)
        wandb_config["lr"] = PythonObject(config.lr)
        wandb_config["batch_size"] = PythonObject(config.batch_size)
        wandb_config["group_size"] = PythonObject(config.group_size)
        wandb_config["max_tokens"] = PythonObject(config.max_tokens)
        wandb_config["temperature"] = PythonObject(config.temperature)
        wandb_config["gtpo_beta"] = PythonObject(config.gtpo_beta)
        wandb_config["hicra_alpha"] = PythonObject(config.hicra_alpha)
        wandb_config["sepa_steps"] = PythonObject(config.sepa_steps)
        wandb_config["max_steps"] = PythonObject(config.max_steps)
        wandb_config["backend"] = PythonObject(config.backend)
        wandb_run = init_wandb(config.wandb_project, run_name, wandb_config)
        print("Wandb initialized: " + config.wandb_project + "/" + run_name)

    # Training loop
    var example_idx = 0
    var total_correct = 0
    var total_completions = 0
    var sepa_lambda_val: Float64 = 0.0

    for batch_idx in range(config.max_steps):
        var step_start_ns = perf_counter_ns()

        # 1. Checkpoint for sampling
        backend.checkpoint_for_sampling("step_" + String(batch_idx))

        # 2. Select prompts
        var batch_prompts = List[String]()
        var batch_prompt_ids = List[List[Int]]()
        var batch_answers = List[String]()

        for _ in range(config.batch_size):
            var ex_idx = example_idx % len(examples)
            example_idx += 1
            batch_prompts.append(examples[ex_idx].prompt)
            batch_prompt_ids.append(pre_encoded_prompts[ex_idx].copy())
            batch_answers.append(examples[ex_idx].reference)

        # 3. Sample completions via backend
        var all_group_sequences = backend.sample_batch(
            batch_prompt_ids,
            config.group_size,
            config.max_tokens,
            config.temperature,
            0.95,
        )

        # Build flat token sequences for batch decode
        var all_token_seqs_flat = List[List[Int]]()
        var group_flat_offsets = List[Int]()

        for f_idx in range(len(all_group_sequences)):
            group_flat_offsets.append(len(all_token_seqs_flat))
            for s_idx in range(len(all_group_sequences[f_idx])):
                all_token_seqs_flat.append(all_group_sequences[f_idx][s_idx].tokens.copy())

        var all_decoded_texts = batch_decode(tokenizer, all_token_seqs_flat)

        # 4. Process groups, compute advantages, collect datum data
        var batch_rewards = List[Float64]()
        var batch_correct = 0
        var batch_max_token_hits = 0
        var batch_total_completions = 0
        var batch_entropy_stats = List[EntropyStats]()
        var all_logprobs = List[List[Float64]]()
        var all_planning_masks = List[List[Int]]()
        var needs_planning = config.transform_mode == "gtpo_hicra" or config.transform_mode == "gtpo_sepa"
        var all_datum_tokens = List[List[Int]]()
        var all_datum_logprobs = List[List[Float64]]()
        var all_datum_advantages = List[List[Float64]]()

        for f_idx in range(len(all_group_sequences)):
            var prompt_text = batch_prompts[f_idx]
            var prompt_ids = batch_prompt_ids[f_idx].copy()
            var answer = batch_answers[f_idx]
            var ob_len = len(prompt_ids)
            var flat_offset = group_flat_offsets[f_idx]
            var n_seqs = len(all_group_sequences[f_idx])

            # Compute rewards + planning masks (pure Mojo)
            var rewards_G = List[Float64]()
            var logprobs_G = List[List[Float64]]()
            var planning_masks_G = List[List[Int]]()

            for s_idx in range(n_seqs):
                var reward = reward_fn.score(all_decoded_texts[flat_offset + s_idx], answer)
                rewards_G.append(reward)
                logprobs_G.append(all_group_sequences[f_idx][s_idx].logprobs.copy())

                if needs_planning:
                    var token_strs = vocab_lookup(
                        vocab_table, all_group_sequences[f_idx][s_idx].tokens
                    )
                    planning_masks_G.append(
                        identify_planning_tokens_native(token_strs, strategic_grams)
                    )
                else:
                    planning_masks_G.append(
                        List[Int](length=len(all_group_sequences[f_idx][s_idx].tokens), fill=0)
                    )

            # Accumulate for SEPA state update
            for s_idx in range(n_seqs):
                all_logprobs.append(logprobs_G[s_idx].copy())
                all_planning_masks.append(planning_masks_G[s_idx].copy())

            for r_idx in range(len(rewards_G)):
                batch_rewards.append(rewards_G[r_idx])
                if rewards_G[r_idx] > 0.5:
                    batch_correct += 1

            # Track max-token truncations
            for s_idx in range(n_seqs):
                batch_total_completions += 1
                if len(all_group_sequences[f_idx][s_idx].tokens) >= config.max_tokens:
                    batch_max_token_hits += 1

            var group_correct = 0
            for r_idx in range(len(rewards_G)):
                if rewards_G[r_idx] > 0.5:
                    group_correct += 1

            var answer_preview = answer
            if len(answer_preview) > 40:
                answer_preview = String(answer_preview[:40])
            print(
                "  group: " + String(group_correct) + "/"
                + String(len(rewards_G)) + " correct | answer=" + answer_preview
            )

            # Skip uninformative groups
            var all_same = True
            if len(rewards_G) > 0:
                var first = rewards_G[0]
                for r_idx in range(1, len(rewards_G)):
                    if rewards_G[r_idx] != first:
                        all_same = False
                        break
            if all_same:
                if len(rewards_G) > 0 and rewards_G[0] > 0.5:
                    print("    -> skipped (all correct)")
                else:
                    print("    -> skipped (all wrong)")
                continue

            # Update SEPA lambda from controller (no-op for non-SEPA transforms)
            if config.transform_mode == "gtpo_sepa":
                token_transform.update_sepa_lambda(
                    sepa_controller.resolve_lambda(step=Float64(batch_idx))
                )

            # Compute advantages via trait-based pipeline
            var episode_advantages = episode_adv.compute(rewards_G)
            var adv_result = token_transform.transform(
                episode_advantages, logprobs_G, planning_masks_G,
            )
            var token_advs_G = adv_result.token_advs.copy()
            if adv_result.has_stats:
                batch_entropy_stats.append(adv_result.stats.copy())

            # Collect datum data (pure Mojo — batch built after loop)
            for s_idx in range(n_seqs):
                var seq_tokens = all_group_sequences[f_idx][s_idx].tokens.copy()
                var seq_logprobs = all_group_sequences[f_idx][s_idx].logprobs.copy()
                var full_tokens = List[Int](capacity=ob_len + len(seq_tokens))
                for p_idx in range(ob_len):
                    full_tokens.append(prompt_ids[p_idx])
                for t_idx in range(len(seq_tokens)):
                    full_tokens.append(seq_tokens[t_idx])

                var padded_logprobs = List[Float64](length=ob_len, fill=0.0)
                for lp_idx in range(len(seq_logprobs)):
                    padded_logprobs.append(seq_logprobs[lp_idx])

                var padded_advantages = List[Float64](length=ob_len, fill=0.0)
                for a_idx in range(len(token_advs_G[s_idx])):
                    padded_advantages.append(token_advs_G[s_idx][a_idx])

                all_datum_tokens.append(full_tokens^)
                all_datum_logprobs.append(padded_logprobs^)
                all_datum_advantages.append(padded_advantages^)

            # Per-generation emergence logging
            for s_idx in range(n_seqs):
                var completion_text = all_decoded_texts[flat_offset + s_idx]
                var reward = rewards_G[s_idx]
                var gen_entries = List[KeyValue]()
                gen_entries.append(json_int("step", batch_idx))
                var prompt_preview = prompt_text
                if len(prompt_preview) > 200:
                    prompt_preview = String(prompt_preview[:200])
                gen_entries.append(json_string("prompt", prompt_preview))
                var comp_preview = completion_text
                if len(comp_preview) > 500:
                    comp_preview = String(comp_preview[:500])
                gen_entries.append(json_string("completion", comp_preview))
                gen_entries.append(json_float("reward", reward))
                gen_entries.append(json_int("num_tokens", len(all_group_sequences[f_idx][s_idx].tokens)))
                generations_logger.log(gen_entries)

        # 5. SEPA state updates
        total_completions += len(batch_rewards)
        total_correct += batch_correct
        var correct_rate: Float64 = 0.0
        if len(batch_rewards) > 0:
            correct_rate = Float64(batch_correct) / Float64(len(batch_rewards))

        if config.transform_mode == "gtpo_sepa":
            sepa_controller.observe_correct_rate(Optional[Float64](correct_rate))

            if sepa_controller.enabled() and sepa_controller.sepa_schedule == "auto":
                # Reuse pre-computed planning masks (no Python calls here)
                for t_idx in range(len(all_logprobs)):
                    var logprobs = all_logprobs[t_idx].copy()
                    var planning_mask = all_planning_masks[t_idx].copy()
                    var exec_entropies = List[Float64]()
                    for e_idx in range(len(logprobs)):
                        if planning_mask[e_idx] == 0:
                            exec_entropies.append(-logprobs[e_idx])
                    sepa_controller.update_auto_state(exec_entropies)

        # 6. Train via backend
        var num_datums = len(all_datum_tokens)
        if num_datums == 0:
            print("Step " + String(batch_idx) + ": no informative datums, skipping.")
            continue

        print(
            "Step " + String(batch_idx) + ": submitting "
            + String(num_datums) + " datums for training..."
        )

        var loss_value = backend.train_step(
            all_datum_tokens, all_datum_logprobs, all_datum_advantages,
            config.lr, config.weight_decay,
        )

        var step_time_ns = perf_counter_ns() - step_start_ns
        var step_time_s = Float64(step_time_ns) / 1_000_000_000.0

        # 7. Logging
        var mean_reward: Float64 = 0.0
        if len(batch_rewards) > 0:
            var reward_sum: Float64 = 0.0
            for r_idx in range(len(batch_rewards)):
                reward_sum += batch_rewards[r_idx]
            mean_reward = reward_sum / Float64(len(batch_rewards))

        var running_correct_rate: Float64 = 0.0
        if total_completions > 0:
            running_correct_rate = Float64(total_correct) / Float64(total_completions)

        var max_token_hit_rate: Float64 = 0.0
        if batch_total_completions > 0:
            max_token_hit_rate = Float64(batch_max_token_hits) / Float64(batch_total_completions)

        # Aggregate entropy stats
        var step_exec_mean: Float64 = 0.0
        var step_exec_var: Float64 = 0.0
        var step_plan_mean: Float64 = 0.0
        var step_plan_var: Float64 = 0.0
        if len(batch_entropy_stats) > 0:
            var n_stats = len(batch_entropy_stats)
            for es_idx in range(n_stats):
                step_exec_mean += batch_entropy_stats[es_idx].exec_mean
                step_exec_var += batch_entropy_stats[es_idx].exec_var
                step_plan_mean += batch_entropy_stats[es_idx].plan_mean
                step_plan_var += batch_entropy_stats[es_idx].plan_var
            step_exec_mean /= Float64(n_stats)
            step_exec_var /= Float64(n_stats)
            step_plan_mean /= Float64(n_stats)
            step_plan_var /= Float64(n_stats)

        if config.transform_mode == "gtpo_sepa":
            sepa_lambda_val = sepa_controller.resolve_lambda(step=Float64(batch_idx))

        var condition_label = config.advantage_mode + "+" + config.transform_mode

        # Metrics JSONL
        var metrics_entries = List[KeyValue]()
        metrics_entries.append(json_int("step", batch_idx))
        metrics_entries.append(json_string("advantage_mode", config.advantage_mode))
        metrics_entries.append(json_string("transform_mode", config.transform_mode))
        metrics_entries.append(json_string("condition", condition_label))
        metrics_entries.append(json_float("loss", loss_value))
        metrics_entries.append(json_float("mean_reward", mean_reward))
        metrics_entries.append(json_float("correct_rate", correct_rate))
        metrics_entries.append(json_float("running_correct_rate", running_correct_rate))
        metrics_entries.append(json_float("sepa_lambda", sepa_lambda_val))
        var sepa_gate = False
        if config.transform_mode == "gtpo_sepa":
            sepa_gate = sepa_controller.gate_open()
        metrics_entries.append(json_bool("sepa_gate_open", sepa_gate))
        metrics_entries.append(json_int("num_datums", num_datums))
        metrics_entries.append(json_float("max_token_hit_rate", max_token_hit_rate))
        metrics_entries.append(json_float("step_time_s", step_time_s))
        if len(batch_entropy_stats) > 0:
            metrics_entries.append(json_float("exec_entropy_mean", step_exec_mean))
            metrics_entries.append(json_float("exec_entropy_var", step_exec_var))
            metrics_entries.append(json_float("plan_entropy_mean", step_plan_mean))
            metrics_entries.append(json_float("plan_entropy_var", step_plan_var))
        metrics_logger.log(metrics_entries)

        print(
            "Step " + String(batch_idx) + " [" + condition_label + "] | loss="
            + String(loss_value) + " | reward=" + String(mean_reward)
            + " | correct=" + String(correct_rate * 100.0) + "%"
            + " | datums=" + String(num_datums)
            + " | sepa_l=" + String(sepa_lambda_val)
            + " | time=" + String(step_time_s) + "s"
        )

        # Wandb
        if wandb_enabled:
            var wandb_metrics = Python.dict()
            wandb_metrics["step"] = PythonObject(batch_idx)
            wandb_metrics["loss"] = PythonObject(loss_value)
            wandb_metrics["mean_reward"] = PythonObject(mean_reward)
            wandb_metrics["correct_rate"] = PythonObject(correct_rate)
            wandb_metrics["running_correct_rate"] = PythonObject(running_correct_rate)
            wandb_metrics["sepa_lambda"] = PythonObject(sepa_lambda_val)
            wandb_metrics["num_datums"] = PythonObject(num_datums)
            wandb_metrics["step_time_s"] = PythonObject(step_time_s)
            log_wandb(wandb_run, wandb_metrics, batch_idx)

        # Step record for emergence analysis
        var step_entries = List[KeyValue]()
        step_entries.append(json_int("step", batch_idx))
        step_entries.append(json_float("mean_reward", mean_reward))
        step_entries.append(json_int("correct_count", batch_correct))
        step_entries.append(json_int("total_count", len(batch_rewards)))
        step_entries.append(json_string("condition", condition_label))
        if len(batch_entropy_stats) > 0:
            step_entries.append(json_float("exec_entropy_mean", step_exec_mean))
            step_entries.append(json_float("exec_entropy_var", step_exec_var))
            step_entries.append(json_float("plan_entropy_mean", step_plan_mean))
            step_entries.append(json_float("plan_entropy_var", step_plan_var))
        steps_logger.log(step_entries)

        # Periodic checkpoint
        if config.save_every > 0 and (batch_idx + 1) % config.save_every == 0:
            var ckpt_name = "checkpoint_step_" + String(batch_idx + 1)
            backend.save(ckpt_name)
            print("Saved checkpoint: " + ckpt_name)

    # Save final checkpoint
    backend.save("final")
    var final_rate: Float64 = 0.0
    if total_completions > 0:
        final_rate = 100.0 * Float64(total_correct) / Float64(total_completions)
    print(
        "Training complete. " + config.advantage_mode + "+"
        + config.transform_mode + ", " + String(config.max_steps)
        + " steps, running correct rate: " + String(final_rate) + "%"
    )
    print("Metrics saved to " + metrics_path)

    if wandb_enabled:
        finish_wandb(wandb_run)


# ---------------------------------------------------------------------------
# Monomorphization dispatch helpers
# ---------------------------------------------------------------------------


fn _run[B: TrainingBackend, E: EpisodeAdvantageFn](
    mut backend: B, ep: E, config: TrainConfig,
) raises:
    """Dispatch over token transform mode, then call train()."""
    var reward = BoxedMathReward()
    var data = MathDataSource(config.max_examples)
    if config.transform_mode == "none":
        var tx = UniformExpand()
        train(backend, reward, data, ep, tx, config)
    elif config.transform_mode == "gtpo":
        var tx = GTPOTransform(config.gtpo_beta)
        train(backend, reward, data, ep, tx, config)
    elif config.transform_mode == "gtpo_hicra":
        var tx = GTPOHicraTransform(config.gtpo_beta, config.hicra_alpha)
        train(backend, reward, data, ep, tx, config)
    else:
        var tx = GTPOSepaTransform(config.gtpo_beta)
        train(backend, reward, data, ep, tx, config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


fn main() raises:
    var config = parse_args()
    if config.backend == "max":
        var backend = MAXBackend(config)
        if config.advantage_mode == "grpo":
            _run(backend, GRPOAdvantage(), config)
        else:
            _run(backend, MaxRLAdvantage(), config)
    else:
        var backend = TinkerBackend(config)
        if config.advantage_mode == "grpo":
            _run(backend, GRPOAdvantage(), config)
        else:
            _run(backend, MaxRLAdvantage(), config)
