"""Tinker GPU training client — Mojo entry point.

Ports the training loop from textpolicy/tinker/train_math.py.
All advantage math runs natively (SIMD); external services
(Tinker, tokenizer, datasets, wandb) go through pybridge.
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
)
from src.sepa import SEPAController
from src.logging import JsonlLogger, KeyValue, json_float, json_int, json_string, json_bool, build_json_line
from src.pybridge import (
    load_tokenizer,
    encode_prompt,
    decode,
    load_math_dataset,
    MathExample,
    create_tinker_client,
    create_lora_training_client,
    save_weights_and_get_sampling_client,
    sample,
    forward_backward,
    optim_step,
    save_state,
    build_datum,
    identify_planning_tokens,
    extract_sample_results,
    SampleSequence,
    to_python_list,
    to_python_float_list,
    init_wandb,
    log_wandb,
    finish_wandb,
    py_int,
    py_float,
    py_len,
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
# Reward / grading (native Mojo)
# ---------------------------------------------------------------------------


fn extract_boxed(text: String) -> String:
    """Extract \\boxed{...} answer from MATH solution text."""
    var marker = "\\boxed{"
    var marker_len = len(marker)

    var idx = text.rfind(marker)
    if idx == -1:
        return String("")

    var start = idx + marker_len
    var depth = 1
    var pos = start
    var text_len = len(text)
    var bytes = text.as_bytes()
    while pos < text_len and depth > 0:
        if bytes[pos] == UInt8(ord("{")):
            depth += 1
        elif bytes[pos] == UInt8(ord("}")):
            depth -= 1
        pos += 1

    var extracted = String(text[start : pos - 1])
    return String(extracted.strip())


fn grade_answer(given: String, expected: String) -> Bool:
    """Simple string-match grading. Strips whitespace."""
    return String(given.strip()) == String(expected.strip())


fn get_reward(response: String, answer: String) -> Float64:
    """Binary correctness reward for MATH problems."""
    var given = extract_boxed(response)
    if grade_answer(given, answer):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Composable advantage pipeline (native)
# ---------------------------------------------------------------------------


struct AdvantageResult:
    """Result from compute_composable_advantages."""

    var token_advs: List[List[Float64]]
    var has_stats: Bool
    var stats: EntropyStats

    fn __init__(out self, var token_advs: List[List[Float64]], has_stats: Bool, var stats: EntropyStats):
        self.token_advs = token_advs^
        self.has_stats = has_stats
        self.stats = stats^


fn compute_composable_advantages(
    rewards_G: List[Float64],
    sampled_tokens_G: List[List[Int]],
    logprobs_G: List[List[Float64]],
    tokenizer: PythonObject,
    *,
    advantage_mode: String = "grpo",
    transform_mode: String = "none",
    gtpo_beta: Float64 = 0.1,
    hicra_alpha: Float64 = 0.2,
    sepa_lambda: Float64 = 0.0,
    strategic_grams: List[String] = List[String](),
) raises -> AdvantageResult:
    """Compute token-level advantages with composable transforms."""
    var grams = strategic_grams.copy()
    if len(grams) == 0:
        grams = default_strategic_grams()

    # Step 1: Episode-level advantages
    var advantages_G: List[Float64]
    if advantage_mode == "maxrl":
        advantages_G = compute_maxrl_advantages(rewards_G)
    else:
        advantages_G = compute_grpo_advantages(rewards_G)

    # Step 2: Token-level expansion
    if transform_mode == "none":
        var all_token_advs = List[List[Float64]]()
        for i in range(len(sampled_tokens_G)):
            var n_tokens = len(sampled_tokens_G[i])
            all_token_advs.append(List[Float64](length=n_tokens, fill=advantages_G[i]))
        return AdvantageResult(all_token_advs^, False, EntropyStats())

    # For all GTPO-based transforms, we need entropies
    var all_token_advs = List[List[Float64]]()
    var all_exec_entropies = List[Float64]()
    var all_plan_entropies = List[Float64]()

    var needs_planning = transform_mode == "gtpo_hicra" or transform_mode == "gtpo_sepa"

    for idx in range(len(sampled_tokens_G)):
        var tokens = sampled_tokens_G[idx].copy()
        var logprobs = logprobs_G[idx].copy()
        var advantage = advantages_G[idx]

        # Entropy proxy: -logprob
        var entropies = List[Float64](capacity=len(logprobs))
        for j in range(len(logprobs)):
            entropies.append(-logprobs[j])

        var planning_mask: List[Int]
        if needs_planning:
            planning_mask = identify_planning_tokens(
                tokens, tokenizer, grams
            )
        else:
            planning_mask = List[Int](length=len(tokens), fill=0)

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
# Training loop
# ---------------------------------------------------------------------------


fn train(config: TrainConfig) raises:
    """Main training loop using the Tinker API."""
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

    # Load tokenizer
    print("Loading tokenizer for " + config.model + " ...")
    var tokenizer = load_tokenizer(config.model)

    # Connect to Tinker
    print("Connecting to Tinker...")
    var service_client = create_tinker_client(config.base_url)

    # Create LoRA training client
    print("Creating LoRA training client (model=" + config.model + ", rank=" + String(config.lora_rank) + ")...")
    var training_client = create_lora_training_client(
        service_client, config.model, config.lora_rank
    )
    print("Training client ready.")

    # Load dataset
    print("Loading MATH dataset...")
    var examples = load_math_dataset(max_examples=config.max_examples)
    print("Loaded " + String(len(examples)) + " examples")

    # SEPA controller
    var sepa_controller = SEPAController(
        sepa_steps=config.sepa_steps,
        sepa_schedule=config.sepa_schedule,
        sepa_delay_steps=config.sepa_delay_steps,
        sepa_correct_rate_gate=config.sepa_correct_rate_gate,
    )

    # Strategic grams
    var strategic_grams = default_strategic_grams()

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
        wandb_run = init_wandb(config.wandb_project, run_name, wandb_config)
        print("Wandb initialized: " + config.wandb_project + "/" + run_name)

    # Training loop
    var example_idx = 0
    var total_correct = 0
    var total_completions = 0
    var sepa_lambda_val: Float64 = 0.0

    for batch_idx in range(config.max_steps):
        var step_start_ns = perf_counter_ns()

        # 1. Get sampling client
        var sampling_client = save_weights_and_get_sampling_client(
            training_client, "step_" + String(batch_idx)
        )

        # 2. Select prompts and submit sample requests
        var batch_prompts = List[String]()
        var batch_prompt_ids = List[List[Int]]()
        var batch_answers = List[String]()
        var sample_futures = List[PythonObject]()

        for _ in range(config.batch_size):
            var ex_idx = example_idx % len(examples)
            var example = examples[ex_idx].copy()
            example_idx += 1

            var prompt_text = example.problem
            var answer = example.answer
            var prompt_ids = encode_prompt(tokenizer, prompt_text)

            var future = sample(
                sampling_client, prompt_ids,
                config.group_size, config.max_tokens,
                config.temperature,
            )
            sample_futures.append(future)
            batch_prompts.append(prompt_text)
            batch_prompt_ids.append(prompt_ids^)
            batch_answers.append(answer)

        # 3. Collect results, compute advantages, build datums
        var datums_list = Python.import_module("builtins").list()
        var batch_rewards = List[Float64]()
        var batch_correct = 0
        var batch_max_token_hits = 0
        var batch_total_completions = 0
        var batch_entropy_stats = List[EntropyStats]()
        var all_sampled_tokens = List[List[Int]]()
        var all_logprobs = List[List[Float64]]()

        for f_idx in range(len(sample_futures)):
            var future = sample_futures[f_idx]
            var prompt_text = batch_prompts[f_idx]
            var prompt_ids = batch_prompt_ids[f_idx].copy()
            var answer = batch_answers[f_idx]
            var ob_len = len(prompt_ids)

            var sample_result = future.result()
            var sequences = extract_sample_results(sample_result)

            var rewards_G = List[Float64]()
            var sampled_tokens_G = List[List[Int]]()
            var logprobs_G = List[List[Float64]]()

            for s_idx in range(len(sequences)):
                var completion_text = decode(tokenizer, sequences[s_idx].tokens)
                var reward = get_reward(completion_text, answer)
                rewards_G.append(reward)
                sampled_tokens_G.append(sequences[s_idx].tokens.copy())
                logprobs_G.append(sequences[s_idx].logprobs.copy())

            for s_idx in range(len(sampled_tokens_G)):
                all_sampled_tokens.append(sampled_tokens_G[s_idx].copy())
                all_logprobs.append(logprobs_G[s_idx].copy())

            for r_idx in range(len(rewards_G)):
                batch_rewards.append(rewards_G[r_idx])
                if rewards_G[r_idx] > 0.5:
                    batch_correct += 1

            # Track max-token truncations
            for s_idx in range(len(sequences)):
                batch_total_completions += 1
                if len(sequences[s_idx].tokens) >= config.max_tokens:
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

            # Compute advantages
            var sepa_lam: Float64 = 0.0
            if config.transform_mode == "gtpo_sepa":
                sepa_lam = sepa_controller.resolve_lambda(step=Float64(batch_idx))

            var adv_result = compute_composable_advantages(
                rewards_G,
                sampled_tokens_G,
                logprobs_G,
                tokenizer,
                advantage_mode=config.advantage_mode,
                transform_mode=config.transform_mode,
                gtpo_beta=config.gtpo_beta,
                hicra_alpha=config.hicra_alpha,
                sepa_lambda=sepa_lam,
                strategic_grams=strategic_grams,
            )
            var token_advs_G = adv_result.token_advs.copy()
            if adv_result.has_stats:
                batch_entropy_stats.append(adv_result.stats.copy())

            # Build Datum objects
            for s_idx in range(len(sequences)):
                # Full sequence: prompt + completion
                var seq_tokens = sequences[s_idx].tokens.copy()
                var seq_logprobs = sequences[s_idx].logprobs.copy()
                var full_tokens = List[Int](capacity=ob_len + len(seq_tokens))
                for p_idx in range(ob_len):
                    full_tokens.append(prompt_ids[p_idx])
                for t_idx in range(len(seq_tokens)):
                    full_tokens.append(seq_tokens[t_idx])

                # Padded logprobs: zeros for prompt
                var padded_logprobs = List[Float64](length=ob_len, fill=0.0)
                for lp_idx in range(len(seq_logprobs)):
                    padded_logprobs.append(seq_logprobs[lp_idx])

                # Padded advantages: zeros for prompt
                var padded_advantages = List[Float64](length=ob_len, fill=0.0)
                for a_idx in range(len(token_advs_G[s_idx])):
                    padded_advantages.append(token_advs_G[s_idx][a_idx])

                var datum = build_datum(full_tokens, padded_logprobs, padded_advantages)
                datums_list.append(datum)

            # Per-generation emergence logging
            for s_idx in range(len(sequences)):
                var completion_text = decode(tokenizer, sequences[s_idx].tokens)
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
                gen_entries.append(json_int("num_tokens", len(sequences[s_idx].tokens)))
                generations_logger.log(gen_entries)

        # 4. SEPA state updates
        total_completions += len(batch_rewards)
        total_correct += batch_correct
        var correct_rate: Float64 = 0.0
        if len(batch_rewards) > 0:
            correct_rate = Float64(batch_correct) / Float64(len(batch_rewards))

        if config.transform_mode == "gtpo_sepa":
            sepa_controller.observe_correct_rate(Optional[Float64](correct_rate))

            if sepa_controller.enabled() and sepa_controller.sepa_schedule == "auto":
                for t_idx in range(len(all_sampled_tokens)):
                    var tokens = all_sampled_tokens[t_idx].copy()
                    var logprobs = all_logprobs[t_idx].copy()
                    var planning_mask = identify_planning_tokens(
                        tokens, tokenizer, strategic_grams
                    )
                    var exec_entropies = List[Float64]()
                    for e_idx in range(len(logprobs)):
                        if planning_mask[e_idx] == 0:
                            exec_entropies.append(-logprobs[e_idx])
                    sepa_controller.update_auto_state(exec_entropies)

        # 5. Train: forward_backward + optim_step
        var num_datums = py_len(datums_list)
        if num_datums == 0:
            print("Step " + String(batch_idx) + ": no informative datums, skipping.")
            continue

        print(
            "Step " + String(batch_idx) + ": submitting "
            + String(num_datums) + " datums for training..."
        )

        var fwd_bwd_future = forward_backward(training_client, datums_list)
        var optim_future = optim_step(training_client, config.lr, config.weight_decay)

        var fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_future.result()

        var step_time_ns = perf_counter_ns() - step_start_ns
        var step_time_s = Float64(step_time_ns) / 1_000_000_000.0

        # 6. Logging
        var mean_reward: Float64 = 0.0
        if len(batch_rewards) > 0:
            var reward_sum: Float64 = 0.0
            for r_idx in range(len(batch_rewards)):
                reward_sum += batch_rewards[r_idx]
            mean_reward = reward_sum / Float64(len(batch_rewards))

        var running_correct_rate: Float64 = 0.0
        if total_completions > 0:
            running_correct_rate = Float64(total_correct) / Float64(total_completions)

        # Extract loss
        var loss_value: Float64 = 0.0
        var builtins = Python.import_module("builtins")
        var has_metrics = builtins.hasattr(fwd_bwd_result, "metrics")
        if Python.is_true(has_metrics) and Python.is_true(fwd_bwd_result.metrics):
            var loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
            loss_value = py_float(loss_sum) / Float64(max(num_datums, 1))

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
            save_state(training_client, ckpt_name)
            print("Saved checkpoint: " + ckpt_name)

    # Save final checkpoint
    save_state(training_client, "final")
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
# Entry point
# ---------------------------------------------------------------------------


fn main() raises:
    var config = parse_args()
    train(config)
