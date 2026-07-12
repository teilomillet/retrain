"""Multiturn environment rollout execution."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import cast

from retrain.backends import TrainHelper
from retrain.config import TrainConfig
from retrain.environments.verifiers import run_multiturn_group
from retrain.io.log import JsonlLogger
from retrain.planning.types import PlanningDetector
from retrain.training import console
from retrain.training.echo import build_rollout_echo_datums, merge_echo_build_stats
from retrain.training.generations import generation_log_indices, top_surprisal_payload
from retrain.training.prompts import PromptBatch
from retrain.training.rollout.state import RolloutAccumulator, accumulate_metric_totals
from retrain.training.rollouts import TokenTextLookup
from retrain.training.signals import CORRECT_THRESHOLD, compute_group_advantages


def run_multiturn(
    config: TrainConfig,
    helper: TrainHelper,
    tokenizer: object,
    verifiers_env: object,
    prompts: PromptBatch,
    acc: RolloutAccumulator,
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
    group_runner=run_multiturn_group,
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
        ) = group_runner(
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
            capture_echo_transitions=config.echo_enabled,
        )
        accumulate_metric_totals(
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

            acc.total_completions += len(turns)
            acc.sampled_completion_token_count += len(seq_token_ids)
            acc.sampled_completion_surprisal_sum += sum(-lp for lp in seq_logprobs)
            acc.max_token_hits += sum(turn.is_truncated for turn in turns)

        acc.logprobs_sepa.extend(logprobs_G)
        acc.planning_masks_sepa.extend(planning_masks_G)

        for r in rewards_G:
            acc.rewards.append(r)
            if r > CORRECT_THRESHOLD:
                acc.correct += 1
            if acc.tl_grpo_ema is not None:
                acc.tl_grpo_ema = (
                    config.tl_grpo_ema_decay * acc.tl_grpo_ema
                    + (1 - config.tl_grpo_ema_decay) * r
                )

        console.print_group_summary(rewards_G, answer)
        reward_tie_stats = acc.ties.observe(rewards_G)
        if reward_tie_stats["is_uniform"] and not config.tl_grpo:
            if not console.keep_uniform_group(
                rewards_G,
                batch_advantage_norm=config.batch_advantage_norm,
                keep_for_echo=config.echo_enabled,
            ):
                continue

        adv_result = compute_group_advantages(
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

            eligible_tokens = sum(len(tokens) for tokens in turn_token_ids)
            acc.eligible_completion_token_count += eligible_tokens
            acc.pre_optimizer_nonzero_advantage_token_count += sum(
                abs(value) > 0.0 for row in seq_advs_by_turn for value in row
            )

            if config.echo_enabled:
                rollout_datums, rollout_echo_build = build_rollout_echo_datums(
                    turns_G[s_idx],
                    completion_advantages=seq_advs_by_turn,
                    weight=config.echo_weight,
                    min_prompt_overlap=config.echo_min_prompt_overlap,
                )
                acc.echo_build = merge_echo_build_stats(
                    acc.echo_build,
                    rollout_echo_build,
                )
                if rollout_datums:
                    acc.echo_eligible_rollout_count += 1
                    for rollout_datum in rollout_datums:
                        acc.datum_tokens.append(rollout_datum.tokens)
                        acc.datum_logprobs.append(rollout_datum.logprobs)
                        acc.datum_advantages.append(rollout_datum.advantages)
                        acc.datum_echo_advantages.append(rollout_datum.echo_advantages)
                        acc.datum_echo_terminal_masks.append(
                            rollout_datum.terminal_observation_mask
                        )
                        acc.datum_echo_full_observation_counts.append(
                            rollout_datum.full_observation_count
                        )
                        acc.rl_completion_token_count += (
                            rollout_datum.action_token_count
                        )
                        acc.rl_completion_surprisal_sum += (
                            rollout_datum.action_surprisal_sum
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
                acc.datum_echo_advantages.append([0.0] * len(full_tokens))
                acc.datum_echo_terminal_masks.append([0] * len(full_tokens))
                acc.datum_echo_full_observation_counts.append(0)
                acc.rl_completion_token_count += len(seq_tokens)
                acc.rl_completion_surprisal_sum += sum(-lp for lp in seq_logprobs)

        generation_entries: list[dict[str, object]] = []
        selected_generation_indices = (
            generation_log_indices(
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
                    # Behavior accumulation.
                    acc.behavior_turns += 1
                    if not tl.get("valid", True):
                        acc.behavior_invalid += 1
                    _op = str(tl.get("operation", "unknown"))
                    acc.behavior_actions[_op] = acc.behavior_actions.get(_op, 0) + 1
                gen_entry["turn_log"] = turn_summary
                acc.behavior_resp_lens.append(len(str(gen_entry.get("completion", ""))))
            if s_idx < len(turn_advantages_G) and turn_advantages_G[s_idx]:
                gen_entry["turn_advantages"] = turn_advantages_G[s_idx]
            if s_idx < len(branch_rewards_G) and branch_rewards_G[s_idx]:
                gen_entry["branch_rewards"] = branch_rewards_G[s_idx]
            if s_idx < len(turns_logprobs_G) and turns_logprobs_G[s_idx]:
                turn_lps = turns_logprobs_G[s_idx]
                gen_entry["turn_mean_logprobs"] = [
                    sum(lps) / len(lps) if lps else 0.0 for lps in turn_lps
                ]
                gen_entry["turn_logprob_var"] = [
                    (sum((x - sum(lps) / len(lps)) ** 2 for x in lps) / len(lps))
                    if len(lps) > 1
                    else 0.0
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
                top_entries = top_surprisal_payload(
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
