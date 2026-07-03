"""Single-turn completion rollout execution."""

from __future__ import annotations

import time
from collections.abc import Mapping

from retrain.backends import EntropySamplingHelper, TrainHelper
from retrain.config import TrainConfig
from retrain.environments.verifiers import score_singleturn_group
from retrain.io.log import JsonlLogger
from retrain.planning.types import PlanningDetector
from retrain.rewards.types import RewardFunction
from retrain.training import console
from retrain.training.generations import generation_log_indices, top_surprisal_payload
from retrain.training.prompts import PromptBatch
from retrain.training.rollout.state import RolloutAccumulator
from retrain.training.rollouts import (
    RuntimeCounters,
    TokenTextLookup,
    decode_sequence_groups,
)
from retrain.training.signals import CORRECT_THRESHOLD, compute_group_advantages


def run_singleturn(
    config: TrainConfig,
    helper: TrainHelper,
    tokenizer: object,
    verifiers_env: object | None,
    reward_fn: RewardFunction | None,
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
    runtime_counters: RuntimeCounters,
    generations_logger: JsonlLogger,
    singleturn_scorer=score_singleturn_group,
) -> None:
    """Sample, score, and datum-ize single-turn completion groups."""
    # 10c. Sample completions
    sample_start = time.perf_counter()
    entropy_helper = (
        helper
        if config.uncertainty_kind == "shannon_entropy"
        and isinstance(helper, EntropySamplingHelper)
        else None
    )
    precomputed_entropies_batch: list[list[list[float]]] | None = None
    if entropy_helper is not None:
        enriched_sequences = entropy_helper.sample_with_entropy(
            prompts.ids,
            group_size,
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
        all_group_sequences = helper.sample(
            prompts.ids,
            group_size,
            config.max_tokens,
            config.temperature,
            config.top_p,
        )
    acc.sample_time_s = time.perf_counter() - sample_start

    decoded_groups = decode_sequence_groups(
        tokenizer,
        all_group_sequences,
        needs_planning=needs_planning,
        token_lookup=token_lookup if needs_planning else None,
        detector=detector if needs_planning else None,
        counters=runtime_counters,
    )

    for f_idx, decoded_group in enumerate(decoded_groups):
        prompt_ids = prompts.ids[f_idx]
        answer = prompts.answers[f_idx]
        task = prompts.tasks[f_idx]
        info = prompts.infos[f_idx]
        prompt_obj = prompts.objs[f_idx]

        rewards_G: list[float] = []
        logprobs_G: list[list[float]] = []
        planning_masks_G: list[list[int]] = []
        completion_texts_G: list[str] = []
        for sample in decoded_group:
            completion_texts_G.append(sample.text)
            logprobs_G.append(sample.logprobs)
            planning_masks_G.append(sample.planning_mask)

        if verifiers_env is None:
            assert reward_fn is not None
            for text in completion_texts_G:
                rewards_G.append(reward_fn.score(text, answer))
        else:
            rewards_G = singleturn_scorer(
                verifiers_env,
                prompt=prompt_obj,
                answer=answer,
                task=task,
                info=info,
                completion_texts=completion_texts_G,
            )

        acc.logprobs_sepa.extend(logprobs_G)
        acc.planning_masks_sepa.extend(planning_masks_G)

        for r in rewards_G:
            acc.rewards.append(r)
            if r > CORRECT_THRESHOLD:
                acc.correct += 1

        for sample in decoded_group:
            acc.total_completions += 1
            if len(sample.token_ids) >= config.max_tokens:
                acc.max_token_hits += 1

        console.print_group_summary(rewards_G, answer)
        reward_tie_stats = acc.ties.observe(rewards_G)
        if reward_tie_stats["is_uniform"] and not config.tl_grpo:
            if not console.keep_uniform_group(
                rewards_G,
                batch_advantage_norm=config.batch_advantage_norm,
                keep_for_echo=False,
            ):
                continue

        # Resolve per-group precomputed entropies
        group_entropies_G: list[list[float]] | None = None
        if precomputed_entropies_batch is not None:
            group_entropies_G = precomputed_entropies_batch[f_idx]

        adv_result = compute_group_advantages(
            config,
            rewards_G,
            logprobs_G,
            planning_masks_G,
            step=step,
            sepa_lambda=sepa_lambda,
            algorithm_params=algorithm_params,
            transform_params=transform_params,
            precomputed_entropies_G=group_entropies_G,
        )
        all_token_advs_G = adv_result.token_advs
        if adv_result.has_stats:
            acc.surprisal_stats.append(adv_result.stats)
        if adv_result.extra_metrics:
            acc.adv_results.append(adv_result)

        for sample, token_advs in zip(decoded_group, all_token_advs_G):
            full_tokens = list(prompt_ids) + list(sample.token_ids)
            padded_logprobs = [0.0] * len(prompt_ids) + list(sample.logprobs)
            padded_advantages = [0.0] * len(prompt_ids) + list(token_advs)
            acc.datum_tokens.append(full_tokens)
            acc.datum_logprobs.append(padded_logprobs)
            acc.datum_advantages.append(padded_advantages)
            acc.datum_echo_advantages.append([0.0] * len(full_tokens))
            acc.datum_echo_full_observation_counts.append(0)
            acc.rl_completion_token_count += len(sample.token_ids)
            acc.rl_completion_surprisal_sum += sum(
                -lp for lp in sample.logprobs
            )

        generation_entries: list[dict[str, object]] = []
        selected_generation_indices = (
            generation_log_indices(
                len(decoded_group),
                samples_per_prompt=config.generation_log_samples_per_prompt,
                rewards=rewards_G,
            )
            if generations_logger.enabled
            else []
        )
        for s_idx in selected_generation_indices:
            sample = decoded_group[s_idx]
            gen_entry: dict[str, object] = {
                "step": step,
                "prompt": prompts.previews[f_idx],
                "completion": sample.text[:500],
                "reward": rewards_G[s_idx],
                "num_tokens": len(sample.logprobs),
            }
            # Top-K highest surprisal tokens with decoded text
            top_entries = top_surprisal_payload(
                sample.logprobs,
                sample.token_ids,
                token_lookup,
                limit=config.generation_top_surprisal_limit,
            )
            if top_entries:
                gen_entry["top_surprisal_tokens"] = top_entries
            generation_entries.append(gen_entry)
        if generation_entries:
            generations_logger.log_many(generation_entries)
