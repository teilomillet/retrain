"""Test-time training to discover over a single problem.

This runner keeps a reusable archive of candidate solutions for one problem,
samples new attempts conditioned on archived candidates, and trains online on
its own search experience.

The initial MVP is intentionally scoped to single-turn tasks. It works with
standard retrain reward functions and single-turn verifiers environments, and
uses the existing backend abstraction so Tinker and local backends work
without special handling.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from transformers import AutoTokenizer

from retrain.advantages import (
    EntropyStats,
    apply_batch_advantage_normalization,
    compute_algorithm_advantages,
    compute_composable_advantages,
)
from retrain.backpressure import StepObservation
from retrain.data import Example
from retrain.flow import build_flow
from retrain.logging_utils import JsonlLogger
from retrain.runtime_support import (
    TokenTextLookup,
    decode_sequence_groups,
)
from retrain.trainer import (
    _CORRECT_THRESHOLD,
    _TRAINER_STATE_FILE,
    _apply_advantage_cap,
    _assert_uniform_completion_advantages_for_non_preserving_backend,
    _format_loss_for_display,
    _prepare_algorithm_params_for_step,
    _prepare_transform_params_for_step,
    _print_backend_capability_summary,
    _print_config_summary,
    _summarize_reward_ties,
)
from retrain.training_runner import (
    TrainingRunResult,
    build_run_result,
    failed_run_result,
)
from retrain.type_defs import ExampleInfoLike, PromptLike
from retrain.verifiers_bridge import (
    encode_prompt_for_sampling,
    is_multiturn_environment,
    load_examples_from_environment,
    load_verifiers_environment,
    prompt_preview,
    score_singleturn_group,
)


def _param_float(params: dict[str, object], key: str, default: float) -> float:
    raw = params.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return default
    return default


def _param_int(params: dict[str, object], key: str, default: int) -> int:
    raw = params.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


@dataclass
class DiscoverEntry:
    """One archived candidate solution."""

    entry_id: int
    text: str
    reward: float
    parent_id: int | None
    created_step: int
    depth: int
    expansions: int = 0
    best_child_reward: float | None = None
    children: list[int] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        if self.best_child_reward is None:
            return self.reward
        return max(self.reward, self.best_child_reward)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class DiscoverArchive:
    """Archive of reusable states for a single discovery problem."""

    def __init__(self, empty_reward: float) -> None:
        self.entries: dict[int, DiscoverEntry] = {
            0: DiscoverEntry(
                entry_id=0,
                text="",
                reward=float(empty_reward),
                parent_id=None,
                created_step=-1,
                depth=0,
            )
        }
        self.total_expansions = 0
        self._next_id = 1

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: int) -> DiscoverEntry:
        return self.entries[entry_id]

    def best_entry(self) -> DiscoverEntry:
        return max(
            self.entries.values(),
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
        )

    def record_selection(self, entry_id: int) -> None:
        self.total_expansions += 1
        current = entry_id
        while current is not None:
            entry = self.entries[current]
            entry.expansions += 1
            current = entry.parent_id

    def add_attempt(
        self,
        *,
        parent_id: int,
        text: str,
        reward: float,
        step: int,
    ) -> DiscoverEntry:
        parent = self.entries[parent_id]
        entry = DiscoverEntry(
            entry_id=self._next_id,
            text=text,
            reward=float(reward),
            parent_id=parent_id,
            created_step=step,
            depth=parent.depth + 1,
        )
        self.entries[entry.entry_id] = entry
        self._next_id += 1
        parent.children.append(entry.entry_id)
        self._update_best_child_chain(parent_id, entry.reward)
        return entry

    def _update_best_child_chain(self, parent_id: int | None, reward: float) -> None:
        current = parent_id
        while current is not None:
            entry = self.entries[current]
            if entry.best_child_reward is None or reward > entry.best_child_reward:
                entry.best_child_reward = reward
            current = entry.parent_id

    def _recompute_best_child_chain(self, parent_id: int | None) -> None:
        current = parent_id
        while current is not None:
            entry = self.entries[current]
            best: float | None = None
            for child_id in entry.children:
                child = self.entries[child_id]
                candidate = child.q_value
                if best is None or candidate > best:
                    best = candidate
            entry.best_child_reward = best
            current = entry.parent_id

    def prune(self, max_entries: int) -> None:
        while len(self.entries) > max_entries:
            candidates = [
                entry
                for entry in self.entries.values()
                if entry.entry_id != 0 and entry.is_leaf
            ]
            if not candidates:
                return
            victim = min(
                candidates,
                key=lambda e: (e.reward, e.expansions, e.created_step, e.depth, e.entry_id),
            )
            parent_id = victim.parent_id
            if parent_id is not None:
                parent = self.entries[parent_id]
                parent.children = [cid for cid in parent.children if cid != victim.entry_id]
            del self.entries[victim.entry_id]
            self._recompute_best_child_chain(parent_id)

    def _rank_priors(self) -> dict[int, float]:
        ordered = sorted(
            self.entries.values(),
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
            reverse=True,
        )
        n = len(ordered)
        denom = n * (n + 1) / 2.0
        priors: dict[int, float] = {}
        for idx, entry in enumerate(ordered):
            priors[entry.entry_id] = (n - idx) / denom
        return priors

    def select(self, batch_size: int, exploration: float) -> list[DiscoverEntry]:
        if batch_size <= 0:
            return []
        priors = self._rank_priors()

        def _score(entry: DiscoverEntry) -> tuple[float, float, float, int]:
            prior = priors.get(entry.entry_id, 0.0)
            bonus = (
                exploration
                * prior
                * math.sqrt(1.0 + float(self.total_expansions))
                / (1.0 + float(entry.expansions))
            )
            return (entry.q_value + bonus, entry.reward, -float(entry.depth), -entry.entry_id)

        ordered = sorted(self.entries.values(), key=_score, reverse=True)
        if not ordered:
            return []
        selected = ordered[:batch_size]
        while len(selected) < batch_size:
            selected.append(ordered[len(selected) % len(ordered)])
        return selected

    def context_entries(self, start_id: int, limit: int) -> list[DiscoverEntry]:
        if limit <= 0:
            return []
        others = [
            entry for entry in self.entries.values() if entry.entry_id != start_id and entry.text
        ]
        return sorted(
            others,
            key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
            reverse=True,
        )[:limit]


def build_discovery_prompt(
    base_prompt: PromptLike,
    *,
    start_text: str,
    context_entries: list[DiscoverEntry],
    candidate_char_limit: int = 1200,
    context_char_limit: int = 600,
) -> PromptLike:
    """Append reusable discovery memory to a prompt."""

    start_text = _truncate_text(start_text.strip(), candidate_char_limit)
    context_entries = [entry for entry in context_entries if entry.text.strip()]
    if not start_text and not context_entries:
        return base_prompt

    blocks: list[str] = []
    if start_text:
        blocks.append(
            "Current candidate to improve:\n"
            f"{start_text}"
        )
    if context_entries:
        rendered = []
        for idx, entry in enumerate(context_entries, start=1):
            rendered.append(
                f"{idx}. reward={entry.reward:.4f}\n"
                f"{_truncate_text(entry.text.strip(), context_char_limit)}"
            )
        blocks.append("Other promising attempts:\n" + "\n\n".join(rendered))
    blocks.append(
        "Improve on the current candidate if possible. Return only the full improved solution."
    )
    memory = "Discovery memory:\n\n" + "\n\n".join(blocks)

    if isinstance(base_prompt, str):
        return base_prompt.rstrip() + "\n\n" + memory

    messages = [dict(msg) for msg in base_prompt]
    messages.append({"role": "user", "content": memory})
    return messages


def _write_discovery_summary(log_dir: Path, archive: DiscoverArchive) -> None:
    best = archive.best_entry()
    top_entries = sorted(
        archive.entries.values(),
        key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
        reverse=True,
    )[:10]
    payload = {
        "archive_size": len(archive),
        "total_expansions": archive.total_expansions,
        "best_entry_id": best.entry_id,
        "best_reward": best.reward,
        "best_depth": best.depth,
        "best_text": best.text,
        "top_entries": [
            {
                "entry_id": entry.entry_id,
                "parent_id": entry.parent_id,
                "reward": entry.reward,
                "depth": entry.depth,
                "expansions": entry.expansions,
                "text_preview": _truncate_text(entry.text, 240),
            }
            for entry in top_entries
        ],
    }
    out_path = log_dir / "ttt_discover.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_discovery_source(config) -> tuple[Example, object | None, object | None]:
    """Load the single problem used by TTT-Discover."""

    from retrain.registry import get_registry

    verifiers_env = None
    reward_fn = None
    if config.environment_provider == "verifiers":
        verifiers_env = load_verifiers_environment(config)
        if is_multiturn_environment(verifiers_env):
            raise NotImplementedError(
                "trainer='ttt_discover' currently supports single-turn problems only. "
                "Multi-turn environment state reuse is not implemented yet."
            )
        examples = load_examples_from_environment(verifiers_env, config)
    else:
        examples = get_registry("data_source").create(config.data_source, config).load()
        reward_fn = get_registry("reward").create(config.reward_type, config)

    if not examples:
        raise RuntimeError("Dataset is empty - cannot run TTT-Discover.")
    if len(examples) > 1:
        print(
            f"TTT-Discover uses one problem per run; loaded {len(examples)} examples "
            "and will optimize on the first one only."
        )
    return examples[0], verifiers_env, reward_fn


def _score_completion_texts(
    *,
    verifiers_env: object | None,
    reward_fn: object | None,
    prompt: PromptLike,
    answer: str,
    task: str,
    info: ExampleInfoLike,
    completion_texts: list[str],
) -> list[float]:
    if verifiers_env is None:
        assert reward_fn is not None
        scorer = cast(object, reward_fn)
        return [
            float(cast(float, scorer.score(text, answer)))  # type: ignore[attr-defined]
            for text in completion_texts
        ]
    return score_singleturn_group(
        verifiers_env,
        prompt=prompt,
        answer=answer,
        task=task,
        info=info,
        completion_texts=completion_texts,
    )

class TTTDiscoverRunner:
    """Online discovery runner with reusable start states."""

    def run(self, config) -> TrainingRunResult:
        metrics_logger: JsonlLogger | None = None
        steps_logger: JsonlLogger | None = None
        generations_logger: JsonlLogger | None = None
        try:
            if config.resume_from:
                raise NotImplementedError(
                    "trainer='ttt_discover' does not support resume_from yet."
                )

            algorithm_params = dict(config.effective_algorithm_params)
            context_k = max(0, _param_int(algorithm_params, "context_k", 3))
            puct_c = _param_float(algorithm_params, "puct_c", 1.0)
            archive_max_entries = max(
                max(64, config.batch_size * config.group_size * 2),
                _param_int(algorithm_params, "archive_max_entries", 4096),
            )
            candidate_char_limit = max(
                64, _param_int(algorithm_params, "candidate_char_limit", 1200)
            )
            context_char_limit = max(
                64, _param_int(algorithm_params, "context_char_limit", 600)
            )

            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            metrics_logger = JsonlLogger(str(log_dir / "metrics.jsonl"))
            steps_logger = JsonlLogger(str(log_dir / "steps.jsonl"))
            generations_logger = JsonlLogger(
                str(log_dir / "generations.jsonl"),
                flush_every=32,
                flush_interval_s=1.0,
            )

            _print_config_summary(config)

            print("Loading discovery problem...")
            example, verifiers_env, reward_fn = _load_discovery_source(config)

            flow = build_flow(config, gpu=True)
            helper = flow.backend
            assert helper is not None
            detector = flow.planning_detector
            sepa_controller = flow.sepa_controller
            assert sepa_controller is not None
            backpressure = flow.backpressure
            assert backpressure is not None
            backend_caps = flow.backend_capabilities
            _print_backend_capability_summary(
                config.backend,
                flow.backend_capability_source,
                backend_caps.reports_sync_loss,
                backend_caps.preserves_token_advantages,
                backend_caps.supports_checkpoint_resume,
                backend_caps.resume_runtime_dependent,
            )

            print(f"Loading tokenizer for {config.model} ...")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                trust_remote_code=True,
            )
            token_lookup = TokenTextLookup(tokenizer)

            empty_reward = _score_completion_texts(
                verifiers_env=verifiers_env,
                reward_fn=reward_fn,
                prompt=example.prompt,
                answer=example.reference,
                task=example.task,
                info=example.info,
                completion_texts=[""],
            )[0]
            archive = DiscoverArchive(empty_reward=empty_reward)

            current_batch_size = config.batch_size
            current_group_size = config.group_size
            total_correct = 0
            total_completions = 0
            delight_eta_ema: float | None = None

            for step in range(config.max_steps):
                step_start = time.perf_counter()
                helper.checkpoint(f"step_{step}")  # type: ignore[unresolved-attribute]

                selected_entries = archive.select(current_batch_size, puct_c)
                batch_prompt_objs: list[PromptLike] = []
                batch_prompt_ids: list[list[int]] = []
                batch_prompt_previews: list[str] = []
                for entry in selected_entries:
                    prompt_obj = build_discovery_prompt(
                        example.prompt,
                        start_text=entry.text,
                        context_entries=archive.context_entries(entry.entry_id, context_k),
                        candidate_char_limit=candidate_char_limit,
                        context_char_limit=context_char_limit,
                    )
                    batch_prompt_objs.append(prompt_obj)
                    batch_prompt_ids.append(
                        encode_prompt_for_sampling(tokenizer, prompt_obj)
                    )
                    batch_prompt_previews.append(prompt_preview(prompt_obj))

                batch_rewards: list[float] = []
                batch_correct = 0
                batch_total_completions = 0
                batch_max_token_hits = 0
                batch_reward_tie_eligible_groups = 0
                batch_reward_tie_groups = 0
                batch_reward_uniform_groups = 0
                batch_reward_tied_pairs = 0
                batch_reward_total_pairs = 0
                batch_reward_unique_fraction_sum = 0.0
                batch_surprisal_stats: list[EntropyStats] = []
                batch_adv_results: list = []
                all_logprobs_sepa: list[list[float]] = []
                all_planning_masks_sepa: list[list[int]] = []
                all_datum_tokens: list[list[int]] = []
                all_datum_logprobs: list[list[float]] = []
                all_datum_advantages: list[list[float]] = []

                step_transform_params = _prepare_transform_params_for_step(
                    config.transform_params,
                    delight_eta_prev=delight_eta_ema,
                )
                step_algorithm_params = _prepare_algorithm_params_for_step(
                    config.effective_algorithm_params,
                    delight_eta_prev=delight_eta_ema,
                )

                sepa_lambda_val = 0.0
                if flow.uses_sepa_controller:
                    sepa_lambda_val = sepa_controller.resolve_lambda(step=float(step))

                sample_start = time.perf_counter()
                use_entropy_sampling = (
                    config.uncertainty_kind == "shannon_entropy"
                    and hasattr(helper, "sample_with_entropy")
                )
                precomputed_entropies_batch: list[list[list[float]]] | None = None
                if use_entropy_sampling:
                    enriched_sequences = helper.sample_with_entropy(  # type: ignore[unresolved-attribute]
                        batch_prompt_ids,
                        current_group_size,
                        config.max_tokens,
                        config.temperature,
                        config.top_p,
                    )
                    all_group_sequences = [
                        [(ids, lps) for ids, lps, _ent in group]
                        for group in enriched_sequences
                    ]
                    precomputed_entropies_batch = [
                        [ent if ent is not None else [] for _ids, _lps, ent in group]
                        for group in enriched_sequences
                    ]
                else:
                    all_group_sequences = helper.sample(  # type: ignore[unresolved-attribute]
                        batch_prompt_ids,
                        current_group_size,
                        config.max_tokens,
                        config.temperature,
                        config.top_p,
                    )
                sample_time = time.perf_counter() - sample_start

                decoded_groups = decode_sequence_groups(
                    tokenizer,
                    all_group_sequences,
                    needs_planning=flow.needs_planning,
                    token_lookup=token_lookup if flow.needs_planning else None,
                    detector=detector if flow.needs_planning else None,
                )

                for f_idx, decoded_group in enumerate(decoded_groups):
                    if not decoded_group:
                        continue
                    start_entry = selected_entries[f_idx]
                    archive.record_selection(start_entry.entry_id)
                    prompt_ids = batch_prompt_ids[f_idx]
                    prompt_obj = batch_prompt_objs[f_idx]

                    rewards_G: list[float] = []
                    logprobs_G: list[list[float]] = []
                    planning_masks_G: list[list[int]] = []
                    completion_texts_G: list[str] = []

                    for sample in decoded_group:
                        completion_texts_G.append(sample.text)
                        logprobs_G.append(sample.logprobs)
                        planning_masks_G.append(sample.planning_mask)

                    rewards_G = _score_completion_texts(
                        verifiers_env=verifiers_env,
                        reward_fn=reward_fn,
                        prompt=prompt_obj,
                        answer=example.reference,
                        task=example.task,
                        info=example.info,
                        completion_texts=completion_texts_G,
                    )

                    all_logprobs_sepa.extend(logprobs_G)
                    all_planning_masks_sepa.extend(planning_masks_G)
                    batch_rewards.extend(rewards_G)
                    batch_correct += sum(1 for r in rewards_G if r > _CORRECT_THRESHOLD)
                    batch_total_completions += len(rewards_G)
                    batch_max_token_hits += sum(
                        1
                        for sample in decoded_group
                        if len(sample.token_ids) >= config.max_tokens
                    )

                    for reward, comp_text in zip(rewards_G, completion_texts_G):
                        archive.add_attempt(
                            parent_id=start_entry.entry_id,
                            text=comp_text,
                            reward=reward,
                            step=step,
                        )
                    archive.prune(archive_max_entries)

                    reward_tie_stats = _summarize_reward_ties(rewards_G)
                    if reward_tie_stats["eligible"]:
                        batch_reward_tie_eligible_groups += 1
                        batch_reward_tie_groups += int(reward_tie_stats["has_tie"])
                        batch_reward_uniform_groups += int(reward_tie_stats["is_uniform"])
                        batch_reward_tied_pairs += reward_tie_stats["tied_pairs"]
                        batch_reward_total_pairs += reward_tie_stats["total_pairs"]
                        batch_reward_unique_fraction_sum += (
                            reward_tie_stats["unique_count"] / len(rewards_G)
                        )

                    if reward_tie_stats["is_uniform"]:
                        if config.batch_advantage_norm:
                            print(
                                f"    -> uniform (reward={rewards_G[0]:.3f}, kept for batch norm)"
                            )
                        else:
                            print(
                                f"    -> skipped (all same, reward={rewards_G[0]:.3f})"
                            )
                            continue

                    group_entropies_G: list[list[float]] | None = None
                    if precomputed_entropies_batch is not None:
                        group_entropies_G = precomputed_entropies_batch[f_idx]

                    if config.algorithm_mode:
                        adv_result = compute_algorithm_advantages(
                            rewards_G,
                            logprobs_G,
                            planning_masks_G,
                            algorithm_mode=config.algorithm_mode,
                            params=step_algorithm_params,
                            gtpo_beta=config.gtpo_beta,
                            hicra_alpha=config.hicra_alpha,
                            sepa_lambda=sepa_lambda_val,
                            step=step,
                            token_distributions_G=None,
                            precomputed_entropies_G=group_entropies_G,
                        )
                    else:
                        adv_result = compute_composable_advantages(
                            rewards_G,
                            logprobs_G,
                            planning_masks_G,
                            advantage_mode=config.advantage_mode,
                            transform_mode=config.transform_mode,
                            gtpo_beta=config.gtpo_beta,
                            hicra_alpha=config.hicra_alpha,
                            sepa_lambda=sepa_lambda_val,
                            advantage_params=config.effective_advantage_params,
                            transform_params=step_transform_params,
                            step=step,
                            post_process_params=config.post_process_params,
                            token_distributions_G=None,
                            precomputed_entropies_G=group_entropies_G,
                        )

                    if adv_result.has_stats:
                        batch_surprisal_stats.append(adv_result.stats)
                    if adv_result.extra_metrics:
                        batch_adv_results.append(adv_result)

                    for s_idx, sample in enumerate(decoded_group):
                        seq_advs = adv_result.token_advs[s_idx]
                        full_tokens = list(prompt_ids) + list(sample.token_ids)
                        padded_logprobs = [0.0] * len(prompt_ids) + list(sample.logprobs)
                        padded_advantages = [0.0] * len(prompt_ids) + list(seq_advs)
                        all_datum_tokens.append(full_tokens)
                        all_datum_logprobs.append(padded_logprobs)
                        all_datum_advantages.append(padded_advantages)
                        generations_logger.log(
                            {
                                "step": step,
                                "start_entry_id": start_entry.entry_id,
                                "start_reward": start_entry.reward,
                                "reward": rewards_G[s_idx],
                                "completion": completion_texts_G[s_idx][:800],
                                "prompt": batch_prompt_previews[f_idx],
                                "num_tokens": len(sample.logprobs),
                            }
                        )

                total_completions += len(batch_rewards)
                total_correct += batch_correct
                correct_rate = (
                    batch_correct / len(batch_rewards) if batch_rewards else 0.0
                )

                if flow.uses_sepa_controller:
                    sepa_controller.observe_correct_rate(correct_rate)
                    if (
                        sepa_controller.enabled()
                        and sepa_controller.sepa_schedule == "auto"
                    ):
                        for logprobs, pmask in zip(
                            all_logprobs_sepa,
                            all_planning_masks_sepa,
                        ):
                            exec_ent = [
                                -logprobs[idx]
                                for idx in range(len(logprobs))
                                if pmask[idx] == 0
                            ]
                            sepa_controller.update_auto_state(exec_ent)

                num_datums = len(all_datum_tokens)
                if num_datums == 0:
                    print(f"Step {step}: no informative datums, skipping.")
                    backpressure.observe(
                        StepObservation(
                            step_time_s=time.perf_counter() - step_start,
                            sample_time_s=sample_time,
                            batch_size=current_batch_size,
                            group_size=current_group_size,
                            skipped=True,
                        )
                    )
                    continue

                if not backend_caps.preserves_token_advantages:
                    _assert_uniform_completion_advantages_for_non_preserving_backend(
                        all_datum_logprobs,
                        all_datum_advantages,
                        backend_name=config.backend,
                    )

                batch_norm_metrics: dict[str, float] = {}
                if config.batch_advantage_norm:
                    all_datum_advantages, batch_norm_metrics = (
                        apply_batch_advantage_normalization(all_datum_advantages)
                    )

                adv_cap_fraction = 0.0
                adv_cap_magnitude = 0.0
                if config.adv_clip_max > 0:
                    all_datum_advantages, adv_cap_fraction, adv_cap_magnitude = (
                        _apply_advantage_cap(all_datum_advantages, config.adv_clip_max)
                    )

                train_start = time.perf_counter()
                loss_value = helper.train_step(  # type: ignore[unresolved-attribute]
                    all_datum_tokens,
                    all_datum_logprobs,
                    all_datum_advantages,
                    config.lr,
                    config.weight_decay,
                )
                train_time = time.perf_counter() - train_start
                clip_fraction = getattr(helper, "_clip_fraction", 0.0)
                step_time = time.perf_counter() - step_start

                obs = StepObservation(
                    step_time_s=step_time,
                    sample_time_s=sample_time,
                    train_time_s=train_time,
                    num_datums=num_datums,
                    batch_size=current_batch_size,
                    group_size=current_group_size,
                    total_tokens=sum(len(t) for t in all_datum_tokens),
                    loss=loss_value,
                    skipped=False,
                )
                backpressure.observe(obs)
                bp_decision = backpressure.recommend()
                if config.bp_enabled and bp_decision.action in ("throttle", "increase"):
                    new_bs = bp_decision.recommended_batch_size
                    if new_bs > 0:
                        current_batch_size = max(
                            config.bp_min_batch_size,
                            min(config.bp_max_batch_size, new_bs),
                        )

                mean_reward = (
                    sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
                )
                best_reward = archive.best_entry().reward
                running_correct_rate = (
                    total_correct / total_completions if total_completions > 0 else 0.0
                )
                reward_tie_group_rate = (
                    batch_reward_tie_groups / batch_reward_tie_eligible_groups
                    if batch_reward_tie_eligible_groups > 0
                    else 0.0
                )
                reward_uniform_group_rate = (
                    batch_reward_uniform_groups / batch_reward_tie_eligible_groups
                    if batch_reward_tie_eligible_groups > 0
                    else 0.0
                )
                reward_tie_pair_rate = (
                    batch_reward_tied_pairs / batch_reward_total_pairs
                    if batch_reward_total_pairs > 0
                    else 0.0
                )
                reward_unique_fraction_mean = (
                    batch_reward_unique_fraction_sum / batch_reward_tie_eligible_groups
                    if batch_reward_tie_eligible_groups > 0
                    else 0.0
                )

                metrics: dict[str, float] = {}
                if batch_adv_results:
                    keys = {
                        k
                        for result in batch_adv_results
                        for k in result.extra_metrics
                    }
                    for key in keys:
                        vals = [
                            result.extra_metrics[key]
                            for result in batch_adv_results
                            if key in result.extra_metrics
                        ]
                        if vals:
                            metrics[key] = sum(vals) / len(vals)
                metrics.update(batch_norm_metrics)

                metric_entry: dict[str, object] = {
                    "step": step,
                    "loss": loss_value,
                    "mean_reward": mean_reward,
                    "best_reward": best_reward,
                    "correct_rate": correct_rate,
                    "running_correct_rate": running_correct_rate,
                    "archive_size": len(archive),
                    "start_state_mean_reward": (
                        sum(entry.reward for entry in selected_entries)
                        / len(selected_entries)
                        if selected_entries
                        else 0.0
                    ),
                    "sample_time_s": sample_time,
                    "train_time_s": train_time,
                    "step_time_s": step_time,
                    "batch_size": current_batch_size,
                    "group_size": current_group_size,
                    "clip_fraction": clip_fraction,
                    "adv_cap_fraction": adv_cap_fraction,
                    "adv_cap_magnitude": adv_cap_magnitude,
                    "reward_tie_group_rate": reward_tie_group_rate,
                    "reward_uniform_group_rate": reward_uniform_group_rate,
                    "reward_tie_pair_rate": reward_tie_pair_rate,
                    "reward_unique_fraction_mean": reward_unique_fraction_mean,
                    "condition": flow.condition_label,
                    "trainer": "ttt_discover",
                }
                metric_entry.update(metrics)
                metrics_logger.log(metric_entry)
                steps_logger.log(metric_entry)

                print(
                    f"Step {step} | loss={_format_loss_for_display(loss_value, backend_caps.reports_sync_loss)}"
                    f" | reward={mean_reward:.4f} | best={best_reward:.4f}"
                    f" | archive={len(archive)} | bs={current_batch_size} | gs={current_group_size}"
                )

                if config.save_every > 0 and (step + 1) % config.save_every == 0:
                    ckpt_name = f"checkpoint_step_{step + 1}"
                    helper.save_adapter(config.adapter_path, ckpt_name)  # type: ignore[unresolved-attribute]
                    _write_discovery_summary(log_dir, archive)
                    print(f"Saved checkpoint: {ckpt_name}")

            final_path = helper.save_adapter(  # type: ignore[unresolved-attribute]
                config.adapter_path,
                "final",
            )
            _write_discovery_summary(log_dir, archive)
            state_file = log_dir / _TRAINER_STATE_FILE
            state_file.write_text(
                json.dumps(
                    {
                        "step": config.max_steps - 1,
                        "example_idx": 1,
                        "total_correct": total_correct,
                        "total_completions": total_completions,
                        "current_batch_size": current_batch_size,
                        "current_group_size": current_group_size,
                        "checkpoint_name": "final",
                        "sepa": sepa_controller.state_dict(),
                        "trainer": "ttt_discover",
                    },
                    indent=2,
                )
                + "\n"
            )

            best = archive.best_entry()
            print(
                f"TTT-Discover complete. best_reward={best.reward:.4f}, "
                f"archive_size={len(archive)}, best_entry_id={best.entry_id}"
            )
            if not final_path:
                return failed_run_result(
                    config,
                    failure_status="missing_policy_ref",
                    error_message=(
                        "TTT-Discover completed without returning a policy reference."
                    ),
                )
            return build_run_result(config, policy_ref=final_path)
        except Exception as exc:
            return failed_run_result(
                config,
                failure_status=f"exception:{type(exc).__name__}",
                error_message=str(exc),
            )
        finally:
            if metrics_logger is not None:
                metrics_logger.close()
            if steps_logger is not None:
                steps_logger.close()
            if generations_logger is not None:
                generations_logger.close()
