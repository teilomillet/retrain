"""Advantage trait abstractions and built-in implementations.

Two traits model the existing two-stage pipeline:
  1. EpisodeAdvantageFn — episode-level advantage (e.g. GRPO, MaxRL)
  2. TokenTransformFn — token-level expansion/weighting (e.g. uniform, GTPO, HICRA, SEPA)

Built-in structs delegate to the pure math functions in advantages.mojo.
The math functions themselves do NOT change.
"""

from src.advantages import (
    compute_grpo_advantages,
    compute_maxrl_advantages,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_entropy_stats,
    EntropyStats,
)


# ---------------------------------------------------------------------------
# AdvantageResult (moved from main.mojo)
# ---------------------------------------------------------------------------


struct AdvantageResult:
    """Result from token-level advantage computation."""

    var token_advs: List[List[Float64]]
    var has_stats: Bool
    var stats: EntropyStats

    fn __init__(out self, var token_advs: List[List[Float64]], has_stats: Bool, var stats: EntropyStats):
        self.token_advs = token_advs^
        self.has_stats = has_stats
        self.stats = stats^


# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------


trait EpisodeAdvantageFn(Movable):
    """Compute episode-level advantages from per-completion rewards."""

    fn compute(self, rewards: List[Float64]) -> List[Float64]:
        ...


trait TokenTransformFn(Movable):
    """Expand episode advantages to token-level, with optional weighting."""

    fn transform(
        self,
        advantages: List[Float64],
        logprobs: List[List[Float64]],
        planning_masks: List[List[Int]],
    ) raises -> AdvantageResult:
        ...

    fn update_sepa_lambda(mut self, sepa_lambda: Float64):
        """Update SEPA pooling strength. No-op for non-SEPA transforms."""
        ...


# ---------------------------------------------------------------------------
# Episode advantage implementations
# ---------------------------------------------------------------------------


struct GRPOAdvantage(EpisodeAdvantageFn):
    """Vanilla reward centering: A_i = r_i - mean(r)."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn compute(self, rewards: List[Float64]) -> List[Float64]:
        return compute_grpo_advantages(rewards)


struct MaxRLAdvantage(EpisodeAdvantageFn):
    """Inverse success-rate reweighting: A_i = (r_i - mean(r)) / (mean(r) + eps)."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn compute(self, rewards: List[Float64]) -> List[Float64]:
        return compute_maxrl_advantages(rewards)


# ---------------------------------------------------------------------------
# Token transform implementations
# ---------------------------------------------------------------------------


struct UniformExpand(TokenTransformFn):
    """Broadcast episode advantage uniformly to all tokens (no entropy weighting)."""

    fn __init__(out self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass

    fn update_sepa_lambda(mut self, sepa_lambda: Float64):
        pass

    fn transform(
        self,
        advantages: List[Float64],
        logprobs: List[List[Float64]],
        planning_masks: List[List[Int]],
    ) raises -> AdvantageResult:
        var all_token_advs = List[List[Float64]]()
        for i in range(len(logprobs)):
            var n_tokens = len(logprobs[i])
            all_token_advs.append(List[Float64](length=n_tokens, fill=advantages[i]))
        return AdvantageResult(all_token_advs^, False, EntropyStats())


struct GTPOTransform(TokenTransformFn):
    """GTPO entropy-weighted credit assignment."""

    var beta: Float64

    fn __init__(out self, beta: Float64 = 0.1):
        self.beta = beta

    fn __moveinit__(out self, deinit take: Self):
        self.beta = take.beta

    fn update_sepa_lambda(mut self, sepa_lambda: Float64):
        pass

    fn transform(
        self,
        advantages: List[Float64],
        logprobs: List[List[Float64]],
        planning_masks: List[List[Int]],
    ) raises -> AdvantageResult:
        var all_token_advs = List[List[Float64]]()
        var all_exec_entropies = List[Float64]()
        var all_plan_entropies = List[Float64]()

        for idx in range(len(logprobs)):
            var lp = logprobs[idx].copy()
            var advantage = advantages[idx]
            var planning_mask = planning_masks[idx].copy()

            var entropies = List[Float64](capacity=len(lp))
            for j in range(len(lp)):
                entropies.append(-lp[j])

            for j in range(len(entropies)):
                if planning_mask[j] != 0:
                    all_plan_entropies.append(entropies[j])
                else:
                    all_exec_entropies.append(entropies[j])

            var token_advs = apply_gtpo_weighting(advantage, entropies, beta=self.beta)
            all_token_advs.append(token_advs^)

        var stats = compute_entropy_stats(all_exec_entropies, all_plan_entropies)
        return AdvantageResult(all_token_advs^, True, stats^)


struct GTPOHicraTransform(TokenTransformFn):
    """GTPO entropy weighting + HICRA planning token amplification."""

    var beta: Float64
    var alpha: Float64

    fn __init__(out self, beta: Float64 = 0.1, alpha: Float64 = 0.2):
        self.beta = beta
        self.alpha = alpha

    fn __moveinit__(out self, deinit take: Self):
        self.beta = take.beta
        self.alpha = take.alpha

    fn update_sepa_lambda(mut self, sepa_lambda: Float64):
        pass

    fn transform(
        self,
        advantages: List[Float64],
        logprobs: List[List[Float64]],
        planning_masks: List[List[Int]],
    ) raises -> AdvantageResult:
        var all_token_advs = List[List[Float64]]()
        var all_exec_entropies = List[Float64]()
        var all_plan_entropies = List[Float64]()

        for idx in range(len(logprobs)):
            var lp = logprobs[idx].copy()
            var advantage = advantages[idx]
            var planning_mask = planning_masks[idx].copy()

            var entropies = List[Float64](capacity=len(lp))
            for j in range(len(lp)):
                entropies.append(-lp[j])

            for j in range(len(entropies)):
                if planning_mask[j] != 0:
                    all_plan_entropies.append(entropies[j])
                else:
                    all_exec_entropies.append(entropies[j])

            var token_advs = apply_gtpo_weighting(advantage, entropies, beta=self.beta)
            token_advs = apply_hicra(token_advs, planning_mask, alpha=self.alpha)
            all_token_advs.append(token_advs^)

        var stats = compute_entropy_stats(all_exec_entropies, all_plan_entropies)
        return AdvantageResult(all_token_advs^, True, stats^)


struct GTPOSepaTransform(TokenTransformFn):
    """GTPO entropy weighting + SEPA selective entropy pooling."""

    var beta: Float64
    var sepa_lambda: Float64

    fn __init__(out self, beta: Float64 = 0.1, sepa_lambda: Float64 = 0.0):
        self.beta = beta
        self.sepa_lambda = sepa_lambda

    fn __moveinit__(out self, deinit take: Self):
        self.beta = take.beta
        self.sepa_lambda = take.sepa_lambda

    fn update_sepa_lambda(mut self, sepa_lambda: Float64):
        self.sepa_lambda = sepa_lambda

    fn transform(
        self,
        advantages: List[Float64],
        logprobs: List[List[Float64]],
        planning_masks: List[List[Int]],
    ) raises -> AdvantageResult:
        var all_token_advs = List[List[Float64]]()
        var all_exec_entropies = List[Float64]()
        var all_plan_entropies = List[Float64]()

        for idx in range(len(logprobs)):
            var lp = logprobs[idx].copy()
            var advantage = advantages[idx]
            var planning_mask = planning_masks[idx].copy()

            var entropies = List[Float64](capacity=len(lp))
            for j in range(len(lp)):
                entropies.append(-lp[j])

            for j in range(len(entropies)):
                if planning_mask[j] != 0:
                    all_plan_entropies.append(entropies[j])
                else:
                    all_exec_entropies.append(entropies[j])

            if self.sepa_lambda > 0.0:
                entropies = apply_sepa_pooling(entropies, planning_mask, self.sepa_lambda)

            var token_advs = apply_gtpo_weighting(advantage, entropies, beta=self.beta)
            all_token_advs.append(token_advs^)

        var stats = compute_entropy_stats(all_exec_entropies, all_plan_entropies)
        return AdvantageResult(all_token_advs^, True, stats^)
