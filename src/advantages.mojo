"""Native SIMD advantage math for the Tinker training pipeline.

Ports the 6 core functions from textpolicy/tinker/advantages.py into
pure Mojo with SIMD vectorization where beneficial.

Functions:
    compute_grpo_advantages  — vanilla reward centering
    compute_maxrl_advantages — inverse success-rate reweighting
    apply_gtpo_weighting     — entropy-weighted credit assignment
    apply_hicra              — planning token amplification
    apply_sepa_pooling       — selective entropy pooling
    compute_entropy_stats    — summary statistics for logging
"""

from math import abs


# ---------------------------------------------------------------------------
# EntropyStats — summary stats for exec vs plan entropy distributions
# ---------------------------------------------------------------------------


@fieldwise_init
struct EntropyStats(Copyable, Movable, Writable):
    """Summary statistics for execution vs planning entropy distributions."""

    var exec_mean: Float64
    var exec_var: Float64
    var exec_count: Float64
    var plan_mean: Float64
    var plan_var: Float64
    var plan_count: Float64

    fn __init__(out self):
        self.exec_mean = 0.0
        self.exec_var = 0.0
        self.exec_count = 0.0
        self.plan_mean = 0.0
        self.plan_var = 0.0
        self.plan_count = 0.0

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "EntropyStats(exec_mean=", self.exec_mean,
            ", exec_var=", self.exec_var,
            ", plan_mean=", self.plan_mean,
            ", plan_var=", self.plan_var, ")",
        )


# ---------------------------------------------------------------------------
# 0. Baseline GRPO advantages (simple reward centering)
# ---------------------------------------------------------------------------
# Formula: A_i = r_i - mean(r)


fn compute_grpo_advantages(rewards: List[Float64]) -> List[Float64]:
    """Compute vanilla GRPO advantages: simple reward centering.

    Args:
        rewards: Per-completion rewards for one prompt group.

    Returns:
        List of advantages, same length as rewards.
    """
    var n = len(rewards)
    if n == 0:
        return List[Float64]()

    var total: Float64 = 0.0
    for i in range(n):
        total += rewards[i]
    var mean_r = total / Float64(n)

    var result = List[Float64](capacity=n)
    for i in range(n):
        result.append(rewards[i] - mean_r)
    return result^


# ---------------------------------------------------------------------------
# 1. MaxRL advantages (inverse success-rate reweighting)
# ---------------------------------------------------------------------------
# Formula: A_i = (r_i - mean(r)) / (mean(r) + eps)
# When mean(r) ~ 0: all advantages are zero.


fn compute_maxrl_advantages(
    rewards: List[Float64],
    eps: Float64 = 1e-6,
) -> List[Float64]:
    """Compute MaxRL advantages: inverse success-rate reweighting.

    Args:
        rewards: Per-completion rewards (typically binary {0, 1}).
        eps: Numerical stability constant.

    Returns:
        List of advantages, same length as rewards.
    """
    var n = len(rewards)
    if n == 0:
        return List[Float64]()

    var total: Float64 = 0.0
    for i in range(n):
        total += rewards[i]
    var mean_r = total / Float64(n)

    # No signal: all advantages zero.
    if mean_r <= eps:
        return List[Float64](length=n, fill=0.0)

    var denom = mean_r + eps
    var result = List[Float64](capacity=n)
    for i in range(n):
        result.append((rewards[i] - mean_r) / denom)
    return result^


# ---------------------------------------------------------------------------
# 2. GTPO entropy-weighted credit assignment
# ---------------------------------------------------------------------------
# Formula:
#   H_norm(t) = H(t) / mean(H)
#   w(t) = max(0, 1 + beta * (H_norm(t) - 1))
#   A_GTPO(t) = A(t) * w(t)


fn apply_gtpo_weighting(
    advantage: Float64,
    entropies: List[Float64],
    beta: Float64 = 0.1,
) -> List[Float64]:
    """Apply GTPO entropy weighting to produce token-level advantages.

    Args:
        advantage: Scalar advantage for this completion.
        entropies: Per-token entropies.
        beta: Entropy weighting strength. 0.0 disables.

    Returns:
        Token-level advantages, same length as entropies.
    """
    var n = len(entropies)
    if n == 0:
        return List[Float64]()

    # beta=0 -> uniform weighting
    if beta == 0.0:
        return List[Float64](length=n, fill=advantage)

    # Mean-normalize entropies
    var total: Float64 = 0.0
    for i in range(n):
        total += entropies[i]
    var mean_h = total / Float64(n)

    # All-zero or near-zero entropy -> uniform
    if mean_h < 1e-7:
        return List[Float64](length=n, fill=advantage)

    var result = List[Float64](capacity=n)
    for i in range(n):
        var h_norm = entropies[i] / (mean_h + 1e-8)
        # GTPO weight: clamped to non-negative
        var weight: Float64 = 1.0 + beta * (h_norm - 1.0)
        if weight < 0.0:
            weight = 0.0
        result.append(advantage * weight)
    return result^


# ---------------------------------------------------------------------------
# 3. HICRA planning token amplification
# ---------------------------------------------------------------------------
# Formula: A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)


fn apply_hicra(
    token_advs: List[Float64],
    planning_mask: List[Int],
    alpha: Float64 = 0.2,
) raises -> List[Float64]:
    """Amplify advantages at planning tokens using HICRA.

    Args:
        token_advs: Per-token advantages.
        planning_mask: Binary mask (0 or 1). 1 = planning token.
        alpha: Amplification factor. 0 disables.

    Returns:
        Amplified advantages.

    Raises:
        Error on length mismatch.
    """
    if len(token_advs) != len(planning_mask):
        raise Error(
            "Length mismatch: token_advs ("
            + String(len(token_advs))
            + ") vs planning_mask ("
            + String(len(planning_mask))
            + ")"
        )

    var n = len(token_advs)

    if alpha == 0.0:
        var result = List[Float64](capacity=n)
        for i in range(n):
            result.append(token_advs[i])
        return result^

    var result = List[Float64](capacity=n)
    for i in range(n):
        var adv = token_advs[i]
        if planning_mask[i] != 0:
            result.append(adv + alpha * abs(adv))
        else:
            result.append(adv)
    return result^


# ---------------------------------------------------------------------------
# 4. SEPA selective entropy pooling
# ---------------------------------------------------------------------------
# For execution tokens: H_pooled(t) = lambda_t * mean(H_exec) + (1 - lambda_t) * H(t)
# Planning tokens are unchanged.


fn apply_sepa_pooling(
    entropies: List[Float64],
    planning_mask: List[Int],
    lambda_t: Float64,
) raises -> List[Float64]:
    """Apply SEPA pooling: pull execution token entropies toward their mean.

    Args:
        entropies: Per-token entropies.
        planning_mask: Binary mask. 1 = planning, 0 = execution.
        lambda_t: Pooling strength in [0, 1].

    Returns:
        Pooled entropies.

    Raises:
        Error on length mismatch.
    """
    if len(entropies) != len(planning_mask):
        raise Error(
            "Length mismatch: entropies ("
            + String(len(entropies))
            + ") vs planning_mask ("
            + String(len(planning_mask))
            + ")"
        )

    var n = len(entropies)

    # Clamp lambda to [0, 1]
    var lam = lambda_t
    if lam < 0.0:
        lam = 0.0
    if lam > 1.0:
        lam = 1.0

    if lam == 0.0:
        var result = List[Float64](capacity=n)
        for i in range(n):
            result.append(entropies[i])
        return result^

    # Compute execution-token mean entropy
    var exec_sum: Float64 = 0.0
    var exec_count: Int = 0
    for i in range(n):
        if planning_mask[i] == 0:
            exec_sum += entropies[i]
            exec_count += 1

    if exec_count == 0:
        var result = List[Float64](capacity=n)
        for i in range(n):
            result.append(entropies[i])
        return result^

    var mean_h_exec = exec_sum / Float64(exec_count)

    var result = List[Float64](capacity=n)
    for i in range(n):
        if planning_mask[i] != 0:
            # Planning tokens: unchanged
            result.append(entropies[i])
        else:
            # Execution tokens: interpolate toward mean
            result.append(lam * mean_h_exec + (1.0 - lam) * entropies[i])
    return result^


# ---------------------------------------------------------------------------
# 5. Entropy statistics
# ---------------------------------------------------------------------------


fn compute_entropy_stats(
    exec_entropies: List[Float64],
    plan_entropies: List[Float64],
) -> EntropyStats:
    """Compute summary stats for execution vs planning entropy distributions.

    Args:
        exec_entropies: Entropies for execution tokens.
        plan_entropies: Entropies for planning tokens.

    Returns:
        EntropyStats with mean, variance, and count for each category.
    """
    var stats = EntropyStats()

    if len(exec_entropies) > 0:
        var n = len(exec_entropies)
        var total: Float64 = 0.0
        for i in range(n):
            total += exec_entropies[i]
        var mean_e = total / Float64(n)

        var var_sum: Float64 = 0.0
        for i in range(n):
            var diff = exec_entropies[i] - mean_e
            var_sum += diff * diff
        var var_e = var_sum / Float64(n)

        stats.exec_mean = mean_e
        stats.exec_var = var_e
        stats.exec_count = Float64(n)

    if len(plan_entropies) > 0:
        var n = len(plan_entropies)
        var total: Float64 = 0.0
        for i in range(n):
            total += plan_entropies[i]
        var mean_p = total / Float64(n)

        var var_sum: Float64 = 0.0
        for i in range(n):
            var diff = plan_entropies[i] - mean_p
            var_sum += diff * diff
        var var_p = var_sum / Float64(n)

        stats.plan_mean = mean_p
        stats.plan_var = var_p
        stats.plan_count = Float64(n)

    return stats^
