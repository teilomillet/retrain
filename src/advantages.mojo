"""Native SIMD advantage math for the Tinker training pipeline.

Ports the 6 core functions from textpolicy/tinker/advantages.py into
pure Mojo with SIMD vectorization where beneficial.

Functions:
    compute_grpo_advantages          — vanilla reward centering
    compute_maxrl_advantages         — inverse success-rate reweighting
    apply_gtpo_weighting             — entropy-weighted credit assignment
    apply_hicra                      — planning token amplification
    apply_sepa_pooling               — selective entropy pooling
    compute_entropy_stats            — summary statistics for logging
    identify_planning_tokens_native  — strategic gram matching (pure Mojo)
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


# ---------------------------------------------------------------------------
# 6. Planning token identification (native Mojo)
# ---------------------------------------------------------------------------
# Replaces the Python regex-based implementation.
# One tokenizer.convert_ids_to_tokens() call remains in pybridge;
# all matching logic runs here with zero Python overhead.


fn _is_word_char(b: UInt8) -> Bool:
    """Check if byte is a regex word character [a-zA-Z0-9_]."""
    if b >= 97 and b <= 122:
        return True  # a-z
    if b >= 65 and b <= 90:
        return True  # A-Z
    if b >= 48 and b <= 57:
        return True  # 0-9
    if b == 95:
        return True  # _
    return False


fn _ascii_lower(b: UInt8) -> UInt8:
    """Lowercase a single ASCII byte. Non-alpha bytes pass through."""
    if b >= 65 and b <= 90:
        return b + 32
    return b


fn _clean_token_fragment(fragment: String) -> String:
    """Clean a tokenizer fragment: replace subword markers with space, strip.

    Handles sentencepiece ▁ (U+2581, bytes E2 96 81) and
    GPT-2/BPE Ġ (U+0120, bytes C4 A0).
    """
    var n = len(fragment)
    if n == 0:
        return String("")

    var result = String()
    var bytes = fragment.as_bytes()
    var i = 0
    while i < n:
        var b = bytes[i]
        # ▁ (U+2581) = 0xE2 0x96 0x81
        if b == 0xE2 and i + 2 < n and bytes[i + 1] == 0x96 and bytes[i + 2] == 0x81:
            result += " "
            i += 3
        # Ġ (U+0120) = 0xC4 0xA0
        elif b == 0xC4 and i + 1 < n and bytes[i + 1] == 0xA0:
            result += " "
            i += 2
        elif b < 0x80:
            # ASCII byte — copy directly
            result += String(fragment[i : i + 1])
            i += 1
        elif b < 0xC0:
            # Stray continuation byte — skip
            i += 1
        elif b < 0xE0:
            # 2-byte UTF-8 sequence
            var end = i + 2
            if end > n:
                end = n
            result += String(fragment[i:end])
            i = end
        elif b < 0xF0:
            # 3-byte UTF-8 sequence
            var end = i + 3
            if end > n:
                end = n
            result += String(fragment[i:end])
            i = end
        else:
            # 4-byte UTF-8 sequence
            var end = i + 4
            if end > n:
                end = n
            result += String(fragment[i:end])
            i = end

    return String(result.strip())


fn _gram_matches_in(text: String, gram: String) -> Bool:
    """Check if gram appears in text with word boundaries (case-insensitive).

    Equivalent to: re.search(r'\\b' + re.escape(gram) + r'\\b', text, re.IGNORECASE)

    Both text and gram are compared case-insensitively at the byte level.
    Only works correctly for ASCII content (sufficient for strategic grams).
    """
    var text_bytes = text.as_bytes()
    var gram_bytes = gram.as_bytes()
    var text_len = len(text)
    var gram_len = len(gram)

    if gram_len == 0 or gram_len > text_len:
        return False

    for i in range(text_len - gram_len + 1):
        # Case-insensitive byte match
        var found = True
        for j in range(gram_len):
            if _ascii_lower(text_bytes[i + j]) != _ascii_lower(gram_bytes[j]):
                found = False
                break
        if not found:
            continue

        # Left word boundary: start of string or preceding char is non-word
        if i > 0 and _is_word_char(text_bytes[i - 1]):
            continue

        # Right word boundary: end of string or following char is non-word
        var end = i + gram_len
        if end < text_len and _is_word_char(text_bytes[end]):
            continue

        return True

    return False


fn identify_planning_tokens_native(
    token_strs: List[String],
    strategic_grams: List[String],
    max_window: Int = 5,
) -> List[Int]:
    """Identify planning tokens via strategic gram matching (pure Mojo).

    Sliding window over token fragments, checking for word-boundary matches
    against each strategic gram. Equivalent to the Python regex implementation
    but with zero Python overhead.

    Args:
        token_strs: Token string fragments (from tokenizer.convert_ids_to_tokens).
        strategic_grams: Phrases to detect (e.g. "let me think").
        max_window: Minimum sliding window size in tokens.

    Returns:
        Binary mask (List[Int] of 0/1), same length as token_strs.
    """
    var n_tokens = len(token_strs)
    if n_tokens == 0 or len(strategic_grams) == 0:
        return List[Int](length=n_tokens, fill=0)

    # Ensure window covers longest gram (by word count)
    var effective_window = max_window
    for g_idx in range(len(strategic_grams)):
        var gram_bytes = strategic_grams[g_idx].as_bytes()
        var word_count = 1
        for bi in range(len(strategic_grams[g_idx])):
            if gram_bytes[bi] == UInt8(32):  # space
                word_count += 1
        if word_count > effective_window:
            effective_window = word_count

    # Pre-clean all fragments once (each token appears in up to
    # effective_window overlapping windows; cleaning once saves ~80% work)
    var cleaned_tokens = List[String](capacity=n_tokens)
    for ci in range(n_tokens):
        cleaned_tokens.append(_clean_token_fragment(token_strs[ci]))

    var mask = List[Int](length=n_tokens, fill=0)

    # Sliding window: shortest-match-first strategy
    for start in range(n_tokens):
        var window_text = String("")
        var window_end = start + effective_window
        if window_end > n_tokens:
            window_end = n_tokens

        for end in range(start, window_end):
            if len(cleaned_tokens[end]) > 0:
                if len(window_text) > 0:
                    window_text += " " + cleaned_tokens[end]
                else:
                    window_text = String(cleaned_tokens[end])

            # Check all grams against current window
            var matched = False
            for g_idx in range(len(strategic_grams)):
                if _gram_matches_in(window_text, strategic_grams[g_idx]):
                    # Mark all tokens in window [start, end]
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break

    return mask^
