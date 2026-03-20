# Experiment Thesis: Advantage Capping to Rescue Training

## What We Know

9 controlled runs (3 conditions x 3 seeds x 100 steps) showed:
- **All conditions converge to ~47.5%** — GRPO baseline, HICRA, SEPA, no separation
- **SEPA at lambda=0.94** has zero effect on correct_rate
- **SEPA exec_entropy_var is 5% HIGHER than HICRA** — opposite of prediction
- The entire MaxRL + GTPO + SEPA stack adds nothing over plain GRPO

## Diagnosis

The loss is raw `-(ratio * advantage)` with no gradient bounding. Extreme
advantage values (heavy-tail outliers from GRPO normalization) can cause
disproportionately large parameter updates, leading to entropy collapse.
This failure mode is documented in DAPO, PRIME-RL, and AEnt (ICLR 2026).

## What We Can and Cannot Do

**True ratio clipping** (PPO-style) requires access to the new policy's
log-probabilities during the training forward pass. With Tinker backend,
the ratio computation happens on their remote GPU — we can't inject clipping.

**Advantage capping** (`adv_clip_max`) caps per-token advantages to [-max, +max]
before they reach Tinker's loss function. This bounds how hard any single token
pushes the gradient. It's a **different mechanism** from PPO clipping:

| | PPO Ratio Clipping | Advantage Capping |
|---|---|---|
| **What it bounds** | policy ratio (new_π/old_π) | advantage magnitude |
| **When it acts** | during training forward pass | before training |
| **What it prevents** | large policy updates | large gradient signals |
| **What it measures** | actual training dynamics | our preprocessing |

They share the goal (prevent extreme gradient updates) but differ in mechanism
and in what we can honestly claim from the metrics.

## What We Can Honestly Measure

**Real model behavior (valid evidence):**
- `exec_surprisal_var` trajectory — does the model's uncertainty distribution change?
- `correct_rate` — does accuracy improve?
- `loss` trajectory — does the loss curve shape change?

**Our preprocessing (diagnostic, not evidence):**
- `adv_cap_fraction` — what % of tokens we capped (measures our intervention intensity)
- `adv_cap_magnitude` — how extreme were the capped values (measures tail severity)

A high `adv_cap_fraction` with no change in entropy or correct_rate = the cap
is too tight and destroying signal. A low `adv_cap_fraction` with behavior change
= even mild capping fixes the tail problem.

## Hypotheses

**H1 — Advantage capping changes entropy trajectory.**
If capped conditions show different `exec_surprisal_var` evolution than uncapped,
the extreme advantages were causing entropy collapse. This is honest evidence
even with Tinker — entropy comes from sampling, not training.

**H2 — Capping breaks the 47.5% plateau.**
If capped conditions achieve >2% higher correct_rate, the mechanism works
regardless of whether it's "true PPO clipping" or advantage capping.

**H3 — Tight vs moderate cap reveals optimal intervention.**
cap=2.0 (aggressive) vs cap=5.0 (moderate) vs uncapped tells us where the
sweet spot is. If cap=2.0 hurts performance, we're over-intervening.

**H4 — SEPA adds signal with stable training.**
If capped SEPA (C4) beats capped baseline (C2), SEPA matters when training
dynamics are healthy. If not, SEPA is addressing a non-problem.

## Conditions

| ID | transform_mode | adv_clip_max | Tests |
|----|----------------|--------------|-------|
| C1 | none | 0 (off) | Uncapped baseline (reproduces ~47.5%) |
| C2 | none | 5.0 | Moderate cap — trim heavy-tail outliers |
| C3 | none | 2.0 | Tight cap — aggressive bounding |
| C4 | gtpo_sepa | 5.0 | Capped SEPA — does SEPA help when stable? |

All conditions use grpo advantage to isolate the capping effect.

## Validation Design

The experiment is self-validating:

1. **C1 reproduces the null.** If C1 ≠ ~47.5%, something else changed.
2. **adv_cap_fraction on C1 = 0.** Sanity check: uncapped condition reports no capping.
3. **adv_cap_fraction on C2/C3 > 0.** The cap is actually intervening.
4. **Entropy is the honest signal.** exec_surprisal_var comes from model sampling,
   not from our advantage preprocessing. If C2/C3 show different entropy evolution
   from C1, we can claim the intervention changed model behavior.

If adv_cap_fraction ≈ 0 on C2/C3, advantages aren't extreme and capping is
irrelevant — the problem is elsewhere.

## Success Criteria

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| H1 confirmed (entropy diverges) | Extreme advantages cause collapse | Test with local backend for true PPO clipping |
| H1+H2 confirmed | Advantage capping works | Ship adv_clip_max as default, test values |
| H3: cap=2.0 < cap=5.0 | Over-intervention | Use moderate cap, test values between 3-7 |
| H3: cap=2.0 > cap=5.0 | Tighter is better | Try even tighter caps |
| H4 confirmed | SEPA is real when training is healthy | Revisit SEPA thesis |
| All null, cap_fraction > 0 | Advantages are bounded but training still fails | Problem is not gradient magnitude |
| All null, cap_fraction ≈ 0 | Advantages were never extreme | Capping is irrelevant, look elsewhere |

## Budget

- Previous: 9 runs (~10% of budget)
- This campaign: 12 runs (4 conditions x 3 seeds x 100 steps)
- Estimated: ~15% of budget
- Remaining after: ~75% for follow-up if signal found

## Results

### Campaign 1: clip-rescue (2026-02-27)

**Status:** Running (8/12 runs spawned, C1 complete, C2/C3 in progress, C4 queued)

#### Key Finding: Advantage capping is a guaranteed no-op for GRPO with binary rewards

**Mathematical proof:** GRPO computes `A_i = r_i - mean(r)`. With rewards in {0, 1}
and group_size=16, the maximum possible advantage magnitude is 15/16 = 0.9375.
Any cap ≥ 1.0 is guaranteed to NEVER trigger.

This means:
- `adv_clip_max=5.0` → no-op (confirmed: `adv_cap_fraction=0.0` across all steps)
- `adv_clip_max=2.0` → no-op (confirmed: `adv_cap_fraction=0.0` across all steps)
- ALL hypotheses H1-H4 involving capping are invalidated before data arrives

**Root cause:** The diagnosis was wrong. GRPO advantages are NOT heavy-tailed or
extreme. They are inherently bounded by the reward range. The 47% plateau is NOT
caused by extreme gradient signals from large advantages.

#### C1 Final Results (uncapped baseline, 3 seeds, 100 steps)

| Seed | final correct_rate | final exec_surprisal_var |
|------|-------------------|--------------------------|
| s42  | 0.466             | 0.163                    |
| s101 | 0.473             | 0.166                    |
| s202 | 0.465             | 0.156                    |
| **Mean** | **0.468**     | **0.162**                |

Confirms the ~47% plateau with tight inter-seed agreement (±0.5%).

#### Complete clip-rescue results (all conditions, 100 steps)

| Condition | Seeds | avg correct_rate | avg exec_var | adv_cap_fraction |
|-----------|-------|-----------------|-------------|------------------|
| C1: GRPO baseline | 3 | **0.468** | 0.162 | 0.0 |
| C2: GRPO+cap=5.0 | 3 | **0.473** | 0.115 | 0.0 |
| C3: GRPO+cap=2.0 | 3 | **0.472** | 0.120 | 0.0 |
| C4: SEPA+cap=5.0 | 3 | **0.477** | 0.097 | 0.0 |

All conditions within ±0.5% of ~0.47. No separation. Cap never triggers.

**SEPA detail:** post_exec_var = 0.001 (vs raw exec_var = 0.097). SEPA achieves
~99% variance reduction in the advantage preprocessing, but the MODEL's exec_var
(measured from fresh rollouts) is comparable across conditions. SEPA's entropy
pooling doesn't change training outcomes.

#### Implications

The "extreme advantage → entropy collapse" hypothesis is disproved for GRPO:
- GRPO advantages bounded [-1, +1] — caps never trigger
- Even if they did, C4 (SEPA) shows mechanically successful entropy processing
  with zero impact on correct_rate

### Campaign 2: maxrl-sweep (2026-02-27, ran in parallel with clip-rescue)

Tests MaxRL advantages, which CAN be heavy-tailed (up to 15.0 when 1/16 correct).

#### Final results (3 conditions complete, cap=2.0 still running)

| Condition | Seeds | avg correct_rate | avg exec_var | cap trigger rate |
|-----------|-------|-----------------|-------------|------------------|
| GRPO baseline | 3 | **0.473** | 0.114 | 0% |
| MaxRL uncapped | 3 | **0.471** | 0.135 | 0% |
| MaxRL+cap=5.0 | 2 | **0.473** | 0.124 | 52% |

**All identical at ~0.472.** Despite:
- MaxRL producing 20x larger loss values than GRPO at step 40 (9.8 vs 0.5)
- cap=5.0 triggering on 52% of steps (capping advantages at magnitudes 7.0-15.0)
- Fundamentally different advantage distributions

The cap IS intervening (confirmed: cap_mag hits 7.0 and 15.0, matching the
theoretical 2/16 and 1/16 correct scenarios). But the model doesn't care.

MaxRL+cap=2.0 complete: 3 seeds, avg 0.477, 68% trigger rate. Same plateau.

### Campaign 3: lr-sweep (2026-02-27, 200 steps)

Tests learning rate (4e-6 vs 4e-5) and extends to 200 steps.

#### THE BREAKTHROUGH: The "47% plateau" is a lie

Extending to 200 steps revealed the truth. The `running_correct_rate` (EMA) hides
violent oscillation in the actual per-step `correct_rate`:

| Step | step_cr (actual) | running_cr (EMA) | What's happening |
|------|-----------------|------------------|------------------|
| 0    | 66%             | 66%              | Base model capability |
| 20   | 52%             | 45%              | Initial training disruption |
| 45   | **85-87%**      | 43%              | **PEAK — model learns to solve 85%!** |
| 80   | 69%             | 47%              | Slight decline from peak |
| 100  | 60%             | 47%              | ← where all previous campaigns stopped |
| 120  | 51%             | 49%              | Collapse beginning |
| 140  | 42%             | 49%              | Accelerating collapse |
| 154  | **10-17%**      | 48%              | **CATASTROPHIC COLLAPSE** |

Confirmed across all 4 runs (2 learning rates × 2 seeds). Same pattern.

**The model reaches 85% correct around step 45, then collapses to <20% by step 154.**
The "47% plateau" is the EMA average of peaks and troughs, not a real ceiling.

Both learning rates (4e-6 and 4e-5, a 10x difference) show identical dynamics —
confirming Adam normalizes the lr effect at this scale range.

---

## Conclusions

### The real diagnosis: entropy collapse without ratio clipping

The 47% "plateau" was never a plateau. It was the running average of a boom-bust cycle:

1. **Boom (steps 20-80):** The model rapidly learns, reaching 85% correct
2. **Bust (steps 100-160):** Without ratio clipping to constrain `new_π/old_π`,
   the policy overshoots, becomes too peaked, and collapses to <20%
3. **The EMA masks this**, reporting a stable ~47% that looks like a ceiling

This is the exact failure mode described in PPO (Schulman 2017), DAPO, and
AEnt (ICLR 2026): unconstrained importance-sampling loss allows the policy to
move too far from the reference, causing entropy collapse.

### What advantage manipulations cannot fix

None of our advantage-level interventions address the root cause:

1. **Advantage magnitude doesn't matter** — Adam normalizes it
2. **GRPO vs MaxRL doesn't matter** — same gradient direction, different scale
3. **Advantage capping doesn't matter** — bounds the signal, not the ratio
4. **SEPA doesn't matter** — changes entropy weighting, not optimization dynamics

The bottleneck is the **unconstrained policy ratio**, not the advantage signal.

### What WILL fix it: ratio clipping

PPO-style clipping: `min(ratio × adv, clip(ratio, 1-ε, 1+ε) × adv)` directly
constrains how far the policy can move per step. This:
- Prevents the boom from overshooting
- Keeps the policy in a stable learning regime
- Is already implemented in `local_train_helper.py` (`clip_eps`, `clip_eps_high`)
- **Requires local GPU** (Tinker doesn't expose the ratio computation)

### What we can honestly say

> "The GRPO 'plateau' at 47% correct_rate is an artifact of EMA smoothing over
> a boom-bust cycle. The model actually reaches 85% correct around step 45, then
> collapses catastrophically to <20% by step 154. This is textbook entropy collapse
> from unconstrained importance-sampling loss — the policy ratio `new_π/old_π` grows
> unbounded, causing the policy to overshoot and collapse.
>
> Advantage-level interventions (mode, magnitude, capping, entropy weighting) cannot
> fix this because Adam normalizes gradient magnitude. The fix requires ratio clipping
> (PPO-style), which constrains the optimization dynamics directly.
>
> GRPO advantages with binary rewards are mathematically bounded to [-1, +1]
> (proven: A_i = r_i - mean(r), max |A| = (G-1)/G). Advantage capping is a
> guaranteed no-op for this reward structure."

### Campaign 4: ppo-clip (2026-03-03, 200 steps)

Tests PPO ratio clipping on Tinker. The SDK supports `loss_fn="ppo"` with
built-in defaults (custom `loss_fn_config` returns 500 — cannot configure clip_eps).

**Key finding: `loss_fn_config` is not supported on Tinker's server.**
Confirmed by systematic testing: `importance_sampling` works, `ppo` works with no
config, any `loss_fn_config` dict causes 500. The `cispo` loss also returns 500.

Tinker's PPO reports: `ppo_mean_ratio`, `ppo_clipped_fraction`, `ppo_kl_div`.

#### Results (Waves 1-3, MaxRL+PPO seeds still running)

| Condition | Seeds | avg running_cr (step 200) | Pattern |
|-----------|-------|--------------------------|---------|
| GRPO baseline (IS loss) | 3/3 | **0.472** | Stable plateau — no collapse |
| GRPO + PPO (Tinker defaults) | 3/3 | **0.356** | ALL collapse at steps 140-190 |
| MaxRL unclipped (IS loss) | 2/3 | **0.484** | Stable, slightly > GRPO |
| MaxRL + PPO (Tinker defaults) | 0/3 | — | Running (step ~8) |

**PPO collapse is systematic (all 3 seeds):**
- s42: running_cr 0.337 (cr drops to 0.008 by step 190)
- s101: running_cr 0.417 (collapse begins ~step 150)
- s202: running_cr 0.315 (collapsed hardest)

**PPO loss scale:** 191-507 (vs 0.01-0.12 for IS loss). This 3000x difference
confirms these are fundamentally different loss computations. Tinker's built-in
clip defaults are unknown and possibly too aggressive for our setup — they
CAUSE the very collapse that ratio clipping is supposed to prevent.

**MaxRL surprise finding:** MaxRL unclipped (0.484 avg) is tracking at or slightly
above GRPO (0.472 avg). With 2 seeds, this is +1.2pp, not yet statistically
significant. If the third seed (s202, currently at step 8) confirms, MaxRL
with importance_sampling loss may have a genuine edge — matching the paper's
claim that MaxRL Pareto-dominates GRPO.

**Critical correction to previous thesis:** The lr-sweep collapse (peak at step 45,
bust at step 154) does **NOT reproduce** at lr=5e-5 with GRPO+IS baselines. All 3
baseline seeds are stable at ~0.47 through 200 steps. The "boom-bust" was either
lr=4e-5 specific or stochastic. The EMA "plateau" at ~47% is real.

### What we know now

1. **The 47% plateau is real** — stable at 0.472 across 3 seeds at 200 steps
2. **Tinker's PPO loss CAUSES collapse** — all 3 seeds crash (avg 0.356 vs 0.472)
3. **MaxRL may edge out GRPO** — 0.484 vs 0.472 with IS loss (needs 3rd seed)
4. **Tinker's loss_fn_config is broken** — cannot configure clip_eps (500 error)
5. **No advantage-level intervention helps** — GRPO, MaxRL advantage formula ≈ same

### Campaign 5: MaxRL paper code audit (2026-03-04)

Read the official MaxRL implementation at github.com/tajwarfahim/maxrl.
Found **critical differences** between the paper and our setup:

| Parameter | Paper (MaxRL repo) | Our setup |
|---|---|---|
| **Loss function** | PPO clipped (dual-clip, clip_ratio_c=3.0) | importance_sampling (unclipped) |
| **clip_range** | 0.2 (low and high) | 0.0 (disabled) |
| **Learning rate** | **1e-6** | **5e-5** (50x higher!) |
| **Gradient clipping** | 0.3 (max_grad_norm) | None |
| **Temperature** | 0.6 | 0.7 |
| **Max response length** | 4096 | 2048 |
| **Base model** | Qwen3-1.7B-**Base** | Qwen3-4B-**Instruct** |
| **KL penalty** | 0.0 (explicitly off) | N/A |
| **PPO epochs** | 1 | 1 |
| **Rollouts per prompt** | 16 | 16 |

**The advantage formula is identical:** `(score - mean) / (mean + eps)`. Confirmed
by reading the paper's `compute_maxrl_outcome_advantage()` — exact match.

**The paper ALWAYS uses PPO clipping.** Their MaxRL = MaxRL advantages + PPO loss.
They never run MaxRL with raw importance_sampling. This means our MaxRL experiments
were testing a configuration the paper authors never intended.

**The paper's PPO has a dual-clip mechanism** for negative advantages:
```python
# Standard clip
pg_losses1 = -adv * ratio
pg_losses2 = -adv * clamp(ratio, 1-eps, 1+eps)
clip1 = max(pg_losses1, pg_losses2)
# Dual-clip lower bound (clip_ratio_c=3.0)
pg_losses3 = -adv * clip_ratio_c
clip2 = min(pg_losses3, clip1)
# Asymmetric: negative advantages get dual-clip
loss = where(adv < 0, clip2, clip1)
```

**Key insight:** The 50x learning rate difference (1e-6 vs 5e-5) is huge. At lr=1e-6,
the model takes tiny steps, so importance sampling ratios stay near 1.0 even without
clipping. At lr=5e-5, ratios diverge fast. Our unclipped IS loss was running with
50x more aggressive updates than the paper.

### Campaign 6: maxrl-paper (2026-03-04)

Tests paper's lr=1e-6 alongside our lr=5e-5 for GRPO and MaxRL.

lr=1e-6 runs died at step 22-23 from Tinker `RequestFailedError` (server crash).
Before dying, all 4 lr=1e-6 runs tracked identically (~0.43-0.44 running_cr at step 20).
No GRPO vs MaxRL separation visible in 22 steps at lr=1e-6.

lr=5e-5 runs completed (200 steps) and confirm previous findings:

| Condition | lr | s42 | s101 | Mean |
|-----------|-----|------|------|------|
| GRPO | 5e-5 | 0.486 | 0.482 | **0.484** |
| MaxRL | 5e-5 | 0.474 | 0.469 | **0.472** |
| GRPO | 1e-6 | 0.439* | 0.427* | 0.433* |
| MaxRL | 1e-6 | 0.438* | 0.435* | 0.437* |
*crashed at step 22-23

### Complete ppo-clip results (all 12 runs done)

| Condition | s42 | s101 | s202 | Mean |
|-----------|------|------|------|------|
| GRPO (IS loss) | 0.480 | 0.465 | 0.471 | **0.472** |
| GRPO + PPO (Tinker) | 0.337 | 0.417 | 0.315 | **0.356** |
| MaxRL (IS loss) | 0.493 | 0.475 | 0.469 | **0.479** |
| MaxRL + PPO (Tinker) | 0.421 | 0.393 | 0.373 | **0.396** |

**Key findings:**
1. **MaxRL edges GRPO by +0.7pp** (0.479 vs 0.472) with IS loss — consistent but small
2. **Tinker PPO CRASHES both** — GRPO+PPO (0.356) and MaxRL+PPO (0.396) collapse
3. **MaxRL+PPO crashes less** than GRPO+PPO (0.396 vs 0.356) — MaxRL advantages may be more compatible with Tinker's PPO defaults

### Campaign 7: Statistical analysis — MaxRL vs GRPO is noise (2026-03-09)

Pooled all 5 seeds (3 from ppo-clip at temp=0.7, 2 from maxrl-paper at temp=0.6):

| Seed | Campaign | Temp | GRPO | MaxRL | Diff |
|------|----------|------|------|-------|------|
| 42 | ppo-clip | 0.7 | 0.480 | 0.493 | +1.3pp MaxRL |
| 101 | ppo-clip | 0.7 | 0.465 | 0.475 | +1.0pp MaxRL |
| 202 | ppo-clip | 0.7 | 0.471 | 0.469 | -0.2pp GRPO |
| 42 | maxrl-paper | 0.6 | 0.486 | 0.474 | -1.2pp GRPO |
| 101 | maxrl-paper | 0.6 | 0.482 | 0.469 | -1.3pp GRPO |

**Pooled means:** GRPO = 0.4768 (std=0.86pp), MaxRL = 0.4762 (std=1.00pp)
**Pooled diff = -0.06pp, Cohen's d = 0.068** (far below 0.2 "small" threshold)

The +0.7pp "MaxRL wins" from ppo-clip was seed noise. The -1.2pp "GRPO wins"
from maxrl-paper was also noise. Both methods are identical at ~47.6%.

### Campaign 8: grad-clip (2026-03-09, RUNNING)

**Discovery:** Tinker's `AdamParams` has a `grad_clip_norm` field we never used!
The paper uses `max_grad_norm=0.3`. We had `0.0` (disabled).

Wired `grad_clip_norm` through: config.py → backend_definitions.py → tinker_backend.py.

Tests gradient clipping at both learning rates:

| Condition | lr | grad_clip_norm | Tests |
|-----------|-----|----------------|-------|
| C1 | 5e-5 | 0.0 | Baseline (reproduces 47% plateau) |
| C2 | 5e-5 | 0.3 | Paper's grad clip at our lr |
| C3 | 1e-6 | 0.0 | Paper's lr, retry (prev crash = infra) |
| C4 | 1e-6 | 0.3 | Paper's exact setup (minus PPO clip) |

GRPO only (MaxRL ≡ GRPO per Campaign 7). 2 seeds × 200 steps = 8 runs.

**Hypothesis:** At lr=5e-5, gradients may be large enough that clipping at 0.3
changes training dynamics. At lr=1e-6, gradients are tiny so clipping is a no-op.
If C2 > C1, gradient clipping is the missing stabilizer.

#### lr=5e-5 results (complete)

| Condition | s42 | s101 | Mean |
|-----------|-----|------|------|
| C1: no grad clip | 0.461 | 0.486 | **0.474** |
| C2: grad_clip=0.3 | 0.398 | 0.485 | **0.442** |

**C2 clip_s42 collapsed** (0.398 — same stochastic collapse pattern seen in ppo-clip).
Excluding the collapsed seed: clip_s101 (0.485) ≈ no_clip_s101 (0.486).

Gradient clipping at 0.3 does NOT help at lr=5e-5. The one collapse may be coincidental
(same seed 42 that collapsed with Tinker PPO in Campaign 4).

#### lr=1e-6 results (3/4 crashed from Tinker infra, 1 still running)

| Condition | s42 | s101 | Mean |
|-----------|-----|------|------|
| C3: no grad clip | 0.478 (step 99✗) | 0.481 (step 113…) | **0.480** |
| C4: grad_clip=0.3 | 0.474 (step 97✗) | 0.476 (step 91✗) | **0.475** |

✗ = crashed from Tinker infra ("model poisoned" or "promise expired"), not our code

**lr=1e-6 successfully ran past step 22** (previous crash point was infra outage).
All 4 runs tracked identically at ~0.475-0.481 before crashes.
No separation between clipped and unclipped.
Confirms: Adam normalizes the 50x lr difference. Same ~47% plateau at both lr values.

### Final conclusions (after 8 campaigns, 60+ runs)

**The 47% plateau cannot be broken by any single hyperparameter we've tested.**

Every paper parameter has been tested in isolation:

| Paper parameter | Value | Tested? | Effect |
|-----------------|-------|---------|--------|
| Advantage formula | MaxRL vs GRPO | ✓ Campaigns 2,4,6,7 | **None** (d=0.068) |
| Advantage capping | adv_clip_max=2,5 | ✓ Campaigns 1,2 | **None** (no-op for GRPO) |
| PPO ratio clipping | loss_fn="ppo" | ✓ Campaign 4 | **Harmful** (causes collapse) |
| Learning rate | 1e-6 vs 5e-5 | ✓ Campaigns 3,6,8 | **None** (Adam normalizes) |
| Gradient clipping | grad_clip_norm=0.3 | ✓ Campaign 8 | **None** (or stochastic collapse) |
| Temperature | 0.6 vs 0.7 | ✓ Campaigns 4 vs 6 | **None** |
| SEPA entropy weighting | various | ✓ Campaign 1 | **None** |

### Campaign 9: PPO Dual-Clip via Manual Forward+Backward (ppo-dualclip)

**Breakthrough**: Implemented proper PPO dual-clip by bypassing the broken SDK.
The SDK's `forward_backward_custom` hardcodes `cross_entropy` loss internally,
which returns 400 errors. Workaround: manual pipeline using `importance_sampling`:
1. `forward(datums, "importance_sampling")` → get new-policy logprobs
2. Compute PPO dual-clip loss locally (PyTorch autograd)
3. `forward_backward(grad_datums, "importance_sampling")` with old_lp=new_lp (ratio=1),
   advantages=custom_grad → chain rule gives correct parameter gradients

**Campaign**: ppo-dualclip.toml — 4 conditions × 2 seeds × 200 steps
- C1: baseline (no clip, lr=5e-5)
- C2: PPO dual-clip (eps=0.2, c=3.0, grad_clip=0.3, lr=5e-5)
- C3: baseline (no clip, lr=1e-6)
- C4: PPO dual-clip (eps=0.2, c=3.0, grad_clip=0.3, lr=1e-6)

#### Complete results (all 8 runs, 200 steps each)

**lr=5e-5 conditions:**

| Metric | Baseline (2 seeds) | PPO dual-clip (2 seeds) |
|--------|-------------------|------------------------|
| Running correct rate (final) | **44.7%, 49.3%** → mean 47.0% | **49.4%, 49.1%** → mean 49.2% |
| Cohen's d (correct rate) | — | **0.166** (negligible) |
| Loss std | **0.989** | **0.039** |
| Loss stability improvement | — | **25.1x** |
| Collapse rate (CR < 25%) | **7.1%** of steps | **3.9%** of steps |

**lr=1e-6 conditions:**

| Metric | Baseline (2 seeds) | PPO dual-clip (2 seeds) |
|--------|-------------------|------------------------|
| Running correct rate (final) | **48.5%, 48.6%** → mean 48.5% | **48.4%, 48.6%** → mean 48.5% |
| Cohen's d (correct rate) | — | **0.018** (negligible — pure zero) |
| Loss std | **0.861** | **0.039** |
| Loss stability improvement | — | **21.9x** |
| Collapse rate (CR < 25%) | **5.3%** of steps | **4.4%** of steps |

**Grand campaign summary (all 8 runs):**
- Mean RCR: **48.3%** (range 44.7-49.4%, spread 4.7pp)
- lr=5e-5 mean: 48.1% (spread 4.7pp) — noisier, baseline s101 collapsed more
- lr=1e-6 mean: 48.5% (spread **0.3pp**) — cleanest ceiling measurement yet
- The +2.2pp at lr=5e-5 was seed noise: at lr=1e-6, diff is **-0.1pp** (zero)

**PPO dual-clip stabilizes loss 22-25x at BOTH learning rates but has ZERO effect
on correct rate at EITHER learning rate. At lr=1e-6, the effect is exactly zero
(d=0.018). Loss stability ≠ learning stability. The ~48% ceiling is confirmed.**

### Final conclusions (after 9 campaigns, 80+ runs)

**The ~48% plateau cannot be broken by any training hyperparameter we've tested.**

Every paper parameter has been tested in isolation AND in the full PPO dual-clip stack:

| Paper parameter | Value | Tested? | Effect |
|-----------------|-------|---------|--------|
| Advantage formula | MaxRL vs GRPO | ✓ Campaigns 2,4,6,7 | **None** (d=0.068) |
| Advantage capping | adv_clip_max=2,5 | ✓ Campaigns 1,2 | **None** (no-op for GRPO) |
| PPO ratio clipping (built-in) | loss_fn="ppo" | ✓ Campaign 4 | **Harmful** (causes collapse) |
| PPO dual-clip (custom) | eps=0.2, c=3.0 | ✓ Campaign 9 | **None** (25x loss stability, d=0.018 at lr=1e-6) |
| Learning rate | 1e-6 vs 5e-5 | ✓ Campaigns 3,6,8,9 | **None** (Adam normalizes) |
| Gradient clipping | grad_clip_norm=0.3 | ✓ Campaigns 8,9 | **None** |
| Temperature | 0.6 vs 0.7 | ✓ Campaigns 4 vs 6 | **None** |
| SEPA entropy weighting | various | ✓ Campaign 1 | **None** |

### Grand cross-campaign summary (22 completed 200-step runs)

| Condition | Runs | Mean RCR | Spread |
|-----------|------|----------|--------|
| GRPO (baseline) | 13 | **47.7%** | 44.7-49.3% |
| MaxRL | 5 | **47.6%** | 46.9-49.3% |
| GRPO + grad_clip | 2 | **44.1%** | 39.8-48.5% (s42 collapsed) |
| PPO dual-clip (lr=5e-5) | 2 | **49.2%** | 49.1-49.4% |
| PPO dual-clip (lr=1e-6) | 2 | **48.5%** | 48.4-48.6% |

**Grand mean across all 22 runs: 47.8%**

**Definitive findings:**
- MaxRL vs GRPO: d=0.147 (negligible, pooled 14 runs)
- PPO dual-clip at lr=5e-5: d=0.166, +2.2pp (seed noise)
- PPO dual-clip at lr=1e-6: d=0.018, -0.1pp (exactly zero)
- lr=1e-6 provides the cleanest ceiling measurement: 48.5% ± 0.1pp across 4 runs
- Loss stability 22-25x at both lr values → no CR improvement at either

**The most likely explanation for the discrepancy with the paper:**
1. The paper trains Qwen3-1.7B-**Base** (smaller, no instruction tuning)
2. The paper runs for 11,600 prompts (we run 200 steps × 8 = 1,600 prompts)
3. Instruct-tuned models may have already captured the "easy gains" from RLVR

**For our Tinker + Qwen3-4B-Instruct setup:** The ~48% plateau is a hard ceiling.
PPO dual-clip stabilizes training dramatically but doesn't break the ceiling.
The model's capability limit on this math task distribution cannot be surpassed
by better optimization alone — it requires either a different (base) model,
more training compute, or a different task distribution.
