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

*(to be filled after running)*
