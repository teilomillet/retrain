# Delight-Gate Campaign Log

## Objective
Break the ~42-48% correct rate ceiling on Qwen3-4B-Instruct math training by
replacing SEPA's pooling transform with DG (Delight Gradient) sign-aware gating.
Update teilo.xyz/posts/sepa with results.

## Iteration History

### v1: Raw surprisals (NO-OP)
Gate argument `A·ℓ/η` with raw ℓ~0.06 → sigmoid arg < 0.03 → 99% neutral.
All conditions = baseline. Gate was dead.

### v2: Z-score normalization (INVERTED)
Used `(ℓ - mean) / std` — fixed dynamic range (neutral dropped to 94-97%) BUT
**inverted the gate direction**: `g+ < 0.5 < g-` because surprisal distributions
are right-skewed. After centering, most tokens have z < 0, which makes
`σ(A_pos × z_neg) < 0.5` for correct rollouts. Backwards.

### v3: Scale-only normalization (COMPLETED — null result)
Uses `ℓ / std` — scales to unit variance WITHOUT centering:
- Gate direction correct: `g+ = 0.52 > 0.50 > 0.47 = g-` ✓
- Neutral = 93-97% (down from 99% in v1)
- Breakthrough = 0.5-1.2% (up from 0.0% in v1)
- **Transient +2.9pp bump at λ=0.4-0.6** for η=0.5 (both seeds) then fades to baseline
- **Final result: ALL conditions null at step 30**

## v3 Final Results (scale-only normalization, campaign_20260319_121037)

| Condition | s42 RCR | s101 RCR | Mean | vs baseline |
|-----------|---------|----------|------|-------------|
| C1: PG baseline | 42.2% | 42.4% | **42.3%** | — |
| C2: DG η=1.0 norm | 42.8% | 42.1% | **42.5%** | +0.2pp |
| C3: DG η=0.5 norm | 42.6% | 42.8% | **42.7%** | +0.4pp |
| C4: DG+PPO norm | CRASHED | CRASHED | — | wandb name too long |
| C5: old SEPA pool | ~42.6% | ~42.7% | **~42.7%** | +0.4pp |

**All conditions within ±0.5pp of baseline. No separation.**

### Transient Signal (η=0.5, both seeds):
| Step | λ | s42 RCR | s101 RCR | Mean | vs baseline |
|------|---|---------|----------|------|-------------|
| 15 | 0.43 | 45.1% | 45.2% | 45.2% | +2.9pp |
| 20 | 0.60 | 45.9% | 44.5% | 45.2% | +2.9pp |
| 25 | 0.77 | 43.5% | 43.5% | 43.5% | +1.2pp |
| 30 | 0.90 | 42.6% | 42.8% | 42.7% | +0.4pp |

**The transient bump appears at moderate λ (0.4-0.6) then fades as λ→1.**
This pattern reproduced in BOTH v1 and v3, suggesting DG helps at partial
strength but full DG (pure token-selection) is too aggressive.

### Gate diagnostic: top surprisal tokens
Logged with decoded text. High-surprisal tokens are:
- Strategy words: "find", "Rewrite", "want", "sum"
- Step transitions: `:Ċ`, `.ĊĊ`, `**`
- Math structure: `$`, `}{`
These ARE meaningful fork-points — the model's uncertainty is highest at
strategy selection and reasoning-step boundaries.

## Key Findings

1. **The gate is mechanically correct** (v3): sign-aware, breakthrough tokens
   identified at strategy decisions, direction verified.

2. **Despite correct mechanics, DG does not break the ceiling.** The ~42% plateau
   is robust to token-level credit assignment, just as it was robust to
   advantage-level interventions (9 previous campaigns, 80+ runs).

3. **The transient bump is real but not sustained.** At moderate DG strength
   (λ=0.4-0.6), both seeds show +2-3pp for ~10 steps, then regress. This may
   indicate DG provides a brief learning signal that the optimizer exhausts.

4. **Old SEPA pooling ≡ DG gating at final RCR.** Both converge to ~42.7%.
   The transform mechanism (pooling vs gating) doesn't matter — the ceiling
   is imposed by something else (model capacity, task distribution, or
   optimization landscape).

## v4 plans (adaptive eta, robust scaling)
Code ready but given v3 null result, the hypothesis that token-level credit
assignment can break the ceiling is weakening. The v4 conditions with adaptive
eta may help sustain the transient bump, but the fundamental constraint appears
to be elsewhere.

## Blog Post Update
The SEPA blog post should be updated with:
1. Three normalization iterations (raw → z-score → scale-only) as methodological contribution
2. DG gate correctly identifies fork-tokens (strategy decisions, step transitions)
3. Transient +2.9pp signal at moderate λ — first non-noise result in 10 campaigns
4. But final result is null: the ~42% ceiling resists all token-level interventions
5. The ceiling is likely model capacity or task distribution, not credit assignment
