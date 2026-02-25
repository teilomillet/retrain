# Advantage Functions

retrain uses a composable advantage pipeline: an episode-level advantage function produces per-completion scores, then optional token-level transforms redistribute credit across individual tokens.

The [5 conditions](campaigns.md) tested in campaigns correspond to a progressive ablation of this pipeline: GRPO baseline, MaxRL, MaxRL+GTPO, MaxRL+GTPO+HICRA, and MaxRL+GTPO+SEPA.

## Pipeline

```
Rewards (per completion)
    │
    ▼
Episode-level advantage (GRPO or MaxRL)
    │
    ▼
Token-level expansion (GTPO token-surprisal weighting)
    │
    ▼
Optional transform (HICRA or SEPA)
    │
    ▼
Token-level advantages (fed to training loss)
```

## Episode-level advantages

### GRPO

Group Relative Policy Optimization. Centers rewards around the group mean:

```
A_i = r_i - mean(r)
```

Simple and effective. Positive reward completions get positive advantage, negative get negative. No normalization.

```toml
[algorithm]
advantage_mode = "grpo"
```

### MaxRL

Inverse success-rate reweighting. Normalizes by the group mean reward:

```
A_i = (r_i - mean(r)) / (mean(r) + eps)
```

When the model is mostly wrong (low mean reward), the denominator is small, amplifying the signal from rare correct completions. When the model is mostly right, advantages shrink -- the model learns less from easy problems.

Returns zero if the group mean is near zero (all wrong).

```toml
[algorithm]
advantage_mode = "maxrl"
```

### Custom episode-level advantage

Set `advantage_mode` to a dotted path. The target can be a plain function:

```python
def hipa_like_advantages(rewards):
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    return [2.0 * (r - mean_r) for r in rewards]
```

```toml
[algorithm]
advantage_mode = "my_advantages.hipa_like_advantages"
```

If your function needs extra knobs, accept a second `params` argument and pass
`advantage_params` when calling `compute_composable_advantages(...)` in Python.

## Token-level transforms

### Custom transform (context-style)

Set `transform_mode` to a dotted path pointing to a function that accepts a
`TransformContext` and returns `TransformOutput`.

```python
from retrain import TransformOutput

def my_transform(ctx):
    scale = float(ctx.params.get("scale", 1.0))
    token_advs = []
    for i, logprobs in enumerate(ctx.logprobs_G):
        token_advs.append([ctx.episode_advantages[i] * scale for _ in logprobs])
    return TransformOutput(token_advs=token_advs)
```

```toml
[algorithm]
transform_mode = "plugins.my_transform.my_transform"

[algorithm.transform_params]
scale = 2.0
```

### GTPO

Group-relative Token-level Policy Optimization. Weights token advantages by normalized token surprisal (`-logprob` of the sampled token):

```
w(t) = max(0, 1 + beta * (H(t)/mean(H) - 1))
A_token(t) = A_episode * w(t)
```

High-surprisal tokens (where the sampled token was unlikely) get amplified. Low-surprisal tokens get dampened.

`uncertainty_kind` controls semantics in TOML:

```toml
[algorithm]
uncertainty_kind = "surprisal"   # default
```

`shannon_entropy` and `varentropy` are parsed but fail fast today; they require full token distributions that current backend sample APIs do not expose.

Controlled by `beta`:

```toml
[gtpo]
beta = 0.1   # 0 = uniform (no GTPO effect)
```

### HICRA

Hierarchical Credit Assignment. Amplifies advantages for planning tokens:

```
A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)
```

Where `mask(t) = 1` for tokens identified as planning (thinking, self-correction, strategy) and `0` for execution tokens. The amplification is proportional to the magnitude of the existing advantage, so it preserves the sign.

```toml
[hicra]
alpha = 0.2   # 0 = no amplification
```

### SEPA

Selective Entropy Pooling of Attention. Pulls execution-token entropies toward their mean before GTPO weighting:

```
H_pooled(t) = lambda * mean(H_exec) + (1 - lambda) * H(t)    if execution token
H_pooled(t) = H(t)                                             if planning token
```

This reduces entropy variance among execution tokens, letting GTPO focus its differentiation on planning tokens. Lambda ramps from 0 to 1 over training. See [SEPA](sepa.md) for scheduling details.

## Valid combinations

| `advantage_mode` | `transform_mode` | What it does |
|-------------------|-------------------|-------------|
| `grpo` | `none` | Baseline GRPO |
| `maxrl` | `none` | MaxRL without token-level transforms |
| `maxrl` | `gtpo` | MaxRL + entropy-weighted credit assignment |
| `maxrl` | `gtpo_hicra` | MaxRL + GTPO + planning token amplification |
| `maxrl` | `gtpo_sepa` | MaxRL + GTPO + selective entropy pooling (recommended) |

These are the 5 conditions used in campaign sweeps. See [Campaigns](campaigns.md).

!!! note
    GRPO can also be combined with `gtpo`, `gtpo_hicra`, or `gtpo_sepa`, but the standard conditions use MaxRL for the non-baseline transforms.

## Full algorithm override

Use `algorithm_mode` when you want to replace the full pipeline in one plugin:

```toml
[algorithm]
algorithm_mode = "plugins.my_algorithm.my_algorithm"
```

When `algorithm_mode` is set, it takes precedence over `advantage_mode` and
`transform_mode`.

## Planning tokens

HICRA and SEPA both rely on identifying which tokens are "planning" (thinking, self-correction) vs "execution" (direct computation). retrain detects planning tokens via strategic gram matching -- a sliding window over token text that checks for patterns like:

- "wait let me", "let me think", "on second thought"
- "let me check", "let me verify", "is this right"
- "double check", "try another approach", "go back and"
- "that's not right", "that doesn't work"
- "the key is", "the key insight", "notice that"

The full default list has 18 grams. You can override them via the `strategic_grams` config field:

```toml
[logging]
strategic_grams = '["wait let me", "let me think", "the key insight"]'
```

Or as a comma-separated string:

```toml
[logging]
strategic_grams = "wait let me, let me think, the key insight"
```

## Uninformative groups

Groups where all completions have the same reward (all correct or all wrong) are skipped -- they produce zero advantage and waste a training step. The trainer logs these as "skipped (all correct)" or "skipped (all wrong)".
