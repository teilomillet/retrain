# SEPA

Selective Entropy Pooling of Attention (SEPA) is an adaptive scheduling strategy that controls how aggressively execution-token entropies are pooled during training.

## What SEPA does

In the [advantage pipeline](advantages.md), GTPO weights token advantages by per-token uncertainty -- high-uncertainty tokens get more credit. SEPA modifies execution-token uncertainty values *before* GTPO weighting by pulling them toward their mean:

```
H_pooled(t) = lambda * mean(H_exec) + (1 - lambda) * H(t)    if execution token
H_pooled(t) = H(t)                                             if planning token
```

When `lambda = 0`, SEPA is off and GTPO sees raw values. When `lambda = 1`, all execution tokens have the same uncertainty (the mean), so GTPO's differentiation is concentrated entirely on planning tokens.

Today `uncertainty_kind = "surprisal"` (sampled-token `-logprob`) is the only available signal. When backends provide full token distributions, `shannon_entropy` becomes usable and SEPA will pool true entropy values.

## Why

Early in training, the model's entropy distribution is noisy -- high variance among execution tokens doesn't carry useful signal. As training progresses, the model becomes more confident on execution tokens, and entropy differences among planning tokens become the meaningful signal.

SEPA ramps pooling strength over training, progressively focusing the entropy signal on planning tokens where it matters most.

## Schedules

### Linear

Lambda ramps from 0 to 1 over `sepa_steps` steps, with an optional `delay_steps` warmup:

```
lambda(step) = max(0, (step - delay_steps) / sepa_steps)
```

```toml
[sepa]
steps = 500
schedule = "linear"
delay_steps = 50
```

For the first 50 steps, lambda stays at 0. Then it ramps linearly to 1 over the next 500 steps.

### Auto

The auto schedule adapts based on execution-token entropy variance decay. It tracks an EMA of batch-level entropy variance and compares it to the initial variance:

```
auto_lambda = 1 - min(var_ema / var_0 / threshold, 1)
lambda = max(auto_lambda, linear_lambda)
```

The linear schedule acts as a floor -- auto can only increase lambda, never decrease it below the linear ramp.

Auto requires a warmup period (default 50 steps) to establish the initial variance baseline.

```toml
[sepa]
steps = 500
schedule = "auto"
delay_steps = 50
```

## Correctness gate

SEPA can be gated on model performance. When `correct_rate_gate > 0`, lambda stays at 0 until the batch correct rate reaches the threshold. Once opened, the gate is sticky -- it never closes again.

```toml
[sepa]
correct_rate_gate = 0.1   # SEPA activates after 10% correct rate
```

This prevents SEPA from interfering with very early training when the model hasn't learned basic problem structure yet.

Set to `0` to disable the gate (SEPA starts immediately).

## TOML configuration

```toml
[algorithm]
transform_mode = "gtpo_sepa"   # must include sepa

[sepa]
steps = 500                    # linear ramp duration
schedule = "linear"            # linear | auto
delay_steps = 50               # delay before ramp starts
correct_rate_gate = 0.1        # min correct rate to enable
```

## State and checkpointing

The SEPA controller maintains internal state (EMA variance, gate status, warmup counter) that is saved and restored across checkpoint resumes. This ensures seamless continuation of the adaptive schedule after preemptions.

## Logged metrics

| Metric | Description |
|--------|-------------|
| `sepa_lambda` | Current pooling strength (0-1) |
| `sepa_gate_open` | Whether the correctness gate has opened |
| `exec_entropy_mean` | Mean execution-token entropy |
| `exec_entropy_var` | Execution-token entropy variance |
| `plan_entropy_mean` | Mean planning-token entropy |
| `plan_entropy_var` | Planning-token entropy variance |
