# Back Pressure

Back pressure uses the Universal Scalability Law (USL) to model throughput and automatically adjust batch size during training. It works with both the local and Tinker backends -- the controller operates at the trainer level, measuring end-to-end step throughput regardless of which backend performs the actual compute.

## What is USL?

The Universal Scalability Law (Gunther 2008) models throughput as a function of concurrency:

```
C(p) = p / (1 + sigma*(p-1) + kappa*p*(p-1))
```

Where:

- **p** = concurrency (batch_size * group_size)
- **sigma** = contention coefficient (serialization overhead)
- **kappa** = coherency coefficient (cross-talk overhead)

At low concurrency, throughput scales linearly. As concurrency increases, contention and coherency overheads cause throughput to plateau and eventually decline (retrograde behavior).

The optimal concurrency is:

```
p* = sqrt((1 - sigma) / kappa)
```

## When to use

Enable back pressure when:

- You're unsure of the optimal batch size for your GPU
- Training on variable hardware (e.g., preemptible instances)
- Running long campaigns where you want the system to auto-tune

Disable it (the default) when you know your hardware well and have tuned batch size manually.

## TOML configuration

```toml
[backpressure]
enabled = true
warmup_steps = 10
ema_decay = 0.9
throttle_margin = 0.85
increase_margin = 0.5
min_batch_size = 1
max_batch_size = 64
peak_gflops = 0.0       # optional: hardware peak for roofline hints
peak_bw_gb_s = 0.0      # optional: memory bandwidth for roofline hints
```

## How it works

### 1. Warmup

During the first `warmup_steps` steps, the controller collects throughput observations at geometrically increasing batch sizes (1, 2, 4, 8, ...) to explore the throughput curve. No adjustments are made.

### 2. Fit

After warmup, the controller fits USL parameters (sigma, kappa) from observed throughput using linearized least-squares with O(1) incremental Cramer sums. A sliding window of 100 observations keeps the fit current.

The fit recovers the serial throughput coefficient (lambda) and the optimal concurrency point (p*).

### 3. Decide

Each step, the controller classifies the current operating regime and recommends an action:

| Regime | Condition | Action |
|--------|-----------|--------|
| `warmup` | `step <= warmup_steps` | Hold (exploring) |
| `retrograde` | `p > throttle_margin * p*` and kappa > 0 | Throttle to `throttle_margin * p*` |
| `memory_bound` | Throughput exceeds USL prediction by >10% | Hold (headroom exists) |
| `optimal` | Within 80-110% of USL prediction | Hold |
| below target | `p < increase_margin * throttle_margin * p*` | Increase to `throttle_margin * p*` |

Both throttle and increase converge toward `throttle_margin * p*` -- the highest safe operating point just below the retrograde cliff.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_steps` | `10` | Steps to collect before fitting |
| `ema_decay` | `0.9` | EMA decay for throughput smoothing |
| `throttle_margin` | `0.85` | Target fraction of p* (safe operating point) |
| `increase_margin` | `0.5` | Hysteresis gap to prevent oscillation |
| `min_batch_size` | `1` | Floor for batch size recommendations |
| `max_batch_size` | `64` | Ceiling for batch size recommendations |
| `peak_gflops` | `0.0` | Hardware peak GFLOPS (enables roofline regime classification) |
| `peak_bw_gb_s` | `0.0` | Hardware peak memory bandwidth in GB/s |

## Logged metrics

| Metric | Description |
|--------|-------------|
| `bp_action` | Current action: `hold`, `throttle`, `increase` |
| `bp_regime` | Current regime: `warmup`, `retrograde`, `memory_bound`, `compute_bound`, `optimal` |
| `bp_p_star` | Optimal concurrency from USL fit |
| `bp_sigma` | USL contention coefficient |
| `bp_kappa` | USL coherency coefficient |
| `bp_utilization` | Actual throughput / predicted peak |
| `bp_throughput` | EMA-smoothed throughput (tokens/s) |
