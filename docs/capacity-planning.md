# Capacity Planning

Use this page to size hardware, wall time, and campaign parallelism before launching long runs.

## Profiles

- **Standard profile (default):** `batch_size=8`, `group_size=16`, `max_tokens=10240`
- **Smoke-test profile (quickstart):** `batch_size=2`, `group_size=8`, `max_tokens=1024` (non-default)

Treat `max_tokens` as an upper bound, not expected generation length. Capacity should be calibrated from real runs.

## Core formulas

### Concurrency per step

```text
concurrency = batch_size * group_size
```

This is the number of completions sampled per training step.

### Total campaign steps

```text
total_campaign_steps = num_conditions * num_seeds * max_steps
```

### Estimated wall time

```text
estimated_wall_time = total_campaign_steps * median_step_time / effective_parallelism
```

Where:

- `median_step_time` comes from recent `step_time_s` logs
- `effective_parallelism = min(max_workers, total_runs)` when `parallel = true`
- `effective_parallelism = 1` when `parallel = false`

## Worker planning for campaigns

Use these campaign keys:

```toml
[campaign]
parallel = true
max_workers = 4
stagger_seconds = 8
```

Guidelines:

- Start with `max_workers = 2` to confirm backend stability.
- Increase workers only after pilot runs show stable latency and no API saturation.
- Use `stagger_seconds` (5-15s) to smooth startup bursts on shared services.
- If all workers slow down together, lower `max_workers` before changing model/training knobs.

## Calibration loop (recommended)

1. Run a pilot: 20-50 steps, 1-2 seeds, target condition(s).
2. Read `step_time_s` and use the median as `median_step_time`.
3. Check `max_token_hit_rate`:
   - If high, completions are being truncated; increase `max_tokens` only if quality requires it.
   - If low, reduce `max_tokens` to lower latency and memory pressure.
4. Check backpressure signals (`bp_action`, `bp_regime`, `bp_p_star`, `bp_throughput`):
   - Frequent `throttle` or persistent `retrograde` means concurrency is too high.
   - Stable `hold` near `optimal` is a good operating zone.
5. Recompute wall time with updated `median_step_time` and chosen `max_workers`.

## Capacity decision checklist

- Is this a smoke test or a full training run?
- Is `max_tokens` set intentionally for the objective?
- Is campaign parallelism sized to backend limits?
- Do pilot metrics confirm expected throughput?
- Is estimated wall time acceptable before scaling seeds/conditions?

## Related pages

- [Getting Started](getting-started.md)
- [Campaigns](campaigns.md)
- [Back Pressure](backpressure.md)
- [Logging & wandb](logging.md)
