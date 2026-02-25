# Capacity Planning for retrain

This page is about planning **retrain** runs, not generic infra sizing.
Use it when deciding batch/group/token settings and campaign worker counts.

## Where capacity shows up in retrain

In retrain, each step does:

1. sample completions
2. score rewards
3. compute advantages/transforms
4. run adapter training

Capacity pressure is mostly driven by:

- `[training].batch_size`
- `[training].group_size`
- `[training].max_tokens`
- `[campaign].parallel`, `max_workers`, `stagger_seconds`
- `[backpressure]` controller settings (optional auto-tuning)

## Profiles used in retrain docs

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

## retrain command flow for capacity sizing

```bash
# 1) Create or inspect config
retrain init --template campaign
retrain explain campaign.toml

# 2) Run a small pilot (edit max_steps in TOML first)
retrain campaign.toml

# 3) Check run matrix and outputs
retrain status logs
```

## Worker planning for retrain campaigns

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

### Tinker API throttling

For the Tinker backend, parallel campaigns automatically throttle concurrent API calls
via a file-based counting semaphore. This means you can safely set `max_workers` higher
than `max_concurrent` — workers will queue for API access instead of overwhelming the endpoint.

```toml
[campaign]
parallel = true
max_workers = 12       # 12 subprocesses run in parallel

[backend]
backend = "tinker"
max_concurrent = 4     # but only 4 hit the Tinker API at once
```

The default `max_concurrent = 4` prevents the 502/504 errors that previously occurred
when many workers called Tinker simultaneously. Tune it based on your endpoint's capacity.

## Calibration loop (recommended for retrain)

1. Run a pilot: 20-50 steps, 1-2 seeds, target condition(s).
2. Read `step_time_s` and use the median as `median_step_time`.
3. Check `max_token_hit_rate`:
   - If high, completions are being truncated; increase `max_tokens` only if quality requires it.
   - If low, reduce `max_tokens` to lower latency and memory pressure.
4. Check backpressure signals (`bp_action`, `bp_regime`, `bp_p_star`, `bp_throughput`):
   - Frequent `throttle` or persistent `retrograde` means concurrency is too high.
   - Stable `hold` near `optimal` is a good operating zone.
5. Recompute wall time with updated `median_step_time` and chosen `max_workers`.

The two files you will read most often are:

- `logs/<run>/metrics.jsonl`
- `logs/<campaign>/manifest.json`

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
