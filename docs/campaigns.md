# Campaigns

The campaign orchestrator runs a full sweep of experimental conditions across multiple seeds — all from a single TOML file.

## Quick start

```bash
retrain campaigns/pilot.toml
```

That's it. One command runs all conditions x seeds, with optional auto-squeeze and wandb logging.

## Campaign TOML format

A campaign TOML is a regular training config with an added `[campaign]` section:

```toml
[campaign]
seeds = [42, 101, 202, 303]
max_steps = 50

[[campaign.conditions]]
advantage_mode = "grpo"
transform_mode = "none"

[[campaign.conditions]]
advantage_mode = "maxrl"
transform_mode = "gtpo_sepa"

# Everything below is the base training config for all runs

[backend]
backend = "tinker"

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
lora_rank = 128

[training]
batch_size = 8
group_size = 16
max_tokens = 2048
lr = 4e-5
save_every = 20

[sepa]
steps = 50
schedule = "linear"
delay_steps = 10

[squeeze]
min_variance_retention = 0.95

[logging]
wandb_project = "sepa-pilot"
```

### `[campaign]` section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seeds` | list[int] | `[42, 101, 202, 303, 404, 505, 606, 707]` | RNG seeds for each run |
| `max_steps` | int | `500` | Training steps per run (overrides `[training].max_steps`) |

### `[[campaign.conditions]]`

Each condition is a table with two required keys:

| Key | Type | Description |
|-----|------|-------------|
| `advantage_mode` | str | `grpo` or `maxrl` |
| `transform_mode` | str | `none`, `gtpo`, `gtpo_hicra`, or `gtpo_sepa` |

If no conditions are specified, defaults to the 5-condition ablation:

| # | `advantage_mode` | `transform_mode` | Label |
|---|-------------------|-------------------|-------|
| 1 | `grpo` | `none` | `grpo+none` |
| 2 | `maxrl` | `none` | `maxrl+none` |
| 3 | `maxrl` | `gtpo` | `maxrl+gtpo` |
| 4 | `maxrl` | `gtpo_hicra` | `maxrl+gtpo_hicra` |
| 5 | `maxrl` | `gtpo_sepa` | `maxrl+gtpo_sepa` |

## Auto-squeeze

Add a `[squeeze]` section to your campaign TOML to automatically find the optimal LoRA rank after the first run:

```toml
[model]
lora_rank = 128          # train at high rank

[squeeze]
min_variance_retention = 0.95
```

After the first training run completes, retrain analyzes the adapter via SVD, prints a variance table, reports the recommended rank, and logs everything to wandb. The remaining campaign runs continue normally.

The recommendation is also saved to `manifest.json` so you can retrieve it programmatically.

See [LoRA-Squeeze](squeeze.md) for the full documentation: algorithm details, standalone usage, compression, configuration reference, and Python API.

## Output structure

Each campaign creates a timestamped directory under `logs/`:

```
logs/campaign_20260222_010127/
├── manifest.json          # Campaign metadata + squeeze recommendation
└── runs/
    ├── grpo+none_s42/
    │   ├── metrics.jsonl
    │   └── emergence/
    ├── grpo+none_s101/
    ├── maxrl+gtpo_sepa_s42/
    └── ...
```

### manifest.json

Contains the full campaign configuration: timestamp, conditions, seeds, max steps, and run details. When auto-squeeze is enabled, also includes:

```json
{
  "squeeze": {
    "recommended_rank": 32
  }
}
```

## wandb integration

When `wandb_project` is set in the base config, each training run gets:

- **Run name**: `{condition}_s{seed}` (e.g., `maxrl+gtpo_sepa_s42`)
- **Group**: `{condition}` (e.g., `maxrl+gtpo_sepa`) — groups runs across seeds
- **Tags**: `{condition},seed{seed}` — for filtering

Plus a **squeeze-analysis** run (if `[squeeze]` is configured) with the variance table and recommended rank.

This makes it easy to compare conditions in the wandb dashboard: group by condition, then see variance across seeds.

## Compute budget

Rough single-GPU estimates for Qwen3-4B with default settings (`batch_size=8, group_size=16`):

| GPU | Per run (500 steps) | Full campaign (40 runs) |
|-----|---------------------|------------------------|
| RTX 4090 | ~6.3 h | ~250 h |
| A100 | ~3.5 h | ~140 h |
| H100 | ~2.1 h | ~84 h |

Start small (`max_steps = 50`, 3-4 seeds) to validate your setup before committing to a full campaign.

## The 5 conditions

These conditions form a progressive ablation — from baseline GRPO to the full MaxRL+GTPO+SEPA pipeline. Each adds one component to isolate its contribution to reasoning performance:

1. **`grpo+none`** — Baseline GRPO advantages, no token-level transforms
2. **`maxrl+none`** — MaxRL advantages (inverse success-rate reweighting)
3. **`maxrl+gtpo`** — Add entropy-weighted credit assignment
4. **`maxrl+gtpo_hicra`** — Add planning token amplification
5. **`maxrl+gtpo_sepa`** — Add selective entropy pooling (full pipeline)
