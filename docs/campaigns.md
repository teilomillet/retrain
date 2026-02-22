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

## Auto-squeeze (LoRA-Squeeze)

Add a `[squeeze]` section to your campaign TOML and retrain will automatically analyze the optimal LoRA rank after the first run completes.

```toml
[squeeze]
min_variance_retention = 0.95
```

### How it works

1. Train at high rank (e.g., `lora_rank = 128`)
2. After the first run finishes, retrain runs memory-efficient SVD on the saved adapter
3. Prints the variance table showing how much information each rank retains
4. Reports the recommended rank (smallest rank that retains >= `min_variance_retention` of variance)
5. Logs results to wandb (if enabled)
6. Continues the remaining campaign runs normally

Based on [LoRA-Squeeze (arXiv 2602.10993)](https://arxiv.org/abs/2602.10993): "it is better to first learn an expressive, higher-rank solution and then compress it."

### Example output

```
============================================================
Auto-squeeze: analyzing tinker://run-abc123/weights/final
  source_rank=128, min_variance_retention=0.95

Source rank: 128
Layers analyzed: 196

  Rank  Mean Var%  Min Var%  Max Var%
-------------------------------------------
     1     12.47%     5.23%    22.81%
     2     23.15%    12.08%    38.42%
     4     42.31%    28.67%    61.05%
     8     68.94%    52.13%    82.47%
    16     85.72%    74.30%    93.18%
    32     95.48%    91.22%    98.15% <--
    64     99.12%    97.84%    99.73%
   128    100.00%   100.00%   100.00%

Recommended rank: 32 (>= 95% variance retained)
============================================================
```

### `[squeeze]` section reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_variance_retention` | float | `0.95` | Minimum fraction of variance to retain. `0.95` = 95% |
| `source_rank` | int | `0` | Expected source rank. `0` = auto-detect from `[model].lora_rank` |

### wandb integration

When `wandb_project` is set, squeeze results are logged as a dedicated **"squeeze-analysis"** run with:

- **Variance table** — rank vs mean/min/max variance retention
- **Variance curves** — line chart of variance vs rank
- **Summary metrics** — `squeeze/recommended_rank`, `squeeze/source_rank`, `squeeze/num_layers`

### Workflow

The recommended workflow is:

1. Pick a high rank (64 or 128) for the campaign
2. Add `[squeeze]` with your desired retention threshold
3. Run the campaign — squeeze runs automatically after the first training run
4. Check the recommended rank in the output or wandb
5. Use the recommended rank for future production runs

## Standalone squeeze

For analyzing an adapter outside of a campaign, create a squeeze-only TOML:

```toml
[squeeze]
adapter_path = "logs/campaign_.../runs/grpo+none_s42/final"
min_variance_retention = 0.95
output_path = "logs/squeezed"    # save compressed adapter here
compress_to = 0                  # 0 = use recommended rank

[model]
lora_rank = 128                  # source rank fallback
```

```bash
retrain squeeze.toml
```

This analyzes the adapter and optionally compresses it to the target rank. For Tinker adapters, use the `tinker://` path directly:

```toml
[squeeze]
adapter_path = "tinker://run-id/weights/final"
```

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
