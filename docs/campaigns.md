# Campaigns

The campaign orchestrator generates and runs a full sweep of all 5 experimental conditions across multiple seeds.

## The 5 conditions

| # | `advantage_mode` | `transform_mode` | Label |
|---|-------------------|-------------------|-------|
| 1 | `grpo` | `none` | `grpo+none` |
| 2 | `maxrl` | `none` | `maxrl+none` |
| 3 | `maxrl` | `gtpo` | `maxrl+gtpo` |
| 4 | `maxrl` | `gtpo_hicra` | `maxrl+gtpo_hicra` |
| 5 | `maxrl` | `gtpo_sepa` | `maxrl+gtpo_sepa` |

These conditions form a progressive ablation: from baseline GRPO to the full MaxRL+GTPO+SEPA pipeline. Each adds one component to isolate its contribution to reasoning performance.

## Usage

```bash
# Dry run -- prints commands without executing
python -m retrain.campaign

# Execute all runs sequentially
python -m retrain.campaign --execute

# Custom seeds
python -m retrain.campaign --seeds 101,102,103,104

# With wandb tracking
python -m retrain.campaign --wandb-project sepa-deep

# Override max steps
python -m retrain.campaign --max-steps 200

# Use a specific config as base
python -m retrain.campaign --config path/to/config.toml
```

Campaigns work with both the local and Tinker backends. The backend is determined by your base TOML config -- set `backend = "tinker"` in your config and pass it with `--config`.

## Default seeds

The default seed set is: `42, 101, 202, 303, 404, 505, 606, 707` (8 seeds).

With 5 conditions and 8 seeds, a full campaign is 40 runs.

## Output structure

Each campaign creates a timestamped directory under `logs/`:

```
logs/campaign_20250215_143022/
├── manifest.json          # Full campaign metadata
├── run_all.sh             # Shell script to re-run everything
└── runs/
    ├── grpo+none_s42/
    │   ├── metrics.jsonl
    │   └── emergence/
    ├── grpo+none_s101/
    ├── maxrl+none_s42/
    ├── maxrl+gtpo_s42/
    ├── maxrl+gtpo_hicra_s42/
    ├── maxrl+gtpo_sepa_s42/
    └── ...
```

### manifest.json

Contains the full campaign configuration: timestamp, conditions, seeds, wandb project, max steps, and the complete command for each run.

### run_all.sh

An executable shell script that re-runs the entire campaign. Useful for reproducing results or running on a different machine.

## wandb integration

When `--wandb-project` is set, each run gets:

- **Run name**: `{condition}_s{seed}` (e.g., `maxrl+gtpo_sepa_s42`)
- **Group**: `{condition}` (e.g., `maxrl+gtpo_sepa`) -- groups runs across seeds
- **Tags**: `{condition},seed{seed}` -- for filtering

This makes it easy to compare conditions in the wandb dashboard: group by condition, then see variance across seeds.

## Compute budget

A full campaign (5 conditions x 8 seeds x 500 steps) is 40 runs. Rough single-GPU estimates:

| GPU | Per run (500 steps) | Full campaign (40 runs) |
|-----|---------------------|------------------------|
| RTX 4090 | ~6.3 h | ~250 h |
| A100 | ~3.5 h | ~140 h |
| H100 | ~2.1 h | ~84 h |

Use `--max-steps 100` and `--seeds 42,101,202` for a smaller sweep (~15 runs) to validate your setup before committing to a full campaign.

## Dataset

All runs train on the [hendrycks/MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) dataset (5 subjects: intermediate algebra, precalculus, number theory, counting & probability, geometry). The dataset auto-downloads from HuggingFace on first run.

## CLI reference

| Flag | Description |
|------|-------------|
| `--execute` | Run all commands (default is dry run) |
| `--seeds` | Comma-separated seed list |
| `--wandb-project` | wandb project name |
| `--max-steps` | Override max training steps |
| `--config` | Base TOML config file |
