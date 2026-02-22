# Logging & wandb

## JSONL files

retrain writes structured logs to JSONL files in the `log_dir` directory:

```
logs/train/
├── metrics.jsonl              # Per-step training metrics
└── emergence/
    ├── steps.jsonl            # Per-step summary for emergence analysis
    └── generations.jsonl      # Individual completions with rewards
```

### metrics.jsonl

One JSON object per training step with all metrics:

| Field | Description |
|-------|-------------|
| `step` | Training step index |
| `condition` | Label like `maxrl+gtpo_sepa` |
| `loss` | Training loss |
| `mean_reward` | Mean reward across the batch |
| `correct_rate` | Batch correct rate |
| `running_correct_rate` | Cumulative correct rate |
| `sepa_lambda` | Current SEPA pooling strength |
| `sepa_gate_open` | Whether SEPA correctness gate is open |
| `num_datums` | Number of datums submitted for training |
| `max_token_hit_rate` | Fraction of completions that hit max_tokens |
| `step_time_s` | Wall time for the step |
| `batch_size` | Current batch size |
| `group_size` | Current group size |
| `bp_action` | Back pressure action |
| `bp_regime` | Back pressure regime |
| `exec_entropy_mean` | Mean execution-token entropy (GTPO modes) |
| `exec_entropy_var` | Execution-token entropy variance |
| `plan_entropy_mean` | Mean planning-token entropy |
| `plan_entropy_var` | Planning-token entropy variance |

### emergence/steps.jsonl

Compact per-step summaries for emergence analysis:

| Field | Description |
|-------|-------------|
| `step` | Training step |
| `mean_reward` | Mean reward |
| `correct_count` | Number of correct completions |
| `total_count` | Total completions |
| `condition` | Condition label |

### emergence/generations.jsonl

Individual completions for qualitative analysis:

| Field | Description |
|-------|-------------|
| `step` | Training step |
| `prompt` | First 200 chars of the prompt |
| `completion` | First 500 chars of the completion |
| `reward` | Reward score |
| `num_tokens` | Completion length in tokens |

## wandb

Enable wandb by setting `wandb_project`:

```toml
[logging]
wandb_project = "my-project"
wandb_run_name = ""          # defaults to condition label
wandb_entity = ""            # team or user
wandb_group = ""             # for grouping related runs
wandb_tags = ""              # comma-separated
```

### Metric prefixes

All wandb metrics use structured prefixes:

| Prefix | Metrics |
|--------|---------|
| `train/` | `loss`, `rewards/mean_reward`, `rewards/correct_rate`, `rewards/running_correct_rate`, `sepa_lambda`, `sepa_gate_open`, `max_token_hit_rate`, `num_datums`, `step_time_s`, `batch_size`, `group_size` |
| `train/entropy/` | `exec_mean`, `exec_var`, `plan_mean`, `plan_var` |
| `train/backpressure/` | `action`, `regime`, `p_star`, `sigma`, `kappa`, `utilization`, `throughput`, `warmup` |

### Run config

The wandb run config records all hyperparameters:

- `advantage_mode`, `transform_mode`, `condition`
- `model`, `lora_rank`, `lr`, `batch_size`, `group_size`
- `max_tokens`, `temperature`, `gtpo_beta`, `hicra_alpha`
- `sepa_steps`, `sepa_delay_steps`, `sepa_correct_rate_gate`
- `max_steps`, `backend`, `seed`

### Squeeze metrics

When a campaign has a `[squeeze]` section, a dedicated **"squeeze-analysis"** wandb run is created after the first training run completes. It logs:

| Prefix | Metrics |
|--------|---------|
| `squeeze/` | `variance_table` (wandb Table), `mean_variance`, `min_variance`, `max_variance`, `rank` |

Summary metrics on the run:

| Key | Description |
|-----|-------------|
| `squeeze/recommended_rank` | Smallest rank meeting the variance threshold |
| `squeeze/source_rank` | Original training rank |
| `squeeze/min_variance_retention` | Configured threshold |
| `squeeze/num_layers` | Number of LoRA layers analyzed |

### Campaign integration

When running campaigns with `wandb_project` set, each training run gets a structured name, group, and tags for easy comparison in the wandb dashboard. The squeeze-analysis run appears in the same project. See [Campaigns](campaigns.md).
