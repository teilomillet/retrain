# LoRA-Squeeze

LoRA-Squeeze finds the optimal LoRA rank for your trained adapter using memory-efficient SVD. Train at high rank, then squeeze to find the smallest rank that retains the information you need.

Based on [LoRA-Squeeze (arXiv 2602.10993)](https://arxiv.org/abs/2602.10993): "it is better to first learn an expressive, higher-rank solution and then compress it."

## How it works

1. Load the trained LoRA adapter (lora_A and lora_B matrices for each layer)
2. For each layer, compute QR decompositions of A and B (never forms the full m x n product)
3. SVD on the small r x r core matrix to get singular values
4. Measure cumulative variance at each target rank: `V(k) = sum(s^2[:k]) / sum(s^2)`
5. Recommend the smallest rank where mean variance across layers >= threshold

The algorithm operates on the small r x r core matrix, so it's fast and memory-efficient even for large models.

## Two ways to use it

### 1. Auto-squeeze in campaigns (recommended)

Add `[squeeze]` to your campaign TOML. After the first training run completes, retrain automatically analyzes the adapter and reports the optimal rank — no manual step needed.

```toml
[campaign]
seeds = [42, 101, 202, 303]
max_steps = 50

[[campaign.conditions]]
advantage_mode = "grpo"
transform_mode = "none"

[model]
lora_rank = 128          # train at high rank

[squeeze]
min_variance_retention = 0.95

[logging]
wandb_project = "my-project"
```

```bash
retrain campaign.toml
```

The campaign will:

1. Run the first training job (e.g., `grpo+none_s42`)
2. Analyze its final adapter via SVD
3. Print the variance table and recommended rank
4. Log results to wandb (if enabled)
5. Save the recommendation to `manifest.json`
6. Continue the remaining campaign runs

### 2. Standalone squeeze

Analyze any saved adapter directly:

```toml
# squeeze.toml
[squeeze]
adapter_path = "logs/campaign_.../runs/grpo+none_s42/final"
min_variance_retention = 0.95

[model]
lora_rank = 128    # source rank fallback (if not detectable from weights)
```

```bash
retrain squeeze.toml
```

For Tinker adapters, use the `tinker://` path directly:

```toml
[squeeze]
adapter_path = "tinker://run-id/weights/final"
min_variance_retention = 0.95
```

#### Compression

To also compress the adapter to the recommended (or a specific) rank:

```toml
[squeeze]
adapter_path = "logs/.../final"
min_variance_retention = 0.95
output_path = "logs/squeezed"    # save compressed adapter here
compress_to = 0                  # 0 = use recommended rank, or set a specific rank
```

The compressed adapter is a valid PEFT adapter with scaled `lora_alpha`:

```
alpha_new = alpha_old * r_target / r_source
```

## Example output

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

Read the table as: "rank 32 retains 95.48% of the mean variance across all 196 LoRA layers, with the worst layer still retaining 91.22%."

## Configuration reference

### `[squeeze]` section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `adapter_path` | str | `""` | Path to adapter directory. Required for standalone, auto-filled in campaigns |
| `min_variance_retention` | float | `0.95` | Minimum variance fraction to retain (0.95 = 95%) |
| `source_rank` | int | `0` | Expected source rank. `0` = detect from `[model].lora_rank` |
| `target_ranks` | list[int] | `[]` | Ranks to evaluate. `[]` = auto power-of-2 sequence |
| `output_path` | str | `""` | Directory to save compressed adapter. Empty = analyze only |
| `compress_to` | int | `0` | Target rank for compression. `0` = use recommended rank |
| `device` | str | `"cpu"` | Torch device for SVD computation |

## wandb integration

When used in a campaign with `wandb_project` set, squeeze creates a dedicated **"squeeze-analysis"** run in the same project with:

| What | Details |
|------|---------|
| **Variance table** | wandb Table with columns: rank, mean_variance, min_variance, max_variance, recommended |
| **Variance curves** | Per-rank line chart data (`squeeze/mean_variance`, `squeeze/min_variance`, `squeeze/max_variance`) |
| **Summary metrics** | `squeeze/recommended_rank`, `squeeze/source_rank`, `squeeze/min_variance_retention`, `squeeze/num_layers` |

The squeeze run appears alongside training runs in your wandb project, tagged with `squeeze` and `rank-{N}`.

## Recommended workflow

1. **Train at high rank** — use `lora_rank = 64` or `128` in your campaign
2. **Add `[squeeze]`** — set `min_variance_retention` to your desired threshold (0.95 is a good default)
3. **Run the campaign** — squeeze runs automatically after the first training run
4. **Check the recommendation** — in the terminal output, `manifest.json`, or wandb
5. **Use the recommended rank** — for future production training runs, set `lora_rank` to the recommended value

### Choosing `min_variance_retention`

| Threshold | Use case |
|-----------|----------|
| `0.99` | Conservative — minimal information loss, higher rank |
| `0.95` | Balanced — good default for most experiments |
| `0.90` | Aggressive — smaller adapter, some information loss acceptable |
| `0.80` | Very aggressive — only for quick iteration or memory-constrained setups |

## Python API

For programmatic use:

```python
from retrain.squeeze import analyze_adapter, compress_adapter

# Analyze
analysis = analyze_adapter(
    adapter_path="logs/.../final",
    min_variance_retention=0.95,
)
print(f"Recommended rank: {analysis.recommended_rank}")
print(f"Variance at rank 32: {analysis.mean_variance[32]:.2%}")

# Compress
compress_adapter(
    adapter_path="logs/.../final",
    output_path="logs/squeezed",
    target_rank=analysis.recommended_rank,
)
```
