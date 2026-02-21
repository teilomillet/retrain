# Research Guide

This page is for researchers running experiments with retrain -- interpreting metrics, comparing conditions, computing statistical significance, and using the analysis tooling.

## Which metrics matter

Not all logged metrics are equally important. Here's what to focus on at each level:

### Primary metrics (compare these across conditions)

| Metric | What it tells you | Where to find it |
|--------|-------------------|-----------------|
| `correct_rate` | Batch-level accuracy -- fraction of completions that solve the problem | `metrics.jsonl`, wandb `train/rewards/correct_rate` |
| `running_correct_rate` | Cumulative accuracy (smoother, less noisy) | `metrics.jsonl`, wandb `train/rewards/running_correct_rate` |
| `mean_reward` | Average reward across the batch (0-1 scale) | `metrics.jsonl`, wandb `train/rewards/mean_reward` |

### Secondary metrics (understand the mechanism)

| Metric | What it tells you |
|--------|-------------------|
| `exec_entropy_var` | Execution-token entropy variance. SEPA should **reduce** this (pooling effect) |
| `plan_entropy_var` | Planning-token entropy variance. Should stay high or increase (diverse thinking) |
| `exec_entropy_mean` | Should decrease over training (model becomes more confident on execution) |
| `plan_entropy_mean` | Should stay relatively high (model keeps "thinking" on planning tokens) |
| `sepa_lambda` | Pooling strength. Ramps 0->1 over training. Compare against correct_rate inflection |
| `sepa_gate_open` | When the correctness gate opened. Earlier = faster learning signal |

### Diagnostic metrics (debug, don't compare)

| Metric | What to watch for |
|--------|-------------------|
| `max_token_hit_rate` | Spikes mean the model is rambling -- consider increasing `max_tokens` |
| `loss` | Should decrease. Can be negative with importance sampling -- that's normal |
| `num_datums` | Drops when groups are uninformative (all correct or all wrong) |
| `bp_regime` | If stuck in `retrograde`, batch size is too high |

## What "good" looks like

There are no absolute benchmarks -- you compare **across conditions**, not against a fixed target. But here's what typical training progression looks like on Qwen3-4B / MATH:

| Phase | Steps | correct_rate | What's happening |
|-------|-------|-------------|-----------------|
| Cold start | 0-20 | ~5-15% | Model is guessing. High entropy everywhere |
| Early learning | 20-100 | ~15-30% | Picks up basic problem patterns. SEPA gate opens |
| Mid training | 100-300 | ~30-50% | Correct rate climbs. Entropy distributions separate |
| Late training | 300-500 | ~45-60%+ | Gains slow. SEPA lambda near 1.0 |

!!! warning "These are rough ranges"
    Actual numbers depend on model, dataset subset, batch/group size, and random seed. The important signal is **relative differences between conditions**, not absolute values.

### What the 5-condition ablation should show

If things are working, you should see this ordering (best to worst):

1. `maxrl+gtpo_sepa` -- highest correct_rate, lowest exec_entropy_var
2. `maxrl+gtpo_hicra` -- close to SEPA, higher exec_entropy_var
3. `maxrl+gtpo` -- no planning-token differentiation
4. `maxrl+none` -- no token-level transforms
5. `grpo+none` -- baseline

The gap between conditions 1 and 2 (SEPA vs HICRA) is the key comparison. Both use planning tokens, but SEPA pools execution entropy while HICRA amplifies planning advantages. Look for:

- **correct_rate**: SEPA >= HICRA (the primary claim)
- **exec_entropy_var**: SEPA < HICRA (the mechanism -- SEPA pools execution entropy)
- **plan_entropy_var**: SEPA ~ HICRA (planning entropy should be similar)

## Comparing conditions in wandb

When you run campaigns with `--wandb-project`, each run gets a **group** label (the condition) and **tags** (condition + seed). To compare:

1. Open your wandb project dashboard
2. Create a panel: x-axis = `step`, y-axis = `train/rewards/correct_rate`
3. **Group by** the `condition` config key to aggregate runs per condition
4. Enable **error bands** (std or min/max) to see seed variance
5. You should see 5 lines (one per condition) with shaded variance bands

Key panels to create:

| Panel | x-axis | y-axis | What it shows |
|-------|--------|--------|--------------|
| Learning curves | `step` | `train/rewards/correct_rate` | Primary comparison |
| Reward curves | `step` | `train/rewards/mean_reward` | Smoother than correct_rate |
| SEPA dynamics | `step` | `train/sepa_lambda` | Only nonzero for `gtpo_sepa` condition |
| Entropy pooling | `step` | `train/entropy/exec_var` | SEPA should show lower variance |
| Planning entropy | `step` | `train/entropy/plan_mean` | Should stay high across conditions |

!!! tip "Parallel coordinates"
    Use wandb's parallel coordinates plot to see how hyperparameters (advantage_mode, transform_mode, seed) relate to final correct_rate. This quickly identifies if one condition dominates.

## How many seeds do I need?

The effect size between conditions (especially SEPA vs HICRA) is modest. More seeds = more statistical power.

| Goal | Seeds | Total runs (5 conditions) | Notes |
|------|-------|--------------------------|-------|
| Quick validation | 3 | 15 | Enough to see directional trends |
| Standard campaign | 8 | 40 | Default. Adequate for large effects |
| Paper-grade claims | 16+ | 80+ | Needed for modest effect sizes (~1-2% absolute lift) |

Use `--max-steps 100` with 3 seeds first to validate your setup. Then scale to 8 seeds / 500 steps for real results.

### Power analysis guidance

From early smoke tests on smaller models: detecting a +1.3% absolute lift in correct_rate with 80% power requires ~2400 episodes per arm. At `batch_size=8, group_size=16` (128 episodes/step), that's ~19 steps x 8 seeds, or fewer steps with more seeds.

Rule of thumb: if your initial 8-seed campaign shows a directional but non-significant effect, double the seeds before concluding "no effect."

## Using the emergence logs

The `emergence/` directory in each run's log folder contains detailed per-step and per-completion data for deeper analysis.

### Loading metrics.jsonl

```python
import json
from pathlib import Path

def load_metrics(run_dir: str) -> list[dict]:
    """Load per-step metrics from a training run."""
    path = Path(run_dir) / "metrics.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# Example: load and plot correct_rate
metrics = load_metrics("logs/train")
steps = [m["step"] for m in metrics]
correct = [m["correct_rate"] for m in metrics]
```

### Loading emergence data

```python
def load_emergence(run_dir: str) -> dict:
    """Load steps and generations from emergence logs."""
    root = Path(run_dir) / "emergence"

    steps = []
    with open(root / "steps.jsonl") as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))

    generations = []
    with open(root / "generations.jsonl") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))

    return {"steps": steps, "generations": generations}
```

### Plotting learning curves across conditions

```python
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def plot_campaign(campaign_dir: str):
    """Plot correct_rate for all conditions in a campaign."""
    root = Path(campaign_dir) / "runs"

    # Group runs by condition
    by_condition = defaultdict(list)
    for run_dir in sorted(root.iterdir()):
        metrics = load_metrics(str(run_dir))
        if not metrics:
            continue
        condition = metrics[0]["condition"]
        by_condition[condition].append(metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, runs in sorted(by_condition.items()):
        # Align steps across seeds, compute mean + std
        all_correct = []
        for run in runs:
            all_correct.append([m["correct_rate"] for m in run])

        min_len = min(len(c) for c in all_correct)
        arr = np.array([c[:min_len] for c in all_correct])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = list(range(min_len))

        ax.plot(steps, mean, label=condition)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Correct Rate")
    ax.set_title("5-Condition Ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_curves.png", dpi=150)

plot_campaign("logs/campaign_20250215_143022")
```

### Comparing entropy distributions (SEPA mechanism)

```python
def compare_entropy(campaign_dir: str, step_range=(400, 500)):
    """Compare exec_entropy_var between SEPA and HICRA in late training."""
    root = Path(campaign_dir) / "runs"

    sepa_vars, hicra_vars = [], []
    for run_dir in sorted(root.iterdir()):
        metrics = load_metrics(str(run_dir))
        if not metrics:
            continue
        condition = metrics[0]["condition"]

        late = [
            m["exec_entropy_var"]
            for m in metrics
            if step_range[0] <= m["step"] < step_range[1]
            and "exec_entropy_var" in m
        ]
        if not late:
            continue

        mean_var = np.mean(late)
        if "sepa" in condition:
            sepa_vars.append(mean_var)
        elif "hicra" in condition:
            hicra_vars.append(mean_var)

    print(f"SEPA  exec_entropy_var (late): {np.mean(sepa_vars):.4f} +/- {np.std(sepa_vars):.4f}")
    print(f"HICRA exec_entropy_var (late): {np.mean(hicra_vars):.4f} +/- {np.std(hicra_vars):.4f}")
    print(f"Reduction: {(1 - np.mean(sepa_vars)/np.mean(hicra_vars))*100:.1f}%")
```

## Statistical testing

### Quick significance check

For a quick two-proportion comparison between two conditions:

```python
from scipy.stats import fisher_exact

# Count correct completions across all seeds for each condition
sepa_correct, sepa_total = 342, 1024
hicra_correct, hicra_total = 298, 1024

table = [
    [sepa_correct, sepa_total - sepa_correct],
    [hicra_correct, hicra_total - hicra_correct],
]
odds_ratio, p_value = fisher_exact(table)
print(f"SEPA: {sepa_correct/sepa_total:.1%}, HICRA: {hicra_correct/hicra_total:.1%}")
print(f"Fisher exact p = {p_value:.4f}")
```

### Using textpolicy analysis tools

The `textpolicy` companion package includes a full statistical testing pipeline. If your project includes textpolicy:

```bash
# Run significance testing on paired campaign results
uv run python scripts/sepa_significance.py \
  --baseline logs/campaign/runs/maxrl+gtpo_hicra_s* \
  --candidate logs/campaign/runs/maxrl+gtpo_sepa_s* \
  --output significance.json \
  --markdown significance.md \
  --resamples 20000 \
  --alpha 0.05
```

This produces a markdown report with:

- **Permutation test** p-values for mean reward and correct rate
- **Bootstrap 95% CI** on the difference (candidate - baseline)
- **Fisher exact test** for correctness proportion
- **Cohen's d** effect size
- **Sample size estimates** for 80% power at the observed effect

The recommendation will be one of:

| Recommendation | Meaning |
|---------------|---------|
| `statistically_significant_improvement` | p < alpha, delta > 0, CI doesn't cross zero |
| `insufficient_statistical_evidence` | p >= alpha or CI crosses zero |
| `significant_degradation` | p < alpha but delta < 0 (candidate is worse) |

### Litmus verdicts

textpolicy also provides rule-based litmus tests with evidence gating:

```bash
uv run python scripts/sepa_litmus.py \
  --baseline logs/campaign/runs/maxrl+gtpo_hicra_s* \
  --candidate logs/campaign/runs/maxrl+gtpo_sepa_s* \
  --output litmus.json \
  --markdown litmus.md
```

Verdicts:

| Verdict | Meaning |
|---------|---------|
| `CONFIRMED` | All threshold checks pass and evidence requirements met |
| `FAILED` | Evidence sufficient but one or more thresholds not met |
| `INCONCLUSIVE` | Not enough data (too few seeds, steps, or completions) |

### When to use which

- **wandb dashboard**: First look, visual comparison, spot obvious trends
- **Fisher exact / permutation test**: Rigorous comparison between two specific conditions
- **Litmus**: Quick pass/fail gate before investing in more seeds
- **Full significance report**: Paper-grade analysis with confidence intervals and power estimates

## Checklist: running a research campaign

1. **Validate setup** (30 min):
    ```bash
    retrain --max-steps 20 --batch-size 2 --group-size 4 --seed 42
    ```
    Check that training runs, metrics are logged, correct_rate > 0.

2. **Small sweep** (a few hours):
    ```bash
    python -m retrain.campaign --max-steps 100 --seeds 42,101,202 --wandb-project test-sweep
    ```
    Check wandb: do the 5 conditions separate? Is the ordering roughly right?

3. **Full campaign** (1-2 days on H100, longer on 4090):
    ```bash
    python -m retrain.campaign --max-steps 500 --seeds 42,101,202,303,404,505,606,707 \
      --wandb-project sepa-full --execute
    ```

4. **Analyze**: Plot learning curves, check entropy distributions, run significance tests.

5. **Scale up if needed**: If SEPA vs HICRA is directional but non-significant, double the seeds (16) and re-run.
