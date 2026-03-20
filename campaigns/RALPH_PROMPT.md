# Objective

You are optimizing GRPO training on Qwen3-4B math. The model plateaus at ~47.5% correct_rate across all conditions. Your job is to find a configuration that breaks this plateau, or definitively prove why it can't be broken with the current setup.

## Rules

- ONLY modify files in campaigns/, logs/, and this campaigns/ directory
- NEVER modify retrain/ source code or tests/
- NEVER exceed 3 concurrent campaign runs
- Track ALL findings in campaigns/EXPERIMENT_THESIS.md under the Results section
- Track iteration state in campaigns/ralph_state.json
- Be honest: if a metric measures YOUR preprocessing (adv_cap_fraction), say so. Only claim signal from model behavior (entropy trajectory, correct_rate, loss).

## Available Mechanisms

| Config Field | What It Does | Backend |
|---|---|---|
| `adv_clip_max` | Caps token advantages to [-max, +max] before training | Any (Tinker, local) |
| `clip_eps` / `clip_eps_high` | PPO-style ratio clipping during training | Local, Tinker |
| `transform_mode` | none, gtpo, gtpo_sepa, gtpo_hicra, entropy_mask, etc. | Any |
| `advantage_mode` | grpo, maxrl | Any |
| `gtpo_beta` | Entropy weighting strength | Any |
| `grad_clip_norm` | Max global gradient norm (0=disabled) | Tinker (via AdamParams) |
| `lr` | Learning rate | Any |
| `temperature` | Sampling temperature | Any |

## Phase Detection (read state, then act)

### 1. Read current state

```bash
# Check ralph state
cat campaigns/ralph_state.json 2>/dev/null || echo '{"iteration": 0, "phase": "init"}'

# Check running campaigns
ps aux | grep retrain | grep -v grep | head -5

# Check completed campaign logs
ls -dt logs/campaign_* 2>/dev/null | head -10

# Check thesis results section
grep -A 999 "^## Results" campaigns/EXPERIMENT_THESIS.md
```

### 2. Act based on phase

**Phase: init** — No campaigns launched yet.
- Verify clip-rescue.toml exists and parses: `uv run python -c "from retrain.config import load_config; load_config('campaigns/clip-rescue.toml'); print('OK')"`
- Launch: `retrain campaigns/clip-rescue.toml`
- Update ralph_state.json: `{"iteration": N, "phase": "running", "campaign": "clip-rescue", "launched_at": "..."}`

**Phase: running** — Campaign is in progress.
- Check status: `uv run retrain status`
- If still running: check partial metrics from completed runs
- Read any available metrics.jsonl files and report early trends
- **PARALLELISM**: `max_workers` in a TOML limits concurrency WITHIN that campaign.
  Multiple campaigns can run simultaneously as separate processes. If you have a
  follow-up campaign designed and the current one is still running, LAUNCH BOTH.
  Don't wait for one to finish before starting the next.
- If all runs complete: move to "analyze"
- Update ralph_state.json

**Phase: analyze** — Campaign finished. Analyze results.
- Read ALL metrics.jsonl files from the campaign
- Compute per-condition mean correct_rate at steps 20, 50, 80, 100
- Compute per-condition exec_surprisal_var trajectory
- Compute adv_cap_fraction (is the cap even intervening?)
- Compare C1 (uncapped) vs C2 (cap=5) vs C3 (cap=2) vs C4 (SEPA+cap=5)
- Write findings to EXPERIMENT_THESIS.md Results section
- Decide next action:

  **If separation found (>2% gap):** Design follow-up to confirm/extend.
  **If entropy diverges but correct_rate doesn't:** Adjust mechanism (try different cap values, lr, temperature).
  **If cap_fraction ≈ 0:** Advantages aren't extreme. Try different approach (lr schedule, temperature, group_size).
  **If all null:** Record conclusion, try fundamentally different approach.

- Create new campaign TOML if needed
- Update ralph_state.json to next phase

**Phase: iterate** — Previous campaign analyzed, new one designed.
- Launch new campaign
- Move to "running"

**Phase: done** — We have a conclusion.
- Write final findings to EXPERIMENT_THESIS.md
- Output: <promise>EXPERIMENT COMPLETE</promise>

### 3. Iteration budget

You have 30 iterations total. Each campaign takes time.
- Iterations 1-5: launch and monitor clip-rescue
- Iterations 6-10: analyze clip-rescue, design follow-up
- Iterations 11-20: run and analyze follow-up experiments
- Iterations 21-30: final experiments or write conclusions

If by iteration 25 you have no signal, write up the null result with what you learned and declare done.

## What "interesting to say" looks like

Rank-ordered by value:
1. "Advantage capping at X breaks the 47.5% plateau" (strongest claim, needs correct_rate evidence)
2. "Advantage capping prevents entropy collapse but doesn't improve accuracy" (mechanistic insight, needs entropy evidence)
3. "GRPO advantages have heavy tails of magnitude Y that correlate with entropy collapse" (diagnostic finding, needs adv_cap_fraction + entropy data)
4. "The 47.5% plateau is not caused by gradient magnitude — here's evidence" (useful null, needs cap_fraction > 0 with no behavior change)
5. "The Tinker backend's importance_sampling loss already bounds ratios internally" (infrastructure finding, needs cap intervention + no loss curve change)

Any of these is a publishable direction. The worst outcome is "we ran stuff and don't know what happened."
