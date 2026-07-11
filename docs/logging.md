# Logging & wandb

## JSONL files

retrain writes structured logs to JSONL files in the `log_dir` directory:

```
logs/train/
├── metrics.jsonl              # Per-step training metrics
├── optimizer_batch_step_000000.safetensors    # optional exact captured rows
├── optimizer_batch_step_000000.manifest.json  # optional manifest-last contract
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
| `reported_loss` | Raw backend-reported loss value (same as `loss`, explicit for tooling) |
| `loss_is_placeholder` | `true` when loss is backend placeholder by design |
| `backend_reports_sync_loss` | Whether backend loss is synchronous optimization loss |
| `backend_preserves_token_advantages` | Whether backend consumes per-token advantages directly |
| `mean_reward` | Mean reward across the batch |
| `correct_rate` | Batch correct rate |
| `running_correct_rate` | Cumulative correct rate |
| `sepa_lambda` | Current SEPA pooling strength |
| `sepa_gate_open` | Whether SEPA correctness gate is open |
| `num_datums` | Number of datums submitted for training |
| `optimizer/logical_batch_sha256` | Canonical digest of trainer-logical tokens, logprobs, post-normalization/cap advantages, and active ECHO fields before backend-specific transforms |
| `optimizer/local_effective_rows_sha256` | Local/Unsloth-only digest of post-crop row tensors after float32 conversion; RL includes logprobs, advantages, active ECHO masks, counts, and rollout denominator, while SFT uses a distinct framing for target weights |
| `optimizer/batch_sha256` | Deprecated compatibility alias of `optimizer/logical_batch_sha256` |
| `optimizer_batch/*` | Capture/replay payload, manifest, source/replay config-contract, and initial/final adapter provenance |
| `max_token_hit_rate` | Fraction of completions that hit max_tokens |
| `step_time_s` | Wall time for the step |
| `batch_size` | Current batch size |
| `group_size` | Current group size |
| `bp_action` | Back pressure action |
| `bp_regime` | Back pressure regime |
| `uncertainty_kind` | Uncertainty variant selected in config (`surprisal`, etc.) |
| `exec_entropy_mean` | Mean execution-token entropy (GTPO modes) |
| `exec_entropy_var` | Execution-token entropy variance |
| `plan_entropy_mean` | Mean planning-token entropy |
| `plan_entropy_var` | Planning-token entropy variance |
| `exec_surprisal_mean` | Mean execution-token surprisal (preferred name) |
| `exec_surprisal_var` | Execution-token surprisal variance (preferred name) |
| `plan_surprisal_mean` | Mean planning-token surprisal (preferred name) |
| `plan_surprisal_var` | Planning-token surprisal variance (preferred name) |

`*_entropy_*` keys are kept for backward compatibility and currently mirror `*_surprisal_*`.

The two optimizer digests answer different questions. Matching
`optimizer/logical_batch_sha256` proves that the trainer handed backends the
same logical batch. It does not prove that backend-specific cropping or numeric
conversion was identical. For matched local or Unsloth ablations, compare
`optimizer/local_effective_rows_sha256`; equality proves only that retrain
constructed the same post-crop rows and row-level scaling inputs. It does not
hash loss configuration, microbatch partitioning, model/optimizer state,
learning rate, or weight decay, so those must be matched independently before
claiming equivalent optimizer inputs or updates. Remote backends report only
the logical digest because retrain cannot observe their internal effective
rows.

The deprecated `optimizer/batch_sha256` and
`train/optimizer/batch_sha256` fields remain aliases of the logical digest for
existing JSONL and W&B consumers.

When `[optimizer_batch].capture = true`, capture metrics include the absolute
manifest path plus payload, manifest, source-config, optimizer-contract, and
initial-adapter SHA256 values. Replay requires that manifest SHA as an external
pin, transitively covering the RNG-bearing payload before JSON parsing.
`trainer = "optimizer_replay"` records those
source hashes again, the replay config/contract hashes, exact allowed and
observed config-difference lists, the final adapter hash, and explicit
`dataset_skipped`, `environment_skipped`, `rollout_skipped`, and
`sampling_skipped` flags. Replay writes
`optimizer_batch_replay_manifest.json` beside `metrics.jsonl` with the same
provenance. See [Exact-Input Optimizer-Batch Replay](optimizer-batch-replay.md).

The final adapter SHA is a diagnostic boundary, not an admission requirement
for an exact-input runtime/memory claim. Bitwise update reproducibility requires
that SHA to match across source and repeated same-condition replays. Replay
adapters are one-batch systems artifacts; quality evaluation uses the
source-run adapter.

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
checkpoint_artifacts = "auto"
```

With the default `checkpoint_artifacts = "auto"`, setting `wandb_project` also
uploads every saved checkpoint and the final adapter as W&B Artifacts. The
artifact contains the adapter payload when it is on local disk, plus
`trainer_state.json`, `latest_sampler_path.txt`, and SFT reproducibility files
when they exist.

For spot or otherwise ephemeral machines, use fail-closed mode:

```toml
[logging]
wandb_project = "my-project"
checkpoint_artifacts = "wandb"
```

`checkpoint_artifacts = "wandb"` requires a live W&B run and raises if artifact
upload is unavailable, W&B is offline, or `save_every = 0`. Without periodic
checkpoints, W&B can still receive the final adapter after a completed run, but
it cannot recover a preempted mid-run job. Without W&B, retrain prints a
local-only warning because checkpoints saved under `adapter_path` and `log_dir`
can disappear with the machine.

To resume after downloading a checkpoint artifact, restore the artifact contents
to a log directory so `trainer_state.json` is present. If the original
checkpoint path from the dead machine no longer exists, retrain automatically
uses the artifact-local `adapter/` directory when it is present. Then run:

```bash
retrain resume-check logs/restored-run --config config.toml
retrain --resume logs/restored-run
```

For local and Unsloth runs this is adapter-only recovery: retrain restores the
trainer counters and LoRA weights, but not optimizer/scaler/RNG state. Check
`retrain status --json` for the saved `resume_mode` and `resume_warning`.
`retrain resume-check` performs the same local preflight before a restart:
it checks `trainer_state.json`, checkpoint payload files, step bounds from
`--config`, resume mode, and local SFT data recoverability without loading the
model or contacting W&B.

### Live Recovery Drill

Run this cheap opt-in drill after changing checkpoint artifact code or W&B
plumbing:

```bash
uv pip install --python .venv/bin/python 'retrain[wandb]'
.venv/bin/python scripts/wandb_recovery_drill.py
```

The drill uses a fake SFT backend and tokenizer, so it does not download a model
or require a GPU. It still uses the real retrain SFT runner and real W&B
Artifacts service. It uploads periodic checkpoints, downloads
`checkpoint_step_1`, deletes the original local `log_dir` and `adapter_path`,
then resumes from the downloaded artifact-local `adapter/`. Success ends with:

```text
DRILL_OK real_wandb_checkpoint_artifact_resume_succeeded
```

### Metric prefixes

All wandb metrics use structured prefixes:

| Prefix | Metrics |
|--------|---------|
| `train/` | `loss`, `rewards/mean_reward`, `rewards/correct_rate`, `rewards/running_correct_rate`, `sepa_lambda`, `sepa_gate_open`, `max_token_hit_rate`, `num_datums`, `step_time_s`, `batch_size`, `group_size` |
| `train/backend/` | `reports_sync_loss`, `preserves_token_advantages` |
| `train/backend/local/` | Optimizer timing, CUDA peak memory, microbatch count, exact padding, attention-work proxy, and sequence-length fields for both RL and standalone SFT |
| `train/sft/` | Backend-independent logical-batch sequence lengths, padding fraction, and supervised-token fraction |
| `train/` (semantics) | `reported_loss`, `loss_is_placeholder`, `train_time_semantics`, and PRIME-RL-only `train_submit_enqueue_time_s` / `train_submit_enqueue_share` |
| `train/rl/` | Action-token coverage, pre-optimizer and optimizer-input nonzero-advantage counts, and ECHO-adjacent RL timing |
| `train/echo/` | ECHO candidates plus native OpenEnv response, exact-bridge, renderer-parity, failure, and terminal-token counters |
| `train/entropy/` | `exec_mean`, `exec_var`, `plan_mean`, `plan_var` |
| `train/surprisal/` | `exec_mean`, `exec_var`, `plan_mean`, `plan_var` |
| `train/backpressure/` | `action`, `regime`, `p_star`, `sigma`, `kappa`, `utilization`, `throughput`, `warmup` |
| `train/recoverability/` | `checkpoint_artifacts_enabled`, `checkpoint_artifacts_live`, `periodic_checkpoints_enabled`, `preemption_resume_ready`, `local_only`, `latest_checkpoint_uploaded` |
| `train/optimizer/` | `logical_batch_sha256`; local/Unsloth also report `local_effective_rows_sha256`; `batch_sha256` is the deprecated logical alias |

Standalone SFT writes ordinary backend counters to JSONL as
`backend/<runtime_key>` and keeps `optimizer/*` evidence at its canonical path.
It projects the same counters to the documented `train/backend/local/*` and
`train/optimizer/*` W&B aliases. `train/sft/*` describes padding to the longest
row in the logical batch; `train/backend/local/*` describes the padding
actually materialized by local microbatches. Keep those two scopes distinct
when comparing batching strategies. SFT also emits canonical `step_time_s`
alongside its legacy rounded `time_s`, so status and benchmark scanners use the
same timing key as RL.

`rl/pre_optimizer_nonzero_advantage_action_tokens` is captured from rollout
advantages before trainer-side batch normalization and clipping.
`rl/optimizer_nonzero_advantage_action_tokens` is recomputed afterwards from
the exact advantages submitted to the backend. The legacy
`rl/nonzero_advantage_action_tokens` key remains as a compatibility alias for
the latter, final count; it no longer reports the pre-transform value.

PRIME-RL optimization is asynchronous. For that backend, `train_time_s` is
retained for schema compatibility but `train_time_semantics` is
`submit_enqueue_latency`; the explicit `train_submit_enqueue_time_s` field has
the same value. `train_share` is suppressed because enqueue time is not remote
optimizer time, and `train_submit_enqueue_share` is emitted instead. Status,
benchmark summaries, JSON export, and Prometheus preserve the missing
synchronous share and surface the enqueue fields rather than converting it to
`0%`. Mixed benchmark suites exclude explicitly labeled enqueue latency from
the synchronous `mean_train_time_s` aggregate and aggregate it under the
enqueue-specific fields instead.

### Run config

The wandb run config records all hyperparameters:

- `advantage_mode`, `transform_mode`, `uncertainty_kind`, `condition`
- `model`, `lora_rank`, `lr`, `batch_size`, `group_size`
- `max_tokens`, `temperature`, `gtpo_beta`, `hicra_alpha`
- `sepa_steps`, `sepa_delay_steps`, `sepa_correct_rate_gate`
- `max_steps`, `backend`, `seed`
- `checkpoint_artifacts`

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
