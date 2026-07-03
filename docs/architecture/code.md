# Code Organization

This guide is for contributors changing retrain internals. The goal is a codebase
that is easy to browse from the tree alone: folders provide the context, files
name one concept, and the training path stays auditable.

## Shape

Prefer package-scoped modules with short names:

```text
retrain/
  config/
    schema.py
    load.py
    override.py
    validate/
  training/
    runner.py
    signals.py
    telemetry.py
    log.py
  advantages/
    episode.py
    algorithm.py
    transform.py
    credit.py
    delight/
      gate.py
      scale.py
      eta.py
      metric.py
      transform.py
    uncertainty.py
    planning.py
    pipeline.py
  backends/
    catalog.py
    local.py
    tinker/
      train.py
      runtime.py
      throttle.py
    unsloth/
      train.py
      runtime.py
  io/
    json.py
    log.py
  metrics/
    scan.py
  process/
    metrics.py
```

Avoid long top-level compound modules such as `training_step_logging.py`. If the
name needs several words, it usually wants a folder. The folder explains the
domain; the file names the local concept.

## Split Rules

Split code when it creates a stable place a maintainer would naturally search:

- `config/schema.py`: `TrainConfig` fields and derived parameter maps.
- `config/load.py`: TOML loading and default overlay.
- `config/override.py`: CLI override parsing and coercion.
- `config/validate/`: focused validation passes for bounds, modes, runtime
  compatibility, defaults, and warnings.
- `training/signals.py`: reward ties, advantage caps, and algorithm parameters.
- `training/telemetry.py`: pure builders for metrics, wandb payloads, and
  emergence rows.
- `training/log.py`: side effects for recording one training step.
- `advantages/episode.py`: episode-level GRPO, MaxRL, and REINFORCE++ modes.
- `advantages/credit.py`: token-credit transforms such as GTPO, HICRA, SEPA,
  and masking.
- `advantages/delight/`: Delight gates, eta resolution, metrics, and transforms.
- `advantages/pipeline.py`: the composable advantage computation path.
- `backends/tinker/train.py`: Tinker backend training implementation.
- `backends/tinker/runtime.py`: Tinker SDK loading and runtime checks.
- `backends/unsloth/train.py`: Unsloth-backed local training implementation.
- `io/json.py`: JSON loading and compact JSONL row encoding.
- `io/log.py`: buffered JSONL append logging.
- `metrics/scan.py`: one-pass metrics JSONL readers and aggregates.
- `process/metrics.py`: process-local runtime measurements such as peak RSS.

Do not split just to hide lines. A private helper is worth keeping only when it
removes real branching, names a non-obvious invariant, or serves at least three
call sites across different concerns. Otherwise, keep the logic local.

## Trainer Boundary

`trainer.py` should read as the orchestration path:

1. build flow
2. load data and backend
3. sample and score rollouts
4. build train datums
5. train
6. record state and logs

It should not own schema details for JSONL metrics, wandb payloads, checkpoint
state, or backend-specific setup. Those belong in narrow package modules that
can be tested directly.

## Import Rules

Internal imports should use the package path that matches the tree. Prefer:

```python
from retrain.training.signals import apply_advantage_cap
from retrain.training.telemetry import build_step_metrics
from retrain.config import TrainConfig, load_config
from retrain.backends.tinker.runtime import load_tinker
from retrain.io.log import JsonlLogger
from retrain.metrics.scan import scan_metrics_file
```

Avoid compatibility shims unless an import path is part of a documented plugin
or public extension surface. If a shim is needed, keep it tiny and document the
removal plan.

## Comments

Comments should explain facts the code cannot show locally: protocol contracts,
determinism requirements, numerical invariants, and intentional tradeoffs. Do
not comment line-by-line mechanics.

## Review Checklist

Before committing an internal refactor:

- The tree shape explains the module names without a glossary.
- The changed files have one primary responsibility each.
- Public behavior is unchanged unless the commit message says otherwise.
- Focused tests cover the moved boundary.
- `make lint`, `make typecheck`, `uv run pytest -q`, and strict docs build pass
  before pushing.
