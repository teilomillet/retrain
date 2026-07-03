# Code Organization

This guide is for contributors changing retrain internals. The goal is a codebase
that is easy to browse from the tree alone: folders provide the context, files
name one concept, and the training path stays auditable.

## Shape

Prefer package-scoped modules with short names:

```text
retrain/
  commands/
    help.py
    name.py
    backends/
      run.py
      capability.py
    doctor/
      run.py
      warn.py
    init/
      run.py
      interactive.py
      templates.py
      customize.py
    plugins/
      run.py
      scaffold.py
      template.py
      kinds.py
      name.py
    status/
      run.py
      top.py
    trace/
      run.py
    tree/
      run.py
      node.py
      note.py
      eval.py
    manual/
      run.py
      render.py
      sync.py
      topic.py
  config/
    schema.py
    load.py
    override.py
    validate/
  types.py
  training/
    trainer.py
    discover.py
    runner.py
    flow.py
    sft.py
    warmup.py
    state.py
    sepa.py
    echo.py
    rollouts.py
    loss.py
    backpressure.py
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
  environments/
    load.py
    prompt.py
    verifiers.py
  rewards/
    boxed.py
    create.py
    custom.py
    types.py
    verifiers.py
  planning/
    regex.py
    semantic.py
    create.py
    tokens.py
    types.py
  registry/
    core.py
    builtin.py
    health.py
  campaign/
    run.py
    sequential.py
    parallel.py
    parse.py
    configs.py
    squeeze.py
    delight.py
  squeeze/
    rank.py
    adapter.py
    run.py
  status/
    scan.py
    format.py
    export/
      scan.py
      prometheus.py
      runs.py
      server.py
      types.py
  tree/
    model.py
    state.py
    eval.py
    format.py
  diff/
    compute.py
    format.py
  benchmark/
    run.py
    summary.py
    format.py
  models/
    gemma4.py
    qwen35.py
  kernels/
    lora.py
    logprobs.py
    accelerators.py
  backends/
    catalog.py
    local/
      train.py
      checkpointing.py
      lora.py
      metrics.py
      memory.py
      sft.py
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
  plugins/
    resolve.py
  process/
    metrics.py
  data/
    source.py
    math.py
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
- `commands/backends/`: backend catalog display and capability summaries used by
  CLI previews.
- `commands/doctor/`: dependency probes and config-backed dependency warnings.
- `commands/init/`: starter config templates, noninteractive init, and the TTY
  interactive setup flow.
- `commands/plugins/`: plugin discovery listing and plugin scaffold generation.
- `commands/status/`: log status scanning and the live `top` dashboard alias.
- `commands/trace/`: flow construction preflight and JSON/text trace output.
- `commands/tree/`: experiment tech-tree dispatch plus node run, note, and
  evaluation side effects.
- `commands/manual/`: manual command parsing, auto-block rendering, topic
  lookup, and sync checks.
- `environments/load.py`: verifiers environment argument parsing, loading, and
  dataset row conversion.
- `environments/prompt.py`: prompt preview, tokenizer encoding, and observation
  token masks.
- `environments/verifiers.py`: multi-turn rollout execution and rubric scoring.
- `registry/`: registry core, built-in factories, and dependency health probes.
- `rewards/`: boxed-match, verifiers-backed, and custom rewards.
- `planning/`: regex and semantic planning-token detectors.
- `training/signals.py`: reward ties, advantage caps, and algorithm parameters.
- `training/flow.py`: flow construction and traceable preflight state.
- `training/telemetry.py`: pure builders for metrics, wandb payloads, and
  emergence rows.
- `training/log.py`: side effects for recording one training step.
- `training/warmup.py`: supervised warmup data loading, stepping, and logging.
- `training/trainer.py`: the main RL orchestration path.
- `training/discover.py`: test-time training over a single problem.
- `advantages/episode.py`: episode-level GRPO, MaxRL, and REINFORCE++ modes.
- `advantages/credit.py`: token-credit transforms such as GTPO, HICRA, SEPA,
  and masking.
- `advantages/delight/`: Delight gates, eta resolution, metrics, and transforms.
- `advantages/pipeline.py`: the composable advantage computation path.
- `campaign/`: sweep config expansion, sequential/parallel execution, and
  post-campaign squeeze/delight summaries.
- `squeeze/`: adapter IO, LoRA-Squeeze rank math, and squeeze workflow command.
- `status/`: run scanning, table formatting, and live export split into scan,
  Prometheus, JSON, server, and snapshot-type modules.
- `tree/`, `diff/`, and `benchmark/`: command-domain packages for experiment
  trees, comparisons, and repeated runs.
- `backends/tinker/train.py`: Tinker backend training implementation.
- `backends/tinker/runtime.py`: Tinker SDK loading and runtime checks.
- `backends/unsloth/train.py`: Unsloth-backed local training implementation.
- `backends/local/`: local PyTorch backend orchestration, checkpointing policy,
  LoRA setup, runtime metrics, GPU memory policy, and SFT row shaping.
- `models/`: model-specific compatibility and kernel selection.
- `kernels/`: GPU-kernel adapters and selected-token logprob math.
- `data/`: example shape, data-source protocol, and built-in datasets.
- `io/json.py`: JSON loading and compact JSONL row encoding.
- `io/log.py`: buffered JSONL append logging.
- `metrics/scan.py`: one-pass metrics JSONL readers and aggregates.
- `plugins/resolve.py`: dotted-path plugin runtime, cache, and discovery.
- `process/metrics.py`: process-local runtime measurements such as peak RSS.

Do not split just to hide lines. A private helper is worth keeping only when it
removes real branching, names a non-obvious invariant, or serves at least three
call sites across different concerns. Otherwise, keep the logic local.

## Trainer Boundary

`training/trainer.py` should read as the orchestration path:

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
from retrain.commands.manual import run as run_manual_command
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
