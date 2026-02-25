# retrain

`retrain` is a TOML-first RLVR (Reinforcement Learning with Verifiable Rewards) trainer for LLMs.

If you are new, start with install -> explore commands -> run a tiny config.

## Install

Requires Python 3.11+.

```bash
# CLI + docs exploration
uv tool install retrain

# Local GPU training (adds torch)
uv tool install "retrain[local]"

# Remote Tinker backend
uv tool install "retrain[tinker]"
```

If you are developing this repo directly:

```bash
pip install -e ".[dev]"
```

## Explore the CLI

Use these first to understand what exists before you train:

```bash
retrain --help
retrain man
retrain man --topic quickstart
retrain man --list-topics
retrain backends
retrain doctor
```

Useful inspection commands while iterating:

```bash
retrain explain retrain.toml   # dry-run: what this config would do
retrain status logs            # summarize runs/campaigns under logs/
retrain man --json --topic quickstart
retrain man --path             # editable bundled manual source
```

## Tiny TOML Demo

Create `mini.toml`:

```toml
[model]
model = "Qwen/Qwen3-4B-Instruct-2507"

[algorithm]
advantage_mode = "grpo"
transform_mode = "none"

[training]
max_steps = 20
batch_size = 2
group_size = 8
max_tokens = 1024
lr = 4e-5

[backend]
backend = "local"
adapter_path = "adapters/mini"

[logging]
log_dir = "logs/mini"
```

Run it:

```bash
retrain mini.toml
```

Override fields from CLI without editing TOML:

```bash
retrain mini.toml --seed 42 --max-steps 40 --wandb-project my-project
```

## Quick Start from Template

```bash
retrain init --template quickstart
retrain retrain.toml
```

Other templates:

```bash
retrain init --list
retrain init --template experiment
retrain init --template campaign
retrain init --interactive
```

## Why retrain

- Composable advantage pipeline: GRPO/MaxRL + GTPO/HICRA/SEPA
- Pluggable backends and inference engines
- Pluggable rewards (match, math, judge, custom)
- Campaign sweeps from one TOML
- LoRA-Squeeze rank analysis/compression
- Checkpoint resume and run status tooling

## Common Config Patterns

Use verifiers environments from TOML:

```toml
[environment]
provider = "verifiers"
id = "primeintellect/gsm8k"
args = { split = "train" }
auto_install = true
max_turns = 8
```

Use a custom transform plugin from TOML:

```toml
[algorithm]
advantage_mode = "maxrl"
transform_mode = "my_transforms.make_transform_spec"
```

## Documentation

Full docs: [retrain.readthedocs.io](https://retrain.readthedocs.io)

- [Getting Started](https://retrain.readthedocs.io/getting-started/)
- [Configuration Reference](https://retrain.readthedocs.io/configuration/)
- [Advantage Functions](https://retrain.readthedocs.io/advantages/)
- [SEPA Scheduling](https://retrain.readthedocs.io/sepa/)
- [Campaigns](https://retrain.readthedocs.io/campaigns/)
- [LoRA-Squeeze](https://retrain.readthedocs.io/squeeze/)
- [Reward Functions](https://retrain.readthedocs.io/rewards/)
- [Inference Engines](https://retrain.readthedocs.io/inference-engines/)

Contributor note: run `retrain man --check` in CI to detect stale auto-generated manual blocks, and `retrain man --sync` locally to refresh them.
