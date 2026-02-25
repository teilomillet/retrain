# retrain

RLVR (Reinforcement Learning with Verifiable Rewards) training framework for LLMs. Train reasoning models on MATH with composable advantage functions and adaptive scheduling.

## Features

- **Composable advantages** -- GRPO, MaxRL, GTPO entropy weighting, HICRA planning amplification, SEPA entropy pooling
- **Pluggable inference** -- PyTorch, MAX, vLLM, SGLang, or any OpenAI-compatible server
- **Pluggable rewards** -- string match, symbolic math, LLM judge, or bring your own
- **Back pressure** -- USL+Roofline adaptive batch sizing
- **Campaign orchestrator** -- sweep conditions x seeds from a single TOML, with wandb logging
- **LoRA-Squeeze** -- auto-analyze optimal LoRA rank after first campaign run (arXiv 2602.10993)
- **Checkpoint resume** -- save and restore full trainer state across preemptions

## Install

```bash
pip install -e .
```

## Quick start

```bash
# 0. Inspect CLI manual (human or agent-friendly)
retrain man
# show a specific section in machine-readable form
retrain man --format json --topic quickstart
# grep/edit the source manual directly
retrain man --path
# refresh auto-generated manual sections
retrain man --sync

# 1. Drop a config
cp retrain.toml my_run.toml

# 2. Train
retrain my_run.toml

# 3. Override from CLI
retrain my_run.toml --seed 42 --wandb-project my-project
```

## Configuration

All configuration lives in a TOML file. See [`retrain.toml`](retrain.toml) for the default config, or run `retrain help` for the full reference.

### Custom Transform (TOML-first)

You can select a custom advantage transform directly from TOML using a dotted Python path:

```toml
[algorithm]
advantage_mode = "maxrl"
transform_mode = "my_transforms.make_transform_spec"
```

Then add an importable Python module (for example `my_transforms.py`) that returns a `TransformSpec`:

```python
from retrain.advantages import TransformSpec

def _entropy_transform(entropies, planning_mask, sepa_lambda):
    return [e if m else e + sepa_lambda for e, m in zip(entropies, planning_mask)]

def make_transform_spec():
    return TransformSpec(
        name="entropy_shift",
        use_gtpo=True,
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=_entropy_transform,
    )
```

### Verifiers Environment (TOML-first)

Use a verifiers environment directly from TOML:

```toml
[environment]
provider = "verifiers"
id = "primeintellect/gsm8k"           # installed env id
args = { split = "train" }             # native TOML object
auto_install = true                    # install from Prime Hub if missing
max_turns = 8                          # only used for multi-turn envs
```

For single-turn envs, retrain scores with the environment rubric.
For multi-turn envs (for example Wordle-style), retrain runs the env loop and
samples each turn with the selected backend/model.
Some Hub environments are eval-only and do not expose training datasets; in that
case retrain now fails fast with guidance. Known trainable examples:
`primeintellect/gsm8k`, `primeintellect/wordle`.

Minimal switchboard in TOML:

```toml
[backend]
backend = "tinker"                     # local | tinker

[algorithm]
transform_mode = "gtpo_sepa"           # or dotted plugin path

[environment]
provider = "verifiers"                 # optional
id = "primeintellect/wordle"           # e.g. wordle or aime
```

## Documentation

Full documentation: [retrain.readthedocs.io](https://retrain.readthedocs.io)

- [Getting Started](https://retrain.readthedocs.io/getting-started/)
- [Configuration Reference](https://retrain.readthedocs.io/configuration/)
- [Advantage Functions](https://retrain.readthedocs.io/advantages/)
- [SEPA Scheduling](https://retrain.readthedocs.io/sepa/)
- [Campaigns](https://retrain.readthedocs.io/campaigns/)
- [LoRA-Squeeze](https://retrain.readthedocs.io/squeeze/)
- [Reward Functions](https://retrain.readthedocs.io/rewards/)
- [Inference Engines](https://retrain.readthedocs.io/inference-engines/)
