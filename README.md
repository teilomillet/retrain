# retrain

RLVR (Reinforcement Learning with Verifiable Rewards) training framework for LLMs. Train reasoning models on MATH with composable advantage functions and adaptive scheduling.

## Features

- **Composable advantages** -- GRPO, MaxRL, GTPO entropy weighting, HICRA planning amplification, SEPA entropy pooling
- **Pluggable inference** -- PyTorch, MAX, vLLM, SGLang, or any OpenAI-compatible server
- **Pluggable rewards** -- string match, symbolic math, LLM judge, or bring your own
- **Back pressure** -- USL+Roofline adaptive batch sizing
- **Campaign orchestrator** -- sweep all 5 conditions across seeds with wandb logging
- **Checkpoint resume** -- save and restore full trainer state across preemptions

## Install

```bash
pip install -e .
```

## Quick start

```bash
# 1. Drop a config
cp retrain.toml my_run.toml

# 2. Train
retrain my_run.toml

# 3. Override from CLI
retrain my_run.toml --seed 42 --wandb-project my-project
```

## Configuration

All configuration lives in a TOML file. See [`retrain.toml`](retrain.toml) for the default config, or run `retrain help` for the full reference.

## Documentation

Full documentation: [retrain.readthedocs.io](https://retrain.readthedocs.io)

- [Getting Started](https://retrain.readthedocs.io/getting-started/)
- [Configuration Reference](https://retrain.readthedocs.io/configuration/)
- [Advantage Functions](https://retrain.readthedocs.io/advantages/)
- [SEPA Scheduling](https://retrain.readthedocs.io/sepa/)
- [Reward Functions](https://retrain.readthedocs.io/rewards/)
- [Inference Engines](https://retrain.readthedocs.io/inference-engines/)

## License

See [LICENSE](LICENSE).
