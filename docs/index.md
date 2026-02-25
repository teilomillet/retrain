# retrain

RLVR (Reinforcement Learning with Verifiable Rewards) training framework for LLMs. Train reasoning models on MATH with composable advantage functions and adaptive scheduling.

!!! info "Hardware"
    **Local backend:** One CUDA GPU with **16+ GB VRAM** (RTX 4090, A100, H100).
    **Tinker backend:** No local GPU -- training runs on the remote Tinker service.
    See [Getting Started](getting-started.md#prerequisites) for details.

## Why retrain?

- **One command** -- `retrain retrain.toml` runs the full pipeline: load model, sample completions, score with verifiable rewards, compute advantages, train with LoRA.
- **Composable algorithms** -- mix and match GRPO/MaxRL advantages with GTPO/HICRA/SEPA transforms. The [5 conditions](campaigns.md) from the SEPA paper are first-class.
- **Pluggable everything** -- inference engines, reward functions, and backends are all swappable via TOML config.
- **Production-ready** -- wandb logging, campaign sweeps, checkpoint resume, adaptive batch sizing.

## Architecture

```
retrain
├── cli.py              # Entry point, TOML + CLI override parsing
├── config.py           # TrainConfig dataclass, TOML loader
├── trainer.py          # Main training loop
├── advantages.py       # GRPO, MaxRL, GTPO, HICRA, SEPA, planning tokens
├── sepa.py             # SEPA scheduler (linear / auto)
├── rewards.py          # match, math, judge, custom reward functions
├── backpressure.py     # USL+Roofline adaptive batch sizing
├── campaign.py         # Sweep orchestrator (conditions x seeds) with auto-squeeze
├── squeeze.py          # LoRA-Squeeze rank analysis and compression
├── local_train_helper.py   # Local GPU backend (PyTorch/PEFT + inference engine)
├── tinker_backend.py       # Remote GPU backend (Tinker API)
├── inference_engine/       # Pluggable inference (PyTorch, MAX, vLLM, SGLang, MLX-LM)
├── data.py             # MATH dataset loader
└── logging_utils.py    # JSONL logger
```

## Features

| Feature | Description |
|---------|-------------|
| **GRPO / MaxRL** | Episode-level advantage functions with inverse success-rate reweighting |
| **GTPO** | Entropy-weighted token-level credit assignment |
| **HICRA** | Planning token amplification via strategic gram detection |
| **SEPA** | Selective Entropy Pooling of Attention -- adaptive scheduling with correctness gate |
| **Inference engines** | PyTorch, MAX, vLLM, SGLang, MLX-LM, OpenAI-compatible servers |
| **Reward functions** | String match, symbolic math (math_verify), LLM judge, custom |
| **Back pressure** | USL model fits throughput curves, auto-adjusts batch size |
| **Campaigns** | Sweep conditions x seeds from a single TOML with wandb groups |
| **LoRA-Squeeze** | Train at high rank, auto-analyze optimal rank via SVD after first run |
| **Checkpoint resume** | Full trainer state (step, SEPA, optimizer) saved and restored |
| **wandb integration** | Structured metric prefixes (`train/`, `train/entropy/`, `train/backpressure/`) |

## Quick links

- [Getting Started](getting-started.md) -- install, configure, run
- [Configuration](configuration.md) -- full TOML reference and CLI overrides
- [Plugins](plugins.md) -- 60-second scaffold flow for custom algorithm/advantage/transform plugins
- [Advantage Functions](advantages.md) -- GRPO, MaxRL, GTPO, HICRA pipeline
- [SEPA](sepa.md) -- selective entropy pooling schedules
- [Reward Functions](rewards.md) -- match, math, judge, custom
- [Inference Engines](inference-engines.md) -- engine selection and multi-GPU setup
- [Back Pressure](backpressure.md) -- adaptive batch sizing
- [Campaigns](campaigns.md) -- sweep orchestrator with auto-squeeze
- [LoRA-Squeeze](squeeze.md) -- optimal rank analysis and compression
- [Backends](backends.md) -- local vs Tinker
- [Logging & wandb](logging.md) -- metrics and experiment tracking
- [Research Guide](research-guide.md) -- interpreting results, statistical testing, analysis code
