# retrain

RLVR (Reinforcement Learning with Verifiable Rewards) training framework for LLMs. retrain is designed to make experiments easier: define a TOML, run training, and compare outcomes with repeatable logs.

!!! info "Hardware"
    **Local backend:** One CUDA GPU with **16+ GB VRAM** (RTX 4090, A100, H100).
    **Tinker backend:** No local GPU -- training runs on the remote Tinker service.
    See [Getting Started](getting-started.md#prerequisites) for details.

## Why retrain?

- **One command** -- `retrain retrain.toml` runs the full pipeline: load model, sample completions, score with verifiable rewards, compute advantages, train with LoRA.
- **Experiment-first** -- built for rapid iteration and reproducible comparisons across configs, seeds, and conditions.
- **Composable algorithms** -- mix and match GRPO/MaxRL advantages with GTPO/HICRA/SEPA transforms. The [5 conditions](campaigns.md) from the SEPA paper are first-class.
- **Pluggable everything** -- inference engines, reward functions, and backends are all swappable via TOML config.
- **Production-ready** -- wandb logging, campaign sweeps, checkpoint resume, adaptive batch sizing.

## Architecture

```
retrain
├── cli.py              # Entry point and command dispatch
├── commands/           # CLI command implementation packages
│   ├── name.py         # CLI display-name resolution
│   ├── help.py         # Top-level help text
│   ├── backends/       # Backend catalog and capability display
│   ├── doctor/         # Dependency probes and warnings
│   ├── init/           # Config templates and init command flows
│   ├── plugins/        # Plugin listing and scaffold commands
│   ├── status/         # Status and live top commands
│   ├── trace/          # Flow trace preflight command
│   ├── tree/           # Experiment tech-tree commands
│   └── manual/         # Manual topics, rendering, sync, and dispatch
├── config/             # TrainConfig schema, TOML loading, CLI overrides
│   ├── schema.py       # TrainConfig fields and derived params
│   ├── load.py         # TOML loader
│   ├── override.py     # CLI override parser
│   ├── migrate.py      # Legacy backend config migration
│   └── validate/       # Defaults, bounds, modes, runtime checks, warnings
├── types.py            # Shared type aliases
├── training/           # The training loop and its support modules
│   ├── console.py      # Terminal summaries for training runs
│   ├── trainer.py      # Main RL loop: sample, score, compute advantages, train
│   ├── discover.py     # Test-time training over a single problem
│   ├── discovery/      # Archive, prompt, and summary helpers for discovery
│   │   ├── archive.py  # Candidate archive ranking, pruning, and selection
│   │   ├── prompt.py   # Discovery memory prompt rendering
│   │   └── summary.py  # Discovery archive JSON summary export
│   ├── examples.py     # Training example loading across providers
│   ├── runner/         # TrainingRunner protocol + built-in runners
│   │   ├── result.py   # Run results, metrics, and artifacts
│   │   ├── protocol.py # Runner protocol
│   │   ├── retain.py   # Built-in retrain loop runner
│   │   ├── command.py  # Shell-command runner
│   │   └── sft.py      # Standalone SFT runner
│   ├── flow.py         # Construct-and-trace of the training flow
│   ├── generations.py  # Generation log selection and surprisal payloads
│   ├── prompts.py      # Prompt batch selection for training steps
│   ├── sft.py          # SFT dataset and tokenization helpers
│   ├── warmup.py       # Supervised warmup phase for the RL trainer
│   ├── state.py        # Checkpoint state serialization
│   ├── sepa.py         # SEPA scheduler (linear / auto)
│   ├── echo.py         # ECHO datums, caps, and shared-step helpers
│   ├── rollouts.py     # Prompt caching and decoded rollout batches
│   ├── loss.py         # Policy-gradient loss variants
│   ├── backpressure.py # USL+Roofline adaptive batch sizing
│   ├── signals.py      # Advantage caps, ties, and algorithm params
│   ├── telemetry.py    # Step metrics, wandb payloads, emergence logs
│   └── log.py          # Side effects for recording one RL step
├── advantages/         # Advantage algorithms, transforms, and token credit
│   ├── episode.py      # GRPO, MaxRL, REINFORCE++ episode advantages
│   ├── algorithm.py    # Full algorithm registry
│   ├── transform.py    # Transform mode registry
│   ├── credit.py       # GTPO, HICRA, SEPA, masking token transforms
│   ├── delight/        # Delight gates, eta, metrics, transforms
│   ├── uncertainty.py  # Token uncertainty signals
│   ├── planning.py     # Strategic planning-token detection
│   └── pipeline.py     # Composable advantage pipeline
├── environments/       # Training environment integrations
│   ├── branch.py       # TL-GRPO branching and turn advantages
│   ├── load.py         # Verifiers args, loading, dataset conversion
│   ├── prompt.py       # Prompt previews, tokenization, observation masks
│   ├── rollout.py      # Multi-turn rollout samples, timing, and scheduling
│   ├── timing.py       # Environment observation timing extraction
│   ├── verifiers.py    # Verifiers bridge: loading, scoring, and rollouts
│   └── openenv/        # Native OpenEnv gym provider (no verifiers needed)
│       ├── client.py       # OpenEnv WebSocket wire-protocol client
│       ├── actions.py      # Completion-to-action parsing
│       ├── render.py       # Observation-to-messages renderers
│       ├── environment.py  # Multi-turn env + episodic-sum rubric
│       └── load.py         # Config loading and seed datasets
├── rewards/            # match, math, judge, custom reward functions
├── planning/           # Planning-token detectors (regex, semantic)
├── registry/           # Component registries, builtins, dependency health
├── campaign/           # Sweep orchestrator (conditions x seeds) with auto-squeeze
├── squeeze/            # LoRA-Squeeze rank analysis and compression
├── status/             # Run/campaign scanning, rendering, live export
├── tree/               # Experiment tech tree (model, state, eval, render)
├── diff/               # Run comparison (compute, render)
├── benchmark/          # Repeated-run benchmark suites
├── models/             # Model-specific quirks (Gemma4 text, Qwen3.5 kernels)
├── kernels/            # GPU kernel helpers (fast LoRA, selected-token CE)
├── backends/           # Backend protocols, catalog, and implementations
│   ├── __init__.py     # TrainHelper protocols + backend contracts
│   ├── catalog.py      # Backend definitions, capabilities, option schemas
│   ├── create/         # Built-in backend constructors
│   │   ├── local.py    # Local backend constructor
│   │   ├── unsloth.py  # Unsloth backend constructor
│   │   ├── tinker.py   # Tinker backend constructor
│   │   ├── prime.py    # PRIME-RL backend constructor
│   │   └── values.py   # Constructor option value coercion
│   ├── options.py      # Backend option schemas, validation, and coercion
│   ├── local/          # Local GPU backend package
│   │   ├── batch.py    # Tensor batch builders for local optimizer steps
│   │   ├── device.py   # Device planning for local train/inference placement
│   │   ├── logprobs.py # Token log-probability paths for local training
│   │   ├── loss.py     # CE accelerator policy and fallback decisions
│   │   ├── sampling.py # Local sampling dispatch and cache policy
│   │   ├── sync.py     # LoRA snapshot sync between trainer and engine
│   │   ├── train.py    # PyTorch/PEFT training + inference engine orchestration
│   │   ├── checkpointing.py  # Gradient-checkpointing policy and layer metrics
│   │   ├── lora.py     # Local-backend LoRA config, patching, and metrics
│   │   ├── metrics.py  # Local backend runtime telemetry and counters
│   │   ├── memory.py   # CUDA allocator, cache, and saved-tensor policy
│   │   ├── state.py    # Adapter state load/save and LoRA snapshots
│   │   ├── sft.py      # SFT row shaping, padding, and context cropping
│   │   └── steps/      # Local optimizer-step implementations
│   ├── prime.py        # PRIME-RL bridge backend
│   ├── tinker/         # Remote Tinker backend
│   │   ├── train.py    # Tinker training client implementation
│   │   ├── runtime.py  # Tinker SDK loading and type access
│   │   └── throttle.py # Optional multi-process Tinker throttle
│   └── unsloth/        # Optional Unsloth backend
│       ├── train.py    # Unsloth-backed local training implementation
│       └── runtime.py  # Unsloth import boundary
├── io/                 # JSON codec and JSONL log writer
│   ├── json.py         # JSON loads + compact JSONL encoding
│   └── log.py          # Buffered JSONL append logger
├── metrics/            # Metrics JSONL readers and summaries
│   └── scan.py         # One-pass metrics file scanning
├── plugins/            # Dotted-path plugin loading and discovery
│   └── resolve.py      # Plugin runtime config, cache, and import resolution
├── process/            # Process-local runtime measurements
│   └── metrics.py      # Peak RSS and related process telemetry
├── inference_engine/       # Pluggable inference (PyTorch, MAX, vLLM, SGLang, TensorRT-LLM, MLX-LM)
└── data/               # Example protocol and built-in MATH source
    ├── source.py       # Example shape and data-source protocol
    └── math.py         # MATH dataset loader
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
| **Capacity Planning** | Formula-driven sizing for memory, worker count, and wall time |
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
- [ECHO Training Optimization](echo-training.md) -- exact sparse ECHO loss path and verification
- [Back Pressure](backpressure.md) -- adaptive batch sizing
- [Capacity Planning](capacity-planning.md) -- estimate memory, wall time, and worker parallelism
- [Campaigns](campaigns.md) -- sweep orchestrator with auto-squeeze
- [LoRA-Squeeze](squeeze.md) -- optimal rank analysis and compression
- [Backends](backends.md) -- local vs Tinker
- [Logging & wandb](logging.md) -- metrics and experiment tracking
- [Research Guide](research-guide.md) -- interpreting results, statistical testing, analysis code
- [Tinker Forecasting Note](tinker-forecasting-note.md) -- what a recent Tinker forecasting result does and does not imply for retrain
