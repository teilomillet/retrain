# Backends

retrain supports three training backends: **local** (PyTorch/PEFT on your GPUs), **tinker** (remote GPU service), and **prime_rl** (external PRIME-RL trainer + inference).

## Backend capabilities

`retrain doctor`, `retrain explain`, and trainer startup report capability metadata:

| Backend | reports_sync_loss | preserves_token_advantages | supports_checkpoint_resume | resume_runtime_dependent |
|---------|-------------------|----------------------------|----------------------------|--------------------------|
| `local` | `true` | `true` | `true` | `false` |
| `tinker` | `true` | `true` | `true` | `true` |
| `prime_rl` | `false` | `false` | `true` | `false` |

- `reports_sync_loss=false` means the backend returns a placeholder loss value by design.
- `preserves_token_advantages=false` means token-level advantages are aggregated before backend transport.
- Uncertainty kinds (e.g. `surprisal`, `shannon_entropy`) are discovered from backend data,
  not declared statically. The advantage pipeline inspects what data the backend provides
  (logprobs, token distributions) and raises a diagnostic error if the required data for the
  configured `uncertainty_kind` is absent. Today all backends provide only per-token logprobs,
  so `surprisal` is the only usable kind. When a backend returns full token distributions,
  `shannon_entropy` becomes available automatically. `flow.trace()` catches mismatches at
  preflight via the synthetic probe.
- Dotted-path custom plugins use conservative defaults and are reported as `source=plugin/default`.
  Unless overridden by plugin capability hooks, this default is `preserves_token_advantages=false`,
  so token-varying flows fail preflight instead of silently degrading.

Inspect backend metadata directly:

```bash
retrain backends --json
```

The JSON payload includes built-in dependency hints, capability flags, option schemas, and plugin hook names.

### Plugin metadata hooks

Dotted-path custom backends can provide metadata hooks on either the backend callable/class
or its module:

- `retrain_backend_capabilities` / `RETRAIN_BACKEND_CAPABILITIES`
- `retrain_backend_option_schema` / `RETRAIN_BACKEND_OPTION_SCHEMA`

Capabilities can be provided as:

- `BackendCapabilities(...)`
- dict with keys: `reports_sync_loss`, `preserves_token_advantages`, `supports_checkpoint_resume`, `resume_runtime_dependent`

Option schema can be provided as:

- mapping to `BackendOptionSpec`
- mapping to dict spec: `{type|value_type, default, choices?, validator?}`

## Local backend

The default. Runs PyTorch/PEFT training directly on local GPUs with a pluggable inference engine.

```toml
[backend]
backend = "local"
devices = "gpu:0"
adapter_path = "/tmp/retrain_adapter"
```

### Single GPU

With one device, the same model handles both inference and training. No weight duplication, minimal overhead.

```toml
[backend]
devices = "gpu:0"
```

### Multi-GPU (split mode)

With multiple devices, inference runs on the first GPU and training on the last:

```toml
[backend]
devices = "gpu:0,gpu:1"
```

- GPU 0: inference (sampling completions)
- GPU 1: training (gradient updates)

In split mode, training runs asynchronously on a background thread. Weight snapshots are taken after each optimizer step for safe cross-thread sync to the inference engine.

### External inference engine

When using MAX, vLLM, SGLang, or MLX-LM, the external engine manages its own GPUs. The `devices` field controls only the training GPU:

```toml
[backend]
devices = "gpu:7"

[inference]
engine = "vllm"
url = "http://localhost:8000"
```

See [Inference Engines](inference-engines.md) for details.

## Tinker backend

Uses the Tinker remote GPU service for both inference and training. The Tinker SDK handles model loading, sampling, and gradient updates on remote hardware. No local GPU required.

### Setup

1. Install with the Tinker extra:

    ```bash
    pip install -e ".[tinker]"
    ```

2. Get your Tinker service URL from your team or infrastructure. If no URL is set, the SDK uses its default endpoint.

3. Configure your TOML:

    ```toml
    [backend]
    backend = "tinker"

    [model]
    model = "Qwen/Qwen3-4B-Instruct-2507"
    base_url = "https://your-tinker-endpoint"  # omit to use SDK default
    lora_rank = 32

    [algorithm]
    advantage_mode = "maxrl"
    transform_mode = "gtpo_sepa"

    [training]
    max_steps = 100
    batch_size = 4
    group_size = 8
    ```

4. Run:

    ```bash
    retrain
    ```

!!! note "What Tinker ignores"
    The `[inference]` section (`engine`, `url`, `attention_kernel`, etc.) and the `devices` and `adapter_path` fields are all ignored when using Tinker. The service manages its own GPU allocation, inference, and checkpoint storage.

### How it works

1. Creates a `ServiceClient` connecting to the Tinker service
2. Creates a `LoRATrainingClient` for the specified model and rank
3. Each step: `checkpoint()` saves weights server-side and returns a sampling client
4. Sampling fires all prompts via the Tinker async API, then collects results
5. Training builds `Datum` objects and calls `forward_backward()` + `optim_step()` on the remote service

The tokenizer and dataset still load locally (for prompt encoding and reward scoring). Only model inference and gradient updates happen remotely.

### Tinker vs Local

| Feature | Local | Tinker |
|---------|-------|--------|
| Local GPU required | Yes | No |
| Model loading | Local HuggingFace download | Server-side |
| Inference | Pluggable engine (PyTorch, MAX, vLLM, SGLang, MLX-LM, ...) | Tinker sampling API |
| Training | PyTorch/PEFT | Tinker training API |
| Weight sync | In-memory or disk | Server-managed |
| Checkpoints | Saved to `adapter_path` | Saved on Tinker service |
| Setup | `pip install -e .` + CUDA GPU | `pip install -e ".[tinker]"` + service URL |

## PRIME-RL backend (experimental)

Uses a running PRIME-RL stack for training + inference while keeping retrain's
trainer loop, rewards, and logging.

```toml
[backend]
backend = "prime_rl"
adapter_path = "/path/to/prime_rl/output_dir"   # MUST match PRIME-RL trainer output_dir

[backend.options]
transport = "filesystem"                         # or "zmq"
strict_advantages = true

[inference]
url = "http://localhost:8000"                    # PRIME-RL inference endpoint
```

Notes:

- retrain sends PRIME-RL `TrainingBatch` messages through PRIME-RL transport.
- `checkpoint()` syncs inference from PRIME-RL broadcast checkpoints via `/update_weights`.
- Optional PRIME-RL settings live under `[backend.options]`: `zmq_host`, `zmq_port`,
  `zmq_hwm`, `sync_wait_s`, and `sync_poll_s`.
- PRIME-RL transport expects one scalar advantage per sample.
  retrain now rejects built-in token-varying modes on `prime_rl` (for example
  `transform_mode=gtpo|gtpo_hicra|gtpo_sepa` and
  `algorithm_mode=maxrl_gtpo|maxrl_gtpo_hicra|maxrl_gtpo_sepa`) to avoid
  silent credit-assignment loss.
  For any backend that reports `preserves_token_advantages=false` (including
  future/custom backends), trainer preflight constructs and probes the configured
  advantage flow and fails before training starts if token-varying advantages are detected.
  `strict_advantages=false` is disallowed to prevent silent aggregation.
- PRIME-RL `train_step()` reports a placeholder loss (`0.0`) because optimization
  runs asynchronously inside the PRIME-RL runtime.

## Device allocation

The `devices` field accepts comma-separated GPU specs:

| Value | Meaning |
|-------|---------|
| `gpu:0` | Single GPU (CUDA device 0) |
| `gpu:0,gpu:1` | Split mode: infer on 0, train on 1 |
| `gpu:7` | Single GPU (CUDA device 7) |
| `cpu` | CPU-only (slow, for testing) |

If CUDA is not available, local backend falls back to CPU automatically.

## Checkpoint resume

Both backends support resuming from checkpoints:

```toml
[resume]
from = "logs/train"
```

Or via CLI:

```bash
retrain --resume logs/train
```

This restores:

- Training step counter
- Dataset position (example index)
- Running accuracy counters
- Batch and group sizes
- SEPA controller state
- LoRA adapter weights

The resume directory must contain a `trainer_state.json` file (created automatically by `save_every` checkpoints and at the end of training).

**Local backend:** Checkpoints are saved to `adapter_path` as subdirectories (e.g., `checkpoint_step_20/`, `final/`). LoRA weights are restored via `safetensors` or `.bin` files. Optimizer state (Adam momentum) is not restored -- the optimizer re-warms.

**Tinker backend:** Checkpoint loading requires the Tinker SDK to support `load_state()` on the training client. If your SDK version doesn't have this method, resume will fail with an `AttributeError`. Check your Tinker SDK version supports checkpoint loading before relying on resume.
