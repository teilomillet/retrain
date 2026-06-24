# Backends

retrain supports four training backends: **local** (PyTorch/PEFT on your GPUs), **unsloth** (Unsloth-patched local model loading with retrain's trainer loop), **tinker** (remote GPU service), and **prime_rl** (external PRIME-RL trainer + inference).

## Backend capabilities

`retrain doctor`, `retrain explain`, and trainer startup report capability metadata:

| Backend | reports_sync_loss | preserves_token_advantages | supports_checkpoint_resume | resume_runtime_dependent | supports_echo_shared_forward |
|---------|-------------------|----------------------------|----------------------------|--------------------------|------------------------------|
| `local` | `true` | `true` | `true` | `false` | `true` |
| `unsloth` | `true` | `true` | `true` | `false` | `true` |
| `tinker` | `true` | `true` | `true` | `true` | `false` |
| `prime_rl` | `false` | `false` | `true` | `false` | `false` |

- `reports_sync_loss=false` means the backend returns a placeholder loss value by design.
- `preserves_token_advantages=false` means token-level advantages are aggregated before backend transport.
- Uncertainty kinds (e.g. `surprisal`, `predictive_variance`, `shannon_entropy`) are discovered
  from backend data, not declared statically. The advantage pipeline inspects what data the
  backend provides (logprobs, token distributions) and raises a diagnostic error if the required
  data for the configured `uncertainty_kind` is absent. Today all backends provide only per-token
  logprobs, so `surprisal` and `predictive_variance` (which derives `p*(1-p)` from logprobs)
  are the usable built-in kinds. When a backend returns full token distributions,
  `shannon_entropy` becomes available automatically. Custom kinds can be provided via dotted
  plugin paths. `flow.trace()` catches mismatches at preflight via the synthetic probe.
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

For tight single-GPU memory budgets, the local backend can microbatch
`train_step()` datums and opt into CUDA allocator cleanup / lower-memory
PyTorch sampling:

```toml
[backend.options]
train_microbatch_size = 1  # 0 disables; positive values reduce train_step VRAM
cuda_empty_cache = true    # release cached CUDA blocks after local sample/train calls
sample_use_cache = true    # faster PyTorch sampling with per-step allocator cleanup
gradient_checkpointing = true  # lower train VRAM at extra forward/backward compute
```

This splits local train-step datums into smaller forward/backward chunks while
preserving the token-weighted loss. Use it when sampling fits but training OOMs
on the full datum batch. `cuda_empty_cache` defaults to `true` for the local
backend because multi-step cache-on runs can otherwise hit allocator
fragmentation even when one-step probes fit. It does not change model math; it
trades allocator reuse for lower fragmentation pressure. `sample_use_cache =
true` keeps PyTorch generation on the fast KV-cache path; disable it only for
low-memory fallback sweeps. `gradient_checkpointing` is enabled by default to
preserve the historical local backend behavior; set it to `false` in throughput
sweeps when VRAM headroom is available.

On the shared-model one-GPU path, retrain temporarily disables gradient
checkpointing during PyTorch sampling when `sample_use_cache = true`, then
re-enables it before the train step. This keeps the train memory win without
letting Hugging Face's checkpointing/use-cache incompatibility silently defeat
KV-cache inference.

For a reproducible 1-GPU sweep over the local backend memory/throughput axes,
start with a dry run:

```bash
uv run python scripts/one_gpu_local_sweep.py config.toml --dry-run
```

Then run a small smoke subset before the full Cartesian product:

```bash
uv run python scripts/one_gpu_local_sweep.py config.toml \
  --max-conditions 4 --isolate-conditions
```

The sweep varies `sample_use_cache`, `cuda_empty_cache`,
`gradient_checkpointing`, `train_microbatch_size`, and `group_size`, and each
condition writes an isolated benchmark suite plus a `sweep_manifest.json`.
Use `--continue-on-error` for full sweeps where OOM or server failures should
be recorded as condition data instead of aborting the remaining matrix.
Use `--isolate-conditions` for GPU-memory sweeps so each condition runs in a
fresh child process with a fresh CUDA context.

To compare local PyTorch inference against already-running vLLM and SGLang
servers under the same config seed and quality gates:

```bash
uv run python scripts/one_gpu_backend_compare.py config.toml \
  --engines pytorch,vllm,sglang \
  --vllm-url http://127.0.0.1:8000 \
  --sglang-url http://127.0.0.1:30000 \
  --group-size 2 --microbatch-size 1
```

The comparison harness preflights `/health` or `/v1/models` for external
servers, enforces token-native prompt usage for vLLM/SGLang by default, and
writes per-engine `condition_status.json` files. It also supports
`--engines trtllm --trtllm-url http://127.0.0.1:31000` for TensorRT-LLM.
It fails vLLM/SGLang/TensorRT-LLM runs by default if the adapter freshness
signal is missing or failing. Use `--allow-adapter-reload-failure` only for
throughput-only server experiments where stale adapter sampling is acceptable.

On one GPU, prefer running `--engines pytorch` first, then one external engine
at a time after starting only that server. vLLM, SGLang, and TensorRT-LLM each
load a second base-model copy, so running multiple servers alongside the trainer
measures memory contention more than inference quality. See
[Inference Engines](inference-engines.md#one-gpu-external-engine-comparison)
for memory-capped launch commands.

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
| ECHO strict shared forward | Yes, per local training microbatch | No; ECHO configs are rejected until the remote API can provide one shared actor pass |
| Setup | `pip install -e .` + CUDA GPU | `pip install -e ".[tinker]"` + service URL |

## Unsloth backend

Uses Unsloth `FastLanguageModel` for local model loading and LoRA/QLoRA patching
while keeping retrain's rollout, reward, token-advantage, ECHO, checkpoint, and
logging contracts. This is the intended one-GPU long-context backend to test
before rewriting retrain internals.

```toml
[backend]
backend = "unsloth"
devices = "gpu:0"

[backend.options]
max_seq_length = 32768       # set explicitly for long-context runs
load_in_4bit = true          # QLoRA default
device_map = "retrain"       # load directly on retrain's train device
train_microbatch_size = 1
cuda_empty_cache = true
sample_use_cache = true
gradient_checkpointing = true
liger_fused_linear_ce = true
qwen35_gated_delta_chunk_size = "auto"  # 4070 Ti-safe Qwen3.5 fallback
train_selective_suffix_logits = true    # only backprop weighted RL/ECHO tokens
train_unsloth_fused_ce = "auto"         # exact SFT fused/chunked CE when eligible
train_unsloth_fused_ce_target_gb = 0.0  # 0 = retrain auto target for the GPU
train_save_on_cpu = true                # exact saved-tensor offload fallback
```

For SFT rows with one constant supervised token weight, `train_unsloth_fused_ce`
routes the loss through Unsloth's dynamic fused/chunked cross-entropy helper.
Use `train_unsloth_fused_ce = "require"` in a smoke when you need proof that
this exact path was used. `auto` falls back and records
`local_train_unsloth_fused_ce_fallback_reason` if the installed Unsloth package
does not expose the helper, if the row uses mixed/fractional token weights, or
if a long row has a sparse supervised-token mask. Dense long supervised
completion/text rows are the intended fused-CE case; long prompts with only a
few supervised completion tokens are usually better served by selective
hidden/chunked logprobs.
RL and ECHO rows with arbitrary token weights still use retrain's weighted
log-probability path because reducing them to a plain CE mask would change the
objective.
When `train_unsloth_fused_ce_target_gb = 0`, retrain picks a conservative CE
chunk target on small GPUs before calling Unsloth: `0.25` GB on <=16 GB cards,
`0.5` GB on <=24 GB cards, and Unsloth's own dynamic target above that.
Do not combine `train_unsloth_fused_ce = "require"` with
`train_save_on_cpu = true`: the installed Unsloth helper uses `torch.func`, and
PyTorch does not support `torch.func` under saved-tensor hooks. In `auto` mode,
retrain falls back and reports `saved_tensor_hooks_incompatible`.

For exact full-context long-row training on small GPUs, selective saved-tensor
offload can keep small autograd-saved tensors on GPU while offloading larger
tensors to CPU. This preserves the same train context and gradient objective as
`train_save_on_cpu = true`; it only changes the saved-tensor placement policy.

```toml
[backend.options]
train_selective_suffix_logits = true
train_save_on_cpu = true
train_save_on_cpu_pin_memory = true
train_save_on_cpu_min_numel = 1048576  # 0 = offload every saved tensor
train_logprob_chunk_size = 256         # force hidden/chunked logprob fallback
```

For ECHO/tool traces with sparse weighted positions inside long rows, this
keeps the full actor forward/backward objective but avoids full-vocab logits for
unweighted suffix positions. Check
`local_train_selective_sparse_suffix_skips` and
`local_train_selective_hidden_logprob_batches` in runtime metrics to confirm the
guard routed the batch to the exact sparse hidden-state path.

You can reproduce the isolated ECHO logprob benchmark without downloading a
model:

```bash
PYTHONPATH=. python scripts/bench_echo_sparse_logprobs.py \
  --device cuda:0 \
  --seq-len 4096 \
  --early-target-pos 16 \
  --selected-tokens 16 \
  --vocab-size 8192 \
  --hidden-size 64 \
  --repeats 7
```

On the 12 GB RTX 4070 Ti smoke host, that shape measured the old suffix-style
LM-head/log-softmax region at `467.3 MB` median peak allocated and `0.00441 s`
median, versus `20.8 MB` and `0.00043 s` for selected hidden logits. The
actual `train_step_with_echo_masks` smoke in the same script reported
`local_train_selective_sparse_suffix_skips = 3`,
`local_train_selective_hidden_logprob_batches = 3`, and zero suffix-logprob
batches, which proves the ECHO train helper took the sparse exact path.

For the fastest one-GPU iteration on very long prompts, use a supervised context
window. This keeps rollout/inference on the full prompt but trains only on a
suffix window before the earliest weighted RL/ECHO token. It is not the same
gradient as full-context RL; use it when iteration speed matters more than
full-context credit assignment.

```toml
[backend.options]
train_selective_suffix_logits = true
train_supervised_context_tokens = 16384  # 0 disables; 8192/16384 are useful probes
train_save_on_cpu = false                # the cropped row should fit without CPU offload
```

Install Unsloth Core in the training environment:

```bash
uv pip install unsloth --torch-backend=auto
```

Then run a real backend smoke before any long Quaero run:

```bash
uv run python scripts/smoke_unsloth_backend.py \
  --model Qwen/Qwen3.5-2B \
  --device gpu:0 \
  --max-seq-length 32768 \
  --output /tmp/retrain_unsloth_smoke.json
```

The smoke imports the installed Unsloth package, verifies the
`FastLanguageModel` API shape, loads the model through retrain's Unsloth
backend, samples one token, runs one RL+ECHO update, and writes JSON evidence
including losses, runtime counters, and CUDA peak-memory counters. A green local
unit suite is not a substitute for this installed-package smoke.

Measured on the remote 11.6 GB RTX 4070 Ti with `Qwen/Qwen3.5-2B`,
`max_seq_length = 32768`, QLoRA rank 8, one 30k-token synthetic prompt, one
sampled token, and one RL+ECHO update: `train_save_on_cpu = true` passed with
about 7.48 GB peak CUDA reserved and 116 s train wall time. The same 30k train
step OOMed without saved-tensor CPU offload in the older weighted-logprob path
even with Unsloth gradient checkpointing, Liger fused CE, selective suffix
logits, and Tiled MLP.

Measured standalone SFT on the same host with the new fused-CE path:
dense 8k, 16k, and 24k supervised-token rows saved adapters with
`train_unsloth_fused_ce = "require"`, `train_unsloth_fused_ce_target_gb = 0`
(effective `0.25` GB), and no CPU offload. Peak reserved VRAM was `4678 MB`,
`7374 MB`, and `10216 MB`. Dense 30k still OOMed on this 11.6 GB card, failing
on a 236 MB allocation with about 219 MB free. Sparse 30k prompt rows are a
different workload: with only two supervised completion tokens, `auto` falls
back from fused CE to selective hidden logprobs; a 4096-token supervised context
window saved an adapter at `3416 MB` peak reserved VRAM.

For shorter rows, keep `train_save_on_cpu = false` unless a smoke proves it is
needed; 16k dense-label SFT fits without it and is much faster than CPU offload.

Measured exact selective-offload results on the same host and prompt:
`train_save_on_cpu_min_numel = 65536` ran in 115.56 s, `1048576` ran in
109.72 s, `1572864` ran in 109.63 s, and `2097152` ran in 109.62 s. These keep
the full 30001-token train row (`local_train_context_tokens_removed = 0`) but
do not reach a 25% speedup on the 4070 Ti; the best measured gain was about
5.7%, with peak reserved memory near the card limit. Use this as a small exact
throughput knob, not as a substitute for more VRAM or working native kernels.

Measured approximate supervised-window results on the same host and prompt:
`train_supervised_context_tokens = 4096` cropped the train row from 30001 to
4098 tokens and reduced train wall time to 5.68 s; `8192` cropped to 8194 tokens
and ran in 12.50 s; `16384` cropped to 16386 tokens and ran in 34.83 s. These
are throughput wins, not proof that the shorter training context is quality
equivalent.

`offload_embedding = true` is exposed for API completeness, but it failed this
shared-model PyTorch path because Unsloth moved embeddings to CPU while retrain
feeds CUDA token IDs during sampling. Keep it `false` until a dedicated
embedding-offload input path is implemented and smoked.

`max_seq_length = 0` falls back to `max(2048, max_tokens)`, which is only a
conservative smoke default. For Quaero long-horizon experiments, set the desired
training sequence length explicitly. The backend rejects contradictory precision
modes such as `load_in_4bit = true` and `load_in_16bit = true` in the same
config. Full fine-tuning is intentionally not exposed yet because retrain's
checkpoint and weight-sync contract is adapter-based.

`device_map = "retrain"` asks Unsloth/HuggingFace to place the model on
retrain's selected training device during load. This matters for 4-bit/8-bit
QLoRA because those quantized models generally cannot be moved safely with a
post-load `.to(device)` call. Advanced runs can pass an Unsloth/HuggingFace
device-map string such as `"sequential"` or `"auto"` instead.

ECHO works on this backend because it reuses retrain's lower-level
`train_step_with_echo_masks` path: RL and environment-token losses are computed
from the same actor forward/backward pass. This is different from handing control
to an opaque external GRPO trainer.

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

**Unsloth backend:** Checkpoint semantics match the local adapter path: retrain
saves and restores LoRA adapter directories under `adapter_path`. Optimizer state
is not restored.

**Tinker backend:** Checkpoint loading requires the Tinker SDK to support `load_state()` on the training client. If your SDK version doesn't have this method, resume will fail with an `AttributeError`. Check your Tinker SDK version supports checkpoint loading before relying on resume.
