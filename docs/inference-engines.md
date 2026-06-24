# Inference Engines

!!! note "Local backend only"
    This page applies to the **local backend** only. When using the **Tinker backend**, the Tinker service handles all inference internally -- the `[inference]` config section is ignored. See [Backends](backends.md) for Tinker setup.

retrain separates inference (sampling completions) from training (gradient updates). The inference engine controls how completions are generated, while PyTorch/PEFT always handles LoRA training.

## Architecture

```text
retrain
  └── LocalTrainHelper
        ├── InferenceEngine (ABC)
        │     ├── PyTorchEngine     ← same model, shared VRAM
        │     ├── MAXLocalEngine    ← in-process MAX pipeline
        │     ├── MAXServeEngine    ← HTTP to max serve
        │     └── OpenAIEngine      ← HTTP to vLLM / SGLang / TensorRT-LLM / MLX-LM / any server
        └── PyTorch/PEFT training (unchanged)
```

All engines implement the same interface: `generate()` returns token IDs + per-token logprobs. The training side never knows which engine produced the samples.

## Engine options

| Engine | TOML value | What it does |
|--------|------------|-------------|
| PyTorch | `pytorch` | Shares the training model for inference. 1x VRAM. Default |
| MAX (auto) | `max` | In-process if no URL, HTTP to `max serve` if `url` set |
| vLLM | `vllm` | HTTP client to a vLLM server |
| SGLang | `sglang` | HTTP client to a SGLang server |
| TensorRT-LLM | `trtllm` | HTTP client to a `trtllm-serve` server |
| MLX-LM | `mlx` | HTTP client to a local `mlx_lm.server` endpoint |
| OpenAI | `openai` | HTTP client to any OpenAI-compatible endpoint |

For `vllm` and `sglang`, retrain sends prompt token IDs directly in
`/v1/completions` requests. This avoids local token decode followed by server
re-tokenization. If a server rejects token prompts with a 400/422 response,
retrain falls back to the text prompt path and increments
`engine_token_prompt_fallbacks`.

For `trtllm`, retrain sends the active LoRA adapter path in each completion
request using TensorRT-LLM's `lora_request` field. The current `trtllm-serve`
documentation shows text prompts for `/v1/completions`, so retrain does not
assume token-ID prompt support until a server version proves it.

For `mlx`, retrain sends the active LoRA adapter path in each completion request using the MLX-LM `adapters` field.

## Why PyTorch is the lowest-VRAM 1-GPU path

With LoRA training, only the adapter weights change -- the base model is frozen. The PyTorch engine exploits this: the same model object serves both training and inference. There is no weight duplication.

Every other engine loads a separate copy of the base model -- either in a different framework (MAX) or a different process (vLLM, SGLang, TensorRT-LLM, MLX-LM). On 1 GPU, that means 2x base model VRAM. Whether a server engine is faster enough to justify that cost is workload-dependent and should be decided by the benchmark sweep.

```text
PyTorch (1 GPU):     [base model + LoRA]  ← shared, 1x VRAM
MAX (1 GPU):         [base model + LoRA]  +  [base model (MAX)]  ← 2x base VRAM
vLLM (1 GPU):        [base model + LoRA]  +  [base model (vLLM)] ← 2x base VRAM
TensorRT-LLM (1 GPU): [base model + LoRA] +  [base model (trtllm)] ← 2x base VRAM
```

## Multi-GPU: when to use MAX / vLLM / TensorRT-LLM

With multiple GPUs, inference and training run on separate devices. Base model duplication is expected and desirable -- each device has its own copy.

```text
8x H100 example:
  GPUs 0-6:  max serve (tensor parallel inference, continuous batching)
  GPU 7:     PyTorch/PEFT training
```

Here MAX, vLLM, SGLang, or TensorRT-LLM provide real benefits: tensor parallelism across inference GPUs, continuous batching for high throughput, and optimized kernels.

## Quick start

```bash
# 1 GPU -- PyTorch (default, no extra setup)
retrain --devices gpu:0

# 1 GPU -- explicit PyTorch
retrain --devices gpu:0 --inference-engine pytorch

# 8 GPUs -- MAX serve on GPUs 0-6, training on GPU 7
max serve --model Qwen/Qwen3-4B-Instruct-2507  # manages its own GPUs
retrain --devices gpu:7 \
    --inference-engine max --inference-url http://localhost:8000

# 8 GPUs -- vLLM server
vllm serve Qwen/Qwen3-4B-Instruct-2507 --tensor-parallel-size 7
retrain --devices gpu:7 \
    --inference-engine vllm --inference-url http://localhost:8000

# 8 GPUs -- TensorRT-LLM server
trtllm-serve Qwen/Qwen3-4B-Instruct-2507 --tp_size 7
retrain --devices gpu:7 \
    --inference-engine trtllm --inference-url http://localhost:31000

# Apple Silicon -- MLX-LM local server
pip install -e ".[mlx]"
python -m mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-4bit
retrain --devices cpu \
    --inference-engine mlx --inference-url http://localhost:8080
```

## One-GPU external-engine comparison

On one GPU, external inference servers duplicate the frozen base model: the
trainer has one copy and the server has another. Treat vLLM/SGLang on one GPU
as an empirical A/B test against the shared-model PyTorch path, not as an
automatic upgrade.

NVIDIA Dynamo is an orchestration layer above engines such as vLLM, SGLang,
and TensorRT-LLM. It is relevant when retrain grows to multiple inference
replicas, multi-GPU or multi-node serving, KV-aware routing, or disaggregated
prefill/decode. It is not a first-line replacement for the one-GPU A/B test:
on a single model running on a single GPU, benchmark the underlying engine
first.

Run only one external server at a time and start with conservative memory caps:

```bash
# vLLM, one-GPU A/B server. Runtime LoRA updating is required because
# retrain overwrites and reloads _live_adapter after each optimizer step.
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
vllm serve Qwen/Qwen3.5-2B \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3.5-2B \
  --dtype half --max-model-len 32768 \
  --gpu-memory-utilization 0.35 \
  --enable-lora --max-lora-rank 8 --max-loras 1 --max-cpu-loras 2 \
  --max-num-seqs 4 --max-num-batched-tokens 4096

# SGLang, one-GPU A/B server. The target modules are required when the
# server starts without an initial --lora-paths adapter.
CUDA_HOME=/tmp/retrain-cuda-home-cu13 \
PATH=/path/to/sglang-env/bin:$CUDA_HOME/bin:$PATH \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-2B \
  --served-model-name Qwen/Qwen3.5-2B \
  --host 127.0.0.1 --port 30000 \
  --dtype half --context-length 32768 \
  --mem-fraction-static 0.35 \
  --attention-backend triton --sampling-backend pytorch --grammar-backend none \
  --disable-cuda-graph \
  --enable-lora --max-lora-rank 8 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --max-loras-per-batch 1

# TensorRT-LLM, one-GPU A/B server. Use a port that can coexist with vLLM,
# and make the LoRA config match the PEFT adapter target modules used by the
# retrain config. TensorRT-LLM receives the live adapter via per-request
# lora_request, so retrain does not call a reload endpoint.
cat > /tmp/retrain-trtllm-lora.yaml <<'YAML'
lora_config:
  max_lora_rank: 8
  max_loras: 1
  max_cpu_loras: 2
  lora_target_modules:
    - attn_q
    - attn_k
    - attn_v
  trtllm_modules_to_hf_modules:
    attn_q: q_proj
    attn_k: k_proj
    attn_v: v_proj
YAML

trtllm-serve Qwen/Qwen3.5-2B \
  --host 127.0.0.1 --port 31000 \
  --config /tmp/retrain-trtllm-lora.yaml
```

For pip-installed CUDA toolkits, SGLang's JIT linker may require a conventional
`$CUDA_HOME/lib64/libcudart.so` linker name. If the toolkit only provides a
versioned runtime such as `libcudart.so.13`, create a temp CUDA home with
`bin`, `include`, `nvvm`, and `lib64/libcudart.so` symlinks before launching
SGLang.

Then run the comparison harness against the live server:

```bash
uv run python scripts/one_gpu_backend_compare.py config.toml \
  --engines vllm \
  --vllm-url http://127.0.0.1:8000 \
  --group-size 2 --microbatch-size 1 \
  --cuda-empty-cache true --gradient-checkpointing true \
  --sample-use-cache true --prefix-caching true
```

Repeat with `--engines sglang` after stopping vLLM and starting SGLang. Keep the
PyTorch baseline in the same output root or note the baseline path in the run
manifest. The harness preflights `/health` or `/v1/models` and fails
vLLM/SGLang runs when token-ID prompts fall back to decoded text, unless
`--allow-token-prompt-fallback` is set. It also fails multi-step
vLLM/SGLang/TensorRT-LLM runs when no adapter freshness signal succeeds, unless
`--allow-adapter-reload-failure` is set.

For TensorRT-LLM, run the same harness after starting `trtllm-serve`:

```bash
uv run python scripts/one_gpu_backend_compare.py config.toml \
  --engines trtllm \
  --trtllm-url http://127.0.0.1:31000 \
  --group-size 2 --microbatch-size 1 \
  --cuda-empty-cache true --gradient-checkpointing true \
  --sample-use-cache true --prefix-caching true
```

## LoRA weight sync

After each training step, updated LoRA weights must reach the inference engine:

| Engine | Sync mechanism | Latency |
|--------|---------------|---------|
| PyTorch (1 GPU) | Same model object, no sync needed | 0 |
| PyTorch (split mode) | In-memory `sync_from_state_dict()` via snapshot | ~1ms |
| vLLM / SGLang | `save_pretrained()` to disk, then server reload endpoint | ~1-2s |
| TensorRT-LLM / MLX-LM | `save_pretrained()` to disk, then per-request adapter path | server-dependent |
| MAX | `save_pretrained()` to disk, then `reload_weights()` | ~1-2s |

The `_weights_dirty` flag avoids redundant saves. In split mode, a weight snapshot is taken after each optimizer step for safe cross-thread access.
For vLLM/SGLang, retrain calls the reload endpoint every time `_live_adapter`
is rewritten, even though the path string is stable; same path does not imply
same adapter contents. vLLM uses `/v1/load_lora_adapter` with `load_inplace`;
SGLang uses `/load_lora_adapter` and unloads the previous `default` LoRA name
before reloading it.
For TensorRT-LLM, retrain records the new `_live_adapter` path after each
optimizer step and sends it as `lora_request` on each completion request.

## Device allocation

| Config | Training | Inference |
|--------|----------|-----------|
| `engine = "pytorch"`, `devices = "gpu:0"` | GPU 0 | GPU 0 (same model) |
| `engine = "pytorch"`, `devices = "gpu:0,gpu:1"` | GPU 1 | GPU 0 (split mode) |
| `engine = "max"`, `devices = "gpu:7"` | GPU 7 | MAX-managed |
| `engine = "vllm"`, `devices = "gpu:7"` | GPU 7 | Server-managed |
| `engine = "trtllm"`, `devices = "gpu:7"` | GPU 7 | Server-managed |
| `engine = "mlx"`, `devices = "cpu"` | CPU | MLX-LM server-managed |

With external engines, `devices` controls only the training GPU. The engine manages its own GPU allocation independently.

## TOML configuration

```toml
[inference]
engine = "pytorch"         # pytorch | max | vllm | sglang | trtllm | mlx | openai
url = ""                   # server URL for non-PyTorch engines
attention_kernel = "default"
dtype = "auto"
kv_cache_dtype = "auto"
prefix_caching = true
```

For the local PyTorch engine, `prefix_caching = true` enables bounded
exact-prefix KV reuse within a rollout sampling phase. This targets multi-turn
environments where the next prompt extends a previous prompt/completion with new
environment observations. The cache is cleared when `checkpoint()` prepares a
new sampling phase so stale KV from old adapter weights is not reused.

## InferenceEngine ABC

All engines implement three methods:

| Method | Purpose |
| --- | --- |
| `generate(prompt_ids_list, num_samples, max_tokens, temperature, top_p)` | Return `[num_prompts][num_samples]` of token IDs and logprobs |
| `reload_weights(adapter_path)` | Reload a LoRA adapter from disk |
| `shutdown()` | Release resources |

`SampleResult` is a dataclass with `token_ids: list[int]` and `logprobs: list[float]`.

`PyTorchEngine` adds `sync_from_state_dict(lora_dict)` for fast in-memory weight sync in split mode.

## Files

| File | Role |
|------|------|
| `retrain/inference_engine/__init__.py` | Exports + `create_engine()` factory |
| `retrain/inference_engine/base.py` | `InferenceEngine` ABC + `SampleResult` dataclass |
| `retrain/inference_engine/pytorch_engine.py` | Local PyTorch engine |
| `retrain/inference_engine/max_engine.py` | MAX engine (in-process vs serve) |
| `retrain/inference_engine/openai_engine.py` | HTTP client for vLLM / SGLang / TensorRT-LLM / MLX-LM / any server |
| `retrain/local_train_helper.py` | Orchestrates engine + training, weight sync |
