# Inference Engines

!!! note "Local backend only"
    This page applies to the **local backend** only. When using the **Tinker backend**, the Tinker service handles all inference internally -- the `[inference]` config section is ignored. See [Backends](backends.md) for Tinker setup.

retrain separates inference (sampling completions) from training (gradient updates). The inference engine controls how completions are generated, while PyTorch/PEFT always handles LoRA training.

## Architecture

```
retrain
  └── LocalTrainHelper
        ├── InferenceEngine (ABC)
        │     ├── PyTorchEngine     ← same model, shared VRAM
        │     ├── MAXLocalEngine    ← in-process MAX pipeline
        │     ├── MAXServeEngine    ← HTTP to max serve
        │     └── OpenAIEngine      ← HTTP to vLLM / SGLang / any server
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
| OpenAI | `openai` | HTTP client to any OpenAI-compatible endpoint |

## Why PyTorch is best on 1 GPU

With LoRA training, only the adapter weights change -- the base model is frozen. The PyTorch engine exploits this: the same model object serves both training and inference. There is no weight duplication.

Every other engine loads a separate copy of the base model -- either in a different framework (MAX) or a different process (vLLM, SGLang). On 1 GPU, that means 2x base model VRAM for no benefit.

```
PyTorch (1 GPU):     [base model + LoRA]  ← shared, 1x VRAM
MAX (1 GPU):         [base model + LoRA]  +  [base model (MAX)]  ← 2x base VRAM
vLLM (1 GPU):        [base model + LoRA]  +  [base model (vLLM)] ← 2x base VRAM
```

## Multi-GPU: when to use MAX / vLLM

With multiple GPUs, inference and training run on separate devices. Base model duplication is expected and desirable -- each device has its own copy.

```
8x H100 example:
  GPUs 0-6:  max serve (tensor parallel inference, continuous batching)
  GPU 7:     PyTorch/PEFT training
```

Here MAX or vLLM provide real benefits: tensor parallelism across inference GPUs, continuous batching for high throughput, and optimized kernels.

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
```

## LoRA weight sync

After each training step, updated LoRA weights must reach the inference engine:

| Engine | Sync mechanism | Latency |
|--------|---------------|---------|
| PyTorch (1 GPU) | Same model object, no sync needed | 0 |
| PyTorch (split mode) | In-memory `sync_from_state_dict()` via snapshot | ~1ms |
| MAX / vLLM / SGLang | `save_pretrained()` to disk, then `reload_weights()` | ~1-2s |

The `_weights_dirty` flag avoids redundant saves. In split mode, a weight snapshot is taken after each optimizer step for safe cross-thread access.

## Device allocation

| Config | Training | Inference |
|--------|----------|-----------|
| `engine = "pytorch"`, `devices = "gpu:0"` | GPU 0 | GPU 0 (same model) |
| `engine = "pytorch"`, `devices = "gpu:0,gpu:1"` | GPU 1 | GPU 0 (split mode) |
| `engine = "max"`, `devices = "gpu:7"` | GPU 7 | MAX-managed |
| `engine = "vllm"`, `devices = "gpu:7"` | GPU 7 | Server-managed |

With external engines, `devices` controls only the training GPU. The engine manages its own GPU allocation independently.

## TOML configuration

```toml
[inference]
engine = "pytorch"         # pytorch | max | vllm | sglang | openai
url = ""                   # server URL for non-PyTorch engines
attention_kernel = "default"
dtype = "auto"
kv_cache_dtype = "auto"
prefix_caching = true
```

## InferenceEngine ABC

All engines implement three methods:

```python
class InferenceEngine(ABC):
    def generate(self, prompt_ids_list, num_samples, max_tokens,
                 temperature, top_p) -> list[list[SampleResult]]:
        """Return [num_prompts][num_samples] of (token_ids, logprobs)."""

    def reload_weights(self, adapter_path: str) -> None:
        """Reload LoRA adapter from disk."""

    def shutdown(self) -> None:
        """Release resources."""
```

`SampleResult` is a dataclass with `token_ids: list[int]` and `logprobs: list[float]`.

`PyTorchEngine` adds `sync_from_state_dict(lora_dict)` for fast in-memory weight sync in split mode.

## Files

| File | Role |
|------|------|
| `retrain/inference_engine/__init__.py` | Exports + `create_engine()` factory |
| `retrain/inference_engine/base.py` | `InferenceEngine` ABC + `SampleResult` dataclass |
| `retrain/inference_engine/pytorch_engine.py` | Local PyTorch engine |
| `retrain/inference_engine/max_engine.py` | MAX engine (in-process vs serve) |
| `retrain/inference_engine/openai_engine.py` | HTTP client for vLLM / SGLang / any server |
| `retrain/local_train_helper.py` | Orchestrates engine + training, weight sync |
