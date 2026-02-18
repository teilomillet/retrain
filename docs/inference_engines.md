# 14. Pluggable Inference Engines

Retrain separates inference (sampling completions) from training (gradient updates). The `--inference-engine` flag selects how sampling is performed, while PyTorch/PEFT always handles LoRA training.

## Architecture

```
LocalBackend (Mojo, trait interface)
  └── LocalTrainHelper (Python)
        ├── InferenceEngine (ABC)
        │     ├── PyTorchEngine     ← same model, shared VRAM
        │     ├── MAXLocalEngine    ← in-process MAX pipeline
        │     ├── MAXServeEngine    ← HTTP to max serve
        │     └── OpenAIEngine      ← HTTP to vLLM / SGLang / any server
        └── PyTorch/PEFT training (unchanged)
```

All engines implement the same interface: `generate()` returns token IDs + per-token logprobs. The training side never knows which engine produced the samples.

## Engine Options

| Engine | Flag | What it does |
|--------|------|-------------|
| PyTorch | `--inference-engine pytorch` | Shares the training model for inference. 1x VRAM. Default. |
| MAX (auto) | `--inference-engine max` | In-process if no URL, HTTP to `max serve` if `--inference-url` set |
| vLLM | `--inference-engine vllm` | HTTP client to a vLLM server |
| SGLang | `--inference-engine sglang` | HTTP client to a SGLang server |
| OpenAI | `--inference-engine openai` | HTTP client to any OpenAI-compatible endpoint |

## Why PyTorch Is Best on 1 GPU

With LoRA training, only the adapter weights change -- the base model is frozen. The PyTorch engine exploits this: the same model object serves both training (forward/backward through base + LoRA) and inference (generate with base + LoRA). There is no weight duplication.

Every other engine loads a separate copy of the base model -- either in a different framework (MAX) or a different process (vLLM, SGLang). On 1 GPU, that means 2x base model VRAM for no benefit. The base weights are identical and frozen; the duplication exists only because different frameworks can't share GPU memory.

```
PyTorch (1 GPU):     [base model + LoRA]  ← shared, 1x VRAM
MAX (1 GPU):         [base model + LoRA]  +  [base model (MAX)]  ← 2x base VRAM
vLLM (1 GPU):        [base model + LoRA]  +  [base model (vLLM)] ← 2x base VRAM
```

## Multi-GPU: When to Use MAX / vLLM

With multiple GPUs, inference and training run on separate devices. The base model duplication is expected and desirable -- each device has its own copy.

```
8x H100 example:
  GPUs 0-6:  max serve (tensor parallel inference, continuous batching)
  GPU 7:     PyTorch/PEFT training
```

Here MAX or vLLM provide real benefits: tensor parallelism across inference GPUs, continuous batching for high throughput, and optimized kernels.

## Quick Start

```bash
# 1 GPU — PyTorch (default, no extra setup)
./retrain-tinker --backend local --devices gpu:0

# 1 GPU — explicit PyTorch
./retrain-tinker --backend local --devices gpu:0 --inference-engine pytorch

# 8 GPUs — MAX serve on GPUs 0-6, training on GPU 7
max serve --model Qwen/Qwen3-4B-Instruct-2507  # manages its own GPUs
./retrain-tinker --backend local --devices gpu:7 \
    --inference-engine max --inference-url http://localhost:8000

# 8 GPUs — vLLM server
vllm serve Qwen/Qwen3-4B-Instruct-2507 --tensor-parallel-size 7
./retrain-tinker --backend local --devices gpu:7 \
    --inference-engine vllm --inference-url http://localhost:8000
```

## LoRA Weight Sync

After each training step, updated LoRA weights must reach the inference engine.

| Engine | Sync mechanism | Latency |
|--------|---------------|---------|
| PyTorch (1 GPU) | Same model object, no sync needed | 0 |
| PyTorch (split mode) | In-memory `sync_from_state_dict()` via snapshot | ~1ms |
| MAX / vLLM / SGLang | `save_pretrained()` to disk, then `reload_weights()` | ~1-2s |

The `_weights_dirty` flag avoids redundant saves. In split mode, a weight snapshot is taken after each optimizer step for safe cross-thread access.

## Device Allocation

| Config | Training | Inference |
|--------|----------|-----------|
| `--inference-engine pytorch --devices gpu:0` | GPU 0 | GPU 0 (same model) |
| `--inference-engine pytorch --devices gpu:0,gpu:1` | GPU 1 | GPU 0 (split mode) |
| `--inference-engine max --devices gpu:7` | GPU 7 | MAX-managed |
| `--inference-engine vllm --devices gpu:7` | GPU 7 | Server-managed |

With external engines, `--devices` controls only the training GPU. The engine manages its own GPU allocation independently.

## InferenceEngine ABC

All engines implement three methods:

```python
class InferenceEngine(ABC):
    def generate(self, prompt_ids_list, num_samples, max_tokens,
                 temperature, top_p) -> List[List[SampleResult]]:
        """Return [num_prompts][num_samples] of (token_ids, logprobs)."""

    def reload_weights(self, adapter_path: str) -> None:
        """Reload LoRA adapter from disk."""

    def shutdown(self) -> None:
        """Release resources."""
```

`SampleResult` is a dataclass with `token_ids: List[int]` and `logprobs: List[float]`.

PyTorchEngine adds `sync_from_state_dict(lora_dict)` for fast in-memory weight sync in split mode.

## TOML Configuration

```toml
[inference]
engine = "max"
url = "http://localhost:8000"
```

CLI args `--inference-engine` and `--inference-url` override TOML values.

## Future: MAX Custom Ops for PyTorch

The ideal single-GPU path: replace PyTorch's inference kernels with MAX's optimized Mojo kernels (Flash Attention 3, fused MLP) via `torch.compile` + `CustomOpLibrary`. This gives MAX-speed inference inside the same PyTorch model -- no second model, no VRAM duplication, autograd still works for training.

```
Future PyTorchEngine (1 GPU):
  Training forward:   PyTorch autograd (unchanged)
  Inference forward:  MAX custom ops via torch.compile (fast)
  VRAM:               1x base model + LoRA (unchanged)
```

This is a drop-in optimization to `PyTorchEngine.generate()` -- the engine ABC and the rest of the pipeline stay untouched.

See: https://docs.modular.com/max/develop/custom-kernels-pytorch/

## Files

| File | Role |
|------|------|
| `retrain/inference_engine/__init__.py` | Exports + `create_engine()` factory |
| `retrain/inference_engine/base.py` | `InferenceEngine` ABC + `SampleResult` dataclass |
| `retrain/inference_engine/pytorch_engine.py` | Local PyTorch engine (extracted from train helper) |
| `retrain/inference_engine/max_engine.py` | MAX engine (auto-detect in-process vs serve) |
| `retrain/inference_engine/openai_engine.py` | HTTP client for vLLM / SGLang / any server |
| `retrain/local_train_helper.py` | Orchestrates engine + training, weight sync |
| `src/local_backend.mojo` | Mojo trait implementation, calls into Python helper |
| `src/config.mojo` | `inference_engine` + `inference_url` config fields |
