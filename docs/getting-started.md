# Getting Started

Goal: run your first retrain experiment quickly, then scale it safely.

## Prerequisites

=== "Local backend (default)"

    **Hardware:** A CUDA GPU with at least **16 GB VRAM** (e.g., RTX 4090, A100, H100). The default model (Qwen3-4B) loads in bf16 (~8 GB) plus LoRA adapter and KV cache overhead.

    !!! tip "Low VRAM?"
        Start with `retrain init --template quickstart` (smoke-test profile with `max_tokens=1024`), then move to the standard training defaults (`max_tokens=10240`) once your run is stable.

    **Software:** Python 3.11+, CUDA 12.x, internet access for the first run (downloads model weights from HuggingFace and the MATH dataset).

=== "Tinker backend"

    **Hardware:** No local GPU required. All model loading, inference, and training run on the Tinker remote GPU service.

    **Software:** Python 3.11+, internet access, a Tinker service URL.

    **Access:** You need access to a running Tinker service. Set the URL in your config or ask your team for the endpoint.

## Install

=== "Local backend"

    ```bash
    pip install -e .
    ```

    This installs retrain and its dependencies (PyTorch, Transformers, PEFT, wandb, verifiers).

=== "Tinker backend"

    ```bash
    pip install -e ".[tinker]"
    ```

    This installs retrain plus the Tinker SDK. No local PyTorch/CUDA required for training -- the Tinker service handles GPU work.

## Minimal config

Create a `retrain.toml` in your working directory:

=== "Local backend"

    ```toml
    [model]
    model = "Qwen/Qwen3-4B-Instruct-2507"

    [algorithm]
    advantage_mode = "maxrl"
    transform_mode = "gtpo_sepa"

    [training]
    max_steps = 100
    batch_size = 4
    group_size = 8
    ```

=== "Tinker backend"

    ```toml
    [backend]
    backend = "tinker"

    [model]
    model = "Qwen/Qwen3-4B-Instruct-2507"
    base_url = "https://your-tinker-endpoint"

    [algorithm]
    advantage_mode = "maxrl"
    transform_mode = "gtpo_sepa"

    [training]
    max_steps = 100
    batch_size = 4
    group_size = 8
    ```

    !!! note
        The `[inference]` section (engine, url, attention_kernel, etc.) is ignored when using Tinker. The Tinker service handles all inference internally.

Everything else uses sensible defaults. See [Configuration](configuration.md) for the full reference.

## Run

```bash
retrain
```

Or point to a specific config:

```bash
retrain path/to/config.toml
```

Override any field from the CLI:

```bash
retrain --seed 42 --lr 1e-4 --wandb-project my-run
```

## Expected output

=== "Local backend"

    ```
    Loaded .env
    Loading tokenizer for Qwen/Qwen3-4B-Instruct-2507 ...
    Loading vocabulary table...
    Vocabulary table: 151936 entries
    Loading dataset...
    Loaded 7500 examples
    Pre-encoded 7500 prompts
    Loading train model: Qwen/Qwen3-4B-Instruct-2507 on cuda:0...
    trainable params: 41,943,040 || all params: 3,951,079,424 || trainable%: 1.0616
    LocalTrainHelper ready (engine=pytorch, split_mode=False).
      group: 3/8 correct | answer=120
      group: 0/8 correct | answer=\frac{1}{4}
    Step 0 [maxrl+gtpo_sepa] | loss=-0.0012 | reward=0.188 | correct=18.8% | datums=16 | bs=4 | gs=8 | sepa_l=0.0000 | time=45.2s
    ...
    ```

=== "Tinker backend"

    ```
    Loaded .env
    Loading tokenizer for Qwen/Qwen3-4B-Instruct-2507 ...
    Loading vocabulary table...
    Vocabulary table: 151936 entries
    Loading dataset...
    Loaded 7500 examples
    Pre-encoded 7500 prompts
    Connecting to Tinker...
    Creating LoRA training client (model=Qwen/Qwen3-4B-Instruct-2507, rank=32)...
    Training client ready.
      group: 3/8 correct | answer=120
      group: 0/8 correct | answer=\frac{1}{4}
    Step 0 [maxrl+gtpo_sepa] | loss=-0.0012 | reward=0.188 | correct=18.8% | datums=16 | bs=4 | gs=8 | sepa_l=0.0000 | time=12.3s
    ...
    ```

Each step shows the condition label, loss, mean reward, correct rate, number of datums submitted for training, batch/group sizes, SEPA lambda, and wall time. For asynchronous backends, loss is annotated as `(... placeholder)` to avoid misreading it as synchronous optimizer loss.

## What happened

1. **Loaded model** -- Qwen3-4B with LoRA adapter (rank 32, ~42M trainable params)
2. **Downloaded MATH dataset** -- 7500 problems from 5 subjects (intermediate algebra, precalculus, number theory, counting & probability, geometry) via the [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) mirror. The dataset auto-downloads on first run and is cached locally by HuggingFace.
3. **For each step:**
    - Selected a batch of prompts
    - Sampled `group_size` completions per prompt
    - Scored each completion by extracting `\boxed{...}` and comparing to the reference answer
    - Computed MaxRL advantages with GTPO+SEPA transforms
    - Trained the LoRA adapter with importance-sampling loss
4. **Saved checkpoints** every `save_every` steps and a final adapter

## Typical training times

Times vary by hardware and config. Rough estimates for **Qwen3-4B** are shown below. These are orientation values for retrain experiments, not hard guarantees.

=== "Local backend"

    | GPU | Per step | 100 steps | 500 steps |
    |-----|----------|-----------|-----------|
    | RTX 4090 (24 GB) | ~45s | ~75 min | ~6.3 h |
    | A100 (80 GB) | ~25s | ~42 min | ~3.5 h |
    | H100 (80 GB) | ~15s | ~25 min | ~2.1 h |

    A full campaign (5 conditions x 8 seeds x 500 steps = 40 runs) takes **~250 h on a single 4090** or **~84 h on a single H100**.

=== "Tinker backend"

    Training times depend on the remote hardware provisioned by the Tinker service. Per-step latency includes network round-trips for sampling and training RPCs. Expect similar per-step times to the equivalent local GPU, plus ~1-3s of network overhead per step.

    When running parallel campaigns, retrain automatically throttles concurrent Tinker API calls (default: 4 at a time) to prevent backend overload. See [Campaigns](campaigns.md#tinker-backend-throttling) for details.

Use `--max-steps 100` for a quick validation run.

!!! tip "Quick sanity check"
    Run 20 steps first to verify everything works:
    ```bash
    retrain --max-steps 20 --batch-size 2 --group-size 4
    ```

## Next steps

- [Configuration](configuration.md) -- tune hyperparameters, enable wandb, switch inference engines
- [Capacity Planning](capacity-planning.md) -- estimate wall time, worker count, and memory before long runs
- [Advantage Functions](advantages.md) -- understand GRPO vs MaxRL and the GTPO/HICRA/SEPA transforms
- [Campaigns](campaigns.md) -- sweep conditions across seeds with auto-squeeze rank analysis
- [Backends](backends.md) -- scale to multiple GPUs or use Tinker remote training
