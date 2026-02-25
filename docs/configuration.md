# Configuration

All configuration lives in a TOML file. By default, retrain loads `retrain.toml` from the current directory. Pass a path to use a different file:

```bash
retrain                    # loads ./retrain.toml
retrain path/to/config.toml
```

## Full annotated config

```toml
[backend]
backend = "local"          # local | tinker
devices = "gpu:0"          # e.g. gpu:0,gpu:1 for split mode
adapter_path = "/tmp/retrain_adapter"

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
base_url = ""              # Tinker service URL (tinker backend only)
lora_rank = 32

[algorithm]
advantage_mode = "maxrl"   # grpo | maxrl
transform_mode = "gtpo_sepa"  # none | gtpo | gtpo_hicra | gtpo_sepa

[training]
seed = -1                  # -1 = no seed
max_steps = 500
batch_size = 8
group_size = 16
max_tokens = 2048
temperature = 0.7
lr = 4e-5
weight_decay = 0.0
max_examples = 0           # 0 = use all examples
save_every = 20

[gtpo]
beta = 0.1                 # entropy weighting strength

[hicra]
alpha = 0.2                # planning token amplification

[sepa]
steps = 500                # ramp duration (linear schedule)
schedule = "linear"        # linear | auto
delay_steps = 50           # steps before SEPA ramp begins
correct_rate_gate = 0.1    # min correct rate to enable SEPA

[inference]
engine = "pytorch"         # pytorch | max | vllm | sglang | mlx | openai
url = ""                   # server URL for max/vllm/sglang/mlx/openai
attention_kernel = "default"  # default | flash | triton | tk | cutlass
dtype = "auto"             # auto | bf16 | fp8 | fp4
kv_cache_dtype = "auto"    # auto | bf16 | fp8 | int8
prefix_caching = true      # share prompt KV across group completions

[data]
source = "math"            # built-in data source (ignored when [environment] is set)

[environment]
provider = ""              # "" | "verifiers"
id = ""                    # verifiers env id (e.g. primeintellect/gsm8k)
args = {}                  # env kwargs as TOML object (or JSON string)
max_turns = -1             # cap turns for multi-turn envs; -1 = no override
auto_install = false       # install Prime Hub env automatically if missing

[reward]
type = "match"             # match | math | judge | custom
judge_model = ""           # LLM for judge reward (e.g. "gpt-4o-mini")
custom_module = ""         # Python module path for custom reward
custom_function = "score"  # function name in custom module

[backpressure]
enabled = false
warmup_steps = 10
ema_decay = 0.9
throttle_margin = 0.85
increase_margin = 0.5
min_batch_size = 1
max_batch_size = 64
peak_gflops = 0.0          # 0 = skip roofline hints
peak_bw_gb_s = 0.0

[squeeze]
# Only needed in campaign TOMLs or standalone squeeze TOMLs
min_variance_retention = 0.95  # threshold for rank recommendation
# adapter_path = ""           # required for standalone squeeze, auto-filled in campaigns
# source_rank = 0             # 0 = detect from [model].lora_rank
# output_path = ""            # set to save compressed adapter
# compress_to = 0             # 0 = use recommended rank

[resume]
from = ""                  # path to log dir with trainer_state.json

[logging]
log_dir = "logs/train"
wandb_project = ""         # set to enable wandb
wandb_run_name = ""
wandb_entity = ""
wandb_group = ""
wandb_tags = ""            # comma-separated
strategic_grams = ""       # custom planning token grams (JSON array or CSV)
```

## Field reference

### `[backend]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `backend` | str | `"local"` | Training backend: `local` (PyTorch/PEFT) or `tinker` (remote GPU) |
| `devices` | str | `"gpu:0"` | Comma-separated device list. Multi-GPU enables split mode (inference on first, training on last) |
| `adapter_path` | str | `"/tmp/retrain_adapter"` | Directory for LoRA adapter checkpoints |

### `[model]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | str | `"Qwen/Qwen3-4B-Instruct-2507"` | HuggingFace model ID or local path |
| `base_url` | str | `""` | Tinker service URL (tinker backend only) |
| `lora_rank` | int | `32` | LoRA rank. Alpha is set to `2 * rank` |

### `[algorithm]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `advantage_mode` | str | `"maxrl"` | Episode-level advantage: `grpo` or `maxrl` |
| `transform_mode` | str | `"gtpo_sepa"` | Token-level transform: `none`, `gtpo`, `gtpo_hicra`, or `gtpo_sepa` |

### `[training]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `seed` | int | `-1` | RNG seed. `-1` disables seeding |
| `max_steps` | int | `500` | Total training steps |
| `batch_size` | int | `8` | Number of prompts per step |
| `group_size` | int | `16` | Completions sampled per prompt |
| `max_tokens` | int | `2048` | Max new tokens per completion |
| `temperature` | float | `0.7` | Sampling temperature |
| `lr` | float | `4e-5` | Learning rate |
| `weight_decay` | float | `0.0` | AdamW weight decay |
| `max_examples` | int | `0` | Limit dataset size. `0` = use all |
| `save_every` | int | `20` | Checkpoint frequency (steps) |

### `[optimizer]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `beta1` | float | `0.9` | AdamW beta1 |
| `beta2` | float | `0.95` | AdamW beta2 |
| `eps` | float | `1e-8` | AdamW epsilon |

### `[lora]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `alpha` | int | `0` | LoRA alpha. `0` = auto (`rank * 2`) |
| `dropout` | float | `0.0` | LoRA dropout rate |

### `[data]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `source` | str | `"math"` | Dataset source. Selects which data loader to use |

### `[environment]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `provider` | str | `""` | Optional environment provider. Use `"verifiers"` to load verifiers environments |
| `id` | str | `""` | Environment ID (for example `primeintellect/gsm8k` or `primeintellect/wordle`) |
| `args` | table / str | `{}` | Environment kwargs. Prefer TOML table (`args = { split = "train" }`); JSON string is also accepted |
| `max_turns` | int | `-1` | Multi-turn safety cap. `-1` means use environment defaults |
| `auto_install` | bool | `false` | If true, auto-install missing Prime Hub environments before loading |

!!! note
    Not every Hub environment is trainable. Some are evaluation-only and do not expose a training dataset.
    retrain will now fail fast with an actionable error in that case.

### `[planning]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `detector` | str | `"regex"` | Planning token detector: `regex` or `semantic` |
| `model` | str | `"all-MiniLM-L6-v2"` | Sentence-transformer model for semantic detector |
| `threshold` | float | `0.02` | Planning detection threshold |

### `[gtpo]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `beta` | float | `0.1` | Entropy weighting strength. `0` disables GTPO weighting |

### `[hicra]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `alpha` | float | `0.2` | Planning token amplification. `0` disables HICRA |

### `[sepa]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `steps` | int | `500` | Ramp duration for linear schedule |
| `schedule` | str | `"linear"` | `linear` ramps lambda 0->1; `auto` adapts based on entropy variance decay |
| `delay_steps` | int | `50` | Steps before linear ramp begins |
| `correct_rate_gate` | float | `0.1` | Min batch correct rate to enable SEPA. Sticky once opened |

### `[inference]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `engine` | str | `"pytorch"` | Inference engine: `pytorch`, `max`, `vllm`, `sglang`, `mlx`, `openai` |
| `url` | str | `""` | Server URL for non-PyTorch engines |
| `attention_kernel` | str | `"default"` | Attention implementation: `default`, `flash`, `triton`, `tk`, `cutlass` |
| `dtype` | str | `"auto"` | Inference dtype: `auto`, `bf16`, `fp8`, `fp4` |
| `kv_cache_dtype` | str | `"auto"` | KV cache dtype: `auto`, `bf16`, `fp8`, `int8` |
| `prefix_caching` | bool | `true` | Share prompt KV cache across group completions |

### `[reward]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `type` | str | `"match"` | Reward type: `match`, `math`, `judge`, `custom` |
| `judge_model` | str | `""` | LLM model for judge reward (e.g. `"gpt-4o-mini"`) |
| `custom_module` | str | `""` | Python module path for custom reward |
| `custom_function` | str | `"score"` | Function name in custom module |

### `[backpressure]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable USL adaptive batch sizing |
| `warmup_steps` | int | `10` | Steps to collect before fitting USL model |
| `ema_decay` | float | `0.9` | EMA decay for throughput tracking |
| `throttle_margin` | float | `0.85` | Target fraction of optimal concurrency |
| `increase_margin` | float | `0.5` | Hysteresis gap for batch size increases |
| `min_batch_size` | int | `1` | Floor for adaptive batch size |
| `max_batch_size` | int | `64` | Ceiling for adaptive batch size |
| `peak_gflops` | float | `0.0` | Hardware peak GFLOPS for roofline hints. `0` = skip |
| `peak_bw_gb_s` | float | `0.0` | Hardware peak memory bandwidth (GB/s) |

### `[squeeze]`

Optional section for LoRA-Squeeze rank analysis. In campaign TOMLs, triggers auto-squeeze after the first run. See [LoRA-Squeeze](squeeze.md) for details.

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `min_variance_retention` | float | `0.95` | Minimum variance fraction to retain (0.95 = 95%) |
| `adapter_path` | str | `""` | Path to adapter directory. Required for standalone squeeze, auto-filled in campaigns |
| `source_rank` | int | `0` | Expected source rank. `0` = detect from `[model].lora_rank` |
| `target_ranks` | list[int] | `[]` | Ranks to evaluate. `[]` = auto power-of-2 |
| `output_path` | str | `""` | Directory to save compressed adapter. Empty = don't compress |
| `compress_to` | int | `0` | Target rank for compression. `0` = use recommended rank |
| `device` | str | `"cpu"` | Torch device for SVD computation |

### `[resume]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `from` | str | `""` | Path to log directory containing `trainer_state.json` |

### `[logging]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `log_dir` | str | `"logs/train"` | Directory for JSONL logs and checkpoints |
| `wandb_project` | str | `""` | wandb project name. Set to enable wandb |
| `wandb_run_name` | str | `""` | wandb run name. Defaults to `{advantage_mode}+{transform_mode}` |
| `wandb_entity` | str | `""` | wandb entity / team |
| `wandb_group` | str | `""` | wandb group for organizing related runs |
| `wandb_tags` | str | `""` | Comma-separated wandb tags |
| `strategic_grams` | str | `""` | Custom planning token grams as JSON array or CSV. Empty = use defaults |

## CLI overrides

Any TOML field can be overridden from the command line using `--flag value`:

```bash
retrain --seed 42 --lr 1e-4 --wandb-project my-run
retrain config.toml --batch-size 4 --advantage-mode grpo
```

| Flag | Config field |
|------|-------------|
| `--advantage-mode` | `advantage_mode` |
| `--transform-mode` | `transform_mode` |
| `--seed` | `seed` |
| `--lr` | `lr` |
| `--batch-size` | `batch_size` |
| `--group-size` | `group_size` |
| `--max-steps` | `max_steps` |
| `--lora-rank` | `lora_rank` |
| `--log-dir` | `log_dir` |
| `--wandb-project` | `wandb_project` |
| `--wandb-entity` | `wandb_entity` |
| `--wandb-group` | `wandb_group` |
| `--wandb-tags` | `wandb_tags` |
| `--wandb-run-name` | `wandb_run_name` |
| `--resume` | `resume_from` |

The `--resume` flag is special: it sets `resume_from` and is not a direct TOML key.

CLI flags use `--kebab-case`, which maps to `snake_case` config fields.

## Dataset

retrain trains on [hendrycks/MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (EleutherAI mirror). The dataset is hardcoded to 5 subjects:

- Intermediate algebra
- Precalculus
- Number theory
- Counting & probability
- Geometry

This yields ~7500 training examples. The dataset auto-downloads from HuggingFace on first run and is cached locally.

Use `max_examples` to cap the dataset size (e.g., for fast debugging):

```toml
[training]
max_examples = 100   # only load first 100 problems
```

!!! note
    To train on a different dataset, you can either:
    1. Use a verifiers environment via `[environment]` (no code changes), or
    2. Add a custom data source in `retrain/data.py` that returns `list[Example]`.

## Environment variables

Place a `.env` file in your working directory. retrain loads it automatically:

```
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```
