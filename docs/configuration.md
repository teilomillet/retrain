# Configuration

All configuration lives in a TOML file. By default, retrain loads `retrain.toml` from the current directory. Pass a path to use a different file:

```bash
retrain                    # loads ./retrain.toml
retrain path/to/config.toml
```

## Full annotated config

```toml
[backend]
backend = "local"          # local | unsloth | tinker | prime_rl
devices = "gpu:0"          # e.g. gpu:0,gpu:1 for split mode
adapter_path = "/tmp/retrain_adapter"

[backend.options]          # backend-specific options (example for prime_rl)
transport = "filesystem"   # filesystem | zmq
zmq_host = "localhost"     # used when transport = "zmq"
zmq_port = 5555            # used when transport = "zmq"
zmq_hwm = 10               # used when transport = "zmq"
strict_advantages = true   # must remain true; false is rejected to prevent silent aggregation
sync_wait_s = 30           # max wait for broadcast weights in checkpoint()
sync_poll_s = 0.2          # polling interval for broadcast weights

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
base_url = ""              # Tinker service URL (tinker backend only)
lora_rank = 32

[algorithm]
algorithm_mode = ""        # optional full algorithm plugin (overrides composable path)
advantage_mode = "maxrl"   # grpo | maxrl | my_module.my_advantage
transform_mode = "gtpo_sepa"  # none | gtpo | ... | my_module.my_transform
uncertainty_kind = "surprisal" # surprisal | predictive_variance | shannon_entropy | my_module.my_unc

[algorithm.params]
# used by algorithm_mode plugins
# alpha = 0.1

[algorithm.advantage_params]
# used by advantage_mode plugins
# scale = 2.0

[algorithm.transform_params]
# used by transform_mode plugins
# cap = 0.2
# uncertainty_kind = "surprisal"  # optional per-transform override

[plugins]
search_paths = ["plugins"] # module prefixes searched before normal dotted imports
strict = true              # fail fast on plugin load/shape errors

[training]
trainer = "retrain"        # retrain | sft | optimizer_replay | command | plugin
seed = -1                  # -1 = no seed
max_steps = 500
batch_size = 8
group_size = 16
max_tokens = 10240
temperature = 0.7
lr = 4e-5
weight_decay = 0.0
max_examples = 0           # 0 = use all examples
save_every = 20
sft_data_path = ""         # JSONL: messages, prompt/completion, or text rows
sft_data_sha256 = ""       # optional exact-data pin from retrain explain
sft_data_rows = 0          # optional row-count pin; 0 = unpinned
sft_audit_path = ""        # optional retrain.sft_audit.v1 JSON
sft_audit_sha256 = ""      # required exact audit-byte pin when path is set
sft_batch_size = 0         # 0 = trainer default
sft_max_tokens = 0         # 0 = trainer default
sft_lr = 0.0               # 0 = use lr
sft_loss_fn = "auto"       # auto | importance_sampling | cross_entropy

[optimizer_batch]
capture = false             # one-step local source-run capture
replay_path = ""            # manifest input for trainer = "optimizer_replay"
expected_logical_sha256 = "" # required external pin for replay
expected_manifest_sha256 = "" # pins manifest plus RNG-bearing payload
allow_config_differences = [] # v1: backend.options.gradient_checkpointing only

[echo]
enabled = false             # train same-rollout environment/tool tokens in multi-turn envs
weight = 0.05               # supervised token weight; kept small to avoid dominating RL
loss_fn = "cross_entropy"   # paper-faithful ECHO cross-entropy loss
max_tokens_per_step = 2048  # hard cap on supervised ECHO tokens per step
max_token_ratio = 0.5       # cap ECHO tokens to this fraction of RL completion tokens
entropy_floor = 0.01        # skip ECHO when completion surprisal falls below this floor
min_prompt_overlap = 0.5    # require stable prompt-prefix overlap before extracting suffix

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
engine = "pytorch"         # pytorch | max | vllm | sglang | trtllm | mlx | openai
url = ""                   # server URL for max/vllm/sglang/trtllm/mlx/openai
attention_kernel = "default"  # default | flash | triton | tk | cutlass
dtype = "auto"             # auto | bf16 | fp8 | fp4
kv_cache_dtype = "auto"    # auto | bf16 | fp8 | int8
prefix_caching = true      # exact-prefix KV reuse for local PyTorch rollouts

[data]
source = "math"            # built-in data source (ignored when [environment] is set)

[environment]
provider = ""              # "" | "verifiers" | "openenv"
id = ""                    # verifiers env id (e.g. primeintellect/gsm8k)
                           # or OpenEnv server URL (e.g. http://localhost:8765)
args = {}                  # env kwargs as TOML object (or JSON string)
max_turns = -1             # cap turns for multi-turn envs; -1 = no override
auto_install = false       # install Prime Hub env automatically if missing
rollout_env_workers = 1    # async env worker cap for multi-turn rollouts
rollout_buffer_size = 0    # 0 = buffer up to group_size active rollouts

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
checkpoint_artifacts = "auto"  # auto | off | wandb
strategic_grams = ""       # custom planning token grams (JSON array or CSV)
```

## Field reference

### `[backend]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `backend` | str | `"local"` | Training backend: `local` (PyTorch/PEFT), `unsloth` (Unsloth-patched local model loading), `tinker` (remote GPU), or `prime_rl` (external PRIME-RL trainer + inference) |
| `devices` | str | `"gpu:0"` | Comma-separated device list. Multi-GPU enables split mode (inference on first, training on last) |
| `adapter_path` | str | `"/tmp/retrain_adapter"` | Directory for LoRA adapter checkpoints |
| `options` | table | backend defaults | Backend-specific options table. For `local`: `train_microbatch_size`, `cuda_empty_cache`, `cuda_expandable_segments`, `strict_deterministic`, `sample_use_cache`, `sample_kv_quantization`, `sample_oscar_repo`, `sample_oscar_bits`, `sample_oscar_quant_mode`, `sample_oscar_group_size`, `sample_oscar_kv_rotation`, `sample_oscar_kv_norm`, `sample_oscar_residual_block_size`, `sample_oscar_attn_implementation`, `gradient_checkpointing`, `cudnn_causal_conv1d_shim`, `qwen35_gated_delta_kernel`, `train_selective_suffix_logits`, `train_compile_selective_ce`, `train_compile_selective_ce_min_tokens`, `train_save_on_cpu`, `train_save_on_cpu_pin_memory`, `train_save_on_cpu_min_numel`, `train_supervised_context_tokens`, `train_unsloth_fused_ce`, `train_unsloth_fused_ce_target_gb`, `train_unsloth_fused_ce_torch_compile`, `lora_detach_input`, `lora_fast_linear`, `lora_freeze_a`. For `unsloth`: `max_seq_length`, `load_in_4bit`, `load_in_8bit`, `load_in_16bit`, `fast_inference`, `gpu_memory_utilization`, `device_map`, `train_microbatch_size`, `qwen35_gated_delta_chunk_size`, `qwen35_gated_delta_kernel`, `train_selective_suffix_logits`, `train_compile_selective_ce`, `train_compile_selective_ce_min_tokens`, `train_save_on_cpu`, `train_save_on_cpu_pin_memory`, `train_save_on_cpu_min_numel`, `train_supervised_context_tokens`, `train_unsloth_fused_ce`, `train_unsloth_fused_ce_target_gb`, `train_unsloth_fused_ce_torch_compile`. For `prime_rl`: `transport`, `zmq_host`, `zmq_port`, `zmq_hwm`, `strict_advantages`, `sync_wait_s`, `sync_poll_s` |

!!! note
    Legacy `prime_rl_*` keys under `[backend]` were removed. Use `[backend.options]` keys instead.

Local backend memory knob:

```toml
[backend]
backend = "local"

[backend.options]
train_microbatch_size = 1  # 0 disables; positive values reduce train_step VRAM
cuda_empty_cache = true    # release cached CUDA blocks after local sample/train calls
strict_deterministic = false  # opt-in strict PyTorch/CUDA update guard
sample_use_cache = true    # faster PyTorch sampling with per-step allocator cleanup
gradient_checkpointing = true  # lower train VRAM at extra forward/backward compute
cudnn_causal_conv1d_shim = false  # opt-in Qwen3.5 GatedDelta fast path via cuDNN frontend
qwen35_gated_delta_kernel = "auto"  # auto | off | torch | flash_qla; explicit supported-GPU experiment
sample_kv_quantization = "off"  # off | oscar; OScaR is experimental sampling-only
sample_oscar_repo = ""  # path to upstream OScaR-KV-Quant checkout when enabled
sample_oscar_bits = 2  # 2 | 4
sample_oscar_quant_mode = "k-channel"
sample_oscar_group_size = 0  # 0 chooses the upstream default for the bit width
sample_oscar_kv_rotation = "hadamard"
sample_oscar_kv_norm = "1"
sample_oscar_residual_block_size = 128
sample_oscar_attn_implementation = "sdpa"
train_selective_suffix_logits = false  # optional: compute logits only for weighted suffix tokens
train_compile_selective_ce = "off"  # off | auto | require; compile selected CE on CUDA
train_compile_selective_ce_min_tokens = 128  # avoid compile overhead for tiny target sets
train_save_on_cpu = false  # optional: offload autograd saved tensors to CPU; much slower
train_save_on_cpu_pin_memory = true  # transfer mode for saved-tensor CPU offload
train_save_on_cpu_min_numel = 0  # 0 = offload all saved tensors; positive = exact selective offload
train_supervised_context_tokens = 0  # 0 = full train row; positive = approximate suffix-window train
train_unsloth_fused_ce = "off"  # off | auto | require; exact SFT-only fused/chunked CE when available
train_unsloth_fused_ce_target_gb = 0.0  # 0 = retrain auto target; explicit GB overrides
train_unsloth_fused_ce_torch_compile = true  # pass through to Unsloth's fused CE helper
lora_fast_linear = false  # opt-in fused PyTorch autograd path for dense LoRA linear modules
lora_detach_input = false  # research gate: stop LoRA branch activation gradients
lora_freeze_a = false  # LoRA-FA-style gate: freeze lora_A and train lora_B only
```

`train_microbatch_size` splits local PyTorch/PEFT training datums into smaller
forward/backward chunks while preserving the token-weighted loss. RL, SFT, and
hybrid RL+ECHO pad each chunk only to that chunk's longest row; a long rollout
therefore no longer forces every later microbatch to its global width. This
trades more forward/backward launches for lower peak VRAM and less padding.
Use the `train/backend/local/*` W&B fields (wall time, microbatch count, peak
memory, padding avoidance, and attention-work proxy) to sweep the largest
microbatch that fits instead of assuming that size 1 is fastest.
Standalone SFT emits the same backend aliases, plus backend-independent
`train/sft/sequence_length_*`, `train/sft/logical_padding_*`, and
`train/sft/supervised_token_fraction` fields. In JSONL, local runtime fields
remain under `backend/local_train_*` so the artifact records the raw backend
counter names.
`cuda_empty_cache` is allocator hygiene and does not change model outputs. It
defaults to `true` for the local backend because multi-step cache-on runs can
fragment the CUDA allocator even when one-step probes fit. `sample_use_cache =
true` keeps the PyTorch engine on the faster generation KV-cache path; disable
it only when rollout sampling OOMs and slower generation is acceptable.
`strict_deterministic = true` establishes a deterministic cuBLAS workspace
before CUDA/model construction and strictly enables PyTorch deterministic
algorithms plus cuDNN deterministic mode. It requires `training.seed >= 0`,
seeds model/adapter initialization before construction, and fails rather than
continuing if CUDA was initialized first or the controls cannot be verified. The
`local_strict_deterministic_*`, `local_cublas_workspace_config`,
`local_attention_implementation_resolved`, and
`local_sdpa_strict_torch_guard_enabled` runtime fields are copied into normal
metrics and SFT/optimizer-replay manifests. The resolved attention value is the
Transformers model configuration, not proof of the CUDA SDPA sub-kernel that
actually ran. These fields do not certify arbitrary
third-party Triton kernels. A causal campaign still requires two identical
captured updates with identical final adapter hashes on the target GPU. Strict
PyTorch/cuDNN/cuBLAS controls are process-global, so mixed strict/non-strict
campaign arms must run in separate processes; retrain rejects a non-strict
local backend constructed after a strict one in the same process.
`gradient_checkpointing` defaults to `true` for compatibility with previous
local backend behavior; set it to `false` during throughput sweeps when the full
train step fits in memory.
`lora_fast_linear` is an opt-in local LoRA runtime path. `lora_freeze_a` applies
a LoRA-FA-style recipe by freezing `lora_A` tensors before optimizer creation,
which changes the adapter optimization surface and must be validated with a
matched quality gate before becoming a training default.
`cudnn_causal_conv1d_shim` is an opt-in Qwen3.5 CUDA fast-path workaround for
hosts where NVIDIA's cuDNN frontend exposes `cudnn.ops.causal_conv1d` but the
standard `causal_conv1d` package is unavailable or cannot build. It does not
replace a real `causal_conv1d` install and records shim availability in backend
metrics.
`qwen35_gated_delta_kernel = "flash_qla"` is an explicit Qwen3.5 accelerator
experiment switch. The default `"auto"` works on any CUDA GPU supported by the
normal model stack by preserving the installed FLA or torch implementation.
Explicit `flash_qla` requires `flash-qla`, an upstream-supported device
(SM90/SM100 as of FlashQLA v0.1.1), and a matched equivalence/performance gate
before being used for claims; unsupported devices fail closed instead of
silently switching kernels.
`sample_kv_quantization = "oscar"` is an experimental OScaR KV-cache
quantization path for local PyTorch sampling only. It is default-off and
fail-closed: it requires `inference.engine = "pytorch"`, `sample_use_cache =
true`, local split mode, an upstream OScaR checkout, and a working `flash-attn`
install exposing `flash_attn_with_kvcache`. The training model remains the
standard HF/PEFT model. Do not use this option for quality or throughput claims
until a full OScaR retrain optimizer step passes on the target host; the first
A100 validation passed dense training but did not complete the OScaR retrain
step because of OScaR train-forward and `flash-attn` ABI/runtime blockers.
`train_selective_suffix_logits` is useful for RL rows where only completion or
ECHO tokens carry weight. Set `train_logprob_chunk_size` to a positive value
such as `256` when you want to force the safer hidden-state/chunked logprob
path instead of the `logits_to_keep` suffix shortcut. When the selected tokens
are sparse inside a long suffix, retrain now automatically skips the suffix
shortcut and uses the exact hidden-state path, so it applies the LM head and
log-softmax only to selected RL/ECHO target positions instead of materializing
`[long_suffix, vocab]` logits. The runtime metrics
`local_train_selective_sparse_suffix_skips`,
`local_train_selective_hidden_logprob_batches`, and
`local_train_selective_suffix_logprob_batches` show which branch ran.
Set `train_compile_selective_ce = "auto"` only after a CUDA smoke shows many
selected targets per row. It compiles the selected LM-head/cross-entropy region
and falls back to eager CE below `train_compile_selective_ce_min_tokens`.
`"require"` is intended for proof smokes and fails if the compiled path cannot
run.
`train_save_on_cpu` is the last-resort exact-autograd
memory path for very long rows on small GPUs; expect substantially slower
backward passes. `train_save_on_cpu_pin_memory` controls the transfer mode for
that exact path; on the measured 4070 Ti Qwen3.5 run, setting it to `false` was
slower. `train_save_on_cpu_min_numel` is also exact: positive values keep small
autograd-saved tensors on GPU and offload only larger tensors. This can recover
some throughput when there is spare VRAM, but it may OOM if set too high.
`train_supervised_context_tokens` is an approximate speed knob: when selective
suffix training is enabled, retrain crops each training row to this many context
tokens before the earliest weighted RL/ECHO token. Inference still sees the full
prompt, but the train gradient no longer represents the full context.
`train_unsloth_fused_ce` is an exact SFT loss path for rows whose supervised
token weights are all the same positive value. The Unsloth backend defaults it
to `auto`; local defaults it to `off` because most local environments do not
install Unsloth. Use `require` in a smoke when you need proof that Unsloth's
dynamic fused/chunked CE was used. If the row has fractional or mixed token
weights, retrain falls back because using a plain CE mask would change the
objective. For long rows with sparse supervised targets, `auto` also falls back
because selective hidden/chunked logprobs use less memory than CE over every
position. If the installed Unsloth helper raises a non-OOM runtime/compiler
failure, `auto` records that runtime reason and falls back to retrain's exact
loss path; CUDA OOM is still propagated as a capacity failure. The runtime metrics
`local_train_unsloth_fused_ce_attempts`,
`local_train_unsloth_fused_ce_batches`,
`local_train_unsloth_fused_ce_available`, and
`local_train_unsloth_fused_ce_fallback_reason` report which branch ran.
`attempts` counts actual fused-helper tries; `batches` counts only fused-CE
attempts that actually survived backward and contributed to the optimizer step.
When `train_unsloth_fused_ce_target_gb = 0`, retrain uses a conservative
small-GPU default before calling Unsloth: `0.25` GB on <=16 GB CUDA cards,
`0.5` GB on <=24 GB CUDA cards, and Unsloth's own target selection above that.
The metric `local_train_unsloth_fused_ce_effective_target_gb` records the value
used for the batch.
The fused CE path is not combined with `train_save_on_cpu` because Unsloth's
helper uses `torch.func`, and PyTorch saved-tensor hooks are incompatible with
that API. In `auto` mode retrain falls back with
`saved_tensor_hooks_incompatible`; in `require` mode it fails early.

### Migrate legacy backend config

Use `migrate-config` to rewrite old `prime_rl_*` keys safely:

```bash
# Preview only (default): shows unified diff, does not write
retrain migrate-config retrain.toml

# CI check mode: exit 1 if migration is needed
retrain migrate-config retrain.toml --check

# Apply in place
retrain migrate-config retrain.toml --write

# Apply in place + backup
retrain migrate-config retrain.toml --write --backup

# Non-destructive output file
retrain migrate-config retrain.toml --output retrain.migrated.toml

# Stream mode (stdin -> stdout)
cat retrain.toml | retrain migrate-config --stdin --stdout
```

`load_config` remains strict: legacy keys still hard-fail until migrated.

### `[model]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | str | `"Qwen/Qwen3-4B-Instruct-2507"` | HuggingFace model ID or local path |
| `base_url` | str | `""` | Tinker service URL (tinker backend only) |
| `lora_rank` | int | `32` | LoRA rank. Alpha is set to `2 * rank` |

### `[algorithm]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `algorithm_mode` | str | `""` | Optional full algorithm selector. Built-ins (`grpo_none`, `maxrl_gtpo`, etc.) or dotted plugin path (`my_module.my_algorithm`). When set, it overrides composable `advantage_mode + transform_mode`. |
| `advantage_mode` | str | `"maxrl"` | Episode-level advantage: built-ins (`grpo`, `maxrl`) or a dotted plugin path (`my_module.my_advantage`) |
| `transform_mode` | str | `"gtpo_sepa"` | Token-level transform: built-ins (`none`, `gtpo`, `gtpo_hicra`, `gtpo_sepa`, …) or dotted plugin path (`my_module.my_transform`) |
| `uncertainty_kind` | str | `"surprisal"` | Uncertainty signal used by GTPO-family transforms. `surprisal` uses sampled-token `-logprob`. `predictive_variance` uses `p*(1-p)` (works with logprobs only). `shannon_entropy` uses true per-position entropy `−Σ pᵢ log pᵢ` computed on GPU — requires `inference_engine = "pytorch"` and `backend = "local"`. Or a dotted plugin path (e.g. `my_module.my_uncertainty`). |
| `surprisal_mask_rho` | float | `0.0` | Top-ρ surprisal masking fraction (Yue et al.). `0` disables masking. TOML key `entropy_mask_rho` is accepted as a backward-compat alias. |

Nested plugin params tables under `[algorithm]`:

- `[algorithm.params]` for `algorithm_mode`
- `[algorithm.advantage_params]` for `advantage_mode`
- `[algorithm.transform_params]` for `transform_mode`

### `[plugins]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `search_paths` | list[str] | `["plugins"]` | Module prefixes searched before normal dotted import resolution |
| `strict` | bool | `true` | If true, plugin load/shape errors fail fast |

### `[training]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `seed` | int | `-1` | RNG seed. `-1` disables seeding |
| `trainer` | str | `"retrain"` | Training loop: `"retrain"` for RL/RLVR, `"sft"` for standalone supervised fine-tuning, `"optimizer_replay"` for a verified one-step local systems replay, `"command"` for an external command, or a dotted plugin path |
| `max_steps` | int | `500` | Total training steps |
| `batch_size` | int | `8` | Number of prompts per step |
| `group_size` | int | `16` | Completions sampled per prompt |
| `max_tokens` | int | `10240` | Max new tokens per completion |
| `temperature` | float | `0.7` | Sampling temperature |
| `lr` | float | `4e-5` | Learning rate |
| `weight_decay` | float | `0.0` | AdamW weight decay |
| `max_examples` | int | `0` | Limit dataset size. `0` = use all |
| `save_every` | int | `20` | Checkpoint frequency (steps) |
| `sft_warmup_steps` | int | `0` | Optional supervised warmup steps inside `trainer = "retrain"` before RL starts |
| `sft_data_path` | str | `""` | SFT JSONL path. Required for `trainer = "sft"`; optional for warmup |
| `sft_data_sha256` | str | `""` | Optional SHA256 pin for the exact SFT JSONL bytes. Mismatches fail before training |
| `sft_data_rows` | int | `0` | Optional loaded-row-count pin for SFT data. `0` leaves the count unpinned |
| `sft_audit_path` | str | `""` | Optional `retrain.sft_audit.v1` JSON. When set, its exact hash, pass status, dataset binding, corpus mode, and patch lineage are verified before SFT |
| `sft_audit_sha256` | str | `""` | Required SHA256 pin for the exact audit JSON bytes when `sft_audit_path` is set |
| `sft_batch_size` | int | `0` | SFT datums per optimizer step. `0` keeps the trainer default |
| `sft_max_tokens` | int | `0` | SFT row token cap. `0` uses `max_tokens` for standalone SFT and `max_tokens + 512` for warmup compatibility |
| `sft_lr` | float | `0.0` | SFT learning rate. `0` uses `lr` |
| `sft_loss_fn` | str | `"auto"` | SFT loss. `"auto"` resolves to `cross_entropy` for `trainer = "sft"` and preserves historical `importance_sampling` warmup behavior for `trainer = "retrain"` |

!!! note
    The quickstart template intentionally uses `max_tokens = 1024` for low-cost smoke tests.
    Treat `10240` as the default for standard training and campaign planning.
    See [Capacity Planning](capacity-planning.md) for sizing guidance.

### Standalone Unsloth SFT then RL

Use `trainer = "sft"` when the user already has supervised examples and does
not want to construct an RL dataset, reward, or environment for the first
phase. The SFT trainer accepts JSONL rows in any of these forms:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
{"prompt":"Question: ...\nAnswer:","completion":" ..."}
{"text":"plain next-token text"}
```

SFT masks are accounted after the causal one-token shift. If suffix truncation
lands inside an assistant target, retrain reserves the first retained token as
context and supervises only later tokens. A row that retains fewer than two
tokens, such as `sft_max_tokens = 1`, has no loss-bearing target and fails fast
instead of logging supervised work that cannot produce a gradient.

For `prompt`/`completion` rows, retrain tokenizes the fields independently and
then concatenates their token IDs. This prevents a tokenizer's BPE merge across
the text boundary from erasing or misclassifying the first completion token.

Example SFT config:

```toml
[backend]
backend = "unsloth"
adapter_path = "logs/my-model-adapter"

[backend.options]
load_in_4bit = true
train_microbatch_size = 1
cuda_empty_cache = true
gradient_checkpointing = true

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
lora_rank = 32

[training]
trainer = "sft"
max_steps = 100
batch_size = 4
max_tokens = 2048
lr = 2e-5
sft_data_path = "data/sft.jsonl"
# Optional once retrain explain reports trusted values:
# sft_data_sha256 = "..."
# sft_data_rows = 1000
# sft_audit_path = "data/sft.audit.json"
# sft_audit_sha256 = "..."
sft_batch_size = 4
sft_loss_fn = "auto"  # cross_entropy for standalone SFT
save_every = 20

[logging]
log_dir = "logs/my-model-sft"
```

The SFT run writes `trainer_state.json` in `log_dir`, including the latest
adapter checkpoint path and step. A later SFT run can resume remaining SFT
steps from that log directory, and a later RL run can continue training the
same LoRA by pointing `[resume].from` at the SFT log directory:

```toml
[backend]
backend = "unsloth"
adapter_path = "logs/my-model-adapter"

[model]
model = "Qwen/Qwen3-4B-Instruct-2507"
lora_rank = 32

[training]
trainer = "retrain"
sft_warmup_steps = 0
max_steps = 200
batch_size = 2
group_size = 4
lr = 1e-5

[resume]
from = "logs/my-model-sft"
```

For `trainer = "sft"`, `resume.from` has two modes. If it points at a log
directory containing `trainer_state.json`, retrain restores the saved checkpoint
and continues from the next SFT step. If it points directly at an adapter
directory, retrain loads that adapter as initialization and starts SFT at step
0. Local and Unsloth SFT resume is `adapter_only`: trainer counters and LoRA
weights are restored, but optimizer/scaler/RNG state is not.

Before restarting a job, run a local preflight:

```bash
retrain resume-check logs/my-model-sft --config training.toml
```

This checks the resume directory, checkpoint payload, target `max_steps`, resume
mode, and SFT data recoverability without loading the model.

The same run also writes:

- `log_dir/sft_manifest.json`
- `adapter_path/final/retrain_sft_manifest.json`
- `log_dir/resolved_config.json`
- `log_dir/sft_data_recoverability.json`
- `log_dir/sft_data.snapshot.jsonl` when the JSONL is small enough to copy

Those manifests record the base model, LoRA rank/alpha, dataset path, dataset
SHA256, row count, byte count, tracking warnings, final adapter checkpoint,
latest resource metrics, and a PEFT loading snippet. The
adapter directory is the Hugging Face/PEFT artifact: it can be loaded with
`PeftModel.from_pretrained(base_model, adapter_path)` or uploaded with
`huggingface-cli upload`.

`resolved_config.json` records the resolved `TrainConfig` including defaults,
with conventional secret-shaped option keys redacted. `sft_data_recoverability.json`
records whether retrain copied the exact JSONL into `sft_data.snapshot.jsonl`;
large files are not copied, but the recoverability file still records their
path, SHA256, row count, byte count, and reason they were left external.

`retrain explain config.toml` loads configured SFT data, prints the resolved
path, SHA256, row count, byte count, and git tracking status, and verifies
`sft_data_sha256` / `sft_data_rows` when they are set. If `sft_audit_path` is
configured, explain also verifies its externally pinned bytes and requires:

```json
{
  "schema": "retrain.sft_audit.v1",
  "status": "pass",
  "audited_dataset": {"sha256": "...", "rows": 1000},
  "corpus_mode": "replacement",
  "lineage": {}
}
```

`audited_dataset` must match the JSONL Retrain actually loaded. A
`corpus_mode` of `"patch"` additionally requires `lineage.base` and
`lineage.patch`, each with a valid `sha256` and positive `rows`. Retrain
does not infer patch-versus-replacement semantics from paths. This makes the
config a portable, fail-closed data contract instead of just a filename.

For RL configs, the same preview includes `+echo` in the condition when ECHO
is enabled, reports its weight and token caps, and shows the environment turn
limit, renderer, and configured task-source/task-ID guards. Use `--json` when
these resolved launch-contract fields must be checked automatically.

### Qwen3.5 Unsloth SFT Ergonomics

For a general low-cost Unsloth SFT starting point, use
`campaigns/qwen35-2b-unsloth-sft.toml` and change only `sft_data_path`,
`max_steps`, and `max_tokens` first. The intended user journey is:

1. Export/choose an SFT JSONL dataset.
2. Run the SFT smoke on the target GPU to measure peak VRAM and artifact shape.
3. Run `retrain campaigns/qwen35-2b-unsloth-sft.toml`.
4. Evaluate the saved adapter from `logs/qwen35-2b-unsloth-sft/sft_manifest.json`.
5. Continue RL with `[resume].from = "logs/qwen35-2b-unsloth-sft"`.

Use the SFT smoke before expensive training:

```bash
python scripts/smoke_unsloth_sft.py \
  --model Qwen/Qwen3.5-2B \
  --max-seq-length 32768 \
  --max-tokens 2048 \
  --batch-size 1 \
  --steps 1 \
  --train-unsloth-fused-ce require \
  --train-unsloth-fused-ce-target-gb 0 \
  --output /tmp/qwen35-sft-smoke.json
```

Use the USL sweep to find the batch/microbatch bottleneck before increasing
the dataset or run length:

```bash
python scripts/usl_unsloth_sft_sweep.py \
  --batch-sizes 1,2,3,4,6,8 \
  --microbatch-sizes 1,0 \
  --steps 2 \
  --synthetic-prompt-tokens 128 \
  --max-tokens 512 \
  --output-root logs/qwen35-sft-usl
```

For long examples, sweep the same controls with the long-row memory knobs rather
than extrapolating from short rows. Use `--synthetic-prompt-tokens` to probe
long-context activation pressure, and `--synthetic-completion-tokens` to probe
long supervised-label CE/logit pressure:

```bash
python scripts/usl_unsloth_sft_sweep.py \
  --batch-sizes 1,2,4 \
  --microbatch-sizes 1,0 \
  --steps 1 \
  --synthetic-prompt-tokens 30000 \
  --synthetic-completion-tokens 0 \
  --max-tokens 32768 \
  --train-supervised-context-tokens 4096 \
  --output-root logs/qwen35-sft-usl-30k-window4096
```

The sweep fits the Universal Scalability Law over `batch_size` for each
microbatch strategy and writes `summary.json` with `sigma`, `kappa`, `p_star`,
stage shares, peak VRAM, and improvement deltas against
`train_microbatch_size = 1`. Interpret high `sigma` plus flat throughput as
serial microbatch or launch overhead. If the full-batch condition
(`train_microbatch_size = 0`) fits in VRAM and improves datums/s, use it for
the real SFT run; keep `train_microbatch_size = 1` for conservative memory
smokes and very long rows.

On the measured 12 GB RTX 4070 Ti smoke for `Qwen/Qwen3.5-2B`, short
128-token synthetic SFT rows benefit from full-batch microbatching. At
`batch_size = 8`, throughput improved from `2.77` datums/s with serial
microbatching to `16.24` datums/s with full-batch microbatching (`+486.7%`),
while peak reserved VRAM rose from `3054 MB` to `3152 MB`. A follow-up
`batch_size = 8,12,16` sweep found the raw peak at `batch_size = 12`
(`17.41` datums/s) with a USL `p_star` around `13.4`; `batch_size = 16`
still fit but was slightly slower (`17.32` datums/s). Treat these as
hardware/workload evidence, not universal defaults.

On the same GPU, exact dense-label SFT now uses Unsloth fused CE when eligible.
With QLoRA rank 8, Tiled MLP, `train_unsloth_fused_ce = "require"`, and
auto CE target (`0.25` GB on this card), dense 8k, 16k, and 24k supervised-token
rows saved adapters at `4678 MB`, `7374 MB`, and `10216 MB` peak reserved VRAM.
The dense 30k row still OOMed, failing on a 236 MB allocation with only about
219 MB free. For sparse long-prompt rows, `auto` falls back from fused CE:
a 30,002-token row with only 2 supervised completion tokens and
`train_supervised_context_tokens = 4096` saved an adapter, trained on a cropped
4,098-token window, peaked at `3416 MB`, and reported
`local_train_unsloth_fused_ce_fallback_reason = "sparse_supervised_tokens"`.
A small long-row sweep over `batch_size = 1,2` showed the opposite microbatch
rule from short rows: serial microbatching stayed best. `batch_size = 2` with
`train_microbatch_size = 1` reached `0.181` datums/s at `4036 MB` peak reserved,
while `batch_size = 2` with full-batch microbatching fell to `0.116` datums/s
and `5304 MB`.

To compare the standalone SFT footprint against the existing full RL+ECHO smoke,
first run `scripts/smoke_unsloth_backend.py --output /tmp/rl-smoke.json`, then
rerun SFT with `--compare-to /tmp/rl-smoke.json`. Treat the result as a measured
claim, not a default guarantee: lower SFT peak VRAM is evidence that the SFT path
avoids rollout/sampling/ECHO memory, while equal peaks can happen on tiny or
model-dominated cases. In both cases, inspect `comparison`,
`backend/local_train_gpu_peak_memory_reserved_mb`, train-token counts, and sample
metrics before drawing a product conclusion.

Memory policy should stay evidence-first:

- Exact first: `train_microbatch_size = 1`, `train_selective_suffix_logits = true`,
  `gradient_checkpointing = true`, `liger_fused_linear_ce = true`. For SFT,
  add `train_unsloth_fused_ce = "require"` to the smoke so the run fails if it
  cannot use Unsloth's dynamic fused/chunked CE helper. Dense-label rows on
  small GPUs should leave `train_unsloth_fused_ce_target_gb = 0` so retrain can
  choose its conservative small-card target. If `require` fails with a runtime
  compiler reason but `off` passes, keep `auto` for training only after checking
  that the metrics report a fallback; do not count that run as a fused-CE proof.
- If long rows OOM and fused CE did not run, use exact saved-tensor CPU offload
  with `train_save_on_cpu = true`. Do not stack saved-tensor CPU offload with
  required fused CE; PyTorch `torch.func` rejects saved-tensor hooks.
- Use `train_supervised_context_tokens` only as an explicit approximation after
  exact full-context SFT has been measured and rejected for the target hardware.
  Record `backend/local_train_context_original_max_tokens`,
  `backend/local_train_context_cropped_max_tokens`, and
  `backend/local_train_context_tokens_removed` with the result so the run is not
  mistaken for exact full-context training.

### `[echo]`

ECHO is an auxiliary world-modeling objective for multi-turn environments. It
composes with any configured retrain algorithm: the algorithm
still determines sampled assistant-token advantages, while ECHO adds a
supervised environment-token mask from the same rollout. The native OpenEnv
provider consumes each newly model-visible response immediately after the
action, including terminal responses, and uses Prime Intellect's pinned
`renderers` bridge to extend the exact sampled prompt/action token prefix. One
combined transition row carries the GRPO mask on action tokens and the ECHO mask
only on that response's body. Next-turn sampling still uses the ordinary full
chat render, so a message-based evaluation harness sees the same protocol.

For verifiers environments without the native one-shot response hook, retrain
keeps the compatibility path: exact prompt-aligned masks for `tool`,
`environment`, or `observation` roles, followed by conservative prompt-suffix
extraction. A failed native token bridge never drops sampled action tokens; it
produces an action-only GRPO row and increments `echo/bridge_failures`.

For `loss_fn = "cross_entropy"`, retrain follows the ECHO normalization shape:
selected environment-token negative log-probabilities are divided by the
full observation length for that rollout row, then scaled by `weight`. With the
current verifiers bridge, the full-observation length is reconstructed from the
same prompt-aligned observation-body mask used for ECHO targets. This is exact
when the mask marks the whole observation body, which is the intended bridge
contract. If a harness wants to train only an `env_only` subspan inside a larger
observation body, the bridge must expose a separate full-body denominator before
that variant can be exact.

The local PyTorch backend gathers RL and ECHO losses from the same actor
forward pass for each training microbatch. The Unsloth backend uses the same
lower-level retrain train step after loading the actor through Unsloth, so it
also declares strict shared-forward support. Tinker's public remote loss API
currently exposes separate RL and ECHO `forward_backward` calls rather than one
shared actor pass, so `retrain explain` rejects ECHO on `backend = "tinker"`
until that backend can declare strict shared-forward support.

For agent/tool workloads, rollout sampling should use KV cache and prefix cache
where available, but ECHO training should not replay a no-grad rollout KV cache
as the train prefix. That would remove the cached prefix from the autograd graph
and change the full-context training objective. The exact memory optimization
for ECHO is `train_selective_suffix_logits = true`: retrain still runs one
actor forward/backward over the train row, but it computes token logprobs only
for weighted RL/ECHO positions. For long sparse tool traces, the sparse-suffix
guard routes to the hidden-state path and avoids full-vocab logits for the
intervening unweighted positions.

The explicit-mask path is the intended ECHO path. Check
`echo/observation_mask_datums` in metrics to confirm a run is using it; a value
of zero means retrain used the compatibility suffix fallback or no eligible
observation tokens were present. If a harness renders warning text and terminal
output inside the same observation-role message, retrain cannot separate them at
token level; render warnings as a different role/message if they should be
excluded from ECHO.

For native OpenEnv, also require `echo/observation_responses > 0`,
`echo/bridged_transition_datums > 0`, and `echo/bridge_failures = 0` in a
mechanism gate. `echo/renderer_parity_failures` means the official renderer did
not reproduce the tokenizer's sampling prompt and retrain refused to guess.
`echo/terminal_candidate_tokens` counts terminal targets before the step cap;
only `echo/terminal_kept_tokens > 0` proves terminal feedback reached the loss.

ECHO candidates are collected before reward-uniform groups are skipped for RL.
This keeps the paper's intended signal: all-failed rollouts can still train the
environment-prediction objective even when they provide no policy-gradient
contrast. In that case retrain sends a same-rollout hybrid batch with zero RL
advantages and nonzero ECHO masks. If both RL advantages and ECHO masks are
zero after caps and guards, the step is skipped as uninformative.

ECHO requires a token-preserving backend with strict shared-forward support.
Today that means `backend = "local"` or `backend = "unsloth"`. `prime_rl` is
rejected because it cannot carry prompt-side token masks without silently
collapsing them to scalar advantages; `tinker` is rejected because it cannot
currently compute the RL and ECHO losses from one shared actor pass.

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable auxiliary ECHO training on multi-turn environment/tool observation tokens |
| `weight` | float | `0.05` | Positive supervised weight for ECHO tokens. Must be in `[0, 1]` |
| `loss_fn` | str | `"cross_entropy"` | ECHO loss. Only `"cross_entropy"` is supported for paper-faithful ECHO |
| `max_tokens_per_step` | int | `2048` | Absolute cap on positive ECHO tokens per step |
| `max_token_ratio` | float | `0.5` | Cap positive ECHO tokens to this fraction of RL completion tokens, so ECHO cannot dominate the step |
| `entropy_floor` | float | `0.01` | Skip the ECHO step when sampled-completion mean surprisal is below this floor; this is a mode-collapse guard |
| `min_prompt_overlap` | float | `0.5` | Compatibility-only minimum overlap for environments that do not expose native one-shot observation messages |

### `[optimizer]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `beta1` | float | `0.9` | AdamW beta1 |
| `beta2` | float | `0.95` | AdamW beta2 |
| `eps` | float | `1e-8` | AdamW epsilon |

### `[optimizer_batch]`

This section supports exact-input one-GPU systems ablations. See
[Exact-Input Optimizer-Batch Replay](optimizer-batch-replay.md) for the full
workflow and evidence gate.

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `capture` | bool | `false` | Capture the exact post-transform logical batch in a normal one-step local RL run |
| `replay_path` | str | `""` | Captured `.manifest.json` input; required for `trainer = "optimizer_replay"` |
| `expected_logical_sha256` | str | `""` | Required 64-character external pin for the captured logical batch |
| `expected_manifest_sha256` | str | `""` | Required 64-character external pin for the exact manifest and its RNG-bearing payload |
| `allow_config_differences` | list[str] | `[]` | Explicit optimizer-contract differences. V1 permits only `backend.options.gradient_checkpointing` |

Capture/replay v1 requires `backend = "local"`, exactly one device,
`inference.engine = "pytorch"`, `max_steps = 1`, `save_every = 0`,
`sft_warmup_steps = 0`, backpressure disabled, and a local adapter identified
by `resume.from`. Capture additionally requires the initialization resume state
to contain `step = -1`, proving that step zero executes once. Replay skips
data, environments, rollouts, and sampling; it is not a quality-evaluation
mode.

Exact manifest, logical/effective rows, RNG, initial adapter, config contract,
and loss admit runtime/memory comparisons with repeated timing. They do not
guarantee bitwise-identical CUDA updates. That stronger claim additionally
requires identical final adapter hashes across source and repeated
same-condition replays. Evaluate the source-run adapter, not replay outputs,
for model quality.

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
| `provider` | str | `""` | Optional environment provider: `"verifiers"` for verifiers environments, `"openenv"` for a running OpenEnv gym server |
| `id` | str | `""` | Environment ID (for example `primeintellect/gsm8k`) or, for `openenv`, the server base URL (for example `http://localhost:8765`) |
| `args` | table / str | `{}` | Environment kwargs. Prefer TOML table (`args = { split = "train" }`); JSON string is also accepted |
| `max_turns` | int | `-1` | Multi-turn safety cap. For OpenEnv, a positive value is also sent to every preload and live `reset`; `-1` uses the server default |
| `auto_install` | bool | `false` | If true, auto-install missing Prime Hub environments before loading |
| `rollout_env_workers` | int | `1` | Max concurrent async environment jobs for multi-turn setup/render/step stages |
| `rollout_buffer_size` | int | `0` | Max in-flight rollout env jobs; `0` uses the current group size |

!!! note
    Not every Hub environment is trainable. Some are evaluation-only and do not expose a training dataset.
    retrain will now fail fast with an actionable error in that case.

#### `provider = "openenv"`

Trains against an already-running [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
gym server (reset/step over its WebSocket wire protocol) — no verifiers
install required. Requires the `websockets` package (`pip install retrain[openenv]`).

- `id` is the server base URL; retrain does not launch or manage the server.
- Seeds are the dataset: example *i* uses `seed + i`, and its prompt is the
  rendered `reset(seed)` observation. `[data] max_examples` caps the count
  (default 100 when unset).
- Rewards are the per-episode sum of step rewards from the server.
- `args` accepts `renderer` — dotted path to a callable turning observations
  into chat messages (renderers written for verifiers' OpenEnv integration work
  unchanged); the default renderer JSON-dumps observations and states the
  action schema on reset. `message_timeout_s` sets the WebSocket response
  timeout (default 300).
- For contamination-sensitive training, `expected_task_source` and
  `expected_task_ids` enable fail-closed reset provenance checks. A guarded
  reset observation must expose consistent `task_id` and `task_source` fields
  at top level, in `info`, or in `metadata`. Preload seeds must cover exactly
  the configured task-ID set. Each live rollout must then return the same task
  identity that its seed returned during preload.
- A positive `max_turns` is forwarded to both preload and live resets so the
  rendered prompt, server episode, and retrain rollout cap share one horizon.
  Non-positive values are not sent, preserving servers that use their own
  default.

Example guarded task set:

```toml
[environment]
provider = "openenv"
id = "http://localhost:8765"
max_turns = 16
args = { expected_task_source = "factory", expected_task_ids = ["task-3000", "task-3001"] }
```

- Completions must be a single JSON action object; malformed completions
  receive a corrective observation instead of crashing the rollout.
- MCP tool environments are not supported on this provider; use the
  verifiers integration for those.

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
| `schedule` | str | `"linear"` | `linear` ramps lambda 0->1; `auto` adapts based on uncertainty variance decay |
| `delay_steps` | int | `50` | Steps before linear ramp begins |
| `correct_rate_gate` | float | `0.1` | Min batch correct rate to enable SEPA. Sticky once opened |

### `[inference]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `engine` | str | `"pytorch"` | Inference engine: `pytorch`, `max`, `vllm`, `sglang`, `trtllm`, `mlx`, `openai` |
| `url` | str | `""` | Server URL for non-PyTorch engines |
| `attention_kernel` | str | `"default"` | Attention implementation: `default`, `flash`, `triton`, `tk`, `cutlass` |
| `dtype` | str | `"auto"` | Inference dtype: `auto`, `bf16`, `fp8`, `fp4` |
| `kv_cache_dtype` | str | `"auto"` | KV cache dtype: `auto`, `bf16`, `fp8`, `int8` |
| `prefix_caching` | bool | `true` | Reuse exact-prefix KV cache entries during local PyTorch rollout sampling; cache is cleared at checkpoint/weight-sync boundaries |

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
| `from` | str | `""` | Path to log directory containing `trainer_state.json`; local/Unsloth restore is adapter-only, not optimizer-exact |

### `[logging]`

| TOML key | Type | Default | Description |
|----------|------|---------|-------------|
| `log_dir` | str | `"logs/train"` | Directory for JSONL logs and checkpoints |
| `wandb_project` | str | `""` | wandb project name. Set to enable wandb |
| `wandb_run_name` | str | `""` | wandb run name. Defaults to `{advantage_mode}+{transform_mode}` |
| `wandb_entity` | str | `""` | wandb entity / team |
| `wandb_group` | str | `""` | wandb group for organizing related runs |
| `wandb_tags` | str | `""` | Comma-separated wandb tags |
| `checkpoint_artifacts` | str | `"auto"` | `auto` uploads checkpoints to W&B Artifacts when `wandb_project` is set; `wandb` requires live artifact upload and `save_every > 0`; `off` leaves checkpoints local only |
| `strategic_grams` | str | `""` | Custom planning token grams as JSON array or CSV. Empty = use defaults |

## CLI overrides

Any TOML field can be overridden from the command line using `--flag value`:

```bash
retrain --seed 42 --lr 1e-4 --wandb-project my-run
retrain config.toml --batch-size 4 --advantage-mode grpo
retrain config.toml --transform-param cap=0.1 --advantage-param scale=2.0
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
| `--checkpoint-artifacts` | `checkpoint_artifacts` |
| `--resume` | `resume_from` |

The `--resume` flag is special: it sets `resume_from` and is not a direct TOML key.
Use `retrain resume-check <log_dir> --config <config.toml>` before a restart to
verify that the checkpoint directory and target config line up.

Repeatable plugin-param flags:

- `--algorithm-param key=value`
- `--advantage-param key=value`
- `--transform-param key=value`

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
    2. Add a custom data source under `retrain/data/` that returns `list[Example]`.

## Environment variables

Place a `.env` file in your working directory. retrain loads it automatically:

```
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```
