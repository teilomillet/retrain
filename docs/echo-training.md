# ECHO Training Optimization

ECHO trains the model to predict environment, tool, or observation tokens from
the same transition row used for policy learning. For native OpenEnv, retrain
uses Prime Intellect `renderers==0.1.7` to preserve the exact sampled
prompt/action prefix, then appends and masks only the current model-visible
environment response. It captures the response immediately after `step`, so
terminal output is not lost. Next-turn behavior sampling remains on the normal
full chat render to match message-based evaluation. Most tokens in these rows
are context, while only a small subset carry RL or ECHO loss.

The optimization in retrain is deliberately exact for the default ECHO path. It
does not replace training prefixes with detached KV cache. Instead, it keeps the
same actor forward/backward graph and avoids full-vocabulary logits for
unweighted positions.

## Why not train from cached KV?

KV cache is correct and useful during rollout sampling. The model can reuse an
exact prefix cache between tool calls because generation only needs the next
token distribution, not gradients through the sampled prefix.

Training is different. If a train step reuses a no-grad rollout KV cache as the
prefix, the cached prefix is no longer part of the autograd graph. That changes
the full-context objective and removes gradient flow through the prompt/tool
history. retrain therefore keeps train-time KV reuse out of the exact ECHO
path. Approximate suffix-window training is available through
`train_supervised_context_tokens`, but it is explicitly documented as an
approximation.

## What the exact path does

Enable the selective path for ECHO:

```toml
[backend.options]
train_selective_suffix_logits = true
train_logprob_chunk_size = 256  # optional: force hidden/chunked logprobs
train_compile_selective_ce = "off"  # off | auto | require
train_compile_selective_ce_min_tokens = 128
```

With `train_selective_suffix_logits = true`, retrain builds a target mask from
tokens that carry RL advantage or positive ECHO weight. The transformer still
runs over the train row, so the hidden states are conditioned on the full
context. The memory win comes after that: retrain computes the LM head and
class-index cross entropy only for selected target positions when the selected
tokens are sparse inside a long suffix. PyTorch defines this cross entropy as
the same objective as `LogSoftmax` followed by `NLLLoss`; using it here avoids
materializing a selected-token log-probability matrix just to gather one class
per row.

This avoids tensors shaped like:

```text
[long_suffix_tokens, vocab_size]
```

and replaces them with:

```text
[selected_target_tokens, vocab_size]
```

For tool traces, `selected_target_tokens` can be orders of magnitude smaller
than `long_suffix_tokens`.

## Implementation change

The selected-hidden-token path used to compute:

```text
LM head -> full selected-token log-softmax -> gather target class
```

It now computes:

```text
LM head -> unreduced class-index cross entropy -> negate to log-probability
```

This is a narrower implementation of the same loss. The LM head still produces
the logits needed for each selected hidden state, but retrain no longer asks
PyTorch to materialize the selected-token log-probability matrix before reading
one target class per row. The helper keeps returning a dense shifted-token
tensor afterward because the rest of the policy/ECHO loss path is already
vectorized and this scatter was measured cheaper than adding sparse branching.

For CUDA runs with many selected targets, `train_compile_selective_ce = "auto"`
can additionally route the selected LM-head/cross-entropy region through
`torch.compile`. This is still exact for the selected class-index CE loss, but
it is opt-in because the first compiled shape pays compilation cost and varied
selected-token counts can create extra compiler work. Use `"require"` only in
proof smokes; it fails closed if the compiled path cannot run. The compiled
path only runs for a plain `torch.nn.Linear` LM head, because it calls the
weight/bias linear operation directly. Wrapped or custom LM heads fall back to
the eager `lm_head(...)` path.

## Correctness boundary

This path is exact for the selected RL/ECHO token log-probabilities:

- same input tokens;
- same attention mask;
- same actor forward/backward step;
- same hidden state at each selected target position;
- same LM head and class-index cross entropy for those selected targets.

It only skips LM-head/loss work for positions whose weight is zero. Those
positions do not contribute to the RL or ECHO loss in the selective path.

The path is not a replacement for activation checkpointing or saved-tensor CPU
offload. For very long full-context rows, those controls may still be needed:

```toml
[backend.options]
train_save_on_cpu = true
train_save_on_cpu_pin_memory = true
train_save_on_cpu_min_numel = 1048576
```

## Metrics to check

Use runtime metrics to prove which branch ran:

| Metric | Meaning |
| --- | --- |
| `local_train_selective_suffix_logits` | The selective target-mask mode is enabled |
| `local_train_selective_sparse_suffix_skips` | A sparse long suffix skipped the suffix-logits shortcut |
| `local_train_selective_hidden_logprob_batches` | The exact selected-hidden-token path ran |
| `local_train_selective_compiled_ce_batches` | The selected-hidden path used `torch.compile` for CE |
| `local_train_selective_suffix_logprob_batches` | The suffix `logits_to_keep` shortcut ran instead |
| `local_train_selective_fallback_logprob_batches` | The selective path could not run and fell back |
| `local_train_compile_selective_ce_fallback_reason` | Why compiled CE did not run in `auto` mode |

For sparse ECHO/tool traces, the expected proof is:

```text
local_train_selective_sparse_suffix_skips > 0
local_train_selective_hidden_logprob_batches > 0
local_train_selective_fallback_logprob_batches = 0
```

For a compiled selected-CE proof, also require:

```text
local_train_compile_selective_ce_mode = require
local_train_selective_compiled_ce_batches > 0
local_train_compile_selective_ce_fallback_reason = ""
```

## Reproducible benchmark

Run the synthetic ECHO benchmark without downloading a model:

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

On the 12 GB RTX 4070 Ti smoke host, that shape measured:

| Region | Median peak allocated | Median time |
| --- | ---: | ---: |
| suffix-style LM head + log-softmax | `467.3 MB` | `0.00517 s` |
| previous selected log-softmax + gather | `20.8 MB` | `0.00041 s` |
| current selected cross-entropy | `20.5 MB` | `0.00025 s` |

That is a 20.7x speedup versus the old suffix-style region and a 1.64x speedup
over the previous selected-token log-softmax path for the measured
LM-head/loss region. A sweep over `selected_tokens = 1, 4, 16, 64, 128, 256`
still showed at least a 1.22x speedup for selected cross-entropy over selected
log-softmax. The same script also runs
`LocalTrainHelper.train_step_with_echo_masks`; in the measured run it reported
`local_train_selective_sparse_suffix_skips = 3`,
`local_train_selective_hidden_logprob_batches = 3`, and zero suffix-logprob
batches.

Interpret the benchmark fields this way:

- `suffix_path` is the old long-suffix LM-head/log-softmax shape that the sparse
  guard avoids.
- `selected_logsoftmax_path` is the previous selected-token implementation.
- `selected_cross_entropy_path` is the current implementation.
- `selected_ce_speedup_vs_logsoftmax` is the relevant before/after comparison
  for this change.
- `selected_compiled_ce_speedup_vs_eager_ce` is present only when
  `--compile-selective-ce` is enabled.
- `echo_train_smoke.runtime_metrics` proves that a real train helper call used
  the sparse hidden path instead of only benchmarking isolated tensors.

For the compiled selected-CE path, use a larger selected-token shape so the
compile region is large enough to matter:

```bash
PYTHONPATH=. python scripts/bench_echo_sparse_logprobs.py \
  --device cuda:0 \
  --seq-len 4096 \
  --early-target-pos 16 \
  --selected-tokens 256 \
  --vocab-size 32768 \
  --hidden-size 256 \
  --repeats 11 \
  --compile-selective-ce auto \
  --compile-selective-ce-min-tokens 128
```

On the same 12 GB RTX 4070 Ti host, that shape measured selected eager CE at
`162.4 MB` and `0.00130 s`, versus compiled selected CE at `50.3 MB` and
`0.00063 s`, a 2.05x speedup for the measured CE region. A `require` proof
with `--compile-selective-ce-min-tokens 1` reported
`local_train_selective_compiled_ce_batches = 3` in the train-helper smoke. The
production threshold stays at `128` so tiny selected-token batches use the
lower-overhead eager path.

Two adjacent ideas were measured but intentionally not promoted to defaults.
Returning sparse selected-token losses instead of scattering into the dense
shifted-token tensor was numerically exact in an isolated policy/ECHO loss
microbenchmark, but it was slower than the current dense vector path on the 12
GB RTX 4070 Ti and saved only about `1.3 MB` at `32768` context, or `5.2 MB` at
`131072` context, in that isolated region. The extra branching is therefore not
worth adding until a full train-step benchmark shows a real end-to-end win.
Changing the selected-token CE chunk default was also rejected: `256` stayed on
the fastest frontier for `1024` to `2048` selected targets, while smaller
chunks saved memory at a large speed cost. Keep `train_logprob_chunk_size` as
the explicit knob when a run is memory-bound.

## Operational recommendation

For Quaero-style ECHO training on long agent/tool traces:

1. Use `backend = "unsloth"` or `backend = "local"` because ECHO requires a
   backend with strict shared-forward support.
2. Set `train_selective_suffix_logits = true`.
3. Keep `train_supervised_context_tokens = 0` for exact full-context training
   until a target GPU proves it cannot fit.
4. Use `train_save_on_cpu = true` only after an exact full-context smoke OOMs.
5. Confirm the runtime metrics above before treating a run as optimized.
