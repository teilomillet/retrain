# ECHO Training Optimization

ECHO trains the model to predict environment, tool, or observation tokens from
the same rollout row used for policy learning. Agent workloads make this row
long: the model emits an action, waits for a tool reply, emits another action,
and so on. Most tokens in that history are context, while only a small subset
carry RL or ECHO loss.

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
```

With `train_selective_suffix_logits = true`, retrain builds a target mask from
tokens that carry RL advantage or positive ECHO weight. The transformer still
runs over the train row, so the hidden states are conditioned on the full
context. The memory win comes after that: retrain computes the LM head and
log-softmax only for selected target positions when the selected tokens are
sparse inside a long suffix.

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

## Correctness boundary

This path is exact for the selected RL/ECHO token log-probabilities:

- same input tokens;
- same attention mask;
- same actor forward/backward step;
- same hidden state at each selected target position;
- same LM head and log-softmax for those selected targets.

It only skips LM-head/log-softmax work for positions whose weight is zero. Those
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
| `local_train_selective_suffix_logprob_batches` | The suffix `logits_to_keep` shortcut ran instead |
| `local_train_selective_fallback_logprob_batches` | The selective path could not run and fell back |

For sparse ECHO/tool traces, the expected proof is:

```text
local_train_selective_sparse_suffix_skips > 0
local_train_selective_hidden_logprob_batches > 0
local_train_selective_fallback_logprob_batches = 0
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
| suffix-style LM head + log-softmax | `467.3 MB` | `0.00441 s` |
| selected hidden-token logits | `20.8 MB` | `0.00043 s` |

That is a 10.2x speedup for the measured LM-head/log-softmax region and a much
smaller peak allocation. The same script also runs
`LocalTrainHelper.train_step_with_echo_masks`; in the measured run it reported
`local_train_selective_sparse_suffix_skips = 3`,
`local_train_selective_hidden_logprob_batches = 3`, and zero suffix-logprob
batches.

## Operational recommendation

For Quaero-style ECHO training on long agent/tool traces:

1. Use `backend = "unsloth"` or `backend = "local"` because ECHO requires a
   backend with strict shared-forward support.
2. Set `train_selective_suffix_logits = true`.
3. Keep `train_supervised_context_tokens = 0` for exact full-context training
   until a target GPU proves it cannot fit.
4. Use `train_save_on_cpu = true` only after an exact full-context smoke OOMs.
5. Confirm the runtime metrics above before treating a run as optimized.
