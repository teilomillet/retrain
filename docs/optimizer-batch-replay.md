# Exact-Input Optimizer-Batch Replay

Fixed sampling seeds do not guarantee a fair systems ablation in a tool
environment. A timestamp, generated path, database diagnostic, or other
model-visible observation can change the next prompt and therefore every
downstream optimizer row. Matching seeds is useful, but matching the actual
optimizer batch is decisive for isolating optimizer inputs.

retrain can capture one real RL step at the exact trainer-logical optimizer
boundary and replay it without loading a dataset, connecting to an environment,
or sampling the model. The v1 surface is intentionally narrow: local backend
with PyTorch inference on exactly one device, one step, a content-pinned
initial adapter, and no SFT warmup or backpressure.
It is a systems-ablation fixture, not a way to reuse benchmark trajectories for
training or to claim deterministic agent behavior or bitwise-deterministic
CUDA updates.

## 1. Capture a source batch

Start from a one-step local RL config. It must resume from a local adapter and
disable periodic checkpoints. V1 accepts only an initialization resume state
whose `trainer_state.json` has `step = -1`; the normal trainer then executes
logical step zero exactly once. A completed checkpoint with `step >= 0` or an
earlier value fails closed instead of capturing zero or multiple steps.

```toml
[training]
trainer = "retrain"
max_steps = 1
save_every = 0

[backend]
devices = "gpu:0"

[inference]
engine = "pytorch"

[resume]
from = "run_logs/init_resume"

[optimizer_batch]
capture = true
```

Run it normally. Capture occurs after batch advantage normalization, advantage
clipping, and the active ECHO token cap, immediately before the existing
logical-batch digest and optimizer call. The source run still performs its
optimizer update.

The log directory receives:

```text
optimizer_batch_step_000000.safetensors
optimizer_batch_step_000000.manifest.json
```

The payload stores ragged token rows, binary64 old logprobs and advantages,
active ECHO masks/counts/denominator, and Torch CPU/CUDA RNG state. The JSON
manifest is committed last and pins the payload (including RNG), logical rows,
resolved config, optimizer contract, and exact initial adapter weight file by
SHA256. Resolved-config and optimizer-contract provenance recursively redacts
secret-shaped keys, opaque trainer commands, environment arguments, and URL
credentials before it is hashed or persisted. Loading uses safetensors, never
pickle.

## 2. Replay a matched condition

Copy the source config, change the runner and output paths, and pin the digest
reported by the capture metrics:

```toml
[training]
trainer = "optimizer_replay"
max_steps = 1
save_every = 0

[optimizer_batch]
replay_path = "run_logs/source/optimizer_batch_step_000000.manifest.json"
expected_logical_sha256 = "<64-character digest from the source run>"
expected_manifest_sha256 = "<64-character manifest digest from the source run>"
allow_config_differences = ["backend.options.gradient_checkpointing"]

[backend.options]
gradient_checkpointing = false
```

Both external pins are required. The logical digest makes the compared rows
explicit; the exact manifest digest transitively pins the RNG-bearing payload
before the manifest is parsed. The initial `resume.from` adapter must have the
same weight-file hash as the source. The resolved optimizer contract must also
match exactly. In v1, the only permitted difference is
`backend.options.gradient_checkpointing`, and it must be explicitly declared.
An undeclared difference fails; a declared difference that is not actually
present also fails. This prevents a stale allowlist from weakening the gate.

Replay constructs the local backend, loads the pinned adapter, restores the
captured Torch RNG state, performs exactly one optimizer update, and saves the
final adapter. It never loads training examples, touches OpenEnv/verifiers, or
calls sampling. `retrain explain replay.toml` verifies the payload, adapter,
and config contract before GPU allocation.

## Evidence gates

Replay supports two different claims. Keep them separate.

### A. Exact-input systems claim

This is the admission gate for runtime and memory comparisons. Require all of
the following:

- both runs report the expected `optimizer/logical_batch_sha256`;
- both report the same `optimizer/local_effective_rows_sha256`;
- `optimizer_batch/initial_adapter_sha256` matches;
- allowed and observed config differences are exactly
  `backend.options.gradient_checkpointing` for the changed arm;
- replay reports dataset, environment, rollout, and sampling as skipped;
- the checkpointing runtime flag has the intended value;
- source and same-condition replay report the same loss;
- each compared condition saves a final adapter whose SHA differs from the
  initial adapter SHA;
- timing and memory conclusions use repeated fresh replays and report their
  spread, rather than treating the source run or one replay as decisive.

Compare `local_train_wall_s`, forward/backward/optimizer timing, peak GPU
memory, and microbatch-padding metrics. Under this claim, differing final
adapter hashes do not invalidate the systems comparison: the artifact proves
the inputs presented to the optimizer were identical, not that CUDA kernels
executed bitwise-identically.

The first live CUDA control exposed this boundary. Source and replay had the
same manifest, logical/effective rows, RNG state, initial adapter, optimizer
contract, and loss, but source versus same-condition replay final weights were
not identical. A second identical replay also differed similarly. Pairwise
update cosine was about `0.9965`, while delta L2 was about `8.2–8.4%` of the
update norm. This is evidence of CUDA/kernel numerical nondeterminism in that
runtime, not evidence that the captured optimizer inputs differed.

### B. Bitwise update claim

A bitwise update claim has a stricter gate: source, same-condition replay, and
repeated identical replays must produce the same final adapter SHA. The live
CUDA control did not pass this gate, so retrain does not currently claim
bitwise-deterministic updates.

Approximate numerical-equivalence claims are separate again. Predeclare their
metrics and tolerances, then report update cosine and relative delta L2 across
repeats. Do not infer such a tolerance after seeing one pair, and do not enable
PyTorch deterministic-algorithm modes as a presumed fix without measuring that
they preserve correctness and improve repeatability on the target model and
kernels.

## Scope boundary

Optimizer-batch replay removes rollout nondeterminism from a systems
comparison. It does not eliminate CUDA numerical nondeterminism, fix
nondeterministic observations, improve reward quality, or show that two agents
behaved identically. A replay output is a diagnostic one-batch adapter, not a
quality candidate. For downstream benchmark or quality evaluation, use the
source-run adapter produced by the normal training path and run fresh,
uncontaminated evaluations.
