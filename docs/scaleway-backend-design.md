# Scaleway Backend — Design Proposal

**Status**: RFC — pending team review  
**Author**: Gireg Roussel  
**Date**: 2026-04-24

---

## Motivation

retrain currently has three backends: `local` (local GPU), `tinker` (proprietary service) and `prime_rl` (external PRIME-RL stack). None of them satisfy the **data sovereignty** constraint: compute and data hosted in France/EU, with no dependency on a US cloud provider.

The goal is to add a `scaleway` backend that uses Scaleway's GPU infrastructure (H100, L40S, A100…) to run inference and training, with automatic provisioning via Terraform.

---

## Overview

```
retrain (local CPU)
    │
    ├─ terraform apply ──────────────────→  Scaleway GPU Instance
    │   (cloud-init at boot)                    ├─ Inference engine  :8000  (vLLM by default, or SGLang)
    │                                           ├─ Training API      :8001
    │                                           └─ LoRA weights      (GPU RAM)
    │
    ├─ sample()     ────────────────────→  Inference engine  /v1/chat/completions
    ├─ train_step() ────────────────────→  Training API  /train_step
    ├─ checkpoint() ────────────────────→  Training API  /checkpoint  →  reload LoRA on inference engine
    └─ end of run   ────────────────────→  terraform destroy
```

**No local GPU required.** Everything runs on the Scaleway instance. retrain only drives the lifecycle and sends data.

---

## Target user experience

```toml
[backend]
backend = "scaleway"

[backend.options]
gpu_type         = "l40s"                              # or "h100", "a100", "L40S-1-48G" (exact type)
zone             = "fr-par-2"
model            = "meta-llama/Llama-3.1-8B-Instruct"
lora_rank        = 32
inference_engine = "vllm"                              # or "sglang" (see dedicated section)
max_model_len    = 4096

[algorithm]
advantage_mode = "maxrl"
transform_mode = "gtpo_sepa"

[training]
max_steps  = 200
batch_size = 4
group_size = 8
```

```bash
export SCW_SECRET_KEY=...
export SCW_DEFAULT_PROJECT_ID=...
retrain
```

retrain provisions the instance, waits for it to be ready, trains, then destroys it. The final LoRA adapter is retrieved locally before teardown.

---

## GPU type resolution

Two forms are supported in `gpu_type`:

| TOML value | Resolved Scaleway instance |
|------------|---------------------------|
| `"h100"` | `H100-1-80G` |
| `"l40s"` | `L40S-1-48G` |
| `"a100"` | `GPU-A100-S` |
| `"l4"` | `L4-1-24G` |
| `"L40S-1-48G"` | passthrough (exact type) |

If the value is not in the table, it is passed as-is to Terraform — which allows using any new Scaleway type without updating retrain.

---

## Inference engine selection

The backend supports **vLLM** (default) and **SGLang**, configurable via `inference_engine`.
TensorRT-LLM is excluded from the MVP: it requires compiling a TRT engine at boot (~10–30 min) and LoRA hot-swap is significantly more complex.

### Comparison for the retrain use case

| Criterion | vLLM | SGLang |
|-----------|------|--------|
| Reliable LoRA hot-swap | ✅ mature (`/v1/load_lora_adapter`) | ⚠️ recent (`/add_lora`) |
| KV cache on shared prefixes | ❌ limited | ✅ **RadixAttention** |
| Sampling throughput | 🟡 average | ✅ 20–40% higher |
| Cloud-init setup | ✅ simple | ✅ simple |
| OpenAI-compatible API | ✅ | ✅ |

**Key point — RadixAttention**: in RL, retrain samples `group_size` completions from the **same prompt** (8, 16…). SGLang reuses the KV cache of the shared prefix across all completions in a group, massively reducing memory usage and sampling time. This is a structural advantage for this workload.

**Recommendation**: start with **vLLM** for the maturity of its LoRA hot-swap. SGLang becomes the natural choice once its LoRA reload API is validated as stable — the architecture supports it with no changes on the training side.

### LoRA reload on each `checkpoint()`

```
Training server                Inference engine
      │                               │
      │  saves LoRA weights           │
      │──────────────────────────────→│ POST /v1/load_lora_adapter  (vLLM)
      │                               │ POST /add_lora               (SGLang)
      │                               │  → engine reloads weights into GPU RAM
```

The endpoint varies by `inference_engine` but the protocol on the retrain side is identical.

---

## Components

### 1. Terraform (`retrain/scaleway/terraform/`)

```
main.tf          # scaleway_instance_server + network rules (ports 8000/8001)
variables.tf     # instance_type, zone, project_id, model, lora_rank, inference_engine, max_model_len
outputs.tf       # inference_url, training_url, instance_ip
cloud-init.yaml  # bootstrap: installs inference engine + training server at startup
versions.tf      # provider constraints
```

`cloud-init` installs dependencies and starts, depending on `inference_engine`:
- `python -m vllm.entrypoints.openai.api_server --model <model> --port 8000 --enable-lora` **(default)**
- or `python -m sglang.launch_server --model <model> --port 8000`
- `python -m retrain.scaleway.training_server --port 8001 --model <model> --lora-rank <rank>`

All packages are installed in a venv at `/opt/retrain`.

### 2. Training Server (`retrain/scaleway/training_server.py`)

Minimal FastAPI server running on the GPU instance, exposing `TrainHelper` over HTTP. It instantiates `LocalTrainHelper` internally (PyTorch/PEFT on the Scaleway GPU).

| Endpoint | Body | Response |
|----------|------|----------|
| `POST /train_step` | tokens, logprobs, advantages, lr, wd | `{loss: float}` |
| `POST /checkpoint` | name | `{}` — syncs training weights → vLLM |
| `POST /save_adapter` | path, name | LoRA weights (tar.gz bytes) |
| `POST /load_state` | name | `{}` |
| `GET  /health` | — | `{status: "ok"}` |

### 3. `ScalewayTrainHelper` (`retrain/scaleway_backend.py`)

Python client on the retrain side:

1. `__init__` → `TerraformRunner.apply()` (blocks ~2–5 min, logs progress)
2. Polls `/health` on both services
3. Delegates `sample()` to the inference engine (OpenAI-compatible, identical for vLLM and SGLang)
4. Delegates `train_step()` / `checkpoint()` to the Training Server
5. `checkpoint()` triggers LoRA reload on the inference engine via the appropriate endpoint:
   - vLLM: `POST /v1/load_lora_adapter`
   - SGLang: `POST /add_lora`
6. `close()` / context manager / `__del__` → `TerraformRunner.destroy()`

### 4. `TerraformRunner` (`retrain/scaleway/terraform_runner.py`)

- Resolves `gpu_type` → exact Scaleway type
- `terraform init && terraform apply -auto-approve`
- Parses outputs (inference_url, training_url)
- `terraform destroy -auto-approve` at teardown
- State file stored in `<log_dir>/.terraform-state/terraform.tfstate`

---

## Capabilities

Identical to the `tinker` backend (remote, synchronous):

| Capability | `scaleway` | `tinker` | `prime_rl` |
|------------|-----------|---------|------------|
| `reports_sync_loss` | `true` | `true` | `false` |
| `preserves_token_advantages` | `true` | `true` | `false` |
| `supports_checkpoint_resume` | `true` | `true` | `true` |
| `resume_runtime_dependent` | `true` | `true` | `false` |

`resume_runtime_dependent = true`: the instance is recreated on each run, so in-memory weights are lost. Resume relies on a checkpoint saved locally (or on Scaleway Object Storage).

---

## Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
scaleway = ["httpx>=0.27", "fastapi>=0.111", "uvicorn[standard]>=0.29", "scaleway>=0.10"]
```

```bash
pip install retrain[scaleway]
# + terraform CLI (>= 1.6) in PATH
```

---

## Open questions

1. **Inference engine default**: vLLM (current) or SGLang? SGLang offers a significant gain via RadixAttention on RL multi-completion workloads, but its LoRA reload API is newer. Should SGLang be validated via a smoke test before making it the default?

2. **Network security**: ports 8000/8001 should not be publicly exposed. Preferred approach?
   - Scaleway VPC (private IP only) + VPN/bastion for retrain → more secure, more complex
   - Automatic SSH tunnel from `TerraformRunner` → simple, light overhead
   - Security group filtered on the retrain machine's public IP → compromise

3. **Terraform state**: local by default (`<log_dir>/.terraform-state/`). For teams with multiple simultaneous runners, a shared Terraform backend will be needed. Scaleway Object Storage is a natural sovereign option — activate?

4. **GPU catalogue**: the `gpu_type → instance_type` resolution table needs to be maintained. Should it be hardcoded in retrain, or fetched from a Scaleway API endpoint at runtime?

5. **Provisioning time**: ~2–5 min of `terraform apply` before the first sample. Acceptable for long campaigns, costly for short tests. Should a "persistent instance" mode be added (no automatic destroy)?

6. **Cross-run checkpoints**: with `resume_runtime_dependent = true`, resuming a run requires the LoRA adapter to have been saved locally before destroy. Should `save_adapter` be forced automatically at teardown?

---

## Files to create / modify (summary)

| File | Type |
|------|------|
| `retrain/scaleway_backend.py` | new |
| `retrain/scaleway/training_server.py` | new |
| `retrain/scaleway/terraform_runner.py` | new |
| `retrain/scaleway/terraform/main.tf` | new |
| `retrain/scaleway/terraform/variables.tf` | new |
| `retrain/scaleway/terraform/outputs.tf` | new |
| `retrain/scaleway/terraform/versions.tf` | new |
| `retrain/scaleway/terraform/cloud-init.yaml` | new |
| `retrain/backend_definitions.py` | modify — add `"scaleway"` |
| `pyproject.toml` | modify — `[scaleway]` extra + entrypoint |
