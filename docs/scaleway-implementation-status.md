# Scaleway Backend — Implementation Status

## Architecture

The Scaleway backend provisions a GPU instance via Terraform, rsyncs the project,
and runs `retrain` with the `local` backend directly on the VM over SSH. This
replaces the earlier HTTP/FastAPI design described in `scaleway-backend-design.md`
— no training server, inference engine sidecar, or ZMQ process is involved.

## Steps

| # | Description | Status | File(s) |
|---|-------------|--------|---------|
| 1 | `TerraformRunner` | ✅ Done | `retrain/scaleway/terraform_runner.py` |
| 2 | Terraform files (SSH-only security group) | ✅ Done | `retrain/scaleway/terraform/` |
| 3 | `ScalewayTrainHelper` (SSH + rsync) | ✅ Done | `retrain/scaleway_backend.py` |
| 4 | Backend registration | ✅ Done | `retrain/backend_definitions.py` |
| 5 | Contract test | ✅ Done | `tests/test_backend_contract.py` |

## Flow

1. `terraform apply` — provision VM, return `instance_ip`
2. Wait for SSH to become available (configurable timeout, default 10 min)
3. `rsync` project root to `/opt/retrain-run` on the VM
4. `pip install -e '.[local]'` into `/opt/retrain-venv`
5. Write campaign config to VM with `backend = "local"` override
6. `retrain <config>` — stream logs back via SSH
7. `scp` adapter from VM to local machine
8. `terraform destroy` on `close()`

## To validate

- [ ] `pytest tests/test_backend_contract.py::test_scaleway_backend_contract`
- [ ] `terraform validate` in `retrain/scaleway/terraform/`
- [ ] End-to-end run on Scaleway with `campaigns/sroie.toml`

## Open questions

1. **Network security** — `caller_ip` is auto-detected via ipify at `terraform apply` time. A dynamic IP that changes between `apply` and the SSH connection will cause a timeout. Set `caller_ip` explicitly in the campaign TOML if needed.
2. **Terraform state** — stored locally in `<log_dir>/.terraform-state/` by default. Scaleway Object Storage would be needed for multi-runner or multi-machine setups.
3. **GPU catalogue** — type resolution table is hardcoded in `TerraformRunner._GPU_TYPE_MAP`.
4. **cloud-init duration** — model pre-download can take 10–20 min. SSH becomes available well before that; training will start before the model is cached if rsync + pip finish first. This is fine: `LocalTrainHelper` calls `from_pretrained` at startup and will wait for the download.
