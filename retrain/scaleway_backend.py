"""Scaleway backend for retrain.

Provisions a GPU instance on Scaleway via Terraform, rsyncs the project,
then runs retrain with the local backend directly on the GPU. Streams logs
via SSH and downloads the adapter when done.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from retrain.scaleway.terraform_runner import TerraformRunner

if TYPE_CHECKING:
    from retrain.config import TrainConfig

logger = logging.getLogger(__name__)

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
_REMOTE_PROJECT = "/opt/retrain-run"
_REMOTE_VENV = "/opt/retrain-venv"
_REMOTE_ADAPTER = "/tmp/retrain-adapter"
_REMOTE_CONFIG = f"{_REMOTE_PROJECT}/.scaleway-run.toml"


class ScalewayTrainHelper:
    """Autonomous backend: provisions a Scaleway GPU and runs retrain locally on it."""

    def __init__(
        self,
        model: str,
        lora_rank: int,
        gpu_type: str = "l40s",
        zone: str = "fr-par-2",
        ssh_timeout_s: int = 600,
        ssh_poll_s: float = 15.0,
        state_dir: str = "",
    ) -> None:
        state_path = Path(state_dir) if state_dir else Path.cwd() / ".terraform-scaleway"
        self._runner = TerraformRunner(
            zone=zone,
            gpu_type=gpu_type,
            model=model,
            lora_rank=lora_rank,
            state_dir=state_path,
        )
        self._instance_ip = self._runner.apply()
        self._state_path = state_path
        self._project_root = Path.cwd()

        try:
            _wait_ssh(self._instance_ip, ssh_timeout_s, ssh_poll_s)
        except Exception:
            self._runner.destroy()
            raise

    # ------------------------------------------------------------------
    # Autonomous execution
    # ------------------------------------------------------------------

    def run(self, config: "TrainConfig") -> str:
        """Rsync project, run retrain on VM with local backend, return adapter path."""
        ip = self._instance_ip

        logger.info("Syncing project to %s:%s …", ip, _REMOTE_PROJECT)
        _rsync(self._project_root, ip, _REMOTE_PROJECT)

        logger.info("Installing retrain on VM …")
        _ssh(ip, f"cd {_REMOTE_PROJECT} && {_REMOTE_VENV}/bin/pip install -e '.[local]' -q")

        logger.info("Writing remote campaign config …")
        toml_content = _config_to_toml(config, adapter_path=_REMOTE_ADAPTER)
        _write_remote_file(ip, _REMOTE_CONFIG, toml_content)

        logger.info("Starting retrain on %s — streaming logs …", ip)
        _ssh_stream(ip, f"cd {_REMOTE_PROJECT} && {_REMOTE_VENV}/bin/retrain {_REMOTE_CONFIG}")

        adapter_path = config.adapter_path or str(self._state_path / "adapter")
        logger.info("Downloading adapter to %s …", adapter_path)
        _download(ip, _REMOTE_ADAPTER, adapter_path)

        return adapter_path

    # ------------------------------------------------------------------
    # TrainHelper stubs — never called for autonomous backends
    # ------------------------------------------------------------------

    def checkpoint(self, name: str) -> None:
        raise NotImplementedError("ScalewayTrainHelper is autonomous — checkpoint() is not callable")

    def sample(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):  # type: ignore[override]
        raise NotImplementedError("ScalewayTrainHelper is autonomous — sample() is not callable")

    def sample_with_entropy(self, prompt_ids_list, num_samples, max_tokens, temperature, top_p):  # type: ignore[override]
        raise NotImplementedError("ScalewayTrainHelper is autonomous — sample_with_entropy() is not callable")

    def train_step(self, all_tokens, all_logprobs, all_advantages, lr, weight_decay):  # type: ignore[override]
        raise NotImplementedError("ScalewayTrainHelper is autonomous — train_step() is not callable")

    def save_adapter(self, path: str, name: str) -> str:
        raise NotImplementedError("ScalewayTrainHelper is autonomous — save_adapter() is not callable")

    def load_state(self, name: str) -> None:
        raise NotImplementedError("ScalewayTrainHelper is autonomous — load_state() is not callable")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Destroy the Scaleway instance."""
        self._runner.destroy()

    def __enter__(self) -> ScalewayTrainHelper:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_runner") and self._runner._applied:
            logger.warning(
                "ScalewayTrainHelper was garbage-collected without explicit close() — "
                "use a context manager or call close() to ensure the instance is destroyed."
            )


# ------------------------------------------------------------------
# Config serialization
# ------------------------------------------------------------------

def _config_to_toml(config: "TrainConfig", adapter_path: str) -> str:
    """Serialize TrainConfig to TOML, overriding backend=local and adapter path."""
    lines: list[str] = []

    def s(v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)

    lines += [
        f'backend = "local"',
        f'adapter_path = "{adapter_path}"',
        "",
        "[model]",
        f"model = {s(config.model)}",
        f"lora_rank = {config.lora_rank}",
        f"lora_alpha = {config.lora_alpha}",
        f"lora_dropout = {config.lora_dropout}",
        "",
        "[algorithm]",
    ]
    if config.algorithm_mode:
        lines.append(f"algorithm_mode = {s(config.algorithm_mode)}")
    else:
        lines.append(f"advantage_mode = {s(config.advantage_mode)}")
        lines.append(f"transform_mode = {s(config.transform_mode)}")
    lines += [
        f"gtpo_beta = {config.gtpo_beta}",
        "",
        "[training]",
        f"max_steps = {config.max_steps}",
        f"batch_size = {config.batch_size}",
        f"group_size = {config.group_size}",
        f"max_tokens = {config.max_tokens}",
        f"temperature = {config.temperature}",
        f"top_p = {config.top_p}",
        f"lr = {config.lr}",
        f"weight_decay = {config.weight_decay}",
        f"save_every = {config.save_every}",
        f"seed = {config.seed}",
        "",
        "[data]",
        f"source = {s(config.data_source)}",
    ]
    if config.max_examples:
        lines.append(f"max_examples = {config.max_examples}")

    lines += [
        "",
        "[reward]",
    ]
    if config.reward_custom_module:
        lines += [
            f'type = "custom"',
            f"custom_module = {s(config.reward_custom_module)}",
            f"custom_function = {s(config.reward_custom_function)}",
        ]
    else:
        lines.append('type = "default"')

    lines += [
        "",
        "[logging]",
        f"log_dir = {s(config.log_dir)}",
    ]

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# SSH helpers
# ------------------------------------------------------------------

def _wait_ssh(ip: str, timeout_s: int, poll_s: float) -> None:
    logger.info("Waiting for SSH on %s …", ip)
    deadline = time.monotonic() + timeout_s
    while True:
        result = subprocess.run(  # nosec B603
            ["ssh", *_SSH_OPTS, f"root@{ip}", "true"],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("SSH ready on %s.", ip)
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timeout waiting for SSH on {ip}")
        time.sleep(poll_s)


def _ssh(ip: str, cmd: str) -> None:
    result = subprocess.run(  # nosec B603
        ["ssh", *_SSH_OPTS, f"root@{ip}", cmd],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH command failed on {ip}:\n{result.stderr.strip()}")


def _ssh_stream(ip: str, cmd: str) -> None:
    """Run a command on the VM, streaming stdout/stderr to local console."""
    result = subprocess.run(  # nosec B603
        ["ssh", *_SSH_OPTS, f"root@{ip}", cmd],
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote retrain failed on {ip} (exit code {result.returncode}).")


def _rsync(local_root: Path, ip: str, remote_dir: str) -> None:
    result = subprocess.run(  # nosec B603
        [
            "rsync", "-az", "--delete",
            "--exclude=.git",
            "--exclude=.terraform-scaleway",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.venv",
            "--exclude=logs/",
            "-e", f"ssh {' '.join(_SSH_OPTS)}",
            f"{local_root}/",
            f"root@{ip}:{remote_dir}/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed:\n{result.stderr.strip()}")


def _write_remote_file(ip: str, remote_path: str, content: str) -> None:
    result = subprocess.run(  # nosec B603
        ["ssh", *_SSH_OPTS, f"root@{ip}", f"cat > {remote_path}"],
        input=content,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to write {remote_path} on {ip}:\n{result.stderr.strip()}")


def _download(ip: str, remote_path: str, local_path: str) -> None:
    Path(local_path).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # nosec B603
        ["scp", "-o", "StrictHostKeyChecking=no", "-r", f"root@{ip}:{remote_path}", local_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scp failed downloading adapter:\n{result.stderr.strip()}")
    logger.info("Adapter downloaded to %s", local_path)
