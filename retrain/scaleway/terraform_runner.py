"""Terraform lifecycle manager for Scaleway GPU instances."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_GPU_TYPE_MAP: dict[str, str] = {
    "h100": "H100-1-80G",
    "l40s": "L40S-1-48G",
    "a100": "GPU-A100-S",
    "l4":   "L4-1-24G",
}

_TF_DIR = Path(__file__).parent / "terraform"


class TerraformError(RuntimeError):
    pass


def _detect_public_ip() -> str:
    """Detect the caller's public IP via ipify. Raises RuntimeError on failure."""
    try:
        ip = httpx.get("https://api4.ipify.org", timeout=5).text.strip()
        return f"{ip}/32"
    except Exception as exc:
        raise RuntimeError(
            "Could not auto-detect public IP to restrict ports 8000/8001. "
            "Set caller_ip explicitly in your campaign config."
        ) from exc


class TerraformRunner:
    """Provisions and tears down a Scaleway GPU instance via Terraform."""

    def __init__(
        self,
        zone: str,
        gpu_type: str,
        model: str,
        lora_rank: int,
        caller_ip: str = "",
        tf_dir: Path | None = None,
        state_dir: Path | None = None,
    ) -> None:
        if not caller_ip:
            caller_ip = _detect_public_ip()
            logger.info("Auto-detected caller IP: %s", caller_ip)
        self._zone = zone
        self._instance_type = _GPU_TYPE_MAP.get(gpu_type.lower(), gpu_type)
        self._model = model
        self._lora_rank = lora_rank
        self._caller_ip = caller_ip
        self._tf_dir = tf_dir or _TF_DIR
        self._state_dir = state_dir or Path(os.getcwd()) / ".terraform-scaleway"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self._state_dir / "resources.tfstate"
        self._applied = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self) -> str:
        """Provision the instance. Returns instance_ip."""
        logger.info("Terraform apply — provisioning %s in %s", self._instance_type, self._zone)
        logger.info("Terraform state → %s", self.state_file)
        self._run_tf("init", "-no-color", "-input=false")
        self._applied = True
        self._run_tf(
            "apply",
            "-auto-approve",
            "-input=false",
            "-no-color",
            *self._state_args(),
            *self._var_args(),
        )
        outputs = self._read_outputs()
        instance_ip = outputs["instance_ip"]["value"]
        logger.info("Instance ready — ip=%s", instance_ip)
        return instance_ip

    def destroy(self) -> None:
        """Tear down the instance."""
        if not self._applied:
            return
        logger.info("Terraform destroy — tearing down instance")
        try:
            self._run_tf(
                "destroy",
                "-auto-approve",
                "-input=false",
                "-no-color",
                *self._state_args(),
                *self._var_args(),
            )
        except TerraformError as exc:
            logger.warning("terraform destroy failed (instance may leak): %s", exc)
        self._applied = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state_args(self) -> list[str]:
        return [f"-state={self.state_file}"]

    def _var_args(self) -> list[str]:
        project_id = os.environ.get("SCW_DEFAULT_PROJECT_ID", "")

        def v(key: str, value: str | int) -> list[str]:
            return ["-var", f"{key}={value}"]

        args: list[str] = []
        for pair in [
            v("instance_type", self._instance_type),
            v("zone", self._zone),
            v("project_id", project_id),
            v("model", self._model),
            v("lora_rank", self._lora_rank),
            v("caller_ip", self._caller_ip),
        ]:
            args.extend(pair)
        return args

    def _run_tf(self, *args: str) -> None:
        cmd = ["terraform", *args]
        env = {**os.environ, "TF_DATA_DIR": str(self._state_dir)}
        result = subprocess.run(  # nosec B603 — list argv, shell=False, no user input reaches here
            cmd,
            cwd=str(self._tf_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug("terraform %s stdout:\n%s", args[0], result.stdout.strip())
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise TerraformError(f"terraform {args[0]} failed:\n{stderr}")

    def _read_outputs(self) -> dict[str, object]:
        cmd = ["terraform", "output", "-json", f"-state={self.state_file}"]
        env = {**os.environ, "TF_DATA_DIR": str(self._state_dir)}
        result = subprocess.run(  # nosec B603 — list argv, shell=False, no user input reaches here
            cmd,
            cwd=str(self._tf_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TerraformError(f"terraform output failed: {result.stderr}")
        return json.loads(result.stdout)
