"""Terraform lifecycle manager for Scaleway GPU instances."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

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


class TerraformRunner:
    """Provisions and tears down a Scaleway GPU instance via Terraform."""

    def __init__(
        self,
        zone: str,
        gpu_type: str,
        model: str,
        lora_rank: int,
        inference_engine: str,
        caller_ip: str = "0.0.0.0/0",
        max_model_len: int = 32768,
        tf_dir: Path | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self._zone = zone
        self._instance_type = _GPU_TYPE_MAP.get(gpu_type.lower(), gpu_type)
        self._model = model
        self._lora_rank = lora_rank
        self._inference_engine = inference_engine
        self._caller_ip = caller_ip
        self._max_model_len = max_model_len
        self._tf_dir = tf_dir or _TF_DIR
        self._state_dir = state_dir or Path(os.getcwd()) / ".terraform-scaleway"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self._state_dir / "terraform.tfstate"
        self._applied = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(self) -> tuple[str, str]:
        """Provision the instance. Returns (inference_url, training_url)."""
        logger.info("Terraform apply — provisioning %s in %s", self._instance_type, self._zone)
        logger.info("Terraform state → %s", self.state_file)
        self._run_tf("init", "-no-color", "-input=false")
        self._run_tf(
            "apply",
            "-auto-approve",
            "-input=false",
            "-no-color",
            *self._state_args(),
            *self._var_args(),
        )
        self._applied = True
        outputs = self._read_outputs()
        inference_url = outputs["inference_url"]["value"]
        training_url = outputs["training_url"]["value"]
        logger.info("Instance ready — inference=%s training=%s", inference_url, training_url)
        return inference_url, training_url

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
        # Values are shell-escaped so user-supplied strings (model name, zone…)
        # cannot break out of the -var=key=value argument.
        def v(key: str, value: str | int) -> str:
            return f"-var={key}={shlex.quote(str(value))}"

        return [
            v("instance_type", self._instance_type),
            v("zone", self._zone),
            v("project_id", project_id),
            v("model", self._model),
            v("lora_rank", self._lora_rank),
            v("inference_engine", self._inference_engine),
            v("caller_ip", self._caller_ip),
            v("max_model_len", self._max_model_len),
        ]

    def _run_tf(self, *args: str) -> None:
        cmd = ["terraform", *args]
        env = {**os.environ, "TF_DATA_DIR": str(self._state_dir)}
        result = subprocess.run(  # nosec B603 — list argv, shell=False, no user input reaches here
            cmd,
            cwd=str(self._tf_dir),
            env=env,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            raise TerraformError(f"terraform {args[0]} exited with code {result.returncode}")

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
