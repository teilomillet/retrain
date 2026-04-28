"""Scaleway backend for retrain.

Provisions a GPU instance on Scaleway via Terraform, waits for the PRIME-RL
inference engine (vLLM) to be ready, then delegates all training protocol
calls to PrimeRLTrainHelper over ZMQ. Tears down the instance on exit.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from retrain.scaleway.terraform_runner import TerraformRunner

if TYPE_CHECKING:
    from retrain.prime_rl_backend import PrimeRLTrainHelper

logger = logging.getLogger(__name__)


class ScalewayTrainHelper:
    """TrainHelper that provisions a Scaleway GPU instance and delegates to PrimeRLTrainHelper."""

    def __init__(
        self,
        model: str,
        lora_rank: int,
        gpu_type: str = "l40s",
        zone: str = "fr-par-2",
        inference_engine: str = "vllm",
        health_timeout_s: int = 300,
        health_poll_s: float = 5.0,
        max_model_len: int = 32768,
        num_train_gpus: int = 1,
        num_infer_gpus: int = 1,
        zmq_port: int = 5555,
        output_dir: str = "",
        state_dir: str = "",
    ) -> None:
        state_path = Path(state_dir) if state_dir else Path.cwd() / ".terraform-scaleway"
        self._runner = TerraformRunner(
            zone=zone,
            gpu_type=gpu_type,
            model=model,
            lora_rank=lora_rank,
            inference_engine=inference_engine,
            max_model_len=max_model_len,
            num_train_gpus=num_train_gpus,
            num_infer_gpus=num_infer_gpus,
            state_dir=state_path,
        )
        inference_url, instance_ip = self._runner.apply()
        self._inference_url = inference_url

        _wait_healthy(inference_url, health_timeout_s, health_poll_s)

        try:
            from retrain.prime_rl_backend import PrimeRLTrainHelper
        except ImportError:
            raise RuntimeError(
                "Scaleway backend requires PRIME-RL.\n"
                "Install it with: pip install prime-rl"
            ) from None

        out_dir = output_dir or str(state_path / "prime_rl_output")
        self._delegate: PrimeRLTrainHelper = PrimeRLTrainHelper(
            model_name=model,
            output_dir=out_dir,
            inference_url=inference_url,
            transport_type="zmq",
            zmq_host=instance_ip,
            zmq_port=zmq_port,
        )

    # ------------------------------------------------------------------
    # TrainHelper protocol — fully delegated to PrimeRLTrainHelper
    # ------------------------------------------------------------------

    def checkpoint(self, name: str) -> None:
        self._delegate.checkpoint(name)

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        return self._delegate.sample(prompt_ids_list, num_samples, max_tokens, temperature, top_p)

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float], list[float] | None]]]:
        return self._delegate.sample_with_entropy(
            prompt_ids_list, num_samples, max_tokens, temperature, top_p
        )

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        return self._delegate.train_step(all_tokens, all_logprobs, all_advantages, lr, weight_decay)

    def save_adapter(self, path: str, name: str) -> str:
        return self._delegate.save_adapter(path, name)

    def load_state(self, name: str) -> None:
        self._delegate.load_state(name)

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
# Health check (standalone so it can be used without a client instance)
# ------------------------------------------------------------------

def _wait_healthy(inference_url: str, timeout_s: int, poll_s: float) -> None:
    url = f"{inference_url}/health"
    logger.info("Waiting for %s …", url)
    deadline = time.monotonic() + timeout_s
    with httpx.Client() as client:
        while True:
            try:
                r = client.get(url, timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                logger.debug("Health check %s failed", url, exc_info=True)
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timeout waiting for {url} to become healthy")
            time.sleep(poll_s)
    logger.info("Inference engine healthy.")
