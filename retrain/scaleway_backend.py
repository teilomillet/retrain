"""Scaleway backend for retrain.

Provisions a GPU instance on Scaleway via Terraform, waits for the inference
engine (vLLM/SGLang) and training server to be ready, then drives training
remotely. Tears down the instance on exit.
"""

from __future__ import annotations

import io
import logging
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from retrain.scaleway.terraform_runner import TerraformRunner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# SampleBatch = list[groups] where each group = list[(token_ids, logprobs)]
SampleBatch = list[list[tuple[list[int], list[float]]]]
EnrichedSampleBatch = list[list[tuple[list[int], list[float], list[float] | None]]]

_DEFAULT_TIMEOUT = 120.0


class ScalewayTrainHelper:
    """TrainHelper that delegates sampling and training to a remote Scaleway GPU instance."""

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
        state_dir: str = "",
    ) -> None:
        self._model = model
        self._lora_rank = lora_rank
        self._inference_engine = inference_engine
        self._health_timeout_s = health_timeout_s
        self._health_poll_s = health_poll_s

        state_dir = Path(state_dir) if state_dir else Path.cwd() / ".terraform-scaleway"
        self._runner = TerraformRunner(
            zone=zone,
            gpu_type=gpu_type,
            model=model,
            lora_rank=lora_rank,
            inference_engine=inference_engine,
            max_model_len=max_model_len,
            state_dir=state_dir,
        )
        self._inference_url, self._training_url = self._runner.apply()
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)
        self._wait_healthy()

    # ------------------------------------------------------------------
    # TrainHelper protocol
    # ------------------------------------------------------------------

    def checkpoint(self, name: str) -> None:
        self._post_training("/checkpoint", {"name": name})

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> SampleBatch:
        tokenizer = self._get_tokenizer()

        results: SampleBatch = []
        for prompt_ids in prompt_ids_list:
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            group: list[tuple[list[int], list[float]]] = []
            for _ in range(num_samples):
                token_ids, logprobs = self._sample_one(
                    prompt_text, max_tokens, temperature, top_p
                )
                group.append((token_ids, logprobs))
            results.append(group)
        return results

    def sample_with_entropy(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> EnrichedSampleBatch:
        # vLLM does not expose per-token entropy natively; return None for entropy
        base = self.sample(prompt_ids_list, num_samples, max_tokens, temperature, top_p)
        return [
            [(token_ids, logprobs, None) for token_ids, logprobs in group]
            for group in base
        ]

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        resp = self._post_training("/train_step", {
            "tokens": all_tokens,
            "logprobs": all_logprobs,
            "advantages": all_advantages,
            "lr": lr,
            "weight_decay": weight_decay,
        })
        return float(resp["loss"])

    def save_adapter(self, path: str, name: str) -> str:
        resp = self._client.post(
            f"{self._training_url}/save_adapter",
            json={"name": name},
            timeout=120,
        )
        resp.raise_for_status()
        local_dir = Path(path) / name
        local_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            self._safe_extractall(tar, local_dir)
        return str(local_dir)

    def load_state(self, name: str) -> None:
        self._post_training("/load_state", {"name": name})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Deterministically destroy the Scaleway instance and release resources."""
        self._runner.destroy()
        self._client.close()

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
    # Internals
    # ------------------------------------------------------------------

    def _wait_healthy(self) -> None:
        for url in (f"{self._inference_url}/health", f"{self._training_url}/health"):
            logger.info("Waiting for %s …", url)
            deadline = time.monotonic() + self._health_timeout_s
            while True:
                try:
                    r = self._client.get(url, timeout=5)
                    if r.status_code == 200:
                        break
                except Exception:
                    logger.debug("Health check %s failed", url, exc_info=True)
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Timeout waiting for {url} to become healthy")
                time.sleep(self._health_poll_s)
        logger.info("Both services healthy.")

    def _sample_one(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[list[int], list[float]]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": 1,
        }
        resp = self._client.post(
            f"{self._inference_url}/v1/chat/completions",
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        token_ids: list[int] = []
        logprobs: list[float] = []
        lp_content = (choice.get("logprobs") or {}).get("content") or []
        for entry in lp_content:
            if "token_id" not in entry:
                continue
            token_ids.append(entry["token_id"])
            top = entry.get("top_logprobs") or []
            logprobs.append(top[0]["logprob"] if top else 0.0)
        return token_ids, logprobs

    @staticmethod
    def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
        dest_resolved = dest.resolve()
        for member in tar.getmembers():
            member_path = (dest / member.name).resolve()
            try:
                member_path.relative_to(dest_resolved)
            except ValueError:
                raise ValueError(f"Unsafe tar path rejected: {member.name}") from None
        tar.extractall(dest)

    def _post_training(self, path: str, body: dict) -> dict:
        resp = self._client.post(
            f"{self._training_url}{path}",
            json=body,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model)
        return self._tokenizer
