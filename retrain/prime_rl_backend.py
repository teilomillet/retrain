"""PrimeRLTrainHelper — bridge retrain backend calls to a running PRIME-RL stack.

This backend is designed for setups where PRIME-RL trainer + inference are already
running, and retrain provides dataset/reward/advantage orchestration.

Key behavior:
- `sample()` calls PRIME-RL/vLLM token endpoint: `/v1/chat/completions/tokens`
- `train_step()` sends PRIME-RL `TrainingBatch` messages through PRIME-RL transport
  (`filesystem` or `zmq`) to the PRIME-RL trainer packer.
- `checkpoint()` updates inference weights from PRIME-RL broadcast checkpoints via
  `/update_weights` when new weight directories become available.

Limit:
PRIME-RL `TrainingSample` stores one scalar advantage per sample. By default this
backend enforces that completion-token advantages are uniform within each sample.
Set `strict_advantages=False` to aggregate token advantages by mean.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

import requests


JSONPayload: TypeAlias = dict[str, object]


class PrimeRLTrainHelper:
    """Training helper using an external PRIME-RL trainer + inference servers."""

    _STEP_RE = re.compile(r"(?:^step_|^checkpoint_step_)(\d+)$")
    _PROMPT_PAD_EPS = 1e-9
    _ADV_UNIFORM_EPS = 1e-6

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        inference_url: str,
        transport_type: str = "filesystem",
        zmq_host: str = "localhost",
        zmq_port: int = 5555,
        zmq_hwm: int = 10,
        strict_advantages: bool = True,
        sync_wait_s: int = 30,
        sync_poll_s: float = 0.2,
    ) -> None:
        """Create PRIME-RL transport sender and inference client."""
        try:
            from prime_rl.configs.shared import (
                FileSystemTransportConfig,
                ZMQTransportConfig,
            )
            from prime_rl.transport import (
                TrainingBatch,
                TrainingSample,
                setup_training_batch_sender,
            )
        except ImportError:
            raise RuntimeError(
                "Backend 'prime_rl' requires PRIME-RL.\n"
                "Install it with: pip install prime-rl"
            ) from None

        if transport_type not in ("filesystem", "zmq"):
            raise ValueError(
                f"Invalid PRIME-RL transport '{transport_type}'. "
                "Expected 'filesystem' or 'zmq'."
            )

        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.strict_advantages = strict_advantages
        self.sync_wait_s = max(0, int(sync_wait_s))
        self.sync_poll_s = max(0.01, float(sync_poll_s))

        base_url = (inference_url or "http://localhost:8000").rstrip("/")
        self._sample_url = f"{base_url}/v1/chat/completions/tokens"
        self._admin_base_url = (
            base_url[:-3] if base_url.endswith("/v1") else base_url
        ).rstrip("/")
        self._session = requests.Session()

        self._last_temperature = 1.0
        self._pending_checkpoint_step: int | None = None
        self._last_sent_step = -1
        self._last_loaded_step = -1

        if transport_type == "filesystem":
            transport = FileSystemTransportConfig()
        else:
            transport = ZMQTransportConfig(
                host=zmq_host,
                port=zmq_port,
                hwm=zmq_hwm,
            )

        self._TrainingBatch = TrainingBatch
        self._TrainingSample = TrainingSample
        self._sender = setup_training_batch_sender(self.output_dir, transport)

        print(
            "PrimeRLTrainHelper ready "
            f"(transport={transport_type}, output_dir={self.output_dir}, "
            f"inference={self._admin_base_url})."
        )

    def checkpoint(self, name: str) -> None:
        """Prepare sampling by syncing to latest available PRIME-RL broadcast weights."""
        step = self._parse_step(name)
        self._pending_checkpoint_step = step
        if step is None or step <= 0:
            return
        self._sync_inference_weights(max_step=step)

    def sample(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        """Sample completions from PRIME-RL inference server."""
        self._last_temperature = float(temperature)
        results: list[list[tuple[list[int], list[float]]]] = []

        for prompt_ids in prompt_ids_list:
            payload: JSONPayload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": ""}],
                "tokens": list(prompt_ids),
                "n": num_samples,
                "max_tokens": int(max_tokens),
                "temperature": max(float(temperature), 1e-7),
                "top_p": float(top_p),
                "logprobs": True,
                "skip_special_tokens": False,
                # PRIME-RL/vLLM extra params.
                "return_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
            }
            response = self._post_json(self._sample_url, payload)
            choices_obj = response.get("choices", [])
            if not isinstance(choices_obj, list):
                raise RuntimeError(
                    "PRIME-RL inference returned malformed 'choices' payload."
                )
            choices = choices_obj
            if len(choices) != num_samples:
                raise RuntimeError(
                    "PRIME-RL inference returned unexpected number of completions: "
                    f"expected {num_samples}, got {len(choices)}."
                )

            group: list[tuple[list[int], list[float]]] = []
            for choice in choices:
                if not isinstance(choice, Mapping):
                    raise RuntimeError("PRIME-RL inference returned non-object choice payload.")
                token_ids, logprobs = self._extract_completion(
                    dict(cast(Mapping[str, object], choice)),
                    prompt_ids,
                )
                group.append((token_ids, logprobs))
            results.append(group)

        return results

    def train_step(
        self,
        all_tokens: list[list[int]],
        all_logprobs: list[list[float]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        """Send one PRIME-RL TrainingBatch. Returns 0.0 (async trainer loss unavailable)."""
        # lr/weight_decay are managed by external PRIME-RL trainer config.
        _ = lr, weight_decay

        if not all_tokens:
            return 0.0

        step = (
            self._pending_checkpoint_step
            if self._pending_checkpoint_step is not None
            else self._last_sent_step + 1
        )
        self._pending_checkpoint_step = None

        examples: list[object] = []
        for tokens, logprobs, advantages in zip(all_tokens, all_logprobs, all_advantages):
            sample = self._to_training_sample(tokens, logprobs, advantages)
            if sample is not None:
                examples.append(sample)

        if not examples:
            return 0.0

        batch = self._TrainingBatch(examples=examples, step=int(step))
        self._sender.send(batch)
        self._last_sent_step = max(self._last_sent_step, int(step))
        return 0.0

    def load_state(self, name: str) -> None:
        """Restore backend-side metadata and re-sync inference to saved step if available."""
        state_file = Path(self.output_dir) / name / "prime_rl_backend_state.json"
        if not state_file.is_file():
            raise FileNotFoundError(
                f"PRIME-RL backend state not found: {state_file}. "
                "Cannot resume without backend metadata."
            )

        saved = json.loads(state_file.read_text())
        self._last_sent_step = int(saved.get("last_sent_step", -1))
        self._last_loaded_step = int(saved.get("last_loaded_step", -1))
        self._pending_checkpoint_step = self._last_sent_step + 1

        if self._last_loaded_step >= 0:
            self._update_inference_from_step(self._last_loaded_step)

        print(f"PrimeRL backend state loaded: {state_file}")

    def save_adapter(self, path: str, name: str) -> str:
        """Persist backend metadata for retrain checkpoints."""
        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "backend": "prime_rl",
            "output_dir": str(self.output_dir),
            "inference_base_url": self._admin_base_url,
            "last_sent_step": self._last_sent_step,
            "last_loaded_step": self._last_loaded_step,
        }
        (save_dir / "prime_rl_backend_state.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )
        print(f"PrimeRL backend state saved: {save_dir}")
        return str(save_dir)

    def _parse_step(self, name: str) -> int | None:
        m = self._STEP_RE.match(name.strip())
        if not m:
            return None
        return int(m.group(1))

    def _sync_inference_weights(self, max_step: int) -> None:
        deadline = time.time() + self.sync_wait_s
        while True:
            step = self._latest_stable_step(max_step=max_step)
            if step is not None and step > self._last_loaded_step:
                self._update_inference_from_step(step)
                return
            if time.time() >= deadline:
                return
            time.sleep(self.sync_poll_s)

    def _latest_stable_step(self, max_step: int | None = None) -> int | None:
        candidates: list[int] = []
        root = self.output_dir / "broadcasts"
        if not root.is_dir():
            return None
        for entry in root.glob("step_*"):
            if not entry.is_dir():
                continue
            step_name = entry.name
            if not step_name.startswith("step_"):
                continue
            try:
                step = int(step_name.split("_", 1)[1])
            except ValueError:
                continue
            if max_step is not None and step > max_step:
                continue
            if (entry / "STABLE").exists():
                candidates.append(step)
        if not candidates:
            return None
        return max(candidates)

    def _candidate_weight_dir(self, step: int) -> Path | None:
        broadcast = self.output_dir / "broadcasts" / f"step_{step}"
        if broadcast.is_dir():
            return broadcast

        ckpt_weight = self.output_dir / "checkpoints" / f"step_{step}" / "weight"
        ckpt_stable = ckpt_weight.parent / "STABLE"
        if ckpt_weight.is_dir() and ckpt_stable.exists():
            return ckpt_weight
        return None

    def _update_inference_from_step(self, step: int) -> None:
        weight_dir = self._candidate_weight_dir(step)
        if weight_dir is None:
            raise FileNotFoundError(
                f"No PRIME-RL weight directory found for step {step} under {self.output_dir}."
            )
        self._post_json(
            f"{self._admin_base_url}/update_weights",
            {"weight_dir": str(weight_dir)},
        )
        self._last_loaded_step = step

    def _to_training_sample(
        self,
        tokens: list[int],
        logprobs: list[float],
        advantages: list[float],
    ) -> object | None:
        if not tokens:
            return None
        if not (len(tokens) == len(logprobs) == len(advantages)):
            n = min(len(tokens), len(logprobs), len(advantages))
            if n == 0:
                return None
            tokens = tokens[:n]
            logprobs = logprobs[:n]
            advantages = advantages[:n]

        prompt_len = 0
        for lp, adv in zip(logprobs, advantages):
            if abs(lp) <= self._PROMPT_PAD_EPS and abs(adv) <= self._PROMPT_PAD_EPS:
                prompt_len += 1
            else:
                break

        prompt_ids = list(tokens[:prompt_len])
        completion_ids = list(tokens[prompt_len:])
        completion_logprobs = [float(x) for x in logprobs[prompt_len:]]
        completion_advantages = [float(x) for x in advantages[prompt_len:]]

        if not completion_ids:
            return None

        adv_mean = self._reduce_advantages(completion_advantages)

        return self._TrainingSample(
            prompt_ids=prompt_ids,
            prompt_mask=[False] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[True] * len(completion_ids),
            completion_logprobs=completion_logprobs,
            completion_temperatures=[self._last_temperature] * len(completion_ids),
            advantage=adv_mean,
            reward=None,
            teacher_logprobs=None,
            pixel_values=None,
            image_grid_thw=None,
        )

    def _reduce_advantages(self, values: list[float]) -> float:
        if not values:
            return 0.0
        lo = min(values)
        hi = max(values)
        if self.strict_advantages and (hi - lo) > self._ADV_UNIFORM_EPS:
            raise RuntimeError(
                "PRIME-RL backend received non-uniform token advantages in one sample. "
                "PRIME-RL transport accepts one scalar advantage per sample. "
                "Use transform_mode='none' to keep uniform per-token advantages, "
                "or set [backend.options] strict_advantages = false to aggregate "
                "token advantages by mean."
            )
        return float(sum(values) / len(values))

    def _extract_completion(
        self,
        choice: dict[str, object],
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        # Path 1: PRIME-RL/verifiers token payload.
        token_block = choice.get("tokens")
        if isinstance(token_block, Mapping):
            token_payload = cast(Mapping[str, object], token_block)
            ids = token_payload.get("completion_ids")
            lps = token_payload.get("completion_logprobs")
            if isinstance(ids, list) and isinstance(lps, list):
                return (
                    [self._coerce_int(t) for t in ids],
                    [self._coerce_float(lp) for lp in lps],
                )

        # Path 2: OpenAI logprobs.content with token_id.
        content = None
        logprobs_obj = choice.get("logprobs")
        if isinstance(logprobs_obj, Mapping):
            logprobs_payload = cast(Mapping[str, object], logprobs_obj)
            content = logprobs_payload.get("content")
        if isinstance(content, list):
            ids: list[int] = []
            lps: list[float] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                item_payload = cast(Mapping[str, object], item)
                lp = item_payload.get("logprob")
                if lp is None:
                    continue
                lps.append(self._coerce_float(lp))
                tid = item_payload.get("token_id")
                if tid is not None:
                    ids.append(self._coerce_int(tid))
            if ids and len(ids) == len(lps):
                return ids, lps

        # Path 3: OpenAI logprobs token arrays.
        if isinstance(logprobs_obj, Mapping):
            logprobs_payload = cast(Mapping[str, object], logprobs_obj)
            token_lps = logprobs_payload.get("token_logprobs")
            tokens = logprobs_payload.get("tokens")
            if isinstance(token_lps, list) and isinstance(tokens, list):
                lps = [
                    0.0 if lp is None else self._coerce_float(lp)
                    for lp in token_lps
                ]
                choice_ids = choice.get("token_ids")
                if isinstance(choice_ids, list):
                    parsed_ids = [self._coerce_int(t) for t in choice_ids]
                    parsed_ids, lps = self._align_ids_and_logprobs(
                        parsed_ids, lps, prompt_ids
                    )
                    return parsed_ids, lps

        # Path 4: explicit id/logprob arrays.
        raw_ids = choice.get("completion_ids") or choice.get("token_ids")
        raw_lps = choice.get("completion_logprobs")
        if isinstance(raw_ids, list) and isinstance(raw_lps, list):
            ids2, lps2 = self._align_ids_and_logprobs(
                [self._coerce_int(t) for t in raw_ids],
                [self._coerce_float(lp) for lp in raw_lps],
                prompt_ids,
            )
            return ids2, lps2

        raise RuntimeError(
            "Could not parse token IDs/logprobs from PRIME-RL inference response. "
            "Ensure the server supports /v1/chat/completions/tokens with return_token_ids."
        )

    @staticmethod
    def _align_ids_and_logprobs(
        ids: list[int],
        lps: list[float],
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        # If ids include prompt+completion while logprobs only cover completion, keep suffix.
        if len(ids) > len(lps) and len(lps) > 0:
            ids = ids[-len(lps) :]
        # If ids include the full prompt prefix, strip it.
        if len(ids) >= len(prompt_ids) and ids[: len(prompt_ids)] == prompt_ids:
            ids = ids[len(prompt_ids) :]
            if len(lps) > len(ids):
                lps = lps[-len(ids) :]
        # Final trim to equal length.
        n = min(len(ids), len(lps))
        return ids[:n], lps[:n]

    @staticmethod
    def _coerce_int(value: object) -> int:
        try:
            return int(cast(str | int | float, value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Expected integer-like token id, got {value!r}."
            ) from exc

    @staticmethod
    def _coerce_float(value: object) -> float:
        try:
            return float(cast(str | int | float, value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Expected float-like logprob, got {value!r}."
            ) from exc

    def _post_json(self, url: str, payload: JSONPayload) -> JSONPayload:
        try:
            resp = self._session.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, Mapping):
                raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
            return dict(data)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"PRIME-RL request failed at {url}: {exc}") from exc
