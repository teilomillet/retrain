"""PRIME-RL backend constructor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, cast

from retrain.backends.options import normalize_option_schema, prime_rl_option_schema

if TYPE_CHECKING:
    from retrain.backends import TrainHelper
    from retrain.config import TrainConfig


class PrimeRLOptions(TypedDict):
    transport: str
    zmq_host: str
    zmq_port: int
    zmq_hwm: int
    strict_advantages: bool
    sync_wait_s: int
    sync_poll_s: float


def _normalize_prime_rl_options(raw_options: Mapping[str, object]) -> PrimeRLOptions:
    options = normalize_option_schema("prime_rl", raw_options, prime_rl_option_schema())
    return {
        "transport": cast(str, options["transport"]),
        "zmq_host": cast(str, options["zmq_host"]),
        "zmq_port": cast(int, options["zmq_port"]),
        "zmq_hwm": cast(int, options["zmq_hwm"]),
        "strict_advantages": cast(bool, options["strict_advantages"]),
        "sync_wait_s": cast(int, options["sync_wait_s"]),
        "sync_poll_s": cast(float, options["sync_poll_s"]),
    }


def create_prime_rl(config: "TrainConfig") -> "TrainHelper":
    try:
        from retrain.backends.prime import PrimeRLTrainHelper
    except ImportError:
        raise RuntimeError(
            "Backend 'prime_rl' requires PRIME-RL.\n"
            "Install it with: pip install prime-rl"
        ) from None

    options = _normalize_prime_rl_options(config.backend_options)
    inference_url = config.inference_url or config.base_url or "http://localhost:8000"

    return PrimeRLTrainHelper(
        model_name=config.model,
        output_dir=config.adapter_path,
        inference_url=inference_url,
        transport_type=options["transport"],
        zmq_host=options["zmq_host"],
        zmq_port=options["zmq_port"],
        zmq_hwm=options["zmq_hwm"],
        strict_advantages=options["strict_advantages"],
        sync_wait_s=options["sync_wait_s"],
        sync_poll_s=options["sync_poll_s"],
    )
