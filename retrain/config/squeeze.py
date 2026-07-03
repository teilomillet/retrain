"""LoRA-Squeeze config loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field


@dataclass
class SqueezeConfig:
    """Configuration for LoRA-Squeeze rank analysis and compression."""

    adapter_path: str = ""
    source_rank: int = 0  # 0 = fallback to [model].lora_rank
    target_ranks: list[int] = field(default_factory=list)  # [] = auto power-of-2
    min_variance_retention: float = 0.95
    output_path: str = ""
    compress_to: int = 0  # 0 = use recommended rank
    device: str = "cpu"


def load_squeeze_config(path: str) -> SqueezeConfig:
    """Load squeeze config from a TOML file with a [squeeze] section.

    Falls back to [model].lora_rank for source_rank when not specified.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    sq = data.get("squeeze", {})
    if not sq:
        raise ValueError(f"No [squeeze] section in {path}")

    adapter_path = sq.get("adapter_path", "")
    if not adapter_path:
        raise ValueError("[squeeze].adapter_path is required")

    source_rank = int(sq.get("source_rank", 0))
    if source_rank == 0:
        # Fallback to [model].lora_rank
        model_sec = data.get("model", {})
        source_rank = int(model_sec.get("lora_rank", 0))

    target_ranks_raw = sq.get("target_ranks", [])
    target_ranks = [int(r) for r in target_ranks_raw]

    return SqueezeConfig(
        adapter_path=adapter_path,
        source_rank=source_rank,
        target_ranks=target_ranks,
        min_variance_retention=float(sq.get("min_variance_retention", 0.95)),
        output_path=str(sq.get("output_path", "")),
        compress_to=int(sq.get("compress_to", 0)),
        device=str(sq.get("device", "cpu")),
    )
