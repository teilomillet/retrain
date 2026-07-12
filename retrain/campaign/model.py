"""Campaign data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict


class CampaignRun(TypedDict):
    """Manifest record for one condition/seed run."""

    condition: str
    advantage_mode: str
    transform_mode: str
    overrides: dict[str, object]
    seed: int
    run_name: str
    log_dir: str
    config_path: NotRequired[str]
    returncode: NotRequired[int]


@dataclass(frozen=True)
class CampaignCondition:
    """A single campaign condition: required modes + optional overrides."""

    advantage_mode: str
    transform_mode: str
    overrides: dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        base = f"{self.advantage_mode}+{self.transform_mode}"
        if not self.overrides:
            return base
        suffix = ",".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))
        return f"{base}~{suffix}"

    def as_legacy_tuple(self) -> tuple[str, str]:
        return (self.advantage_mode, self.transform_mode)
