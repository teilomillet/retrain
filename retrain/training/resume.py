"""Resume contract labels shared by runners and status surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

RESUME_UNSUPPORTED = "unsupported"
RESUME_ADAPTER_ONLY = "adapter_only"
RESUME_EXACT = "exact"
RESUME_MODES = {RESUME_UNSUPPORTED, RESUME_ADAPTER_ONLY, RESUME_EXACT}

ADAPTER_ONLY_RESUME_WARNING = (
    "resume_mode=adapter_only: restores trainer counters and checkpoint "
    "weights, but not optimizer/scaler/RNG state; optimizer dynamics re-warm "
    "after resume."
)
UNSUPPORTED_RESUME_WARNING = (
    "resume_mode=unsupported: this backend does not report checkpoint resume "
    "support."
)
RUNTIME_DEPENDENT_EXACT_RESUME_WARNING = (
    "resume_mode=exact is runtime-dependent: verify this backend runtime and "
    "SDK can load checkpoint state before relying on exact continuation."
)


class ResumeCapabilityLike(Protocol):
    supports_checkpoint_resume: bool
    resume_runtime_dependent: bool
    checkpoint_resume_mode: str


@dataclass(frozen=True)
class ResumeContract:
    """User-visible semantics for continuing from a saved checkpoint."""

    mode: str
    warning: str = ""

    @property
    def exact(self) -> bool:
        return self.mode == RESUME_EXACT


def normalize_resume_mode(raw: object, *, supports_resume: bool = True) -> str:
    """Normalize backend-provided resume mode text."""
    if not supports_resume:
        return RESUME_UNSUPPORTED
    if not isinstance(raw, str):
        return RESUME_ADAPTER_ONLY
    mode = raw.strip().lower()
    if mode in RESUME_MODES:
        return mode
    return RESUME_ADAPTER_ONLY


def _contract_for_mode(
    mode: str,
    *,
    resume_runtime_dependent: bool,
) -> ResumeContract:
    if mode == RESUME_ADAPTER_ONLY:
        return ResumeContract(mode=mode, warning=ADAPTER_ONLY_RESUME_WARNING)
    if mode == RESUME_UNSUPPORTED:
        return ResumeContract(mode=mode, warning=UNSUPPORTED_RESUME_WARNING)
    if resume_runtime_dependent:
        return ResumeContract(
            mode=mode,
            warning=RUNTIME_DEPENDENT_EXACT_RESUME_WARNING,
        )
    return ResumeContract(mode=mode)


def contract_for_capabilities(caps: ResumeCapabilityLike) -> ResumeContract:
    """Build the checkpoint resume contract from backend capabilities."""
    mode = normalize_resume_mode(
        caps.checkpoint_resume_mode,
        supports_resume=caps.supports_checkpoint_resume,
    )
    return _contract_for_mode(
        mode,
        resume_runtime_dependent=caps.resume_runtime_dependent,
    )


def contract_for_capability_payload(
    payload: Mapping[str, object],
) -> ResumeContract:
    """Build the checkpoint resume contract from a serialized capability payload."""
    supports_resume = payload.get("supports_checkpoint_resume") is True
    mode = normalize_resume_mode(
        payload.get("checkpoint_resume_mode"),
        supports_resume=supports_resume,
    )
    return _contract_for_mode(
        mode,
        resume_runtime_dependent=payload.get("resume_runtime_dependent") is True,
    )
