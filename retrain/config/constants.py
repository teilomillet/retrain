"""Shared config constants."""

from __future__ import annotations

_VALID_ENVIRONMENT_PROVIDERS = {"", "verifiers", "openenv"}
_DEFAULT_ADAPTER_PATH = "/tmp/retrain_adapter"
_MAX_REPRODUCIBLE_SEED = (1 << 32) - 1
