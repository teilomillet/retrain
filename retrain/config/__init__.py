"""Train configuration package."""

from __future__ import annotations

from retrain.config.constants import _DEFAULT_ADAPTER_PATH, _VALID_ENVIRONMENT_PROVIDERS
from retrain.config.load import load_config
from retrain.config.migrate import (
    BackendConfigMigrationResult,
    detect_legacy_prime_rl_backend_keys,
    migrate_legacy_backend_keys_toml_text,
)
from retrain.config.override import parse_cli_overrides
from retrain.config.schema import TrainConfig
from retrain.config.sections import (
    _CLI_FLAG_MAP,
    _FIELD_TYPES,
    _MAPPING_OVERRIDE_FIELDS,
    _TOML_MAP,
)
from retrain.config.squeeze import SqueezeConfig, load_squeeze_config

__all__ = [
    "BackendConfigMigrationResult",
    "SqueezeConfig",
    "TrainConfig",
    "detect_legacy_prime_rl_backend_keys",
    "load_config",
    "load_squeeze_config",
    "migrate_legacy_backend_keys_toml_text",
    "parse_cli_overrides",
]
