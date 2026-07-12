"""Parse the [campaign] TOML section: seeds, settings, and conditions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import cast

from retrain.campaign.model import CampaignCondition
from retrain.config import TrainConfig
from retrain.config.constants import _MAX_REPRODUCIBLE_SEED

DEFAULT_CONDITIONS: list[tuple[str, str]] = [
    ("grpo", "none"),
    ("maxrl", "none"),
    ("maxrl", "gtpo"),
    ("maxrl", "gtpo_hicra"),
    ("maxrl", "gtpo_sepa"),
]

DEFAULT_SEEDS: list[int] = [42, 101, 202, 303, 404, 505, 606, 707]

_CONDITION_EXTRA_KEYS = frozenset({"advantage_mode", "transform_mode"})
_TRAIN_CONFIG_FIELDS = frozenset(f.name for f in fields(TrainConfig))


def object_dict(value: object, name: str) -> dict[str, object]:
    """Return a typed object mapping for TOML tables."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a TOML table")
    return cast(dict[str, object], value)


def optional_object_dict(value: object, name: str) -> dict[str, object] | None:
    if value is None:
        return None
    return object_dict(value, name)


def int_from_object(raw: object, name: str) -> int:
    if isinstance(raw, (str, bytes, bytearray, int, float)):
        return int(raw)
    raise ValueError(f"{name} must be int-coercible")


def float_from_object(raw: object, name: str) -> float:
    if isinstance(raw, (str, bytes, bytearray, int, float)):
        return float(raw)
    raise ValueError(f"{name} must be float-coercible")


def campaign_int(campaign: Mapping[str, object], key: str, default: int) -> int:
    raw = campaign.get(key, default)
    try:
        return int_from_object(raw, f"campaign.{key}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"campaign.{key} must be int-coercible") from exc


def campaign_float(campaign: Mapping[str, object], key: str, default: float) -> float:
    raw = campaign.get(key, default)
    try:
        return float_from_object(raw, f"campaign.{key}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"campaign.{key} must be float-coercible") from exc


def campaign_seeds(campaign: Mapping[str, object]) -> list[int]:
    raw = campaign.get("seeds")
    if raw is None:
        return list(DEFAULT_SEEDS)
    if not isinstance(raw, list):
        raise ValueError("campaign.seeds must be a list of integers")
    seeds: list[int] = []
    for idx, seed in enumerate(raw):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"campaign.seeds[{idx}] must be an integer")
        if seed < -1 or seed > _MAX_REPRODUCIBLE_SEED:
            raise ValueError(
                f"campaign.seeds[{idx}] must be -1 (random) or between 0 and "
                f"{_MAX_REPRODUCIBLE_SEED} inclusive"
            )
        seeds.append(seed)
    return seeds


def _validate_condition_config(
    *,
    advantage_mode: str,
    transform_mode: str,
    overrides: Mapping[str, object],
) -> None:
    unknown = sorted(set(overrides) - _TRAIN_CONFIG_FIELDS)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown TrainConfig field(s): {joined}")

    cfg = TrainConfig(
        advantage_mode=advantage_mode,
        transform_mode=transform_mode,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.__post_init__()


def parse_campaign_conditions(
    raw_conditions: object, campaign_path: str
) -> list[CampaignCondition]:
    """Parse campaign conditions and fail fast on malformed entries."""
    if not raw_conditions:
        return [
            CampaignCondition(advantage_mode=a, transform_mode=t)
            for a, t in DEFAULT_CONDITIONS
        ]
    if not isinstance(raw_conditions, list):
        raise ValueError(
            f"campaign.conditions must be a list in {campaign_path}"
        )

    conditions: list[CampaignCondition] = []
    for idx, raw_condition in enumerate(raw_conditions):
        if not isinstance(raw_condition, dict):
            raise ValueError(
                f"campaign.conditions[{idx}] must be a table in {campaign_path}"
            )
        condition = object_dict(
            raw_condition,
            f"campaign.conditions[{idx}]",
        )

        adv_mode = condition.get("advantage_mode")
        tx_mode = condition.get("transform_mode")
        if not isinstance(adv_mode, str) or not adv_mode:
            raise ValueError(
                f"campaign.conditions[{idx}].advantage_mode must be a non-empty string in {campaign_path}"
            )
        if not isinstance(tx_mode, str) or not tx_mode:
            raise ValueError(
                f"campaign.conditions[{idx}].transform_mode must be a non-empty string in {campaign_path}"
            )

        overrides = {
            k: v for k, v in condition.items()
            if k not in _CONDITION_EXTRA_KEYS
        }

        try:
            _validate_condition_config(
                advantage_mode=adv_mode,
                transform_mode=tx_mode,
                overrides=overrides,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid campaign condition at index {idx} in {campaign_path}: {exc}"
            ) from exc

        conditions.append(
            CampaignCondition(
                advantage_mode=adv_mode,
                transform_mode=tx_mode,
                overrides=overrides,
            )
        )

    return conditions
