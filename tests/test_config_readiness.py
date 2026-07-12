"""Fail-closed identity checks for readiness-bound runtime configs."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from retrain.config import load_config
from retrain.config.readiness import assert_readiness_runtime_matches_file


def _write_bound_config(path: Path, *, seed: int = 123) -> None:
    args = json.dumps({"readiness_config": str(path)})
    path.write_text(
        "[environment]\n"
        'provider = "verifiers"\n'
        'id = "quaero-dbt"\n'
        f"args = {json.dumps(args)}\n"
        "\n[training]\n"
        f"seed = {seed}\n"
    )


def test_bound_runtime_matches_exact_loaded_toml(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run.toml"
    _write_bound_config(path)
    config = load_config(str(path))
    monkeypatch.setenv("RETRAIN_CONFIG_PATH", str(path))

    assert_readiness_runtime_matches_file(config)


def test_bound_runtime_rejects_in_memory_override(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run.toml"
    _write_bound_config(path)
    config = replace(load_config(str(path)), seed=124)
    monkeypatch.setenv("RETRAIN_CONFIG_PATH", str(path))

    with pytest.raises(RuntimeError, match="seed"):
        assert_readiness_runtime_matches_file(config)


def test_bound_runtime_rejects_file_change_after_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "run.toml"
    _write_bound_config(path)
    config = load_config(str(path))
    _write_bound_config(path, seed=124)
    monkeypatch.setenv("RETRAIN_CONFIG_PATH", str(path))

    with pytest.raises(RuntimeError, match="seed"):
        assert_readiness_runtime_matches_file(config)


def test_bound_runtime_requires_cli_path(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "run.toml"
    _write_bound_config(path)
    config = load_config(str(path))
    monkeypatch.delenv("RETRAIN_CONFIG_PATH", raising=False)

    with pytest.raises(RuntimeError, match="requires RETRAIN_CONFIG_PATH"):
        assert_readiness_runtime_matches_file(config)
