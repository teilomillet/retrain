"""Regression checks for the repository-level Ordeal configuration."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

from ordeal import ChaosTest


_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_TARGET_MODULES = {
    "retrain.registry",
    "retrain.training.backpressure",
    "retrain.training.sepa",
    "retrain.training.state",
    "tests.test_ordeal_backend_workflow",
}


def _load_ordeal_config() -> dict[str, object]:
    return tomllib.loads((_ROOT / "ordeal.toml").read_text(encoding="utf-8"))


def test_ordeal_target_modules_import() -> None:
    config = _load_ordeal_config()
    explorer = config["explorer"]
    assert isinstance(explorer, dict)
    target_modules = explorer["target_modules"]
    assert isinstance(target_modules, list)
    assert _REQUIRED_TARGET_MODULES <= set(target_modules)

    for module_name in target_modules:
        assert isinstance(module_name, str)
        importlib.import_module(module_name)


def test_ordeal_state_machines_resolve_and_include_registry() -> None:
    config = _load_ordeal_config()
    configured_tests = config["tests"]
    assert isinstance(configured_tests, list)
    class_paths: set[str] = set()

    for entry in configured_tests:
        assert isinstance(entry, dict)
        class_path = entry["class"]
        assert isinstance(class_path, str)
        class_paths.add(class_path)
        module_name, class_name = class_path.rsplit(":", 1)
        state_machine = getattr(importlib.import_module(module_name), class_name)
        assert issubclass(state_machine, ChaosTest)

    assert "tests.test_ordeal_registry:RegistryChaos" in class_paths
