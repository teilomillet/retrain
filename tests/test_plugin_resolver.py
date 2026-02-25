"""Tests for shared plugin resolver behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrain.plugin_resolver import (
    discover_plugin_modules,
    get_plugin_runtime,
    resolve_dotted_attribute,
    set_plugin_runtime,
)


class TestPluginResolver:
    def test_search_path_precedence(self, tmp_path, monkeypatch):
        (tmp_path / "plugins").mkdir()
        (tmp_path / "plugins" / "__init__.py").write_text("")
        (tmp_path / "plugins" / "my_mod.py").write_text(
            "def make(config=None):\n"
            "    return 'from_plugins'\n"
        )
        (tmp_path / "my_mod.py").write_text(
            "def make(config=None):\n"
            "    return 'from_root'\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        set_plugin_runtime(["plugins"], strict=True)

        resolved = resolve_dotted_attribute(
            "my_mod.make",
            selector="transform_mode",
            expected="callable",
        )
        assert resolved.resolved_module == "plugins.my_mod"
        assert callable(resolved.obj)

    def test_resolver_cache_and_runtime_reset(self, tmp_path, monkeypatch):
        import sys

        (tmp_path / "plugins").mkdir()
        (tmp_path / "plugins" / "__init__.py").write_text("")
        (tmp_path / "plugins" / "cached_mod.py").write_text(
            "def make(config=None):\n"
            "    return 'x'\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("plugins", None)
        sys.modules.pop("plugins.cached_mod", None)
        set_plugin_runtime(["plugins"], strict=True)
        first = resolve_dotted_attribute(
            "cached_mod.make",
            selector="advantage_mode",
            expected="callable",
        )
        second = resolve_dotted_attribute(
            "cached_mod.make",
            selector="advantage_mode",
            expected="callable",
        )
        assert first is second

        set_plugin_runtime(["plugins"], strict=False)
        cfg = get_plugin_runtime()
        assert cfg.strict is False

    def test_standardized_import_error_message(self):
        set_plugin_runtime(["plugins"], strict=True)
        with pytest.raises(ImportError, match="Failed to resolve transform_mode plugin"):
            resolve_dotted_attribute(
                "definitely_not_real.mod",
                selector="transform_mode",
                expected="TransformSpec or callable",
            )

    def test_discovery_lists_python_modules(self, tmp_path, monkeypatch):
        (tmp_path / "plugins").mkdir()
        (tmp_path / "plugins" / "a.py").write_text("x = 1\n")
        (tmp_path / "plugins" / "_ignore.py").write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        found = discover_plugin_modules(["plugins"])
        assert "plugins.a" in found
        assert all(not m.endswith("._ignore") for m in found)
