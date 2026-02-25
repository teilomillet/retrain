"""Tests for plugin scaffolding CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrain.cli import _run_init_plugin, _run_plugins


class TestInitPlugin:
    def test_scaffold_transform_plugin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init_plugin(["--kind", "transform", "--name", "my_transform"])
        p = tmp_path / "plugins" / "my_transform.py"
        assert p.is_file()
        content = p.read_text()
        assert "TransformOutput" in content
        assert "def my_transform" in content

    def test_scaffold_with_test(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_init_plugin(
            ["--kind", "advantage", "--name", "my_adv", "--with-test"]
        )
        assert (tmp_path / "plugins" / "my_adv.py").is_file()
        assert (tmp_path / "tests" / "test_my_adv_plugin.py").is_file()

    def test_refuse_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "my_adv.py").write_text("existing\n")
        with pytest.raises(SystemExit):
            _run_init_plugin(["--kind", "advantage", "--name", "my_adv"])

    def test_invalid_kind_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            _run_init_plugin(["--kind", "unknown", "--name", "x"])


class TestPluginsCommand:
    def test_plugins_json(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "toy.py").write_text("def x():\n    return 1\n")

        _run_plugins(["--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "builtins" in payload
        assert "algorithm_mode" in payload["builtins"]
        modules = {m["module"] for m in payload["discovered"]}
        assert "plugins.toy" in modules
