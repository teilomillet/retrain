"""Tests for retrain migrate-config CLI command."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from retrain.cli import _run_migrate_config


_LEGACY_CFG = """\
[backend]
backend = "prime_rl"
adapter_path = "adapters/test"
prime_rl_transport = "zmq"
prime_rl_zmq_port = 7777
"""


def test_preview_prints_diff_and_does_not_modify_file(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg)])
    out = capsys.readouterr().out

    assert "--- " in out
    assert "+++ " in out
    assert "prime_rl_transport" in out
    assert "[backend.options]" in out
    assert cfg.read_text() == _LEGACY_CFG


def test_check_returns_1_when_migration_is_required(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    with pytest.raises(SystemExit) as exc_info:
        _run_migrate_config([str(cfg), "--check"])
    assert exc_info.value.code == 1


def test_check_returns_0_when_config_is_already_clean(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[backend]\nbackend = "prime_rl"\n\n[backend.options]\ntransport = "filesystem"\n'
    )

    _run_migrate_config([str(cfg), "--check"])
    out = capsys.readouterr().out
    assert "No migration needed" in out


def test_write_updates_file_in_place(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg), "--write"])
    text = cfg.read_text()

    assert "prime_rl_transport" not in text
    assert "prime_rl_zmq_port" not in text
    assert "[backend.options]" in text
    assert 'transport = "zmq"' in text
    assert "zmq_port = 7777" in text


def test_output_writes_target_only(tmp_path):
    cfg = tmp_path / "config.toml"
    out_cfg = tmp_path / "migrated.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg), "--output", str(out_cfg)])

    assert cfg.read_text() == _LEGACY_CFG
    migrated = out_cfg.read_text()
    assert "[backend.options]" in migrated
    assert "zmq_port = 7777" in migrated


def test_existing_backend_options_take_precedence(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[backend]\n"
        'backend = "prime_rl"\n'
        'prime_rl_transport = "filesystem"\n'
        "prime_rl_zmq_port = 7777\n\n"
        "[backend.options]\n"
        'transport = "zmq"\n'
    )

    _run_migrate_config([str(cfg), "--write"])
    text = cfg.read_text()

    assert 'transport = "zmq"' in text
    assert "zmq_port = 7777" in text
    assert "prime_rl_transport" not in text


def test_malformed_usage_errors(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    with pytest.raises(SystemExit) as exc_info:
        _run_migrate_config([])
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        _run_migrate_config([str(cfg), "--check", "--write"])
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        _run_migrate_config([str(cfg), "--nope"])
    assert exc_info.value.code == 1

    err = capsys.readouterr().err
    assert "Unknown migrate-config flag" in err or "Usage:" in err


def test_json_report_contains_migration_payload(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg), "--json"])
    out = capsys.readouterr().out
    assert '"needs_migration": true' in out
    assert '"legacy_keys"' in out


def test_write_with_backup_creates_bak(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg), "--write", "--backup"])

    backup = Path(str(cfg) + ".bak")
    assert backup.is_file()
    assert "prime_rl_transport" in backup.read_text()
    assert "prime_rl_transport" not in cfg.read_text()


def test_stdin_stdout_mode(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(_LEGACY_CFG))
    _run_migrate_config(["--stdin", "--stdout"])
    out = capsys.readouterr().out
    assert "[backend.options]" in out
    assert "prime_rl_transport" not in out


def test_stdin_check_exit_code(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(_LEGACY_CFG))
    with pytest.raises(SystemExit) as exc_info:
        _run_migrate_config(["--stdin", "--check"])
    assert exc_info.value.code == 1


def test_migration_is_idempotent(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_LEGACY_CFG)

    _run_migrate_config([str(cfg), "--write"])
    once = cfg.read_text()
    _run_migrate_config([str(cfg), "--write"])
    twice = cfg.read_text()
    assert once == twice
