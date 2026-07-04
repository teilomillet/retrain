"""Guard behavior for the hf_transfer auto-enable (retrain/io/hub.py)."""

from __future__ import annotations

import importlib.util
import os

import pytest

import retrain.io.hub as hft

FLAG = "HF_HUB_ENABLE_HF_TRANSFER"


@pytest.fixture(autouse=True)
def _isolate_flag():
    saved = os.environ.get(FLAG)
    os.environ.pop(FLAG, None)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved


def _pretend_installed(monkeypatch, present: bool):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "hf_transfer":
            return object() if present else None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_enables_when_present_and_unset(monkeypatch):
    _pretend_installed(monkeypatch, True)
    assert hft.enable_if_available() is True
    assert os.environ[FLAG] == "1"


def test_noop_when_absent(monkeypatch):
    _pretend_installed(monkeypatch, False)
    assert hft.enable_if_available() is False
    assert FLAG not in os.environ


def test_respects_explicit_off(monkeypatch):
    os.environ[FLAG] = "0"
    _pretend_installed(monkeypatch, True)  # present, but the user said off
    assert hft.enable_if_available() is False
    assert os.environ[FLAG] == "0"


def test_respects_explicit_on_even_without_package(monkeypatch):
    os.environ[FLAG] = "1"
    _pretend_installed(monkeypatch, False)
    assert hft.enable_if_available() is False
    assert os.environ[FLAG] == "1"
