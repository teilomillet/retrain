from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from retrain.process_metrics import process_max_rss_mb


@pytest.fixture
def fake_resource(monkeypatch):
    def install(raw_rss: int) -> None:
        module = SimpleNamespace(
            RUSAGE_SELF=object(),
            getrusage=lambda _target: SimpleNamespace(ru_maxrss=raw_rss),
        )
        monkeypatch.setitem(sys.modules, "resource", module)

    return install


def test_process_max_rss_mb_uses_kib_on_linux(monkeypatch, fake_resource) -> None:
    fake_resource(2048)
    monkeypatch.setattr(sys, "platform", "linux")

    assert process_max_rss_mb() == pytest.approx(2.0)


def test_process_max_rss_mb_uses_bytes_on_darwin(monkeypatch, fake_resource) -> None:
    fake_resource(2 * 1024 * 1024)
    monkeypatch.setattr(sys, "platform", "darwin")

    assert process_max_rss_mb() == pytest.approx(2.0)


def test_process_max_rss_mb_clamps_nonpositive_values(monkeypatch, fake_resource) -> None:
    fake_resource(0)
    monkeypatch.setattr(sys, "platform", "linux")

    assert process_max_rss_mb() == 0.0
