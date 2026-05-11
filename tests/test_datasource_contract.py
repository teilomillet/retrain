"""Contract tests for custom DataSource plugins.

Verifies that any class registered as a data_source satisfies:
  1. factory(config) — accepts TrainConfig as first argument
  2. .load() — returns a non-empty list of Example instances
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrain.config import TrainConfig
from retrain.data import Example


def _make_config(**kwargs) -> TrainConfig:
    return TrainConfig(**kwargs)


def _assert_datasource_contract(instance: object) -> None:
    assert hasattr(instance, "load") and callable(getattr(instance, "load")), (
        f"{type(instance).__name__} does not implement the DataSource protocol "
        "(missing callable .load() method)"
    )


# ---------------------------------------------------------------------------
# SROIEDataSource
# ---------------------------------------------------------------------------

_FAKE_HF_ROW = {
    "words": ["BOOK", "TA", "25/12/2018", "TOTAL", "9.00"],
    "entities": {
        "company": "BOOK TA",
        "date": "25/12/2018",
        "address": "NO.53 JALAN SAGU",
        "total": "9.00",
    },
}


def _fake_load_dataset(*args, **kwargs):
    return [_FAKE_HF_ROW]


def test_sroie_datasource_accepts_train_config():
    """factory(config) must not raise even though config is not an int."""
    from campaigns.sroie_dataset import SROIEDataSource

    cfg = _make_config()
    instance = SROIEDataSource(cfg)
    _assert_datasource_contract(instance)


def test_sroie_datasource_load_returns_examples():
    from campaigns.sroie_dataset import SROIEDataSource

    cfg = _make_config()
    try:
        import datasets  # noqa: F401
    except ImportError:
        pytest.skip("datasets not installed")

    with patch("datasets.load_dataset", _fake_load_dataset):
        examples = SROIEDataSource(cfg).load()

    assert isinstance(examples, list)
    assert len(examples) > 0
    assert all(isinstance(e, Example) for e in examples)
    first = examples[0]
    assert isinstance(first.prompt, list)
    assert isinstance(first.reference, str)
    assert "company" in first.reference
