"""Training data sources."""

from __future__ import annotations

from retrain.data.math import MathDataSource
from retrain.data.source import DataSource, Example

__all__ = [
    "DataSource",
    "Example",
    "MathDataSource",
]
