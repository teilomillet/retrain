"""User-provided ``module:function`` reward."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Coroutine
from typing import SupportsFloat, SupportsIndex, cast


class CustomReward:
    """Loads a user-provided ``module:function`` as the reward function."""

    def __init__(self, module_path: str, function_name: str) -> None:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, function_name, None)
        if fn is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{function_name}'"
            )
        self._fn = fn

    def score(self, response: str, reference: str) -> float:
        result = self._fn(response, reference)
        if asyncio.iscoroutine(result):
            result = asyncio.run(cast(Coroutine[object, object, object], result))
        return _float_score(result)


def _float_score(value: object) -> float:
    if isinstance(value, str):
        return float(value)
    return float(cast(SupportsFloat | SupportsIndex, value))
