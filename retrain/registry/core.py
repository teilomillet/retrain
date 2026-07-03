"""Dict-based registry for one component type.

Stores lazy factory functions ``(config) -> object`` keyed by short name.
Unknown names containing a ``.`` are treated as dotted-path imports
(``mypackage.module.MyClass``), enabling third-party plugins with zero
boilerplate.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar, cast

from retrain.config import TrainConfig
from retrain.plugins.resolve import resolve_dotted_attribute

T = TypeVar("T")


class Registry(Generic[T]):
    """Dict-based registry for a single component type.

    Stores *lazy factory functions* ``(config: TrainConfig) -> object`` keyed
    by short name (e.g. ``"local"``, ``"match"``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[[TrainConfig], T]] = {}

    # -- public API --------------------------------------------------------

    def register(
        self, name: str
    ) -> Callable[[Callable[[TrainConfig], T]], Callable[[TrainConfig], T]]:
        """Decorator to register a lazy factory under *name*."""
        def decorator(
            fn: Callable[[TrainConfig], T],
        ) -> Callable[[TrainConfig], T]:
            self._factories[name] = fn
            return fn
        return decorator

    def create(self, name: str, config: TrainConfig) -> T:
        """Look up *name* and call its factory with *config*.

        Falls back to dotted-path import when *name* contains a ``.``.
        """
        factory = self._factories.get(name)
        if factory is not None:
            return factory(config)

        # Dotted-path fallback: ``mypackage.module.ClassName``
        if "." in name:
            return self._import_dotted(name, config)

        available = sorted(self._factories)
        raise ValueError(
            f"Unknown {self.kind} '{name}'. "
            f"Built-in options: {available}. "
            f"For a third-party plugin, use a dotted import path "
            f"(e.g. 'mypackage.MyClass')."
        )

    @property
    def builtin_names(self) -> list[str]:
        """Sorted list of registered built-in names."""
        return sorted(self._factories)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _import_dotted(dotted: str, config: TrainConfig) -> T:
        """Import ``module.attr`` and call ``attr(config)``."""
        resolved = resolve_dotted_attribute(
            dotted,
            selector="registry component",
            expected="a callable factory(config)",
        )
        factory = resolved.obj
        if not callable(factory):
            raise TypeError(
                f"Dotted import target '{dotted}' is not callable."
            )
        typed_factory = cast(Callable[[TrainConfig], T], factory)
        return typed_factory(config)
