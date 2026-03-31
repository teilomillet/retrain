"""Ordeal ChaosTest for retrain.registry.

Stateful testing of the Registry component: registration via decorator,
creation, override, dotted-path fallback, and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import hypothesis.strategies as st
from ordeal import ChaosTest, always, invariant, rule, sometimes

from retrain.registry import Registry


# ── Test-scoped helpers ──


@dataclass
class FakeConfig:
    """Minimal config stub for registry factories."""

    name: str = "test"


def make_factory(tag: str):
    """Create a factory that returns a tagged string."""

    def factory(config: FakeConfig) -> str:
        return f"{tag}:{config.name}"

    return factory


# ── ChaosTest ──


class RegistryChaos(ChaosTest):
    """Stateful chaos test for the Registry component.

    Tests registration (decorator API), creation, override semantics,
    and error handling for unknown names.
    """

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.registry: Registry[str] = Registry("chaos_test")
        self.registered: dict[str, str] = {}  # name → latest tag
        self.config = FakeConfig()

    @rule(
        name=st.text(min_size=1, max_size=8, alphabet="abcdefgh"),
        tag=st.text(min_size=1, max_size=4, alphabet="wxyz"),
    )
    def register_factory(self, name: str, tag: str) -> None:
        """Register a factory via the decorator API."""
        # Registry.register(name) returns a decorator
        decorator = self.registry.register(name)
        decorator(make_factory(tag))
        self.registered[name] = tag

    @rule(name=st.text(min_size=1, max_size=8, alphabet="abcdefgh"))
    def create_registered(self, name: str) -> None:
        """Create from a name that may or may not be registered."""
        if name in self.registered:
            result = self.registry.create(name, self.config)
            expected_tag = self.registered[name]
            always(
                result == f"{expected_tag}:{self.config.name}",
                "factory output matches latest registered tag",
            )
        else:
            try:
                self.registry.create(name, self.config)
                always(False, "unregistered name should raise")
            except ValueError:
                pass  # Expected

    @rule(
        name=st.text(min_size=1, max_size=8, alphabet="abcdefgh"),
        new_tag=st.text(min_size=1, max_size=4, alphabet="1234"),
    )
    def override_factory(self, name: str, new_tag: str) -> None:
        """Override a factory — latest registration wins."""
        decorator = self.registry.register(name)
        decorator(make_factory(new_tag))
        self.registered[name] = new_tag

        result = self.registry.create(name, self.config)
        always(
            result == f"{new_tag}:{self.config.name}",
            "override produces new tag output",
        )

    @invariant()
    def registered_names_in_builtins(self) -> None:
        """All registered names appear in builtin_names."""
        builtins = set(self.registry.builtin_names)
        for name in self.registered:
            assert name in builtins

    @invariant()
    def builtin_names_sorted(self) -> None:
        """builtin_names is always sorted."""
        names = self.registry.builtin_names
        assert names == sorted(names)

    def teardown(self) -> None:
        super().teardown()


TestRegistryChaos = RegistryChaos.TestCase
