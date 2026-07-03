"""Client shims for verifiers environments."""

from __future__ import annotations

import types
from collections.abc import Callable

from retrain.environments import load as env_load


_NULL_CLIENT_MSG = (
    "retrain performs sampling via TrainHelper; the verifiers client must never be used"
)


def optional(
    require_fn: Callable[[], types.ModuleType] = env_load.require_verifiers,
) -> types.ModuleType | None:
    """Return the verifiers module when installed.

    Multi-turn rollouts are structural, so native providers such as OpenEnv do
    not require verifiers. Optional niceties like ``TrajectoryStep`` still come
    from verifiers when it is available.
    """
    try:
        return require_fn()
    except ImportError:
        return None


def make() -> object | None:
    """Inert client satisfying ``Environment.init_state``.

    retrain samples through ``TrainHelper.sample()``, never through a verifiers
    client. Newer verifiers validate the client argument in ``init_state``; this
    shim fails loudly if that sampling surface is ever reached.
    """
    try:
        from verifiers.clients import Client  # type: ignore[unresolved-import]
    except ImportError:
        return None

    class _RetrainNullClient(Client):
        def setup_client(self, config: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def to_native_tool(self, tool: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def to_native_prompt(self, messages: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def get_native_response(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def raise_from_native_response(self, response: object) -> None:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def from_native_response(self, response: object) -> object:
            raise NotImplementedError(_NULL_CLIENT_MSG)

        async def close(self) -> None:
            return None

    return _RetrainNullClient(None)
