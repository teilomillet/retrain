"""Minimal client for the OpenEnv WebSocket wire protocol.

retrain talks to an already-running OpenEnv server (started by the user or
an outer harness) and deliberately implements the wire protocol directly
instead of depending on ``openenv-core``: the verifiers extra pins
``openenv-core==0.2.1`` while current servers ship ``openenv`` 0.3.x under
the same import namespace, so a shared-venv dependency would force an
unresolvable pin. The wire protocol is the stable contract between those
versions (verified in quaero's integration record): JSON messages over a
WebSocket at ``<base>/ws``:

    -> {"type": "reset", "data": {...kwargs}}
    -> {"type": "step",  "data": {...action}}
    <- {"type": <str>,   "data": {"observation": ..., "reward": ..., "done": ...}}
    <- {"type": "error", "data": {"message": ..., "code": ...}}

The action schema is served over plain HTTP at ``<base>/schema``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast

from retrain.types import JSONObject


class Transport(Protocol):
    """The subset of a WebSocket connection the client uses."""

    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


ConnectFn = Callable[[str], Awaitable[Transport]]


@dataclass(frozen=True)
class StepResult:
    """One reset/step outcome from the environment server."""

    observation: object
    reward: float | None
    done: bool


def ws_url(base_url: str) -> str:
    """Convert an http(s)/ws(s) base URL to the ``/ws`` endpoint URL."""
    url = base_url.strip().rstrip("/")
    if url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    if not url.startswith(("ws://", "wss://")):
        raise ValueError(
            f"OpenEnv base URL must be http(s):// or ws(s)://, got '{base_url}'."
        )
    return f"{url}/ws"


def http_url(base_url: str) -> str:
    """Convert an http(s)/ws(s) base URL to its plain-HTTP form."""
    url = base_url.strip().rstrip("/")
    if url.startswith("ws://"):
        url = "http://" + url[len("ws://"):]
    elif url.startswith("wss://"):
        url = "https://" + url[len("wss://"):]
    return url


async def _default_connect(url: str) -> Transport:
    try:
        from websockets.asyncio.client import (
            connect,
        )
    except ImportError:
        raise ImportError(
            "OpenEnv environments require the websockets package.\n"
            "Install it with: pip install retrain[openenv]"
        ) from None
    return cast(Transport, await connect(url))


class OpenEnvClient:
    """One WebSocket conversation with an OpenEnv server.

    Each rollout owns its own client so episodes stay isolated; the server
    keys episode state to the connection.
    """

    def __init__(
        self,
        base_url: str,
        *,
        message_timeout_s: float = 300.0,
        connect: ConnectFn | None = None,
    ) -> None:
        self._url = ws_url(base_url)
        self._message_timeout_s = float(message_timeout_s)
        self._connect = connect or _default_connect
        self._transport: Transport | None = None

    async def connect(self) -> None:
        if self._transport is None:
            self._transport = await self._connect(self._url)

    async def reset(self, **kwargs: object) -> StepResult:
        return self._parse_result(
            await self._request({"type": "reset", "data": dict(kwargs)})
        )

    async def step(self, action: JSONObject) -> StepResult:
        return self._parse_result(
            await self._request({"type": "step", "data": dict(action)})
        )

    async def close(self) -> None:
        transport = self._transport
        self._transport = None
        if transport is not None:
            await transport.close()

    async def _request(self, message: JSONObject) -> JSONObject:
        await self.connect()
        transport = self._transport
        assert transport is not None
        await transport.send(json.dumps(message))
        raw = await asyncio.wait_for(
            transport.recv(),
            timeout=self._message_timeout_s,
        )
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise RuntimeError(
                f"OpenEnv server sent a non-object response: {response!r}"
            )
        response_obj = cast(JSONObject, response)
        if response_obj.get("type") == "error":
            data = response_obj.get("data")
            details = data if isinstance(data, dict) else {}
            raise RuntimeError(
                "OpenEnv server error: "
                f"{details.get('message', 'unknown error')} "
                f"(code: {details.get('code', 'UNKNOWN')})"
            )
        return response_obj

    @staticmethod
    def _parse_result(response: JSONObject) -> StepResult:
        data = response.get("data")
        payload = cast(JSONObject, data) if isinstance(data, dict) else {}
        raw_reward = payload.get("reward")
        reward = (
            float(cast(float, raw_reward))
            if isinstance(raw_reward, int | float) and not isinstance(raw_reward, bool)
            else None
        )
        return StepResult(
            observation=payload.get("observation", {}),
            reward=reward,
            done=bool(payload.get("done", False)),
        )


def fetch_action_schema(base_url: str, *, timeout_s: float = 10.0) -> JSONObject:
    """Fetch the env's action JSON schema from the HTTP ``/schema`` endpoint."""
    import requests

    try:
        response = requests.get(f"{http_url(base_url)}/schema", timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch OpenEnv action schema from '{base_url}/schema': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenEnv /schema response must be a JSON object.")
    action_schema = cast(JSONObject, payload).get("action", {})
    if not isinstance(action_schema, dict):
        raise RuntimeError("OpenEnv /schema response missing object `action` schema.")
    return cast(JSONObject, action_schema)
