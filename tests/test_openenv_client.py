"""Wire-protocol tests for the OpenEnv client (fake transport, no network)."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from retrain.environments.openenv.client import (
    OpenEnvClient,
    StepResult,
    _default_connect,
    http_url,
    ws_url,
)


class _FakeTransport:
    def __init__(self, responses: list[dict]) -> None:
        self.sent: list[dict] = []
        self._responses = list(responses)
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        return json.dumps(self._responses.pop(0))

    async def close(self) -> None:
        self.closed = True


def _client(responses: list[dict]) -> tuple[OpenEnvClient, _FakeTransport]:
    transport = _FakeTransport(responses)

    async def connect(url: str) -> _FakeTransport:
        del url
        return transport

    return OpenEnvClient("http://localhost:8765", connect=connect), transport


class TestUrls:
    def test_ws_url_converts_http_and_appends_ws(self):
        assert ws_url("http://host:1234") == "ws://host:1234/ws"
        assert ws_url("https://host") == "wss://host/ws"
        assert ws_url("ws://host/") == "ws://host/ws"

    def test_ws_url_rejects_other_schemes(self):
        with pytest.raises(ValueError, match="http"):
            ws_url("ftp://host")

    def test_http_url_converts_ws_schemes(self):
        assert http_url("ws://host:1") == "http://host:1"
        assert http_url("wss://host") == "https://host"
        assert http_url("http://host") == "http://host"


class TestRequests:
    def test_reset_sends_typed_message_and_parses_result(self):
        client, transport = _client(
            [{"type": "reset", "data": {"observation": {"x": 1}, "reward": None, "done": False}}]
        )
        result = asyncio.run(client.reset(seed=7))
        assert transport.sent == [{"type": "reset", "data": {"seed": 7}}]
        assert result == StepResult(observation={"x": 1}, reward=None, done=False)

    def test_step_sends_action_and_parses_reward(self):
        client, transport = _client(
            [{"type": "step", "data": {"observation": "ok", "reward": 1.5, "done": True}}]
        )
        result = asyncio.run(client.step({"tool": "run"}))
        assert transport.sent == [{"type": "step", "data": {"tool": "run"}}]
        assert result == StepResult(observation="ok", reward=1.5, done=True)

    def test_boolean_reward_is_not_a_reward(self):
        # JSON true would satisfy isinstance(x, int); guard against it.
        client, _ = _client(
            [{"type": "step", "data": {"observation": {}, "reward": True, "done": False}}]
        )
        assert asyncio.run(client.step({})).reward is None

    def test_server_error_raises_with_message_and_code(self):
        client, _ = _client(
            [{"type": "error", "data": {"message": "bad action", "code": "E42"}}]
        )
        with pytest.raises(RuntimeError, match="bad action.*E42"):
            asyncio.run(client.step({}))

    def test_close_is_idempotent_and_closes_transport(self):
        client, transport = _client(
            [{"type": "reset", "data": {"observation": {}, "done": False}}]
        )
        asyncio.run(client.reset())
        asyncio.run(client.close())
        asyncio.run(client.close())
        assert transport.closed


class TestDefaultConnect:
    def test_default_connect_disables_websocket_keepalive(self, monkeypatch):
        transport = _FakeTransport([])
        calls: list[tuple[str, dict[str, object]]] = []

        async def connect(url: str, **kwargs: object) -> _FakeTransport:
            calls.append((url, kwargs))
            return transport

        module = types.ModuleType("websockets.asyncio.client")
        module.connect = connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "websockets.asyncio.client", module)

        result = asyncio.run(_default_connect("ws://localhost:8771/ws"))

        assert result is transport
        assert calls == [("ws://localhost:8771/ws", {"ping_interval": None})]

    def test_default_connect_keeps_older_websockets_compatible(self, monkeypatch):
        transport = _FakeTransport([])
        calls: list[tuple[str, dict[str, object]]] = []

        async def connect(url: str, **kwargs: object) -> _FakeTransport:
            calls.append((url, kwargs))
            if kwargs:
                raise TypeError("unexpected keyword argument")
            return transport

        module = types.ModuleType("websockets.asyncio.client")
        module.connect = connect  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "websockets.asyncio.client", module)

        result = asyncio.run(_default_connect("ws://localhost:8771/ws"))

        assert result is transport
        assert calls == [
            ("ws://localhost:8771/ws", {"ping_interval": None}),
            ("ws://localhost:8771/ws", {}),
        ]
