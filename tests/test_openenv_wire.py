"""Real-socket wire-protocol test: OpenEnvClient against a live WS server.

The other openenv tests inject fake transports; this one exercises the real
``websockets`` connect path end-to-end. Skips when the optional dependency
is absent.
"""

from __future__ import annotations

import asyncio
import json

import pytest

websockets = pytest.importorskip("websockets")

from retrain.environments.openenv.client import OpenEnvClient  # noqa: E402


async def _serve_episode(connection) -> None:
    """One scripted gym episode: reset, then reward-1.0 done step."""
    async for raw in connection:
        message = json.loads(raw)
        if message["type"] == "reset":
            payload = {
                "observation": {"board": f"seed-{message['data'].get('seed')}"},
                "reward": None,
                "done": False,
            }
        elif message["type"] == "step":
            payload = {
                "observation": {"board": "solved"},
                "reward": 1.0,
                "done": True,
            }
        else:
            await connection.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": "unknown type", "code": "BAD_TYPE"},
                    }
                )
            )
            continue
        await connection.send(json.dumps({"type": message["type"], "data": payload}))


async def _run_against_live_server() -> tuple:
    from websockets.asyncio.server import serve

    async with serve(_serve_episode, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = OpenEnvClient(f"http://127.0.0.1:{port}")
        try:
            reset = await client.reset(seed=11)
            step = await client.step({"tool": "solve"})
            with pytest.raises(RuntimeError, match="BAD_TYPE"):
                await client._request({"type": "bogus"})
        finally:
            await client.close()
        return reset, step


def test_client_speaks_wire_protocol_over_real_socket():
    reset, step = asyncio.run(_run_against_live_server())
    assert reset.observation == {"board": "seed-11"}
    assert reset.reward is None and reset.done is False
    assert step.observation == {"board": "solved"}
    assert step.reward == 1.0 and step.done is True
