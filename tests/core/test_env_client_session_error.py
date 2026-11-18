"""EnvClient treats a SESSION_ERROR frame (a reaped session) as a lost session."""

import socket
import threading
import time

import pytest
import uvicorn
from echo_env.server.echo_environment import EchoEnvironment
from fastapi import FastAPI
from openenv.core import GenericEnvClient
from openenv.core.env_client import SessionLostError
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import ConcurrencyConfig


def echo(message):
    return {
        "type": "call_tool",
        "tool_name": "echo_message",
        "arguments": {"message": message},
    }


@pytest.fixture
def reaping_server():
    """An echo server whose idle sessions are reaped after 1 s.

    The reaper polls every `max(timeout / 4, 5)` seconds, so its first check
    runs 5 s after startup.
    """
    server = HTTPEnvServer(
        EchoEnvironment,
        CallToolAction,
        CallToolObservation,
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=1, session_timeout=1.0
        ),
    )
    app = FastAPI()
    server.register_routes(app)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    uv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", server
    finally:
        uv.should_exit = True
        thread.join(timeout=10)


def test_reaped_session_raises_session_lost_until_reset(reaping_server):
    base_url, server = reaping_server
    with GenericEnvClient(base_url=base_url).sync() as client:
        client.reset()
        client.step(echo("before"))

        deadline = time.monotonic() + 10
        while server.active_sessions and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.active_sessions == 0

        # The server answers with SESSION_ERROR and closes the socket.
        with pytest.raises(SessionLostError, match="SESSION_ERROR"):
            client.step(echo("lost"))
        # A retried step does not reconnect into a session it never reset.
        with pytest.raises(SessionLostError):
            client.step(echo("lost again"))

        # The next episode reconnects as before.
        client.reset()
        client.step(echo("after"))
        assert client.state()["step_count"] == 1
