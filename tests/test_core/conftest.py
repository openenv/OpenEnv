# SPDX-License-Identifier: BSD-3-Clause

import socket
import threading
import time

import pytest
import uvicorn


@pytest.fixture
def serve():
    """Serve an ASGI app on a free local port and return its URL."""
    servers = []

    def start(app) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(
            target=lambda: server.run(sockets=[sock]), daemon=True
        )
        thread.start()
        servers.append((server, thread))
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)
