# SPDX-License-Identifier: BSD-3-Clause

"""Plain synchronous clients release their internally-created event-loop thread."""

from unittest.mock import AsyncMock, patch

import pytest
from openenv.core.generic_client import GenericEnvClient
from websockets.protocol import State


@pytest.fixture
def connected_client():
    socket = AsyncMock()
    socket.state = State.OPEN
    with patch(
        "openenv.core.env_client.ws_connect",
        new_callable=AsyncMock,
        return_value=socket,
    ):
        client = GenericEnvClient(base_url="http://localhost:9999")
        client.connect()
        try:
            yield client
        finally:
            client._sync_client._stop_loop()


def test_plain_close_stops_background_thread(connected_client):
    thread = connected_client._sync_client._loop_thread
    connected_client.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("raises", [False, True])
def test_plain_context_exit_stops_background_thread(raises):
    socket = AsyncMock()
    socket.state = State.OPEN
    client = GenericEnvClient(base_url="http://localhost:9999")
    try:
        with patch(
            "openenv.core.env_client.ws_connect",
            new_callable=AsyncMock,
            return_value=socket,
        ):
            try:
                with client:
                    thread = client._sync_client._loop_thread
                    if raises:
                        raise ValueError("body failed")
            except ValueError:
                if not raises:
                    raise
        assert not thread.is_alive()
    finally:
        client._sync_client._stop_loop()


def test_plain_close_failure_still_stops_background_thread(connected_client):
    thread = connected_client._sync_client._loop_thread
    with patch.object(
        connected_client,
        "_close_async",
        new_callable=AsyncMock,
        side_effect=ValueError("close failed"),
    ):
        with pytest.raises(ValueError, match="close failed"):
            connected_client.close()
    assert not thread.is_alive()


def test_repeated_plain_close_stops_every_background_thread(connected_client):
    first_thread = connected_client._sync_client._loop_thread
    assert connected_client.close() is None
    assert connected_client.close() is None
    assert not first_thread.is_alive()
    assert connected_client._sync_client._loop_thread is None
