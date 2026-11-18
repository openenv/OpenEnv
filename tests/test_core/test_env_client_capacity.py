# SPDX-License-Identifier: BSD-3-Clause

"""A server at capacity refuses a session with an error frame the client must surface."""

from unittest.mock import AsyncMock, patch

import pytest
from openenv.core.env_server.exceptions import SessionCapacityError
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.generic_client import GenericEnvClient


class _Action(Action):
    message: str = ""


class _Environment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs) -> Observation:
        return Observation()

    def step(self, action: _Action) -> Observation:
        return Observation()

    @property
    def state(self) -> State:
        return State()


@pytest.fixture
def server_url(serve):
    return serve(create_app(_Environment, _Action, Observation, max_concurrent_envs=1))


async def test_capacity_error_survives_close_before_send(server_url):
    async with GenericEnvClient(base_url=server_url) as holder:
        await holder.reset()

        refused = GenericEnvClient(base_url=server_url)
        await refused.connect()
        try:
            # The server sends CAPACITY_REACHED and closes. Make the close land
            # between the client's reconnect check and its send, as it does
            # under load: the send fails but the error frame is still queued.
            await refused._ws.wait_closed()
            with patch.object(refused, "_ensure_connected", AsyncMock()):
                with pytest.raises(RuntimeError, match="CAPACITY_REACHED") as exc:
                    await refused.reset()
            assert "1/1 sessions active" in str(exc.value)
            # The refused client also learns which server setting to raise.
            assert "MAX_CONCURRENT_ENVS" in str(exc.value)
        finally:
            await refused.close()


def test_capacity_error_names_the_setting_to_raise():
    message = str(SessionCapacityError(active_sessions=1, max_sessions=1))

    assert "1/1 sessions active" in message
    assert "max_concurrent_envs" in message
    assert "MAX_CONCURRENT_ENVS" in message
    assert "SUPPORTS_CONCURRENT_SESSIONS=True" in message


def test_capacity_error_keeps_a_custom_message():
    error = SessionCapacityError(active_sessions=2, max_sessions=2, message="busy")

    assert str(error) == "busy"
    assert (error.active_sessions, error.max_sessions) == (2, 2)
