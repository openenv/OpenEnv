# SPDX-License-Identifier: BSD-3-Clause

"""Provider cleanup retries must not lose ownership of an allocated resource."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from openenv.core.generic_client import GenericEnvClient
from websockets.protocol import State


class ResourceLedger:
    """Model a provider whose next start overwrites its sole resource handle."""

    def __init__(self, start_name, stop_name):
        self.events = []
        self.live_resources = set()
        self.current_resource = None
        self.allocations = 0
        self.start_error = None
        self.readiness_error = None
        self.stop_error = None
        self.provider = Mock(spec=[start_name, stop_name, "wait_for_ready"])
        getattr(self.provider, start_name).side_effect = self.start
        getattr(self.provider, stop_name).side_effect = self.stop
        self.provider.wait_for_ready.side_effect = self.wait_for_ready

    def start(self, *args, **kwargs):
        self.allocations += 1
        self.current_resource = self.allocations
        self.live_resources.add(self.current_resource)
        self.events.append(("start", self.current_resource))
        if self.start_error is not None:
            # A provider can acquire a resource before startup raises.
            raise self.start_error
        return f"http://localhost:{8000 + self.current_resource}"

    def wait_for_ready(self, *args):
        self.events.append(("ready", self.current_resource))
        if self.readiness_error is not None:
            raise self.readiness_error

    def stop(self):
        self.events.append(("stop", self.current_resource))
        if self.stop_error is not None:
            raise self.stop_error
        # An accidental repeated stop fails, as with non-idempotent providers.
        self.live_resources.remove(self.current_resource)


@pytest.fixture(params=[("start_container", "stop_container"), ("start", "stop")])
def ledger(request):
    return ResourceLedger(*request.param)


@pytest.fixture(params=["async", "sync"])
def execution_mode(request):
    return request.param


@pytest.fixture
def client_type():
    class SessionClient(GenericEnvClient):
        child_error = None

        def __init__(self, base_url=None, provider=None, **kwargs):
            if base_url is not None and self.child_error is not None:
                raise self.child_error
            super().__init__(base_url=base_url, provider=provider, **kwargs)

    return SessionClient


@pytest.fixture
def websocket_connect(monkeypatch, ledger):
    async def connect(*args, **kwargs):
        websocket = Mock(spec=["send", "recv", "close", "state"])
        websocket.state = State.OPEN
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "response",
                    "data": {
                        "observation": {"resource": ledger.current_resource},
                        "done": False,
                    },
                }
            )
        )
        websocket.close = AsyncMock()
        return websocket

    mocked_connect = AsyncMock(side_effect=connect)
    monkeypatch.setattr("openenv.core.env_client.ws_connect", mocked_connect)
    return mocked_connect


def is_container_provider(ledger):
    return hasattr(ledger.provider, "start_container")


def make_client(client_type, ledger, execution_mode):
    client = client_type(provider=ledger.provider)
    return client.sync() if execution_mode == "sync" else client


async def invoke(client, execution_mode, method, *args):
    result = getattr(client, method)(*args)
    return await result if execution_mode == "async" else result


def fail_setup(ledger, client_type, failure):
    error = ValueError(f"{failure} failed")
    if failure == "start":
        ledger.start_error = error
    elif failure == "readiness":
        ledger.readiness_error = error
    else:
        client_type.child_error = error
    return error


def allow_setup(ledger, client_type):
    ledger.start_error = None
    ledger.readiness_error = None
    client_type.child_error = None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness", "constructor"])
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
async def test_pending_cleanup_blocks_allocation_and_retries_stop_once(
    ledger, execution_mode, client_type, websocket_connect, failure, retry_method
):
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    cleanup_error = RuntimeError("resource is still running")
    ledger.stop_error = cleanup_error
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        assert ledger.events.count(("stop", 1)) == 1
        allow_setup(ledger, client_type)

        # Repeated user retries may retry stop, but never allocate or connect.
        for _ in range(2):
            previous_events = list(ledger.events)
            with pytest.raises(RuntimeError) as caught:
                await invoke(client, execution_mode, retry_method)
            assert caught.value is cleanup_error
            assert ledger.events == previous_events + [("stop", 1)]
            assert ledger.live_resources == {1}
            websocket_connect.assert_not_called()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness", "constructor"])
async def test_recovered_cleanup_stops_original_resource_before_restart(
    ledger, execution_mode, client_type, websocket_connect, failure
):
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    ledger.stop_error = RuntimeError("temporary cleanup failure")
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        allow_setup(ledger, client_type)
        ledger.stop_error = None
        previous_events = list(ledger.events)

        child = await invoke(client, execution_mode, "new_session")

        assert ledger.events == previous_events + [
            ("stop", 1),
            ("start", 2),
            ("ready", 2),
        ]
        assert ledger.live_resources == {2}
        assert child.base_url == "http://localhost:8002"
        websocket_connect.assert_awaited_once()
        await invoke(client, execution_mode, "close")
        await invoke(client, execution_mode, "close")
        assert ledger.events.count(("stop", 2)) == 1
        assert ledger.live_resources == set()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
async def test_failed_explicit_close_blocks_restart_until_cleanup_succeeds(
    ledger, execution_mode, client_type, websocket_connect, retry_method
):
    client = make_client(client_type, ledger, execution_mode)
    cleanup_error = RuntimeError("close failed")
    try:
        await invoke(client, execution_mode, "connect")
        ledger.stop_error = cleanup_error
        with pytest.raises(RuntimeError) as caught:
            await invoke(client, execution_mode, "close")
        assert caught.value is cleanup_error
        previous_events = list(ledger.events)

        with pytest.raises(RuntimeError) as caught:
            await invoke(client, execution_mode, retry_method)
        assert caught.value is cleanup_error
        assert ledger.events == previous_events + [("stop", 1)]
        assert websocket_connect.await_count == 1
        assert ledger.live_resources == {1}

        ledger.stop_error = None
        previous_events = list(ledger.events)
        await invoke(client, execution_mode, retry_method)
        assert ledger.events == previous_events + [
            ("stop", 1),
            ("start", 2),
            ("ready", 2),
        ]
        await invoke(client, execution_mode, "close")
        assert ledger.live_resources == set()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "readiness"])
async def test_connect_preserves_setup_error_when_cleanup_also_fails(
    ledger, execution_mode, client_type, websocket_connect, failure
):
    client = make_client(client_type, ledger, execution_mode)
    original_error = fail_setup(ledger, client_type, failure)
    ledger.stop_error = RuntimeError("cleanup failed")
    try:
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "connect")
        assert caught.value is original_error
        assert ledger.events.count(("stop", 1)) == 1
        assert ledger.live_resources == {1}
        websocket_connect.assert_not_called()
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
async def test_sibling_constructor_failure_keeps_live_child_usable(
    ledger, execution_mode, client_type, websocket_connect
):
    client = make_client(client_type, ledger, execution_mode)
    try:
        # The provider is owned by a parent with no WebSocket of its own.
        child = await invoke(client, execution_mode, "new_session")
        original_error = fail_setup(ledger, client_type, "constructor")
        with pytest.raises(ValueError) as caught:
            await invoke(client, execution_mode, "new_session")
        assert caught.value is original_error
        assert ledger.events == [("start", 1), ("ready", 1)]
        result = await invoke(child, execution_mode, "step", {})
        assert result.observation == {"resource": 1}
        websocket_connect.assert_awaited_once()
        await invoke(client, execution_mode, "close")
        assert ledger.live_resources == set()
    finally:
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
async def test_multiple_failure_recovery_cycles_leave_no_orphaned_resources(
    ledger, execution_mode, client_type, websocket_connect
):
    client = make_client(client_type, ledger, execution_mode)
    try:
        for cycle, failure in enumerate(["start", "readiness", "constructor"]):
            original_error = fail_setup(ledger, client_type, failure)
            ledger.stop_error = RuntimeError("cleanup temporarily unavailable")
            with pytest.raises(ValueError) as caught:
                await invoke(client, execution_mode, "new_session")
            assert caught.value is original_error
            allow_setup(ledger, client_type)
            ledger.stop_error = None

            await invoke(client, execution_mode, "new_session")
            current_resource = 2 * cycle + 2
            assert ledger.live_resources == {current_resource}
            await invoke(client, execution_mode, "close")
            await invoke(client, execution_mode, "close")
            assert ledger.live_resources == set()
            assert ledger.events.count(("stop", current_resource)) == 1
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_method", ["connect", "new_session"])
@pytest.mark.parametrize("cleanup_recovers", [False, True])
async def test_factory_pending_cleanup_requires_explicit_close(
    ledger, execution_mode, websocket_connect, retry_method, cleanup_recovers
):
    bootstrap = GenericEnvClient.from_env(
        "audit/test-environment",
        provider=ledger.provider,
        use_docker=is_container_provider(ledger),
    )
    client = bootstrap.sync() if execution_mode == "sync" else await bootstrap
    try:
        ledger.stop_error = RuntimeError("cleanup temporarily unavailable")
        with pytest.raises(RuntimeError, match="cleanup temporarily unavailable"):
            await invoke(client, execution_mode, "close")
        if cleanup_recovers:
            ledger.stop_error = None
        previous_events = list(ledger.events)
        websocket_connect.reset_mock()

        # A factory URL cannot be reused after an implicit cleanup retry.
        # Refuse before touching either the provider or the cached endpoint.
        for _ in range(2):
            with pytest.raises(RuntimeError, match=r"close\(\)"):
                await invoke(client, execution_mode, retry_method)
        assert ledger.events == previous_events
        assert ledger.live_resources == {1}
        websocket_connect.assert_not_called()

        ledger.stop_error = None
        await invoke(client, execution_mode, "close")
        await invoke(client, execution_mode, "close")
        assert ledger.events == previous_events + [("stop", 1)]
        assert not ledger.live_resources
    finally:
        ledger.stop_error = None
        await invoke(client, execution_mode, "close")


@pytest.fixture(params=["from_docker_image", "from_env"])
def factory(request, ledger):
    if request.param == "from_docker_image" and not is_container_provider(ledger):
        pytest.skip("from_docker_image requires a container provider")
    return request.param


def start_factory(factory, ledger, client_type=GenericEnvClient):
    if factory == "from_docker_image":
        return client_type.from_docker_image("fixture", provider=ledger.provider)
    return client_type.from_env(
        "audit/test-environment",
        provider=ledger.provider,
        use_docker=is_container_provider(ledger),
    )


@pytest.fixture
def pending_websocket_connect(monkeypatch):
    """Hold the initial WebSocket connection open until the test cancels it."""
    connecting = asyncio.Event()

    async def connect(*args, **kwargs):
        connecting.set()
        await asyncio.Future()

    monkeypatch.setattr("openenv.core.env_client.ws_connect", connect)
    return connecting


def start_factory_task(factory, ledger, client_type=GenericEnvClient):
    """Resolve a factory in a task, recording the cancellation it raises.

    Python 3.10 drops the cancellation message when `await task` re-raises,
    so the exception is captured inside the task instead.
    """
    raised = []

    async def resolve():
        try:
            return await start_factory(factory, ledger, client_type)
        except asyncio.CancelledError as exc:
            raised.append(exc)
            raise

    return asyncio.create_task(resolve()), raised


async def cancel_factory_during_connect(factory, ledger, connecting):
    task, raised = start_factory_task(factory, ledger)
    await asyncio.wait_for(connecting.wait(), timeout=5)
    task.cancel("factory cancelled")
    with pytest.raises(asyncio.CancelledError):
        await task
    return task, raised[0]


@pytest.mark.asyncio
async def test_factory_cancelled_during_connect_stops_provider_once(
    ledger, pending_websocket_connect, factory
):
    task, cancellation = await cancel_factory_during_connect(
        factory, ledger, pending_websocket_connect
    )

    assert cancellation.args == ("factory cancelled",)
    assert task.cancelled()
    assert ledger.events == [("start", 1), ("ready", 1), ("stop", 1)]
    assert ledger.live_resources == set()


@pytest.mark.asyncio
async def test_factory_cancelled_again_during_cleanup_keeps_original_cancellation(
    ledger, pending_websocket_connect, factory
):
    closing = asyncio.Event()

    class SuspendingCloseClient(GenericEnvClient):
        async def _close_async(self):
            # Teardown that suspends, so a second cancel can land inside it.
            closing.set()
            try:
                await asyncio.Future()
            finally:
                await super()._close_async()

    task, raised = start_factory_task(factory, ledger, SuspendingCloseClient)
    await asyncio.wait_for(pending_websocket_connect.wait(), timeout=5)
    task.cancel("factory cancelled")
    await asyncio.wait_for(closing.wait(), timeout=5)
    task.cancel("cancelled during cleanup")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert raised[0].args == ("factory cancelled",)
    assert task.cancelled()
    assert ledger.events == [("start", 1), ("ready", 1), ("stop", 1)]
    assert ledger.live_resources == set()


@pytest.mark.asyncio
async def test_factory_cancellation_survives_provider_cleanup_failure(
    ledger, pending_websocket_connect, factory
):
    ledger.stop_error = RuntimeError("cleanup failed")

    task, cancellation = await cancel_factory_during_connect(
        factory, ledger, pending_websocket_connect
    )

    # Cancellation is not translated into the cleanup error. The caller still
    # holds the provider and is responsible for the resource left running.
    assert cancellation.args == ("factory cancelled",)
    assert task.cancelled()
    assert ledger.events.count(("stop", 1)) == 1
    assert ledger.live_resources == {1}


@pytest.mark.asyncio
async def test_factory_connect_failure_stops_provider_once(
    ledger, monkeypatch, factory
):
    connect_error = OSError("connection refused")
    monkeypatch.setattr(
        "openenv.core.env_client.ws_connect", AsyncMock(side_effect=connect_error)
    )

    with pytest.raises(ConnectionError) as caught:
        await start_factory(factory, ledger)

    assert caught.value.__cause__ is connect_error
    assert ledger.events == [("start", 1), ("ready", 1), ("stop", 1)]
    assert ledger.live_resources == set()
