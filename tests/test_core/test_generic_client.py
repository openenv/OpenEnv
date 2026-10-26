# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for GenericEnvClient and GenericAction
===================================================

Tests cover:
1. GenericEnvClient instantiation and basic operations
2. Dictionary-based action/observation handling
3. from_docker_image() inheritance
4. from_env() inheritance (HuggingFace registry)
5. AutoEnv integration with skip_install parameter
6. Comparison with typed clients
7. GenericAction class
8. AutoAction with skip_install parameter
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openenv.core.client_types import StepResult
from openenv.core.env_client import _is_localhost_ws_url
from openenv.core.generic_client import GenericAction, GenericEnvClient
from openenv.core.sync_client import SyncEnvClient
from websockets.protocol import State


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.recv.return_value = '{"type": "response", "data": {"observation": {"output": "hello"}, "reward": 1.0, "done": false}}'
    return ws


@pytest.fixture
def mock_provider():
    """Create a mock container provider."""
    provider = Mock()
    provider.start_container.return_value = "http://localhost:8000"
    provider.wait_for_ready.return_value = None
    return provider


# ============================================================================
# GenericEnvClient Unit Tests
# ============================================================================


@pytest.fixture(params=[("start_container", "stop_container"), ("start", "stop")])
def session_provider(request):
    start_name, stop_name = request.param
    provider = Mock(spec=[start_name, stop_name, "wait_for_ready"])
    start = getattr(provider, start_name)
    stop = getattr(provider, stop_name)
    start.return_value = "http://localhost:8000"
    return provider, start, stop


class FailingSessionClient(GenericEnvClient):
    def __init__(self, base_url=None, provider=None, **kwargs):
        if base_url is not None:
            raise ValueError("child constructor failed")
        super().__init__(base_url=base_url, provider=provider, **kwargs)


class TestSessionProviderCleanup:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["readiness", "constructor"])
    async def test_failed_session_is_cleaned_up_only_once(
        self, session_provider, failure
    ):
        provider, start, stop = session_provider
        stop.side_effect = [None, RuntimeError("already stopped")]
        if failure == "readiness":
            error = TimeoutError("not ready")
            provider.wait_for_ready.side_effect = error
            client = GenericEnvClient(provider=provider)
        else:
            client = FailingSessionClient(provider=provider)

        with patch("openenv.core.env_client.ws_connect", AsyncMock()) as connect:
            with pytest.raises((TimeoutError, ValueError)) as caught:
                await client.new_session()

        if failure == "readiness":
            assert caught.value is error
        else:
            assert str(caught.value) == "child constructor failed"
        start.assert_called_once_with()
        stop.assert_called_once_with()
        connect.assert_not_called()
        assert client.base_url is None
        assert client._ws_url is None
        assert client._child_clients == []
        await client.close()
        await client.close()
        stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_failed_cleanup_can_be_retried(self, session_provider):
        provider, start, stop = session_provider
        error = TimeoutError("not ready")
        provider.wait_for_ready.side_effect = error
        stop.side_effect = [RuntimeError("cleanup failed"), None]
        client = GenericEnvClient(provider=provider)
        with pytest.raises(TimeoutError) as caught:
            await client.new_session()
        assert caught.value is error
        assert client._provider is provider
        stop.assert_called_once_with()
        await client.close()
        assert stop.call_count == 2
        await client.close()
        assert stop.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_parent", [False, True])
    async def test_constructor_failure_preserves_running_provider(
        self, session_provider, connect_parent
    ):
        provider, start, stop = session_provider
        parent = FailingSessionClient(provider=provider)
        ws = AsyncMock()
        with patch("openenv.core.env_client.ws_connect", AsyncMock(return_value=ws)):
            if connect_parent:
                await parent.connect()
            else:
                parent._start_provider_if_needed()
            with pytest.raises(ValueError, match="child constructor failed"):
                await parent.new_session()
            start.assert_called_once_with()
            stop.assert_not_called()
            assert parent.base_url == "http://localhost:8000"
            if connect_parent:
                assert parent._ws is ws
                ws.close.assert_not_called()
            await parent.close()
        stop.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_provider_can_restart_after_successful_cleanup(
        self, session_provider
    ):
        provider, start, stop = session_provider
        provider.wait_for_ready.side_effect = TimeoutError("not ready")
        parent = GenericEnvClient(provider=provider)
        with pytest.raises(TimeoutError):
            await parent.new_session()
        stop.assert_called_once_with()
        provider.wait_for_ready.side_effect = None
        with patch(
            "openenv.core.env_client.ws_connect",
            AsyncMock(return_value=AsyncMock()),
        ):
            child = await parent.new_session()
            assert child.base_url == "http://localhost:8000"
            assert start.call_count == 2
            await parent.close()
        assert stop.call_count == 2

    @pytest.mark.parametrize("failure", ["readiness", "constructor"])
    def test_sync_session_failure_is_cleaned_up_only_once(
        self, session_provider, failure
    ):
        provider, start, stop = session_provider
        stop.side_effect = [None, RuntimeError("already stopped")]
        if failure == "readiness":
            provider.wait_for_ready.side_effect = TimeoutError("not ready")
            parent = GenericEnvClient(provider=provider)
        else:
            parent = FailingSessionClient(provider=provider)
        client = parent.sync()
        try:
            with pytest.raises((TimeoutError, ValueError)):
                client.new_session()
            stop.assert_called_once_with()
        finally:
            client.close()
        stop.assert_called_once_with()


class TestGenericEnvClientInstantiation:
    """Test GenericEnvClient instantiation."""

    def test_instantiation_with_http_url(self):
        """Test that GenericEnvClient can be instantiated with HTTP URL."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_https_url(self):
        """Test that GenericEnvClient can be instantiated with HTTPS URL."""
        client = GenericEnvClient(base_url="https://example.com")
        assert client._ws_url == "wss://example.com/ws"

    def test_instantiation_with_ws_url(self):
        """Test that GenericEnvClient can be instantiated with WS URL."""
        client = GenericEnvClient(base_url="ws://localhost:8000")
        assert client._ws_url == "ws://localhost:8000/ws"

    def test_instantiation_with_custom_timeouts(self):
        """Test custom timeout parameters."""
        client = GenericEnvClient(
            base_url="http://localhost:8000",
            connect_timeout_s=30.0,
            message_timeout_s=120.0,
        )
        assert client._connect_timeout == 30.0
        assert client._message_timeout == 120.0

    def test_instantiation_with_provider_only(self, mock_provider):
        """Test provider-owned startup without an initial base URL."""
        client = GenericEnvClient(provider=mock_provider)
        assert client._ws_url is None

    def test_instantiation_requires_base_url_or_provider(self):
        """Test that clients need either a URL or a provider."""
        with pytest.raises(ValueError, match="requires either base_url or provider"):
            GenericEnvClient()

    @pytest.mark.asyncio
    async def test_connect_starts_provider_when_base_url_omitted(self, mock_provider):
        """The outer client can start a provider that owns its image/source."""
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()

        with patch("openenv.core.env_client.ws_connect", AsyncMock(return_value=ws)):
            client = GenericEnvClient(provider=mock_provider)
            await client.connect()

            mock_provider.start_container.assert_called_once_with()
            mock_provider.wait_for_ready.assert_called_once_with(
                "http://localhost:8000"
            )
            assert client._ws_url == "ws://localhost:8000/ws"

            await client.close()

        mock_provider.stop_container.assert_called_once_with()
        assert client._ws_url is None

    @pytest.mark.asyncio
    async def test_provider_owned_startup_requires_no_arg_start_container(self):
        """Providers that require an image fail with actionable guidance."""

        class _RequiresImageProvider:
            def start_container(self, image):
                raise AssertionError("start_container should not be called")

            def wait_for_ready(self, base_url):
                raise AssertionError("wait_for_ready should not be called")

            def stop_container(self):
                pass

        client = GenericEnvClient(provider=_RequiresImageProvider())

        with pytest.raises(ValueError, match=r"start_container\(\) requires image"):
            await client.connect()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_parent", [False, True])
    async def test_new_session_reuses_provider_server(
        self, mock_provider, connect_parent
    ):
        """Child sessions connect to the same server without owning the provider."""
        websockets = []

        async def fake_ws_connect(*args, **kwargs):
            ws = AsyncMock()
            websockets.append(ws)
            return ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            client = GenericEnvClient(provider=mock_provider)
            if connect_parent:
                await client.connect()
            session = await client.new_session()

            assert isinstance(session, GenericEnvClient)
            assert session is not client
            assert session._provider is None
            assert session._base_url == "http://localhost:8000"
            assert session._ws_url == "ws://localhost:8000/ws"
            assert len(websockets) == 1 + int(connect_parent)
            mock_provider.start_container.assert_called_once_with()

            await client.close()

        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "start_method,stop_method",
        [("start_container", "stop_container"), ("start", "stop")],
    )
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_new_session_stops_provider_when_readiness_fails(
        self, start_method, stop_method, cleanup_fails
    ):
        provider = Mock(spec=[start_method, stop_method, "wait_for_ready"])
        start = getattr(provider, start_method)
        stop = getattr(provider, stop_method)
        start.return_value = "http://localhost:8000"
        error = TimeoutError("environment never became ready")
        provider.wait_for_ready.side_effect = error
        if cleanup_fails:
            stop.side_effect = RuntimeError("cleanup failed")
        client = GenericEnvClient(provider=provider)

        with patch("openenv.core.env_client.ws_connect", AsyncMock()) as connect:
            with pytest.raises(TimeoutError) as exc_info:
                await client.new_session()

        assert exc_info.value is error
        start.assert_called_once_with()
        stop.assert_called_once_with()
        connect.assert_not_called()
        assert client.base_url is None
        assert client._child_clients == []

    @pytest.mark.asyncio
    async def test_close_stops_provider_when_child_close_raises(self, mock_provider):
        """Parent cleanup continues even when a child session close fails."""
        client = GenericEnvClient(provider=mock_provider)
        child = MagicMock()
        child.close = AsyncMock(side_effect=RuntimeError("child close failed"))
        client._child_clients.append(child)

        await client.close()

        child.close.assert_awaited_once_with()
        assert client._child_clients == []
        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_cancelled_child_close_still_closes_all_children_and_parent(self):
        """Cancellation waits for every child before parent teardown."""

        class FakeRuntimeProvider:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        first_close_started = asyncio.Event()
        release_first_close = asyncio.Event()
        second_close_completed = asyncio.Event()

        class FirstChild:
            def __init__(self):
                self.closed = False

            async def close(self):
                first_close_started.set()
                await release_first_close.wait()
                self.closed = True

        class SecondChild:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True
                second_close_completed.set()

        class ParentSocket:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def close(self):
                self.state = State.CLOSED

        provider = FakeRuntimeProvider()
        client = GenericEnvClient(provider=provider)
        client._base_url = "http://localhost:8000"
        client._ws_url = "ws://localhost:8000/ws"
        parent_ws = ParentSocket()
        client._ws = parent_ws
        client._ws_loop = asyncio.get_running_loop()
        first_child = FirstChild()
        second_child = SecondChild()
        client._child_clients.extend([first_child, second_child])

        close_call = asyncio.create_task(client._close_async())
        await first_close_started.wait()
        close_call.cancel()

        await asyncio.wait_for(second_close_completed.wait(), timeout=1)
        assert not close_call.done()
        assert parent_ws.state != State.CLOSED
        assert not provider.stopped

        release_first_close.set()
        close_results = await asyncio.wait_for(
            asyncio.gather(close_call, return_exceptions=True), timeout=1
        )

        assert len(close_results) == 1
        assert isinstance(close_results[0], asyncio.CancelledError)
        assert first_child.closed
        assert second_child.closed
        assert client._child_clients == []
        assert client._ws is None
        assert parent_ws.state == State.CLOSED
        assert provider.stopped
        assert client._base_url is None
        assert client._ws_url is None

    def test_session_client_filters_constructor_kwargs(self):
        """Child creation respects subclasses with narrower __init__ signatures."""

        class MinimalGenericClient(GenericEnvClient):
            def __init__(self, base_url):
                super().__init__(base_url=base_url)

        client = MinimalGenericClient(base_url="http://localhost:8000")
        child = client._create_session_client()

        assert isinstance(child, MinimalGenericClient)
        assert child._base_url == "http://localhost:8000"


class TestGenericEnvClientStepPayload:
    """Test _step_payload method."""

    def test_step_payload_passthrough(self):
        """Test that action dict passes through unchanged."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {"code": "print('hello')", "timeout": 30}
        payload = client._step_payload(action)

        assert payload == action
        assert payload["code"] == "print('hello')"
        assert payload["timeout"] == 30

    def test_step_payload_empty_dict(self):
        """Test with empty action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {}
        payload = client._step_payload(action)

        assert payload == {}

    def test_step_payload_nested_dict(self):
        """Test with nested action dict."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        action = {
            "command": "execute",
            "params": {"file": "test.py", "args": ["--verbose"]},
        }
        payload = client._step_payload(action)

        assert payload == action
        assert payload["params"]["file"] == "test.py"


class TestGenericEnvClientParseResult:
    """Test _parse_result method."""

    def test_parse_result_full_payload(self):
        """Test parsing a complete result payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"stdout": "hello", "stderr": ""},
            "reward": 1.5,
            "done": True,
        }
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {"stdout": "hello", "stderr": ""}
        assert result.reward == 1.5
        assert result.done is True

    def test_parse_result_minimal_payload(self):
        """Test parsing a minimal payload with defaults."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        assert result.observation == {}
        assert result.reward is None
        assert result.done is False

    def test_parse_result_missing_reward(self):
        """Test parsing payload without reward."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {"observation": {"data": "test"}, "done": False}
        result = client._parse_result(payload)

        assert result.observation == {"data": "test"}
        assert result.reward is None
        assert result.done is False


class TestGenericEnvClientParseState:
    """Test _parse_state method."""

    def test_parse_state_full_payload(self):
        """Test parsing a complete state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "episode_id": "ep-123",
            "step_count": 5,
            "custom_field": "value",
        }
        state = client._parse_state(payload)

        assert state == payload
        assert state["episode_id"] == "ep-123"
        assert state["step_count"] == 5

    def test_parse_state_empty_payload(self):
        """Test parsing empty state payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {}
        state = client._parse_state(payload)

        assert state == {}


class TestGenericEnvClientFromDockerImage:
    """Test from_docker_image class method."""

    @pytest.mark.asyncio
    async def test_from_docker_image_creates_client(self, mock_provider):
        """Test that from_docker_image creates a connected client."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with("coding-env:latest")
            mock_provider.wait_for_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_docker_image_with_env_vars(self, mock_provider):
        """Test from_docker_image with environment variables."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
                env_vars={"DEBUG": "1"},
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once_with(
                "coding-env:latest", env_vars={"DEBUG": "1"}
            )


class TestGenericEnvClientFromEnv:
    """Test from_env class method (HuggingFace registry)."""

    @pytest.mark.asyncio
    async def test_from_env_with_docker(self, mock_provider):
        """Test from_env with use_docker=True pulls from HF registry."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = await GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            )

            assert isinstance(client, GenericEnvClient)
            # Should construct HF registry URL
            mock_provider.start_container.assert_called_once()
            call_args = mock_provider.start_container.call_args
            assert "registry.hf.space/user-my-env" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_from_env_with_docker_normalizes_image_name(self, mock_provider):
        """Registry image names follow the Space subdomain (lowercase, '_' -> '-')."""
        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            await GenericEnvClient.from_env(
                "openenv/Echo_Env",
                use_docker=True,
                provider=mock_provider,
            )

            image = mock_provider.start_container.call_args[0][0]
            assert image == "registry.hf.space/openenv-echo-env:latest"

    @pytest.mark.asyncio
    async def test_from_env_uv_wait_uses_remaining_context_timeout(self):
        """A slow `start()` must consume part of the UV readiness budget."""
        provider = Mock()
        provider.context_timeout_s = 10.0
        provider.start.return_value = "http://localhost:8000"
        provider.wait_for_ready.return_value = None

        with (
            patch.object(GenericEnvClient, "connect", new_callable=AsyncMock),
            patch("openenv.core.env_client.time.monotonic", side_effect=[100.0, 103.5]),
        ):
            client = await GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            )

        assert isinstance(client, GenericEnvClient)
        provider.wait_for_ready.assert_called_once_with(timeout_s=6.5)


class TestSyncBootstrapConstructors:
    """Test the synchronous `.sync()` bootstrap chain on the client factories.

    `from_docker_image(...).sync()` / `from_env(...).sync()` let a synchronous
    consumer (e.g. a TRL GRPO rollout loop) bootstrap a container without an
    event loop and without hand-rolling provider.start_container() /
    wait_for_ready(), while `await from_docker_image(...)` keeps the async path.
    See issue #935.
    """

    @staticmethod
    def _fake_ws_connect():
        """A ws_connect stand-in that records every socket it opens."""
        sockets = []

        async def _connect(*args, **kwargs):
            ws = AsyncMock()
            sockets.append(ws)
            return ws

        return _connect, sockets

    def test_from_docker_image_is_lazy_until_resolved(self, mock_provider):
        """The factory returns a handle and starts nothing until driven/.sync()'d."""
        handle = GenericEnvClient.from_docker_image(
            image="coding-env:latest",
            provider=mock_provider,
        )
        # Nothing is started at call time — the container starts only on resolve.
        mock_provider.start_container.assert_not_called()
        # Clean up the undriven coroutine to avoid a "never awaited" warning.
        handle.close()

    def test_bootstrap_handle_is_single_use(self, mock_provider):
        """Resolving a handle twice raises instead of silently double-starting."""
        fake_connect, _ = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            handle = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )
            client = handle.sync()
            with pytest.raises(RuntimeError, match="already been resolved"):
                handle.sync()
            client.close()

        # Only one container was ever started.
        mock_provider.start_container.assert_called_once()

    def test_sync_bootstrap_stops_provider_if_loop_setup_fails(self, mock_provider):
        """If sync setup fails after the container starts, the provider is released.

        Exercises the failure window in `.sync()` before `_connect_async` (and its
        own cleanup) runs — here `_run_sync` itself raises.
        """
        with patch.object(
            GenericEnvClient, "_run_sync", side_effect=RuntimeError("loop boom")
        ):
            with pytest.raises(RuntimeError, match="loop boom"):
                GenericEnvClient.from_docker_image(
                    image="coding-env:latest",
                    provider=mock_provider,
                ).sync()

        mock_provider.start_container.assert_called_once()
        mock_provider.stop_container.assert_called_once_with()

    def test_factory_handle_is_coroutine_compatible(self, mock_provider):
        """The handle is a real coroutine, so asyncio.run / run_async_safely accept it.

        The previous async factories returned a coroutine; the handle must remain a
        drop-in so `asyncio.run(from_docker_image(...))` and
        `run_async_safely(from_docker_image(...))` keep working (regression guard).
        """
        import asyncio as _asyncio

        from openenv.core.utils import run_async_safely

        fake_connect, _ = self._fake_ws_connect()

        # asyncio.run(...) strictly requires a coroutine; a bare awaitable is rejected.
        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            handle = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            )
            assert _asyncio.iscoroutine(handle)
            client = _asyncio.run(handle)
            assert isinstance(client, GenericEnvClient)

        # run_async_safely (asyncio.run under the hood) must accept it too.
        fake_connect, _ = self._fake_ws_connect()
        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client2 = run_async_safely(
                GenericEnvClient.from_docker_image(
                    image="coding-env:latest",
                    provider=mock_provider,
                )
            )
            assert isinstance(client2, GenericEnvClient)

    def test_from_docker_image_sync_returns_connected_client(self, mock_provider):
        """from_docker_image(...).sync() returns a connected SyncEnvClient (no await)."""
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            ).sync()

            # A concrete sync wrapper, consistent with the instance .sync().
            assert isinstance(client, SyncEnvClient)
            mock_provider.start_container.assert_called_once_with("coding-env:latest")
            mock_provider.wait_for_ready.assert_called_once()
            # Connection was actually established on the sync background loop.
            assert len(sockets) == 1
            # base_url proxies through to the wrapped async client.
            assert client.base_url == "http://localhost:8000"

            client.close()

        mock_provider.stop_container.assert_called_once_with()

    def test_from_docker_image_sync_forwards_start_kwargs(self, mock_provider):
        """Extra kwargs reach provider.start_container()."""
        fake_connect, _ = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
                env_vars={"DEBUG": "1"},
            ).sync()
            mock_provider.start_container.assert_called_once_with(
                "coding-env:latest", env_vars={"DEBUG": "1"}
            )
            client.close()

    def test_from_env_sync_with_docker(self, mock_provider):
        """from_env(...).sync() with use_docker=True pulls from the HF registry, no await."""
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            ).sync()

            assert isinstance(client, SyncEnvClient)
            call_args = mock_provider.start_container.call_args
            assert "registry.hf.space/user-my-env" in call_args[0][0]
            assert len(sockets) == 1

            client.close()

    def test_from_env_sync_uv_stops_provider_on_start_failure(self):
        """A UV start() failure releases the provider before re-raising (sync path)."""
        provider = Mock()
        provider.context_timeout_s = None
        provider.start.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            ).sync()

        provider.stop.assert_called_once_with()

    def test_from_env_uv_stops_provider_on_construction_failure(self, monkeypatch):
        """A client-construction failure after a successful start() releases the UV provider.

        The provider start()/wait succeed, but `EnvClient.__init__` rejects an
        invalid OPENENV_CLIENT_MODE — the spawned process must still be stopped.
        """
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")
        provider = Mock()
        provider.context_timeout_s = None
        provider.start.return_value = "http://localhost:8000"
        provider.wait_for_ready.return_value = None

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=False,
                provider=provider,
            ).sync()

        provider.stop.assert_called_once_with()

    def test_from_env_docker_stops_container_on_construction_failure(
        self, mock_provider, monkeypatch
    ):
        """A client-construction failure after start_container releases the container."""
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_env(
                "user/my-env",
                use_docker=True,
                provider=mock_provider,
            ).sync()

        mock_provider.stop_container.assert_called_once_with()

    def test_from_docker_image_stops_container_on_construction_failure(
        self, mock_provider, monkeypatch
    ):
        """from_docker_image releases the container if client construction fails."""
        monkeypatch.setenv("OPENENV_CLIENT_MODE", "not-a-valid-mode")

        with pytest.raises(ValueError, match="Invalid mode"):
            GenericEnvClient.from_docker_image(
                image="coding-env:latest",
                provider=mock_provider,
            ).sync()

        mock_provider.stop_container.assert_called_once_with()

    _RESET_RESPONSE = (
        '{"type": "response", "data": {"observation": {"ready": true}, '
        '"reward": 0.0, "done": false}}'
    )

    def test_documented_sync_usage_with_block_and_reset(self, mock_provider):
        """The Environment Builder docs' sync example works end-to-end (issue #1118).

        `client = MyEnv.from_docker_image(...).sync()` followed by
        `with client: client.reset()` must connect on the sync loop, return a
        concrete StepResult (not a coroutine), and release the container on exit.
        """
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = GenericEnvClient.from_docker_image(
                image="my-env:latest",
                provider=mock_provider,
            ).sync()
            sockets[0].recv.return_value = self._RESET_RESPONSE

            with client:
                result = client.reset()

        if asyncio.iscoroutine(result):
            result.close()
            pytest.fail("client.reset() returned a coroutine in synchronous code")
        assert isinstance(result, StepResult)
        assert result.observation == {"ready": True}
        assert result.done is False
        # `with` on the eagerly connected sync handle must reuse its socket.
        assert len(sockets) == 1
        mock_provider.start_container.assert_called_once_with("my-env:latest")
        mock_provider.stop_container.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_documented_async_usage_async_with_and_reset(self, mock_provider):
        """The Environment Builder docs' async example works end-to-end (issue #1118).

        `client = await MyEnv.from_docker_image(...)` followed by
        `async with client: await client.reset()` must reuse the connection
        opened by the handle and release the container on exit.
        """
        fake_connect, sockets = self._fake_ws_connect()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_connect):
            client = await GenericEnvClient.from_docker_image(
                image="my-env:latest",
                provider=mock_provider,
            )
            assert isinstance(client, GenericEnvClient)
            sockets[0].recv.return_value = self._RESET_RESPONSE

            async with client:
                result = await client.reset()

        assert isinstance(result, StepResult)
        assert result.observation == {"ready": True}
        # `async with` on an already-connected client must not open a second socket.
        assert len(sockets) == 1
        mock_provider.start_container.assert_called_once_with("my-env:latest")
        mock_provider.stop_container.assert_called_once_with()


class TestBaseUrlProperty:
    """Test the public read-only base_url property (issue #935)."""

    def test_base_url_returns_normalized_url(self):
        """base_url exposes the normalized URL with any trailing slash stripped."""
        client = GenericEnvClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_base_url_is_none_before_provider_start(self, mock_provider):
        """A provider-only client has no URL until it connects/starts its container."""
        client = GenericEnvClient(provider=mock_provider)
        assert client.base_url is None

    def test_base_url_is_read_only(self):
        """base_url is a read-only property; callers must not reassign it."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        with pytest.raises(AttributeError):
            client.base_url = "http://evil:9999"  # type: ignore[misc]


# ============================================================================
# AutoEnv skip_install Integration Tests
# ============================================================================


class TestAutoEnvSkipInstall:
    """Test AutoEnv.from_env() with skip_install parameter."""

    def test_skip_install_with_base_url(self):
        """Test skip_install=True with explicit base_url."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            client = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_unavailable_server(self):
        """Test skip_install=True with unavailable server raises error."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                AutoEnv.from_env(
                    "echo",
                    base_url="http://localhost:8000",
                    skip_install=True,
                )

            assert "Server not available" in str(exc_info.value)

    def test_skip_install_with_hub_url_and_running_space(self):
        """Test skip_install=True with HF Space that is running."""
        from openenv.auto.auto_env import AutoEnv

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=True),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
        ):
            client = AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

    def test_skip_install_with_hub_url_and_docker(self, mock_provider):
        """Test skip_install=True with HF Space not running uses Docker."""
        from openenv.auto.auto_env import AutoEnv

        # AutoEnv resolves the from_env bootstrap handle via run_async_safely, so
        # an awaitable stand-in returning a connected client is a valid double.
        async def mock_from_env_async(*args, **kwargs):
            return GenericEnvClient(base_url="http://localhost:8000")

        with (
            patch.object(AutoEnv, "_check_space_availability", return_value=False),
            patch.object(
                AutoEnv,
                "_resolve_space_url",
                return_value="https://user-my-env.hf.space",
            ),
            patch.object(
                GenericEnvClient,
                "from_env",
                side_effect=mock_from_env_async,
            ) as mock_from_env,
        ):
            AutoEnv.from_env(
                "user/my-env",
                skip_install=True,
            )

            mock_from_env.assert_called_once()
            call_kwargs = mock_from_env.call_args[1]
            assert call_kwargs.get("use_docker") is True

    def test_skip_install_local_env_without_docker_image_raises(self):
        """Test skip_install=True for local env without docker_image raises error."""
        from openenv.auto.auto_env import AutoEnv

        with pytest.raises(ValueError) as exc_info:
            AutoEnv.from_env(
                "echo",  # Local name, not Hub URL
                skip_install=True,
            )

        error_msg = str(exc_info.value)
        assert "skip_install=True" in error_msg
        assert "base_url" in error_msg or "docker_image" in error_msg

    def test_skip_install_local_env_with_docker_image(self, mock_provider):
        """Test skip_install=True for local env with docker_image."""
        from openenv.auto.auto_env import AutoEnv

        with patch.object(GenericEnvClient, "connect", new_callable=AsyncMock):
            client = AutoEnv.from_env(
                "echo",
                docker_image="echo-env:latest",
                container_provider=mock_provider,
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)
            mock_provider.start_container.assert_called_once()

    def test_skip_install_false_still_works(self):
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)

        with (
            patch("openenv.auto.auto_env.get_discovery", return_value=mock_discovery),
            patch.object(AutoEnv, "_check_server_availability", return_value=True),
        ):
            result = AutoEnv.from_env(
                "echo",
                base_url="http://localhost:8000",
                skip_install=False,  # Explicit False
            )

            # Should return the typed client, not GenericEnvClient
            assert result is mock_client_instance
            assert not isinstance(result, GenericEnvClient)


# ============================================================================
# Comparison Tests: GenericEnvClient vs Typed Client
# ============================================================================


class TestGenericVsTypedComparison:
    """Compare behavior of GenericEnvClient vs typed clients."""

    def test_step_payload_generic_vs_typed(self):
        """Compare step payload generation."""
        # GenericEnvClient - pass through
        generic_client = GenericEnvClient(base_url="http://localhost:8000")
        generic_payload = generic_client._step_payload({"message": "hello"})

        # Should be identical to what a typed client would produce
        assert generic_payload == {"message": "hello"}

    def test_parse_result_generic_returns_dict(self):
        """GenericEnvClient returns dict observation."""
        generic_client = GenericEnvClient(base_url="http://localhost:8000")

        payload = {
            "observation": {"echoed_message": "hello", "length": 5},
            "reward": 1.0,
            "done": False,
        }
        result = generic_client._parse_result(payload)

        # Observation is a dict, not a typed object
        assert isinstance(result.observation, dict)
        assert result.observation["echoed_message"] == "hello"
        # Access is via dict keys, not attributes
        assert result.observation.get("length") == 5


# ============================================================================
# Import Tests
# ============================================================================


class TestGenericEnvClientImports:
    """Test that GenericEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import GenericEnvClient as GC1

        assert GC1 is GenericEnvClient

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import GenericEnvClient as GC2

        assert GC2 is GenericEnvClient

    def test_import_from_generic_client_module(self):
        """Test direct import from module."""
        from openenv.core.generic_client import GenericEnvClient as GC3

        assert GC3 is GenericEnvClient


class TestSyncEnvClientImports:
    """Test that SyncEnvClient can be imported from various locations."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import SyncEnvClient as SC1

        assert SC1 is SyncEnvClient

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import SyncEnvClient as SC2

        assert SC2 is SyncEnvClient

    def test_import_from_sync_client_module(self):
        """Test direct import from module."""
        from openenv.core.sync_client import SyncEnvClient as SC3

        assert SC3 is SyncEnvClient


class TestSyncEnvClientWrapper:
    """Test SyncEnvClient wrapper functionality."""

    def test_sync_method_returns_sync_client(self):
        """Test that .sync() returns a SyncEnvClient."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = client.sync()

        assert isinstance(sync_client, SyncEnvClient)
        assert sync_client._async is client

    def test_sync_client_has_async_client_property(self):
        """Test that SyncEnvClient exposes async_client property."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        assert sync_client.async_client is async_client

    def test_sync_client_delegates_payload_methods(self):
        """Test that SyncEnvClient delegates _step_payload to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        action = {"code": "print('hello')"}
        payload = sync_client._step_payload(action)

        assert payload == action

    def test_sync_client_delegates_parse_result(self):
        """Test that SyncEnvClient delegates _parse_result to async client."""
        async_client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = async_client.sync()

        payload = {
            "observation": {"output": "hello"},
            "reward": 1.0,
            "done": False,
        }
        result = sync_client._parse_result(payload)

        assert result.observation == {"output": "hello"}
        assert result.reward == 1.0

    def test_sync_client_new_session_reuses_provider_server(self, mock_provider):
        """The sync wrapper exposes child sessions without another context manager."""
        websockets = []

        async def fake_ws_connect(*args, **kwargs):
            ws = AsyncMock()
            websockets.append(ws)
            return ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            with GenericEnvClient(provider=mock_provider).sync() as client:
                session = client.new_session()

                assert isinstance(session, SyncEnvClient)
                assert isinstance(session.async_client, GenericEnvClient)
                assert session.async_client._provider is None
                assert session.async_client._base_url == "http://localhost:8000"
                assert len(websockets) == 2

        mock_provider.start_container.assert_called_once_with()
        mock_provider.stop_container.assert_called_once_with()

    def test_sync_close_stops_provider_when_child_close_raises(self, mock_provider):
        """Sync parent cleanup continues when child close fails."""
        client = GenericEnvClient(provider=mock_provider).sync()
        child = Mock()
        child.close.side_effect = RuntimeError("child close failed")
        client._child_clients.append(child)

        client.close()

        child.close.assert_called_once_with()
        assert client._child_clients == []
        mock_provider.stop_container.assert_called_once_with()

    def test_sync_call_after_async_connect_blocks_for_result(self, monkeypatch):
        """Sync calls after async setup return concrete results, not coroutines."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def connect_client():
            with patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect:
                await client.connect()
                mock_connect.assert_awaited_once()

        asyncio.run(connect_client())
        assert client._execution_mode == "async"

        expected = StepResult(
            observation={"ready": True},
            reward=0.0,
            done=False,
        )

        async def fake_reset(**kwargs):
            return expected

        monkeypatch.setattr(client, "_reset_async", fake_reset)

        result = client.reset()
        if asyncio.iscoroutine(result):
            result.close()
            pytest.fail("client.reset() returned a coroutine in synchronous code")

        assert result is expected

    @pytest.mark.asyncio
    async def test_sync_locked_dispatch_inside_running_loop_uses_sync_loop(
        self, monkeypatch
    ):
        """Sync-locked clients keep work on the sync wrapper loop."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        sync_client = client.sync()

        async def fake_connect():
            return client

        monkeypatch.setattr(client, "_connect_async", fake_connect)
        sync_client.connect()

        expected = StepResult(
            observation={"ready": True},
            reward=0.0,
            done=False,
        )
        observed = {}

        async def fake_reset(**kwargs):
            observed["loop"] = asyncio.get_running_loop()
            return expected

        monkeypatch.setattr(client, "_reset_async", fake_reset)

        try:
            current_loop = asyncio.get_running_loop()
            result = client.reset()
            if asyncio.iscoroutine(result):
                result.close()
                pytest.fail("sync-locked client.reset() returned a coroutine")

            assert result is expected
            assert observed["loop"] is sync_client._loop
            assert observed["loop"] is not current_loop
        finally:
            sync_client.close()


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestGenericEnvClientContextManager:
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_enter_exit(self):
        """Test that async context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "close", new_callable=AsyncMock
            ) as mock_close,
        ):
            async with GenericEnvClient(base_url="http://localhost:8000") as client:
                assert isinstance(client, GenericEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()

    def test_sync_context_manager_enter_exit(self):
        """Test that sync context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "_close_async", new_callable=AsyncMock
            ) as mock_close,
        ):
            client = GenericEnvClient(base_url="http://localhost:8000")

            with client as active_client:
                assert active_client is client
                mock_connect.assert_called_once()

            mock_close.assert_called_once()

    def test_sync_wrapper_context_manager(self):
        """Test SyncEnvClient context manager works correctly."""
        with (
            patch.object(
                GenericEnvClient, "_connect_async", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                GenericEnvClient, "_close_async", new_callable=AsyncMock
            ) as mock_close,
        ):
            async_client = GenericEnvClient(base_url="http://localhost:8000")
            sync_client = async_client.sync()

            with sync_client as client:
                assert isinstance(client, SyncEnvClient)
                mock_connect.assert_called_once()

            mock_close.assert_called_once()


# ============================================================================
# Connection Helper Tests
# ============================================================================


class TestIsLocalhostWsUrl:
    """Test the loopback-host detection used to decide whether to bypass proxies."""

    @pytest.mark.parametrize(
        "ws_url",
        [
            "ws://localhost:8000/ws",
            "ws://127.0.0.1:8000/ws",
            "ws://127.5.5.5:8000/ws",  # anywhere in 127.0.0.0/8 is loopback
            "ws://[::1]:8000/ws",
            "wss://LocalHost:8000/ws",  # case-insensitive
        ],
    )
    def test_loopback_hosts_are_localhost(self, ws_url):
        assert _is_localhost_ws_url(ws_url) is True

    @pytest.mark.parametrize(
        "ws_url",
        [
            "ws://my-localhost-proxy.example.com:8000/ws",  # substring, not the host
            "ws://127.0.0.1.example.com:8000/ws",  # substring, not the host
            "ws://localhost.attacker.com:8000/ws",
            "wss://remote.example.com:8000/ws",
        ],
    )
    def test_remote_hosts_are_not_localhost(self, ws_url):
        assert _is_localhost_ws_url(ws_url) is False


class TestGenericEnvClientConnection:
    """Test connection setup helpers."""

    @pytest.mark.asyncio
    async def test_localhost_connect_disables_proxy_per_connection(self, monkeypatch):
        """Localhost connections pass proxy=None instead of mutating NO_PROXY."""
        monkeypatch.setenv("NO_PROXY", "example.com")
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            observed["proxy"] = kwargs.get("proxy", "MISSING")
            observed["no_proxy_during"] = os.environ.get("NO_PROXY")
            return MagicMock()

        client = GenericEnvClient(base_url="http://localhost:8000")

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        # Proxy is disabled via the per-connection argument, and the global
        # NO_PROXY env var is never touched.
        assert observed["proxy"] is None
        assert observed["no_proxy_during"] == "example.com"
        assert os.environ["NO_PROXY"] == "example.com"

    @pytest.mark.asyncio
    async def test_remote_connect_uses_default_proxy(self, monkeypatch):
        """Non-localhost connections leave proxy handling at the default."""
        monkeypatch.setenv("NO_PROXY", "example.com")
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            observed["proxy"] = kwargs.get("proxy", "MISSING")
            return MagicMock()

        client = GenericEnvClient(base_url="ws://remote.example.com:8000")

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        # No explicit proxy override: websockets uses its default (env-derived).
        assert observed["proxy"] == "MISSING"
        assert os.environ["NO_PROXY"] == "example.com"

    @pytest.mark.asyncio
    async def test_websocket_ping_options_forward_to_connect(self):
        observed = {}

        async def fake_ws_connect(*args, **kwargs):
            observed["ping_interval"] = kwargs.get("ping_interval", "MISSING")
            observed["ping_timeout"] = kwargs.get("ping_timeout", "MISSING")
            return MagicMock()

        client = GenericEnvClient(
            base_url="ws://remote.example.com:8000",
            websocket_ping_interval_s=None,
            websocket_ping_timeout_s=None,
        )

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        assert observed == {"ping_interval": None, "ping_timeout": None}

    @pytest.mark.asyncio
    async def test_concurrent_localhost_connects_do_not_leak_no_proxy(
        self, monkeypatch
    ):
        """Concurrent connects must not corrupt the process-global NO_PROXY.

        The previous implementation mutated os.environ["NO_PROXY"] around the
        connect await; interleaved gather() calls could restore a stale value
        and leak the localhost entry. Disabling the proxy per-connection makes
        this race impossible.
        """
        monkeypatch.delenv("NO_PROXY", raising=False)

        async def fake_ws_connect(*args, **kwargs):
            # Yield control so the event loop interleaves the two connects.
            await asyncio.sleep(0)
            return MagicMock()

        clients = [
            GenericEnvClient(base_url="http://localhost:8000"),
            GenericEnvClient(base_url="http://127.0.0.1:8000"),
        ]

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await asyncio.gather(*(c.connect() for c in clients))

        assert "NO_PROXY" not in os.environ


class TestForeignLoopReconnect:
    """`connect()` must not silently no-op onto a websocket bound to a
    different (often already-closed) event loop -- e.g. a client connected
    via `await Client.from_env(...)` inside `asyncio.run(...)`, then driven
    afterwards through `.sync()`'s own dedicated background loop.
    """

    @pytest.mark.asyncio
    async def test_same_loop_reconnect_is_a_noop(self):
        """Calling connect() again on the loop that created _ws is still a
        cheap no-op -- only a genuinely different loop should reconnect.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def fake_ws_connect(*args, **kwargs):
            return MagicMock()

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client.connect()
            await client.connect()

        mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_foreign_loop_triggers_reconnect_not_noop(self):
        """If _ws was bound to a different loop than the one connect() is
        now running on, connect() must drop the stale reference and
        establish a fresh connection on the current loop, rather than
        returning early as if already connected.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = MagicMock()

        async def fake_ws_connect_first(*args, **kwargs):
            return first_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_first
        ):
            await client.connect()

        assert client._ws is first_ws

        # Simulate "connected on a different loop": the real scenario is two
        # different asyncio event loops, which isn't practical to spin up
        # for a unit test, so we directly substitute a sentinel loop object
        # that isn't the one currently running.
        client._ws_loop = object()

        second_ws = MagicMock()

        async def fake_ws_connect_second(*args, **kwargs):
            return second_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_second
        ) as mock_connect:
            await client.connect()

        mock_connect.assert_called_once()
        assert client._ws is second_ws
        assert client._ws_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_disconnect_clears_ws_loop(self):
        """disconnect() must clear _ws_loop along with _ws so a later
        connect() on the same loop doesn't think it's still connected.
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        async def fake_ws_connect(*args, **kwargs):
            return MagicMock()

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        await client.disconnect()

        assert client._ws is None
        assert client._ws_loop is None

    @pytest.mark.asyncio
    async def test_disconnect_does_not_reconnect_foreign_loop_websocket(self):
        """disconnect() must not open a replacement socket just to close it."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = AsyncMock()

        async def fake_ws_connect(*args, **kwargs):
            return first_ws

        with patch("openenv.core.env_client.ws_connect", side_effect=fake_ws_connect):
            await client.connect()

        client._ws_loop = object()

        with patch("openenv.core.env_client.ws_connect") as mock_connect:
            await client.disconnect()

        mock_connect.assert_not_called()
        first_ws.send.assert_not_called()
        first_ws.close.assert_not_called()
        assert client._ws is None
        assert client._ws_loop is None

    @pytest.mark.asyncio
    async def test_lazy_send_reconnects_without_explicit_connect_call(self):
        """A caller that never explicitly calls `.connect()` -- e.g.
        `client.sync().reset()` right after `from_env()`, which goes straight
        to `_send()` -> `_ensure_connected()` -- must still reconnect on a
        foreign loop. `_ensure_connected()` previously only called `connect()`
        when `self._ws is None`, which skipped the reconnect logic entirely
        whenever `_ws` was already set (the exact foreign-loop case).
        """
        client = GenericEnvClient(base_url="http://localhost:8000")

        first_ws = MagicMock()

        async def fake_ws_connect_first(*args, **kwargs):
            return first_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_first
        ):
            await client.connect()

        # Simulate the foreign-loop state without an explicit connect() call
        # in between, mirroring from_env() handing off to .sync().
        client._ws_loop = object()

        second_ws = AsyncMock()

        async def fake_ws_connect_second(*args, **kwargs):
            return second_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect_second
        ) as mock_connect:
            await client._ensure_connected()

        mock_connect.assert_called_once()
        assert client._ws is second_ws
        assert client._ws_loop is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_receive_does_not_reconnect_after_request_was_sent(self):
        """A response must be read from the socket that carried its request."""

        class SocketClosedAfterSend:
            state = State.OPEN

            async def send(self, _message):
                self.state = State.CLOSED

            async def recv(self):
                raise ConnectionError("original socket closed after send")

        client = GenericEnvClient(base_url="http://localhost:8000")
        original_ws = SocketClosedAfterSend()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.side_effect = AssertionError(
            "must not receive a response on a replacement socket"
        )

        with patch(
            "openenv.core.env_client.ws_connect", return_value=replacement_ws
        ) as mock_connect:
            with pytest.raises(
                ConnectionError, match="original socket closed after send"
            ):
                await client._send_and_receive({"type": "state"})

        mock_connect.assert_not_called()
        assert client._ws is original_ws

    @pytest.mark.asyncio
    async def test_receive_timeout_drops_socket_so_next_call_reconnects(self):
        """A response that arrives after we time out must not be read by the
        *next* call.

        Regression test for #1143: leaving the timed-out socket open let the
        server's late response for the abandoned request sit in the buffer,
        where the next `_receive()` silently read it as its own -- and every
        response after that was shifted by one for the life of the connection.
        """

        class NeverResponds:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def recv(self):
                await asyncio.sleep(10)

            async def close(self):
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.05
        )
        original_ws = NeverResponds()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})

        assert client._ws is None
        # The socket drop no longer awaits close() inline (a slow close
        # handshake would otherwise stall the caller's own deadline -- see
        # test_receive_cancelled_by_outer_deadline_...); it's scheduled as
        # a background task instead. Give the loop one tick to run it.
        await asyncio.sleep(0)
        assert original_ws.state == State.CLOSED

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client._send_and_receive({"type": "state"})

        mock_connect.assert_called_once()
        assert client._ws is replacement_ws

    @pytest.mark.asyncio
    async def test_receive_cancelled_by_outer_deadline_drops_socket_so_next_call_reconnects(
        self,
    ):
        """An outer `asyncio.wait_for` around the whole call cancels
        `_receive()` with `asyncio.CancelledError`, not the client's own
        `asyncio.TimeoutError` -- a caller-imposed deadline (e.g. a step
        call wrapped in its own timeout) hits this path even when
        `message_timeout_s` itself never fires.

        Regression for the gap in the #1143 fix: only `asyncio.TimeoutError`
        was caught, so this path left the socket open and desynced exactly
        like the original bug.
        """

        class NeverResponds:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def recv(self):
                await asyncio.sleep(10)

            async def close(self):
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=10
        )
        original_ws = NeverResponds()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client._send_and_receive({"type": "state"}), timeout=0.05
            )

        assert client._ws is None
        # Background close task, same as above -- give the loop one tick.
        await asyncio.sleep(0)
        assert original_ws.state == State.CLOSED

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            await client._send_and_receive({"type": "state"})

        mock_connect.assert_called_once()
        assert client._ws is replacement_ws

    @pytest.mark.asyncio
    async def test_slow_close_handshake_does_not_stall_callers_deadline(self):
        """A slow close handshake on the dropped socket must not eat into
        the caller's own deadline.

        Regression for the Bugbot finding on #1149: `_receive()` used to
        `await ws.close()` inline before re-raising, so an outer
        `asyncio.wait_for(..., timeout=50ms)` didn't actually return in
        50ms if the server was slow to ack the close (websockets' default
        `close_timeout` is 10s) -- the caller was blocked on a handshake it
        never asked to wait on. The close is now fire-and-forget.
        """

        close_started = asyncio.Event()

        class NeverRespondsSlowClose:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def recv(self):
                await asyncio.sleep(10)

            async def close(self):
                close_started.set()
                await asyncio.sleep(2)  # simulate a stalled close handshake
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=10
        )
        original_ws = NeverRespondsSlowClose()
        client._ws = original_ws
        client._ws_loop = asyncio.get_running_loop()

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client._send_and_receive({"type": "state"}), timeout=0.05
            )
        elapsed = loop.time() - start

        # The 2s close handshake must not have been awaited inline: the
        # caller's own 50ms deadline should be honored, not the socket's.
        assert elapsed < 1.0, (
            f"caller's 50ms deadline took {elapsed:.2f}s -- the close "
            "handshake was awaited inline instead of backgrounded"
        )
        assert client._ws is None

        # The close still actually happens, just not on the caller's time.
        await close_started.wait()
        await asyncio.sleep(2.1)
        assert original_ws.state == State.CLOSED

    @pytest.mark.asyncio
    async def test_close_waits_for_pending_background_close(self):
        """`close()` must not return until a backgrounded socket close (the
        fire-and-forget cleanup in `_receive()`) has actually finished.

        Regression for a second Bugbot finding on #1149:
        `_pending_close_tasks` was never awaited anywhere, so
        `SyncEnvClient.close()` could call `_stop_loop()` and tear down the
        event loop while a background close was still mid-handshake,
        abandoning it -- the server-side session could stay allocated.
        """
        close_completed = False

        class SlowBackgroundClose:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def recv(self):
                await asyncio.sleep(10)

            async def close(self):
                nonlocal close_completed
                await asyncio.sleep(0.05)
                close_completed = True
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.01
        )
        dropped_ws = SlowBackgroundClose()
        client._ws = dropped_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})

        assert client._pending_close_tasks  # background close was scheduled
        assert not close_completed  # ...but hasn't run yet

        await client._close_async()

        assert close_completed, (
            "close() returned before the backgrounded socket close finished"
        )

    @pytest.mark.asyncio
    async def test_reconnect_waits_for_dropped_socket_to_release_capacity(self):
        """A replacement connection must wait for the old session to close.

        A timed-out socket is detached immediately and closed in the background.
        Reconnecting before that handshake finishes races the server's session
        accounting; with the default capacity of one, the retry is rejected.
        """
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class SlowClose:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def recv(self):
                await asyncio.sleep(10)

            async def close(self):
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        client = GenericEnvClient(
            base_url="http://localhost:8000", message_timeout_s=0.01
        )
        dropped_ws = SlowClose()
        client._ws = dropped_ws
        client._ws_loop = asyncio.get_running_loop()

        with pytest.raises(asyncio.TimeoutError):
            await client._send_and_receive({"type": "state"})
        await close_started.wait()

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN
        replacement_ws.recv.return_value = '{"type": "state", "data": {}}'

        async def fake_ws_connect(*args, **kwargs):
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            reconnect = asyncio.create_task(client._connect_async())
            await asyncio.sleep(0)
            assert not reconnect.done()
            mock_connect.assert_not_called()

            release_close.set()
            await asyncio.wait_for(reconnect, timeout=1)

        mock_connect.assert_called_once()
        assert dropped_ws.state == State.CLOSED
        assert client._ws is replacement_ws
        await client._close_async()

    @pytest.mark.asyncio
    async def test_cancelled_close_tracks_current_socket_until_handshake_finishes(
        self,
    ):
        """Cancelling the current close must not abandon its server session."""
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class SlowCurrentSocket:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def close(self):
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        client = GenericEnvClient(base_url="http://localhost:8000")
        current_ws = SlowCurrentSocket()
        client._ws = current_ws
        client._ws_loop = asyncio.get_running_loop()

        close_call = asyncio.create_task(client._close_async())
        await close_started.wait()
        assert client._pending_close_tasks

        close_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_call

        assert client._ws is None
        assert current_ws.state == State.OPEN
        assert any(not task.cancelled() for task in client._pending_close_tasks)

        replacement_ws = AsyncMock()
        replacement_ws.state = State.OPEN

        async def fake_ws_connect(*args, **kwargs):
            return replacement_ws

        with patch(
            "openenv.core.env_client.ws_connect", side_effect=fake_ws_connect
        ) as mock_connect:
            reconnect = asyncio.create_task(client._connect_async())
            await asyncio.sleep(0)
            assert not reconnect.done()
            mock_connect.assert_not_called()

            release_close.set()
            await asyncio.wait_for(reconnect, timeout=1)

        assert current_ws.state == State.CLOSED
        mock_connect.assert_called_once()
        assert client._ws is replacement_ws
        await client._close_async()

    @pytest.mark.asyncio
    async def test_cancelled_close_still_closes_current_and_pending_sockets(self):
        """Cancellation while draining an old socket must not leak either one."""

        class FakeRuntimeProvider:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        close_started = asyncio.Event()
        release_close = asyncio.Event()

        class PendingSocket:
            state = State.OPEN

            async def close(self):
                close_started.set()
                await release_close.wait()
                self.state = State.CLOSED

        class CurrentSocket:
            state = State.OPEN

            async def send(self, _message):
                pass

            async def close(self):
                self.state = State.CLOSED

        provider = FakeRuntimeProvider()
        client = GenericEnvClient(provider=provider)
        client._base_url = "http://localhost:8000"
        client._ws_url = "ws://localhost:8000/ws"

        dropped_ws = PendingSocket()
        pending = asyncio.create_task(dropped_ws.close())
        client._pending_close_tasks.add(pending)
        pending.add_done_callback(client._pending_close_tasks.discard)
        await close_started.wait()

        current_ws = CurrentSocket()
        client._ws = current_ws
        client._ws_loop = asyncio.get_running_loop()

        close_task = asyncio.create_task(client._close_async())
        await asyncio.sleep(0)  # let close enter the shielded pending-close drain
        assert not close_task.done()

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert provider.stopped
        assert client._base_url is None
        assert client._ws_url is None
        assert client._ws is None
        assert current_ws.state == State.CLOSED
        assert not pending.cancelled()

        release_close.set()
        await pending
        assert dropped_ws.state == State.CLOSED


# ============================================================================
# Integration Tests (require running server)
# ============================================================================


@pytest.mark.integration
class TestGenericEnvClientIntegration:
    """
    Integration tests that require a running server.

    These tests require a server to be running on localhost.

    Start a server first:
        cd envs/echo_env/server && uvicorn app:app --host 0.0.0.0 --port 8000

    Run these tests with:
        pytest -m integration tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def local_echo_server(self):
        """Check if local echo server is running."""
        import requests

        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Local echo server not healthy")
            return base_url
        except requests.RequestException:
            pytest.skip(
                "Local echo server not running. "
                "Start it with: cd envs/echo_env/server && uvicorn app:app"
            )

    def test_generic_client_with_local_server(self, local_echo_server):
        """Test GenericEnvClient with a real local server using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            # Reset
            result = client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from GenericEnvClient!"}
            step_result = client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )

    def test_generic_client_multiple_steps(self, local_echo_server):
        """Test multiple steps with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            messages = ["First", "Second", "Third"]
            for msg in messages:
                result = client.step({"message": msg})
                assert msg in result.observation.get("echoed_message", "")

    def test_generic_client_state(self, local_echo_server):
        """Test getting state with GenericEnvClient using sync wrapper."""
        with GenericEnvClient(base_url=local_echo_server).sync() as client:
            client.reset()

            # Execute some steps
            client.step({"message": "step 1"})
            client.step({"message": "step 2"})

            # Get state
            state = client.state()

            assert isinstance(state, dict)
            # State should have step_count
            assert "step_count" in state or len(state) > 0

    @pytest.mark.asyncio
    async def test_generic_client_async_with_local_server(self, local_echo_server):
        """Test GenericEnvClient with async API."""
        async with GenericEnvClient(base_url=local_echo_server) as client:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step with dict action
            action = {"message": "Hello from async GenericEnvClient!"}
            step_result = await client.step(action)

            assert step_result is not None
            assert isinstance(step_result.observation, dict)
            assert "Hello from async GenericEnvClient!" in step_result.observation.get(
                "echoed_message", ""
            )


@pytest.mark.integration
@pytest.mark.docker
class TestGenericEnvClientDocker:
    """
    Docker integration tests for GenericEnvClient.

    These tests require Docker to be running.

    Run these tests with:
        pytest -m "integration and docker" tests/test_core/test_generic_client.py -v
    """

    @pytest.fixture
    def check_docker_and_image(self):
        """Check if Docker is available and echo-env image exists."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            pytest.skip("Docker is not installed")

        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except Exception:
            pytest.skip("Cannot access Docker")

        # Check for echo-env image
        result = subprocess.run(
            ["docker", "images", "-q", "echo-env:latest"],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            pytest.skip("Docker image 'echo-env:latest' not found")

    @pytest.mark.asyncio
    async def test_generic_client_from_docker_image(self, check_docker_and_image):
        """Test GenericEnvClient.from_docker_image() with real Docker."""
        client = await GenericEnvClient.from_docker_image("echo-env:latest")

        try:
            # Reset
            result = await client.reset()
            assert result is not None
            assert isinstance(result.observation, dict)

            # Step
            step_result = await client.step({"message": "Docker test!"})
            assert "Docker test!" in step_result.observation.get("echoed_message", "")

            print("GenericEnvClient.from_docker_image() works!")
        finally:
            await client.close()


# ============================================================================
# GenericAction Tests
# ============================================================================


class TestGenericAction:
    """Test GenericAction class."""

    def test_create_from_kwargs(self):
        """Test creating GenericAction from keyword arguments."""
        action = GenericAction(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_is_dict_subclass(self):
        """Test that GenericAction is a dict subclass."""
        action = GenericAction(message="test")

        assert isinstance(action, dict)
        assert isinstance(action, GenericAction)

    def test_dict_methods_work(self):
        """Test that dict methods work on GenericAction."""
        action = GenericAction(a=1, b=2)

        assert action.get("a") == 1
        assert action.get("c", "default") == "default"
        assert list(action.keys()) == ["a", "b"]
        assert list(action.values()) == [1, 2]

    def test_empty_action(self):
        """Test creating empty GenericAction."""
        action = GenericAction()

        assert len(action) == 0
        assert dict(action) == {}

    def test_nested_values(self):
        """Test GenericAction with nested values."""
        action = GenericAction(
            command="run",
            params={"file": "test.py", "args": ["--verbose"]},
        )

        assert action["command"] == "run"
        assert action["params"]["file"] == "test.py"
        assert action["params"]["args"] == ["--verbose"]

    def test_repr(self):
        """Test GenericAction repr."""
        action = GenericAction(code="x=1")

        repr_str = repr(action)
        assert "GenericAction" in repr_str
        assert "code=" in repr_str

    def test_can_be_used_with_generic_client(self):
        """Test that GenericAction works with GenericEnvClient._step_payload."""
        client = GenericEnvClient(base_url="http://localhost:8000")
        action = GenericAction(message="hello")

        payload = client._step_payload(action)

        assert payload == {"message": "hello"}


class TestGenericActionImports:
    """Test GenericAction imports."""

    def test_import_from_core(self):
        """Test import from openenv.core."""
        from openenv.core import GenericAction as GA1

        assert GA1 is GenericAction

    def test_import_from_openenv(self):
        """Test import from openenv package."""
        from openenv import GenericAction as GA2

        assert GA2 is GenericAction

    def test_import_from_module(self):
        """Test direct import from module."""
        from openenv.core.generic_client import GenericAction as GA3

        assert GA3 is GenericAction


# ============================================================================
# AutoAction skip_install Tests
# ============================================================================


class TestAutoActionSkipInstall:
    """Test AutoAction.from_env() with skip_install parameter."""

    def test_skip_install_returns_generic_action(self):
        """Test skip_install=True returns GenericAction class."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/any-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_works_for_local_names(self):
        """Test skip_install=True works for local environment names."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("echo", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_from_hub_alias(self):
        """Test skip_install works with from_hub alias."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_hub("user/my-env", skip_install=True)

        assert ActionClass is GenericAction

    def test_skip_install_action_can_be_instantiated(self):
        """Test that returned GenericAction can be instantiated."""
        from openenv.auto.auto_action import AutoAction

        ActionClass = AutoAction.from_env("user/repo", skip_install=True)

        # Create an action
        action = ActionClass(code="print('hello')", timeout=30)

        assert action["code"] == "print('hello')"
        assert action["timeout"] == 30

    def test_skip_install_false_still_works(self):
        """Test that skip_install=False (default) still works as before."""
        from openenv.auto._discovery import EnvironmentInfo, reset_discovery
        from openenv.auto.auto_action import AutoAction

        reset_discovery()

        mock_env_info = EnvironmentInfo(
            env_key="echo",
            name="echo_env",
            package_name="openenv-echo-env",
            version="0.1.0",
            description="Echo environment",
            client_module_path="echo_env.client",
            client_class_name="EchoEnv",
            action_class_name="EchoAction",
            observation_class_name="EchoObservation",
            default_image="echo-env:latest",
            spec_version=1,
        )

        mock_discovery = Mock()
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_discovery.discover.return_value = {"echo": mock_env_info}

        mock_action_class = Mock()
        mock_env_info.get_action_class = Mock(return_value=mock_action_class)

        with patch(
            "openenv.auto.auto_action.get_discovery", return_value=mock_discovery
        ):
            result = AutoAction.from_env("echo", skip_install=False)

            # Should return the typed action class, not GenericAction
            assert result is mock_action_class
            assert result is not GenericAction


# ============================================================================
# End-to-End: AutoEnv + AutoAction with skip_install
# ============================================================================


class TestAutoEnvAutoActionSkipInstallIntegration:
    """Test AutoEnv and AutoAction work together with skip_install."""

    def test_both_skip_install_returns_generic_types(self):
        """Test that both AutoEnv and AutoAction with skip_install work together."""
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            # Get action class without installing package
            ActionClass = AutoAction.from_env("user/my-env", skip_install=True)

            # Both should be generic types
            assert isinstance(client, GenericEnvClient)
            assert ActionClass is GenericAction

            # They should work together
            action = ActionClass(code="test")
            payload = client._step_payload(action)
            assert payload == {"code": "test"}

    def test_mixed_skip_install_raises_warning_scenario(self):
        """
        Test scenario where user forgets skip_install on AutoAction.

        This documents the expected behavior - if user uses skip_install
        on AutoEnv but not on AutoAction, AutoAction will try to install.
        """
        import os

        from openenv.auto._discovery import reset_discovery
        from openenv.auto.auto_action import AutoAction
        from openenv.auto.auto_env import AutoEnv

        reset_discovery()

        with patch.object(AutoEnv, "_check_server_availability", return_value=True):
            # Get client without installing package
            client = AutoEnv.from_env(
                "user/my-env",
                base_url="http://localhost:8000",
                skip_install=True,
            )

            assert isinstance(client, GenericEnvClient)

            # Now if user forgets skip_install on AutoAction...
            # It will try to install from Hub and fail (no confirmation in tests)
            # Set env var to bypass confirmation, but mock installation to fail
            with (
                patch.dict(os.environ, {"OPENENV_TRUST_REMOTE_CODE": "1"}),
                patch(
                    "openenv.auto.auto_action.AutoEnv._ensure_package_from_hub"
                ) as mock_ensure,
            ):
                # Make _ensure_package_from_hub raise an error
                mock_ensure.side_effect = ValueError("Installation failed")

                # This should raise ValueError from installation attempt
                with pytest.raises(ValueError) as exc_info:
                    AutoAction.from_env("user/my-env")  # forgot skip_install=True!

                # Installation was attempted
                assert "Installation failed" in str(exc_info.value)
