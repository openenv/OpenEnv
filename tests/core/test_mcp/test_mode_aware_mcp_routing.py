# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for mode-aware MCP tool discovery and execution via HTTPEnvServer (Issue #1212).

Verifies:
1. Production tools/list exposes production tools + shared tools, and excludes simulation tools.
2. Simulation tools/list exposes simulation tools + shared tools, and excludes production tools.
3. Production mode tools can be called successfully.
4. Simulation mode tools can be called successfully.
5. Mode-specific tools shadow shared FastMCP tools with the same name.
6. HTTP /mcp transport works end-to-end.
7. Direct WebSocket /mcp transport works with parity.
8. /ws transport with MCP messages works with parity.
9. MCPToolClient works end-to-end for mode-aware tools.
10. Cross-mode session reuse and WebSocket attachment are rejected.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Context, FastMCP
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.mcp_client import MCPToolClient


class ModeAwareTestEnvironment(MCPEnvironment):
    """Test environment defining mode-agnostic and mode-specific tools."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("mode-aware-test")
        super().__init__(mcp)

        @mcp.tool
        def shared_tool(value: int) -> int:
            """Tool available in all modes."""
            return value * 2

        @self.tool()
        def all_mode_tool(ctx: Context, message: str) -> str:
            """Tool registered with @self.tool() (no mode) requiring FastMCP Context."""
            ctx.info(f"Processing {message}")
            return f"ECHO: {message}"

        @self.tool(mode="production")
        def search_live(query: str) -> str:
            """Production-only tool searching live API."""
            return f"LIVE: {query}"

        @self.tool(mode="simulation")
        def search_mock(query: str) -> str:
            """Simulation-only tool querying local database."""
            return f"MOCK: {query}"

        @mcp.tool
        def override_me() -> str:
            """Base FastMCP tool."""
            return "FASTMCP_BASE"

        @self.tool(mode="production")
        def override_me() -> str:  # noqa: F811
            """Overridden in production mode."""
            return "PRODUCTION_OVERRIDE"

        self._state = State(episode_id="test-ep", step_count=0)

    def reset(self, **kwargs) -> Observation:
        return Observation(done=False, reward=None)

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        return Observation(done=False, reward=None)

    @property
    def state(self) -> State:
        return self._state


@pytest.fixture
def prod_server_app() -> FastAPI:
    """Create a FastAPI app configured in production mode."""
    app = FastAPI()
    server = HTTPEnvServer(
        env=ModeAwareTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="production")
    return app


@pytest.fixture
def sim_server_app() -> FastAPI:
    """Create a FastAPI app configured in simulation mode."""
    app = FastAPI()
    server = HTTPEnvServer(
        env=ModeAwareTestEnvironment,
        action_cls=CallToolAction,
        observation_cls=CallToolObservation,
    )
    server.register_routes(app, mode="simulation")
    return app


class TestModeAwareMCPRouting:
    """Tests verifying mode-aware MCP tool discovery, execution, and session isolation."""

    def test_production_mode_lists_production_and_shared_tools(self, prod_server_app):
        """Production tools/list includes production and shared tools, excluding simulation tools."""
        client = TestClient(prod_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["result"]["session_id"]

        list_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "result" in data
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_live" in tool_names
        assert "search_mock" not in tool_names

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "input_schema" not in tool

    def test_simulation_mode_lists_simulation_and_shared_tools(self, sim_server_app):
        """Simulation tools/list includes simulation and shared tools, excluding production tools."""
        client = TestClient(sim_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["result"]["session_id"]

        list_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {"session_id": session_id},
                "id": 2,
            },
        )
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert "result" in data
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        assert "shared_tool" in tool_names
        assert "search_mock" in tool_names
        assert "search_live" not in tool_names

    def test_production_mode_calls_production_and_shared_tools(self, prod_server_app):
        """Production tools/call successfully executes production and shared tools, rejecting simulation tools."""
        client = TestClient(prod_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        # Call shared tool
        call_shared = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "shared_tool",
                    "arguments": {"value": 21},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_shared.status_code == 200
        res_shared = call_shared.json()
        assert "result" in res_shared
        assert res_shared["result"]["data"] == 42

        # Call production tool
        call_prod = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {"query": "weather"},
                    "session_id": session_id,
                },
                "id": 3,
            },
        )
        assert call_prod.status_code == 200
        res_prod = call_prod.json()
        assert "result" in res_prod
        assert res_prod["result"]["data"] == "LIVE: weather"

        # Calling simulation tool in production returns error
        call_sim = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_mock",
                    "arguments": {"query": "weather"},
                    "session_id": session_id,
                },
                "id": 4,
            },
        )
        assert call_sim.status_code == 200
        res_sim = call_sim.json()
        assert "error" in res_sim
        assert res_sim["error"]["code"] == -32602
        assert "search_mock" in res_sim["error"]["message"]

    def test_simulation_mode_calls_simulation_tools(self, sim_server_app):
        """Simulation tools/call successfully executes simulation tools, rejecting production tools."""
        client = TestClient(sim_server_app)

        create_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        session_id = create_resp.json()["result"]["session_id"]

        # Call simulation tool
        call_sim = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_mock",
                    "arguments": {"query": "mock-weather"},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_sim.status_code == 200
        res_sim = call_sim.json()
        assert "result" in res_sim
        assert res_sim["result"]["data"] == "MOCK: mock-weather"

        # Calling production tool in simulation returns error
        call_prod = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {"query": "weather"},
                    "session_id": session_id,
                },
                "id": 3,
            },
        )
        assert call_prod.status_code == 200
        res_prod = call_prod.json()
        assert "error" in res_prod
        assert res_prod["error"]["code"] == -32602
        assert "search_live" in res_prod["error"]["message"]

    def test_mode_aware_tool_shadows_fastmcp_shared_tool(self, prod_server_app):
        """Production mode tool overrides a shared FastMCP tool with the same name."""
        client = TestClient(prod_server_app)

        # tools/list should return exactly one override_me tool
        list_resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
        )
        assert list_resp.status_code == 200
        tools = list_resp.json()["result"]["tools"]
        matches = [t for t in tools if t["name"] == "override_me"]
        assert len(matches) == 1
        assert matches[0]["description"] == "Overridden in production mode."

        # Calling it returns the production implementation
        call_resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "override_me", "arguments": {}},
                "id": 2,
            },
        )
        assert call_resp.status_code == 200
        assert call_resp.json()["result"]["data"] == "PRODUCTION_OVERRIDE"

    def test_all_mode_tool_registered_without_mode_takes_fastmcp_path(
        self, prod_server_app, sim_server_app
    ):
        """A tool registered with @self.tool() (no mode) uses FastMCP execution path with Context."""
        # Test in production mode
        prod_client = TestClient(prod_server_app)
        create_resp = prod_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["result"]["session_id"]

        call_resp = prod_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "all_mode_tool",
                    "arguments": {"message": "hello-prod"},
                    "session_id": session_id,
                },
                "id": 2,
            },
        )
        assert call_resp.status_code == 200
        res = call_resp.json()
        assert "result" in res, (
            f"Expected successful FastMCP call, got error: {res.get('error')}"
        )
        assert res["result"]["data"] == "ECHO: hello-prod"

        # FastMCP argument validation rejects invalid parameters
        invalid_resp = prod_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "all_mode_tool",
                    "arguments": {},
                    "session_id": session_id,
                },
                "id": 3,
            },
        )
        assert invalid_resp.status_code == 200
        invalid_data = invalid_resp.json()
        assert "error" in invalid_data
        assert invalid_data["error"]["code"] == -32603
        assert "validation error" in invalid_data["error"]["message"].lower()
        assert "result" not in invalid_data

        # Test in simulation mode
        sim_client = TestClient(sim_server_app)
        create_sim = sim_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 4},
        )
        assert create_sim.status_code == 200
        sim_session_id = create_sim.json()["result"]["session_id"]

        call_sim = sim_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "all_mode_tool",
                    "arguments": {"message": "hello-sim"},
                    "session_id": sim_session_id,
                },
                "id": 5,
            },
        )
        assert call_sim.status_code == 200
        res_sim = call_sim.json()
        assert "result" in res_sim, (
            f"Expected successful FastMCP call, got error: {res_sim.get('error')}"
        )
        assert res_sim["result"]["data"] == "ECHO: hello-sim"

    def test_direct_websocket_mcp_parity(self, prod_server_app):
        """Direct /mcp WebSocket transport lists and calls mode-aware tools with parity."""
        client = TestClient(prod_server_app)

        with client.websocket_connect("/mcp") as ws:
            ws.send_text(
                json.dumps(
                    {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}
                )
            )
            list_msg = json.loads(ws.receive_text())
            tool_names = [t["name"] for t in list_msg["result"]["tools"]]
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            ws.send_text(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "search_live",
                            "arguments": {"query": "ws-test"},
                        },
                        "id": 2,
                    }
                )
            )
            call_msg = json.loads(ws.receive_text())
            assert call_msg["result"]["data"] == "LIVE: ws-test"

    def test_ws_mcp_message_parity(self, prod_server_app):
        """Persistent /ws WebSocket with WSMCPMessage routes mode-aware tools."""
        client = TestClient(prod_server_app)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/list",
                            "params": {},
                            "id": 1,
                        },
                    }
                )
            )
            resp_list = json.loads(ws.receive_text())
            assert resp_list["type"] == "mcp"
            tools = resp_list["data"]["result"]["tools"]
            tool_names = [t["name"] for t in tools]
            assert "search_live" in tool_names
            assert "search_mock" not in tool_names

            ws.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "search_live",
                                "arguments": {"query": "ws-mcp"},
                            },
                            "id": 2,
                        },
                    }
                )
            )
            resp_call = json.loads(ws.receive_text())
            assert resp_call["type"] == "mcp"
            assert resp_call["data"]["result"]["data"] == "LIVE: ws-mcp"

    @pytest.mark.asyncio
    async def test_mcp_tool_client_end_to_end(self, prod_server_app):
        """MCPToolClient discovers and executes production mode tools."""
        transport = httpx.ASGITransport(app=prod_server_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as http_client:
            mcp_client = MCPToolClient(base_url="http://testserver")
            mcp_client._http_client = http_client

            tools = await mcp_client.list_tools()
            tool_names = [t.name for t in tools]
            assert "search_live" in tool_names
            assert "shared_tool" in tool_names
            assert "search_mock" not in tool_names

            res = await mcp_client.call_tool("search_live", query="client-test")
            assert res == "LIVE: client-test"

    @pytest.mark.parametrize("source_mode", ["production", "simulation"])
    @pytest.mark.parametrize("production_first", [True, False])
    def test_cross_app_session_reuse_rejected(self, source_mode, production_first):
        """Registration order must not change a session's mode binding."""
        server = HTTPEnvServer(
            ModeAwareTestEnvironment,
            CallToolAction,
            CallToolObservation,
            max_concurrent_envs=4,
        )
        apps = {"production": FastAPI(), "simulation": FastAPI()}
        modes = ["production", "simulation"]
        for mode in modes if production_first else reversed(modes):
            server.register_routes(apps[mode], mode=mode)
        other_mode = "simulation" if source_mode == "production" else "production"
        source, other = TestClient(apps[source_mode]), TestClient(apps[other_mode])
        session_id = source.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1}
        ).json()["result"]["session_id"]
        tool = "search_live" if source_mode == "production" else "search_mock"
        try:
            for method in ("tools/list", "tools/call"):
                request = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "id": 2,
                    "params": {
                        "session_id": session_id,
                        "name": tool,
                        "arguments": {"query": "test"},
                    },
                }
                rejected = other.post("/mcp", json=request).json()
                assert rejected["error"]["code"] == -32602
                accepted = source.post("/mcp", json=request).json()
                assert "error" not in accepted
                if method == "tools/list":
                    assert tool in {t["name"] for t in accepted["result"]["tools"]}
                else:
                    expected = (
                        "LIVE: test" if source_mode == "production" else "MOCK: test"
                    )
                    assert accepted["result"]["data"] == expected
            with other.websocket_connect(f"/ws?session_id={session_id}") as ws:
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "belongs to mode" in response["data"]["message"]
        finally:
            source.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "openenv/session/close",
                    "id": 3,
                    "params": {"session_id": session_id},
                },
            )

    def test_cross_app_session_close_rejected(self):
        """A session created in one mode cannot be closed from the other mode.

        Rejected cross-mode close requests do not destroy the session, and
        same-mode close requests succeed.
        """
        server = HTTPEnvServer(
            env=ModeAwareTestEnvironment,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
            max_concurrent_envs=4,
        )
        prod_app, sim_app = FastAPI(), FastAPI()
        server.register_routes(prod_app, mode="production")
        server.register_routes(sim_app, mode="simulation")
        prod_client, sim_client = TestClient(prod_app), TestClient(sim_app)

        # 1. Production session cannot be closed from simulation mode
        prod_sid = prod_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 1},
        ).json()["result"]["session_id"]

        sim_close = sim_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": prod_sid},
                "id": 2,
            },
        )
        assert sim_close.json()["error"]["code"] == -32602
        assert (
            "belongs to mode 'production', not 'simulation'"
            in sim_close.json()["error"]["message"]
        )

        # Verify production session was not destroyed and still works
        call_prod = prod_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_live",
                    "arguments": {"query": "test"},
                    "session_id": prod_sid,
                },
                "id": 3,
            },
        )
        assert call_prod.json()["result"]["data"] == "LIVE: test"

        # 2. Simulation session cannot be closed from production mode
        sim_sid = sim_client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "openenv/session/create", "id": 4},
        ).json()["result"]["session_id"]

        prod_close = prod_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sim_sid},
                "id": 5,
            },
        )
        assert prod_close.json()["error"]["code"] == -32602
        assert (
            "belongs to mode 'simulation', not 'production'"
            in prod_close.json()["error"]["message"]
        )

        # Verify simulation session was not destroyed and still works
        call_sim = sim_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "search_mock",
                    "arguments": {"query": "test"},
                    "session_id": sim_sid,
                },
                "id": 6,
            },
        )
        assert call_sim.json()["result"]["data"] == "MOCK: test"

        # 3. Same-mode session close still succeeds
        prod_ok = prod_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": prod_sid},
                "id": 7,
            },
        )
        assert prod_ok.json()["result"] == {
            "session_id": prod_sid,
            "closed": True,
        }

        sim_ok = sim_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "openenv/session/close",
                "params": {"session_id": sim_sid},
                "id": 8,
            },
        )
        assert sim_ok.json()["result"] == {
            "session_id": sim_sid,
            "closed": True,
        }

    def test_tools_list_propagates_handler_failure(self):
        """When _async_handle_list_tools fails, the /mcp JSON-RPC response returns an error instead of an empty list."""

        class FailingListToolsEnv(ModeAwareTestEnvironment):
            async def _async_handle_list_tools(self):
                return ListToolsObservation(
                    tools=[],
                    metadata={
                        "error": "FastMCP client connection failed",
                        "error_type": "list_tools_failed",
                    },
                )

        app = FastAPI()
        server = HTTPEnvServer(
            env=FailingListToolsEnv,
            action_cls=CallToolAction,
            observation_cls=CallToolObservation,
        )
        server.register_routes(app, mode="production")
        client = TestClient(app)

        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == -32603
        assert "FastMCP client connection failed" in data["error"]["message"]
        assert "result" not in data


@pytest.mark.parametrize(
    "mode,expected",
    [("production", "PRODUCTION_OVERRIDE"), ("simulation", "FASTMCP_BASE")],
)
async def test_shared_tool_remains_available_outside_override_mode(mode, expected):
    env = ModeAwareTestEnvironment()
    env.set_mode(mode)
    try:
        listed = await env._async_handle_list_tools()
        assert [t.name for t in listed.tools].count("override_me") == 1
        called = await env._async_handle_call_tool(
            CallToolAction(tool_name="override_me", arguments={})
        )
        assert called.error is None
        assert called.result.data == expected
    finally:
        env.close()


@pytest.mark.parametrize(
    "mode,expected",
    [("production", "PRODUCTION_OVERRIDE"), ("simulation", "FASTMCP_BASE")],
)
def test_http_shared_tool_remains_available_outside_override_mode(mode, expected):
    server = HTTPEnvServer(
        ModeAwareTestEnvironment, CallToolAction, CallToolObservation
    )
    app = FastAPI()
    server.register_routes(app, mode=mode)
    with TestClient(app) as client:
        listed = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        ).json()
        assert [t["name"] for t in listed["result"]["tools"]].count("override_me") == 1
        called = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "override_me", "arguments": {}},
                "id": 2,
            },
        ).json()
        assert "error" not in called
        assert called["result"]["data"] == expected
