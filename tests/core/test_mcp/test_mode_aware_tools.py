# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for mode-aware tool registration (Task #5 of Issue #359).

These tests verify that environment authors can register different implementations
of the same tool for production vs simulation modes. This enables patterns like:

- Production: Real API calls to external services (e.g., Expedia)
- Simulation: Mocked/local versions for training (e.g., local database)

The API should be intuitive and follow FastMCP's @mcp.tool() decorator pattern,
but with mode awareness.

Test Strategy:
- High signal: Test actual mode switching behavior (this is the core feature)
- High signal: Test that tools are correctly filtered by mode
- Medium signal: Test API ergonomics (decorator usage, error cases)
- Low signal (skip): Testing FastMCP internals

References:
- Issue #359: Enable registering different tools for prod mode vs sim mode
- PR #348: Production mode implementation
"""

import asyncio

import pytest
from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Observation


# ============================================================================
# Test Fixtures
# ============================================================================


class MinimalMCPEnv(MCPEnvironment):
    """Minimal MCP environment for testing."""

    def __init__(self, mcp_server):
        super().__init__(mcp_server)
        self._mode = "simulation"  # Default mode

    def reset(self, seed=None, episode_id=None, **kwargs):
        return Observation()

    def _step_impl(self, action, timeout_s=None, **kwargs):
        return Observation()

    @property
    def state(self):
        return {}

    def set_mode(self, mode: str):
        """Set the environment mode for testing."""
        self._mode = mode


# ============================================================================
# Mode-Aware Registration API Tests
# ============================================================================


class TestModeAwareRegistrationAPI:
    """Test that a mode-aware registration API exists and is usable."""

    def test_mcp_environment_has_mode_aware_tool_decorator(self):
        """Test that MCPEnvironment provides a mode-aware tool decorator."""
        # This test will fail until we implement the API
        # Expected usage pattern:
        #
        #   class MyEnv(MCPEnvironment):
        #       def __init__(self):
        #           mcp = FastMCP("my-server")
        #           super().__init__(mcp)
        #
        #           @self.tool(mode="production")
        #           def expedia_search(query: str) -> dict:
        #               return call_real_expedia_api(query)
        #
        #           @self.tool(mode="simulation")
        #           def expedia_search(query: str) -> dict:
        #               return query_local_database(query)
        #
        # Alternative: Use a method on the mcp server itself
        #   @self.mcp_tool(mode="production")
        #
        # The decorator should be accessible and not raise an error
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        # FAILING TEST: Check that mode-aware decorator exists
        assert hasattr(env, "tool") or hasattr(env, "mcp_tool"), (
            "MCPEnvironment should provide a mode-aware tool decorator"
        )

    def test_tool_decorator_accepts_mode_parameter(self):
        """Test that the tool decorator accepts a 'mode' parameter."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        # FAILING TEST: Decorator should accept mode parameter
        # We expect something like:
        #   @env.tool(mode="production")
        #   def my_tool(): pass
        #
        # This should not raise an error
        try:
            if hasattr(env, "tool"):

                @env.tool(mode="production")
                def test_tool_prod(x: int) -> int:
                    return x * 2

                success = True
            else:
                success = False
        except (AttributeError, TypeError):
            success = False

        assert success, "tool decorator should accept mode parameter"

    def test_can_register_production_mode_tool(self):
        """Test that a tool can be registered for production mode only."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        # FAILING TEST: Register a production-only tool
        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="production")
        def prod_only_tool(value: str) -> str:
            return f"prod: {value}"

        # Tool should be registered
        # We'll verify this in integration tests

    def test_can_register_simulation_mode_tool(self):
        """Test that a tool can be registered for simulation mode only."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        # FAILING TEST: Register a simulation-only tool
        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="simulation")
        def sim_only_tool(value: str) -> str:
            return f"sim: {value}"

        # Tool should be registered
        # We'll verify this in integration tests


# ============================================================================
# Same Tool Name, Different Modes Tests
# ============================================================================


class TestSameToolDifferentModes:
    """Test registering different implementations for the same tool name."""

    def test_can_register_same_tool_name_for_different_modes(self):
        """Test that the same tool name can have different implementations per mode."""
        # This is the core use case from Issue #359:
        # - expedia_search in production → calls real API
        # - expedia_search in simulation → calls local mock
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # FAILING TEST: Register same tool name with different modes
        @env.tool(mode="production")
        def expedia_search(query: str) -> str:
            return "real_api_result"

        @env.tool(mode="simulation")
        def expedia_search(query: str) -> str:  # noqa: F811
            return "mock_result"

        # Both should register without conflict
        # The system should track them separately by (name, mode) pair

    def test_different_mode_implementations_are_tracked_separately(self):
        """Test that prod and sim implementations don't override each other."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="production")
        def my_tool(x: int) -> str:
            return f"prod_{x}"

        @env.tool(mode="simulation")
        def my_tool(x: int) -> str:  # noqa: F811
            return f"sim_{x}"

        # FAILING TEST: Both implementations should exist
        # Internal state should track: {"my_tool": {"production": fn1, "simulation": fn2}}
        # We verify this through behavior in integration tests


# ============================================================================
# Tool Discovery by Mode Tests
# ============================================================================


class TestToolDiscoveryByMode:
    """Test that list_tools returns tools filtered by current mode."""

    def test_list_tools_shows_only_production_tools_in_prod_mode(self):
        """Test that production mode only shows production tools."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register tools for different modes
        @env.tool(mode="production")
        def prod_tool(x: int) -> int:
            return x

        @env.tool(mode="simulation")
        def sim_tool(x: int) -> int:
            return x * 2

        @env.tool(mode="production")
        def another_prod_tool(x: int) -> int:
            return x + 1

        # FAILING TEST: Set mode to production and list tools
        env.set_mode("production")
        obs = env.step(ListToolsAction())

        assert isinstance(obs, ListToolsObservation)
        tool_names = {tool.name for tool in obs.tools}

        # Should only see production tools
        assert "prod_tool" in tool_names, "Production tool should be visible"
        assert "another_prod_tool" in tool_names, (
            "Another production tool should be visible"
        )
        assert "sim_tool" not in tool_names, "Simulation tool should NOT be visible"

    def test_list_tools_shows_only_simulation_tools_in_sim_mode(self):
        """Test that simulation mode only shows simulation tools."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register tools for different modes
        @env.tool(mode="production")
        def prod_tool(x: int) -> int:
            return x

        @env.tool(mode="simulation")
        def sim_tool(x: int) -> int:
            return x * 2

        @env.tool(mode="simulation")
        def another_sim_tool(x: int) -> int:
            return x + 1

        # FAILING TEST: Set mode to simulation and list tools
        env.set_mode("simulation")
        obs = env.step(ListToolsAction())

        assert isinstance(obs, ListToolsObservation)
        tool_names = {tool.name for tool in obs.tools}

        # Should only see simulation tools
        assert "sim_tool" in tool_names, "Simulation tool should be visible"
        assert "another_sim_tool" in tool_names, (
            "Another simulation tool should be visible"
        )
        assert "prod_tool" not in tool_names, "Production tool should NOT be visible"

    def test_list_tools_shows_both_mode_versions_of_same_tool(self):
        """Test that same tool name appears in both modes with correct implementation."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register same tool name for both modes
        @env.tool(mode="production")
        def expedia_search(query: str) -> str:
            return "real"

        @env.tool(mode="simulation")
        def expedia_search(query: str) -> str:  # noqa: F811
            return "mock"

        # FAILING TEST: Tool should appear in both modes
        env.set_mode("production")
        obs_prod = env.step(ListToolsAction())
        prod_tools = {tool.name for tool in obs_prod.tools}
        assert "expedia_search" in prod_tools

        env.set_mode("simulation")
        obs_sim = env.step(ListToolsAction())
        sim_tools = {tool.name for tool in obs_sim.tools}
        assert "expedia_search" in sim_tools


# ============================================================================
# Tool Execution by Mode Tests
# ============================================================================


class TestToolExecutionByMode:
    """Test that calling a tool executes the correct mode-specific implementation."""

    def test_call_tool_executes_production_implementation_in_prod_mode(self):
        """Test that prod mode executes the production implementation."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register different implementations
        @env.tool(mode="production")
        def compute(x: int) -> str:
            return f"PROD_{x}"

        @env.tool(mode="simulation")
        def compute(x: int) -> str:  # noqa: F811
            return f"SIM_{x}"

        # FAILING TEST: Call in production mode
        env.set_mode("production")
        obs = env.step(CallToolAction(tool_name="compute", arguments={"x": 42}))

        assert isinstance(obs, CallToolObservation)
        assert obs.error is None, f"Should not error: {obs.error}"

        # Should execute production implementation
        # Result handling depends on FastMCP's CallToolResult wrapper
        result = obs.result
        if hasattr(result, "data"):
            result = result.data
        elif isinstance(result, dict) and "data" in result:
            result = result["data"]

        assert result == "PROD_42", f"Expected PROD_42, got {result}"

    def test_call_tool_executes_simulation_implementation_in_sim_mode(self):
        """Test that sim mode executes the simulation implementation."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register different implementations
        @env.tool(mode="production")
        def compute(x: int) -> str:
            return f"PROD_{x}"

        @env.tool(mode="simulation")
        def compute(x: int) -> str:  # noqa: F811
            return f"SIM_{x}"

        # FAILING TEST: Call in simulation mode
        env.set_mode("simulation")
        obs = env.step(CallToolAction(tool_name="compute", arguments={"x": 42}))

        assert isinstance(obs, CallToolObservation)
        assert obs.error is None, f"Should not error: {obs.error}"

        # Should execute simulation implementation
        result = obs.result
        if hasattr(result, "data"):
            result = result.data
        elif isinstance(result, dict) and "data" in result:
            result = result["data"]

        assert result == "SIM_42", f"Expected SIM_42, got {result}"


# ============================================================================
# Mode Switching Tests
# ============================================================================


class TestModeSwitching:
    """Test that switching modes correctly toggles between tool implementations."""

    def test_switching_mode_toggles_tool_implementation(self):
        """Test that switching between modes executes the correct implementation."""
        # This is the key requirement from Issue #359:
        # "We should have tests that check that switching back and forth
        # keeps toggling tools correctly."
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # Register mode-specific implementations
        @env.tool(mode="production")
        def api_call(param: str) -> str:
            return f"real_api_{param}"

        @env.tool(mode="simulation")
        def api_call(param: str) -> str:  # noqa: F811
            return f"mock_api_{param}"

        # FAILING TEST: Toggle between modes multiple times
        # Start in production
        env.set_mode("production")
        obs1 = env.step(CallToolAction(tool_name="api_call", arguments={"param": "A"}))
        result1 = obs1.result
        if hasattr(result1, "data"):
            result1 = result1.data
        elif isinstance(result1, dict) and "data" in result1:
            result1 = result1["data"]
        assert result1 == "real_api_A", "First call should use production"

        # Switch to simulation
        env.set_mode("simulation")
        obs2 = env.step(CallToolAction(tool_name="api_call", arguments={"param": "B"}))
        result2 = obs2.result
        if hasattr(result2, "data"):
            result2 = result2.data
        elif isinstance(result2, dict) and "data" in result2:
            result2 = result2["data"]
        assert result2 == "mock_api_B", "Second call should use simulation"

        # Switch back to production
        env.set_mode("production")
        obs3 = env.step(CallToolAction(tool_name="api_call", arguments={"param": "C"}))
        result3 = obs3.result
        if hasattr(result3, "data"):
            result3 = result3.data
        elif isinstance(result3, dict) and "data" in result3:
            result3 = result3["data"]
        assert result3 == "real_api_C", "Third call should use production again"

        # Switch back to simulation again
        env.set_mode("simulation")
        obs4 = env.step(CallToolAction(tool_name="api_call", arguments={"param": "D"}))
        result4 = obs4.result
        if hasattr(result4, "data"):
            result4 = result4.data
        elif isinstance(result4, dict) and "data" in result4:
            result4 = result4["data"]
        assert result4 == "mock_api_D", "Fourth call should use simulation again"

    def test_mode_switch_updates_list_tools(self):
        """Test that switching mode updates the list of available tools."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="production")
        def prod_only(x: int) -> int:
            return x

        @env.tool(mode="simulation")
        def sim_only(x: int) -> int:
            return x

        # FAILING TEST: List tools should change when mode switches
        env.set_mode("production")
        prod_tools = env.step(ListToolsAction()).tools
        prod_names = {tool.name for tool in prod_tools}

        env.set_mode("simulation")
        sim_tools = env.step(ListToolsAction()).tools
        sim_names = {tool.name for tool in sim_tools}

        assert "prod_only" in prod_names
        assert "prod_only" not in sim_names
        assert "sim_only" in sim_names
        assert "sim_only" not in prod_names


# ============================================================================
# Default Behavior Tests
# ============================================================================


class TestDefaultBehavior:
    """Test behavior when mode is not specified."""

    def test_tool_without_mode_available_in_all_modes(self):
        """Test that tools without mode parameter work in both modes."""
        # From Issue #359: "Whenever a mapping is not specified,
        # obviously the same tool will be used in every mode."
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # FAILING TEST: Register a tool without specifying mode
        @env.tool()
        def universal_tool(x: int) -> int:
            return x * 10

        # Should be available in production mode
        env.set_mode("production")
        prod_tools = env.step(ListToolsAction()).tools
        prod_names = {tool.name for tool in prod_tools}
        assert "universal_tool" in prod_names, "Should be available in production"

        # Should be available in simulation mode
        env.set_mode("simulation")
        sim_tools = env.step(ListToolsAction()).tools
        sim_names = {tool.name for tool in sim_tools}
        assert "universal_tool" in sim_names, "Should be available in simulation"

        # Should execute same implementation in both modes
        env.set_mode("production")
        obs_prod = env.step(
            CallToolAction(tool_name="universal_tool", arguments={"x": 5})
        )
        result_prod = obs_prod.result
        if hasattr(result_prod, "data"):
            result_prod = result_prod.data
        elif isinstance(result_prod, dict) and "data" in result_prod:
            result_prod = result_prod["data"]

        env.set_mode("simulation")
        obs_sim = env.step(
            CallToolAction(tool_name="universal_tool", arguments={"x": 5})
        )
        result_sim = obs_sim.result
        if hasattr(result_sim, "data"):
            result_sim = result_sim.data
        elif isinstance(result_sim, dict) and "data" in result_sim:
            result_sim = result_sim["data"]

        assert result_prod == result_sim == 50, (
            "Should have same behavior in both modes"
        )


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error cases for mode-aware tool registration."""

    def test_calling_production_tool_in_simulation_mode_fails(self):
        """Test that calling a production-only tool in sim mode returns error."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="production")
        def prod_only_tool(x: int) -> int:
            return x

        # FAILING TEST: Calling prod-only tool in sim mode should fail
        env.set_mode("simulation")
        obs = env.step(CallToolAction(tool_name="prod_only_tool", arguments={"x": 1}))

        assert isinstance(obs, CallToolObservation)
        assert obs.error is not None, "Should return an error"
        assert obs.error.error_type.value == "tool_not_found", (
            "Should be tool_not_found error"
        )

    def test_calling_simulation_tool_in_production_mode_fails(self):
        """Test that calling a simulation-only tool in prod mode returns error."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        @env.tool(mode="simulation")
        def sim_only_tool(x: int) -> int:
            return x

        # FAILING TEST: Calling sim-only tool in prod mode should fail
        env.set_mode("production")
        obs = env.step(CallToolAction(tool_name="sim_only_tool", arguments={"x": 1}))

        assert isinstance(obs, CallToolObservation)
        assert obs.error is not None, "Should return an error"
        assert obs.error.error_type.value == "tool_not_found", (
            "Should be tool_not_found error"
        )

    def test_invalid_mode_value_raises_error(self):
        """Test that registering a tool with invalid mode raises error."""
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # FAILING TEST: Invalid mode should raise ValueError
        with pytest.raises(ValueError, match="mode"):

            @env.tool(mode="invalid_mode")
            def bad_tool(x: int) -> int:
                return x

    def test_reserved_name_raises_error(self):
        """Test that registering a tool with a reserved name raises ValueError.

        The @self.tool() decorator should validate against RESERVED_TOOL_NAMES
        to prevent conflicts with MCPEnvironment's core methods (reset, step, state, close).
        """
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # FAILING TEST: Reserved name "reset" should raise ValueError
        with pytest.raises(ValueError, match="reserved"):

            @env.tool()
            def reset(x: int) -> int:
                return x

        # FAILING TEST: Reserved name "step" should raise ValueError
        with pytest.raises(ValueError, match="reserved"):

            @env.tool()
            def step(x: int) -> int:
                return x

        # FAILING TEST: Reserved name "state" should raise ValueError
        with pytest.raises(ValueError, match="reserved"):

            @env.tool()
            def state(x: int) -> int:
                return x

        # FAILING TEST: Reserved name "close" should raise ValueError
        with pytest.raises(ValueError, match="reserved"):

            @env.tool()
            def close(x: int) -> int:
                return x

    def test_async_mode_specific_tool_is_awaited(self):
        """Test that async mode-specific tools are properly awaited.

        Mode-specific tool execution must handle async functions correctly,
        awaiting coroutines instead of returning them raw.
        """
        mcp = FastMCP("test-server")
        env = MinimalMCPEnv(mcp)

        if not hasattr(env, "tool"):
            pytest.skip("Mode-aware tool decorator not yet implemented")

        # FAILING TEST: Register async tools for different modes
        @env.tool(mode="production")
        async def async_compute(x: int) -> str:
            # Simulate async work
            await asyncio.sleep(0.001)
            return f"ASYNC_PROD_{x}"

        @env.tool(mode="simulation")
        async def async_compute(x: int) -> str:  # noqa: F811
            # Simulate async work
            await asyncio.sleep(0.001)
            return f"ASYNC_SIM_{x}"

        # Test production mode
        env.set_mode("production")
        obs_prod = env.step(
            CallToolAction(tool_name="async_compute", arguments={"x": 99})
        )

        assert isinstance(obs_prod, CallToolObservation)
        assert obs_prod.error is None, f"Should not error: {obs_prod.error}"

        result_prod = obs_prod.result
        if hasattr(result_prod, "data"):
            result_prod = result_prod.data
        elif isinstance(result_prod, dict) and "data" in result_prod:
            result_prod = result_prod["data"]

        # This should NOT be a coroutine object
        assert not asyncio.iscoroutine(result_prod), (
            "Result should be awaited, not a coroutine"
        )
        assert result_prod == "ASYNC_PROD_99", (
            f"Expected ASYNC_PROD_99, got {result_prod}"
        )

        # Test simulation mode
        env.set_mode("simulation")
        obs_sim = env.step(
            CallToolAction(tool_name="async_compute", arguments={"x": 99})
        )

        assert isinstance(obs_sim, CallToolObservation)
        assert obs_sim.error is None, f"Should not error: {obs_sim.error}"

        result_sim = obs_sim.result
        if hasattr(result_sim, "data"):
            result_sim = result_sim.data
        elif isinstance(result_sim, dict) and "data" in result_sim:
            result_sim = result_sim["data"]

        # This should NOT be a coroutine object
        assert not asyncio.iscoroutine(result_sim), (
            "Result should be awaited, not a coroutine"
        )
        assert result_sim == "ASYNC_SIM_99", f"Expected ASYNC_SIM_99, got {result_sim}"
