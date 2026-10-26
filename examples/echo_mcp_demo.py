#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause

"""
Echo MCP Demo - Demonstrating MCP tool usage via the canonical SIM mode pattern.

This example shows how to interact with MCP tools through OpenEnv's step() API,
which is the canonical pattern for SIM mode (training/simulation).

Usage:
    PYTHONPATH=src:envs uv run python examples/echo_mcp_demo.py

Key concepts:
    - ListToolsAction: Discover available MCP tools
    - CallToolAction: Invoke a specific MCP tool with arguments
    - step() is the unified API for both MCP and custom actions
"""

import sys
import os

# Add paths for in-repo execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))

from echo_env.server.echo_environment import EchoEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)


def main():
    """Demonstrate MCP tool usage with EchoEnvironment."""
    print("=" * 60)
    print("Echo MCP Demo - Canonical SIM Mode Pattern")
    print("=" * 60)
    print()

    # Create the environment
    env = EchoEnvironment()
    env.reset()

    # =================================================================
    # Step 1: List available tools via ListToolsAction
    # =================================================================
    print("Step 1: List available MCP tools")
    print("-" * 40)

    obs = env.step(ListToolsAction())

    assert isinstance(obs, ListToolsObservation)
    print(f"Found {len(obs.tools)} tools:")
    for tool in obs.tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # =================================================================
    # Step 2: Call echo_message tool via CallToolAction
    # =================================================================
    print("Step 2: Call 'echo_message' tool")
    print("-" * 40)

    obs = env.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello from MCP!"},
        )
    )

    assert isinstance(obs, CallToolObservation)
    print(f"Tool: {obs.tool_name}")
    # Extract the actual result value from CallToolResult
    result_value = obs.result.data if hasattr(obs.result, "data") else obs.result
    print(f"Result: {result_value}")
    print(f"Error: {obs.error}")
    print()

    # =================================================================
    # Step 3: Call echo_with_length tool via CallToolAction
    # =================================================================
    print("Step 3: Call 'echo_with_length' tool")
    print("-" * 40)

    obs = env.step(
        CallToolAction(
            tool_name="echo_with_length",
            arguments={"message": "Count my characters"},
        )
    )

    assert isinstance(obs, CallToolObservation)
    print(f"Tool: {obs.tool_name}")
    # Extract the actual result value from CallToolResult
    result_value = obs.result.data if hasattr(obs.result, "data") else obs.result
    print(f"Result: {result_value}")
    print(f"Error: {obs.error}")
    print()

    # =================================================================
    # Step 4: Handle tool errors gracefully
    # =================================================================
    print("Step 4: Handle non-existent tool")
    print("-" * 40)

    obs = env.step(
        CallToolAction(
            tool_name="nonexistent_tool",
            arguments={},
        )
    )

    assert isinstance(obs, CallToolObservation)
    print(f"Tool: {obs.tool_name}")
    print(f"Result: {obs.result}")
    print(f"Error: {obs.error}")
    print()

    # =================================================================
    # Summary
    # =================================================================
    print("=" * 60)
    print("Summary: The Canonical SIM Mode Pattern")
    print("=" * 60)
    print("""
In SIM mode (training/simulation), all MCP interactions go through step():

    # Discover tools
    obs = env.step(ListToolsAction())

    # Call a tool
    obs = env.step(CallToolAction(
        tool_name="tool_name",
        arguments={"key": "value"}
    ))

This unified API allows:
- Infrastructure to control time (step-by-step execution)
- Logging/recording of all agent actions
- Reward computation based on tool usage
- Consistent interface across all environments
""")


if __name__ == "__main__":
    main()
