# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenClaw Environment Client.

This module provides the client for connecting to an OpenClaw Environment server.
OpenClawEnv extends MCPToolClient to provide tool-calling style interactions
with OpenClaw's agentic capabilities.

Example:
    >>> with OpenClawEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Execute shell command
    ...     result = env.call_tool("exec", command="ls -la")
    ...     print(result)
    ...
    ...     # Read a file
    ...     result = env.call_tool("read", path="README.md")
    ...     print(result)
    ...
    ...     # Search the web
    ...     result = env.call_tool("web_search", query="PyTorch tutorials")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class OpenClawEnv(MCPToolClient):
    """
    Client for the OpenClaw Environment.

    This client provides an interface for interacting with OpenClaw's
    agentic capabilities via MCP tools. It inherits all functionality
    from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Available Tools:
        - exec: Execute shell commands in a sandboxed environment
        - read: Read file contents
        - write: Write content to files
        - edit: Make precise edits to files
        - web_search: Search the web using Brave Search API
        - web_fetch: Fetch and extract content from URLs
        - memory_search: Search memory/context files
        - memory_get: Get snippets from memory files

    Example:
        >>> # Connect to a running server
        >>> with OpenClawEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Execute a shell command
        ...     result = env.call_tool("exec", command="pwd")
        ...     print(result)
        ...
        ...     # Read a file
        ...     result = env.call_tool("read", path="setup.py")
        ...     print(result)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> env = OpenClawEnv.from_docker_image("openclaw-env:latest")
        >>> try:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("exec", command="echo hello")
        ... finally:
        ...     env.close()

    Example with HuggingFace Space:
        >>> # Run from HuggingFace Space
        >>> env = OpenClawEnv.from_env("openenv/openclaw-env")
        >>> try:
        ...     env.reset()
        ...     result = env.call_tool("web_search", query="reinforcement learning")
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
