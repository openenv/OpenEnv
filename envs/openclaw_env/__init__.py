# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenClaw Environment - An MCP environment for training agents on OpenClaw's tool ecosystem.

OpenClaw is a personal AI assistant framework with access to local tools:
- File system operations (read, write, edit)
- Shell command execution
- Web search and fetch
- Memory/context management

This environment exposes OpenClaw's capabilities as MCP tools for RL training,
enabling agents to learn agentic workflows like coding, research, and automation.

Example:
    >>> from openclaw_env import OpenClawEnv
    >>>
    >>> with OpenClawEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("exec", command="echo hello")
    ...     print(result)
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import OpenClawEnv

__all__ = ["OpenClawEnv", "CallToolAction", "ListToolsAction"]
