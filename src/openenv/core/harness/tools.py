# SPDX-License-Identifier: BSD-3-Clause

"""Tool name conflict resolution for harness tool injection (RFC 005)."""

from __future__ import annotations

from typing import Collection

from ..env_server.mcp_types import RESERVED_TOOL_NAMES, Tool


def resolve_tool_conflicts(
    env_tools: list[Tool],
    harness_builtin_tools: Collection[str],
    prefix: str = "env_",
) -> list[Tool]:
    """
    Detect and resolve name conflicts between environment and harness tools.

    Environment tools whose names collide with the harness's built-in tools
    are renamed with `prefix`. Ambiguous situations fail loudly instead of
    silently overriding.

    Args:
        env_tools (`list[Tool]`):
            Environment MCP tool definitions to inject into the harness.
        harness_builtin_tools (`Collection[str]`):
            Names of the harness's built-in tools.
        prefix (`str`, *optional*, defaults to `"env_"`):
            Prefix applied to conflicting environment tool names.

    Returns:
        `list[Tool]` of conflict-free tool definitions. The input list is not
        mutated.

    Raises:
        `ValueError`: If an environment tool uses a reserved orchestration
            name, if environment tool names are duplicated, or if a prefixed
            name still collides with a built-in or another environment tool.
    """
    reserved = sorted({tool.name for tool in env_tools} & RESERVED_TOOL_NAMES)
    if reserved:
        raise ValueError(
            "Tool names are reserved for orchestration controls: " + ", ".join(reserved)
        )

    seen: dict[str, Tool] = {}
    for tool in env_tools:
        if tool.name in seen:
            raise ValueError(f"Duplicate environment tool name: {tool.name}")
        seen[tool.name] = tool

    builtin_names = set(harness_builtin_tools)
    env_names = set(seen)
    resolved: list[Tool] = []
    taken: set[str] = set()
    for tool in env_tools:
        name = tool.name
        if name in builtin_names:
            renamed = f"{prefix}{name}"
            if renamed in builtin_names or renamed in env_names:
                raise ValueError(
                    f"Cannot resolve tool name conflict: '{name}' collides with a "
                    f"harness built-in tool and '{renamed}' is also taken"
                )
            name = renamed
        if name in taken:
            raise ValueError(f"Tool name conflict after prefixing: {name}")
        taken.add(name)
        resolved.append(tool.model_copy(update={"name": name}))
    return resolved


__all__ = ["resolve_tool_conflicts"]
