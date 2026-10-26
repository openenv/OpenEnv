# SPDX-License-Identifier: BSD-3-Clause

"""Core components for agentic environments."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import env_server
from .env_server import *  # noqa: F403

if TYPE_CHECKING:
    from .env_client import EnvClient
    from .generic_client import GenericAction, GenericEnvClient
    from .llm_client import (
        AnthropicClient,
        create_llm_client,
        LLMClient,
        LLMResponse,
        OpenAIClient,
        ToolCall,
    )
    from .mcp_client import MCPClientBase, MCPToolClient
    from .sync_client import SyncEnvClient

__all__ = [
    "EnvClient",
    "SyncEnvClient",
    "GenericEnvClient",
    "GenericAction",
    "MCPClientBase",
    "MCPToolClient",
    "AnthropicClient",
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "ToolCall",
    "create_llm_client",
] + env_server.__all__  # type: ignore


_LAZY_ATTRS = {
    "EnvClient": (".env_client", "EnvClient"),
    "SyncEnvClient": (".sync_client", "SyncEnvClient"),
    "GenericEnvClient": (".generic_client", "GenericEnvClient"),
    "GenericAction": (".generic_client", "GenericAction"),
    "MCPClientBase": (".mcp_client", "MCPClientBase"),
    "MCPToolClient": (".mcp_client", "MCPToolClient"),
    "AnthropicClient": (".llm_client", "AnthropicClient"),
    "LLMClient": (".llm_client", "LLMClient"),
    "LLMResponse": (".llm_client", "LLMResponse"),
    "OpenAIClient": (".llm_client", "OpenAIClient"),
    "ToolCall": (".llm_client", "ToolCall"),
    "create_llm_client": (".llm_client", "create_llm_client"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    try:
        value = getattr(env_server, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
