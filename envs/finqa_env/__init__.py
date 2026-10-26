# envs/finqa_env/__init__.py
"""
FinQA Environment for OpenEnv.

A financial question-answering environment that evaluates LLMs on their ability
to answer complex financial questions using tool calls on SEC 10-K filing data.

Example:
    >>> from envs.finqa_env import FinQAEnv
    >>>
    >>> async with FinQAEnv(base_url="http://localhost:8000") as env:
    ...     await env.reset()
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("get_descriptions", company_name="alphabet")
    ...     result = await env.call_tool("submit_answer", answer="6.118")
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import FinQAEnv
from .models import FinQAState

__all__ = ["FinQAEnv", "FinQAState", "CallToolAction", "ListToolsAction"]
