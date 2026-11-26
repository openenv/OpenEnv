"""Remote MCP client. This module never imports the server or trace data."""

from openenv.core.mcp_client import MCPToolClient


class LLMCapacityEnv(MCPToolClient):
    """OpenEnv client for the LLM capacity reservation tools.

    Infrastructure may call reset/step; an agent should receive only
    observe_demand_history and reserve_capacity via MCP.
    """
