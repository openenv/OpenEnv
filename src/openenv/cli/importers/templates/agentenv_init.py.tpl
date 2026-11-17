"""${env_name} environment: an AgentEnv bundle served through agentenv-framework."""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import ${client}

__all__ = ["${client}", "CallToolAction", "ListToolsAction"]
