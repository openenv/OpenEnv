"""
Agent World Model Environment Client.

Provides a client for connecting to an AWM Environment server.
AWMEnv extends MCPToolClient with AWM-specific helpers.

Example:
    >>> with AWMEnv(base_url="http://localhost:8000") as env:
    ...     # List all available scenarios
    ...     result = env.call_tool("__list_scenarios__")
    ...
    ...     # Start a scenario
    ...     env.reset(scenario="e_commerce_33", task_idx=0)
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Call tools
    ...     result = env.call_tool("search_products", query="headphones")
    ...
    ...     # End episode and get verification result
    ...     result = env.call_tool("done")
"""

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.mcp_client import MCPToolClient

from .models import AWMListToolsObservation, AWMObservation


class AWMEnv(MCPToolClient):
    """
    Client for the Agent World Model Environment.
    Inherits all functionality from MCPToolClient.
    AWM-specific reset parameters:
        scenario (str): Required. Name of the AWM scenario to load.
        task_idx (int): Optional. Index of the pre-defined task/verifier pair.
        task (str): Optional. Custom task description (no verifier).
        verifier_mode (str): "sql" or "code". Default: "sql".
        llm_base_url (str): LLM endpoint for sql verifier using any OpenAI compatible API service.
        llm_api_key (str): LLM API key.
        llm_model (str): LLM model name.

    Hidden tools (not in list_tools):
        done: End the episode and trigger verification.
        __list_scenarios__: List all 1,000 available scenarios.

    Accessing AWM-specific fields:
        After reset/step, access fields via observation attributes:
        >>> result = await env.reset(scenario="marketplace_1", task_idx=0)
        >>> print(result.observation.reward_type)  # "reset_ok"
        >>> print(result.observation.scenario)     # "marketplace_1"
        >>> print(result.observation.task)         # "Update my volunteer profile..."
    """

    def _parse_result(self, payload: dict[str, Any]) -> StepResult:
        """
        This override ensures AWM fields (reward_type, scenario, task, etc.)
        are available as observation attributes instead of being lost.
        """
        obs_data = payload.get("observation", {})

        # Check if this is a ListToolsObservation (has "tools" key)
        if "tools" in obs_data:
            observation = AWMListToolsObservation(**obs_data)
        else:
            observation = AWMObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
