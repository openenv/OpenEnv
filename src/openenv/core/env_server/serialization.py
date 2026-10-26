# SPDX-License-Identifier: BSD-3-Clause

"""
Shared serialization and deserialization utilities for OpenEnv HTTP servers.

This module provides common utilities for converting between JSON dictionaries
and Pydantic models (Action/Observation) to eliminate code duplication across
HTTP server and web interface implementations.
"""

import json
from typing import Any, Dict, Type

from .mcp_types import CallToolAction, ListToolsAction
from .types import Action, Observation

# MCP action types keyed by their "type" discriminator value.
# These are checked before the environment's own action_cls so that
# ListToolsAction / CallToolAction payloads are never rejected by an
# unrelated Pydantic model.
_MCP_ACTION_TYPES: Dict[str, Type[Action]] = {
    "list_tools": ListToolsAction,
    "call_tool": CallToolAction,
}


def _deserialize_mcp_action(
    action_data: Dict[str, Any], action_cls: Type[Action]
) -> Action | None:
    # Only intercept when action_cls is the generic Action base or itself an
    # MCP type. This keeps env-specific action validation authoritative.
    action_type = action_data.get("type")
    if action_type not in _MCP_ACTION_TYPES:
        return None

    mcp_cls = _MCP_ACTION_TYPES[action_type]
    if action_cls is Action or action_cls in _MCP_ACTION_TYPES.values():
        return mcp_cls.model_validate(action_data)

    return None


def deserialize_action(action_data: Dict[str, Any], action_cls: Type[Action]) -> Action:
    """
    Convert JSON dict to Action instance using Pydantic validation.

    MCP action types (``list_tools``, ``call_tool``) are recognised
    automatically via the ``"type"`` discriminator field, regardless of
    the environment's configured ``action_cls``.  All other payloads
    fall through to ``action_cls.model_validate()``.

    For special cases (e.g., tensor fields, custom type conversions),
    use deserialize_action_with_preprocessing().

    Args:
        action_data (`dict`):
            Dictionary containing action data.
        action_cls (`type`):
            The Action subclass to instantiate.

    Returns:
        `Action` instance.

    Raises:
        `ValidationError`: If `action_data` is invalid for the action class.
    """
    mcp_action = _deserialize_mcp_action(action_data, action_cls)
    if mcp_action is not None:
        return mcp_action

    return action_cls.model_validate(action_data)


def deserialize_action_with_preprocessing(
    action_data: Dict[str, Any], action_cls: Type[Action]
) -> Action:
    """
    Convert JSON dict to Action instance with preprocessing for special types.

    This version handles common type conversions needed for web interfaces:
    - Converting JSON string arguments to dict for MCP call_tool actions
    - Converting lists/strings to tensors for 'tokens' field
    - Converting string action_id to int
    - Other custom preprocessing as needed

    Args:
        action_data (`dict`):
            Dictionary containing action data.
        action_cls (`type`):
            The Action subclass to instantiate.

    Returns:
        `Action` instance.

    Raises:
        `ValidationError`: If `action_data` is invalid for the action class.
    """
    mcp_data = action_data
    if action_data.get("type") == "call_tool" and isinstance(
        action_data.get("arguments"), str
    ):
        mcp_data = dict(action_data)
        try:
            mcp_data["arguments"] = json.loads(action_data["arguments"])
        except Exception:
            pass

    mcp_action = _deserialize_mcp_action(mcp_data, action_cls)
    if mcp_action is not None:
        return mcp_action

    processed_data = {}

    for key, value in action_data.items():
        if key == "tokens" and isinstance(value, (list, str)):
            # Convert list or string to tensor
            if isinstance(value, str):
                # If it's a string, try to parse it as a list of numbers
                try:
                    value = json.loads(value)
                except Exception:
                    # If parsing fails, treat as empty list
                    value = []
            if isinstance(value, list):
                try:
                    import torch  # type: ignore

                    processed_data[key] = torch.tensor(value, dtype=torch.long)
                except ImportError:
                    # If torch not available, keep as list
                    processed_data[key] = value
            else:
                processed_data[key] = value
        elif key == "action_id" and isinstance(value, str):
            # Convert action_id from string to int
            try:
                processed_data[key] = int(value)
            except ValueError:
                # If conversion fails, keep original value
                processed_data[key] = value
        else:
            processed_data[key] = value

    return action_cls.model_validate(processed_data)


def serialize_observation(observation: Observation) -> Dict[str, Any]:
    """
    Convert Observation instance to JSON-compatible dict using Pydantic.

    Args:
        observation (`Observation`):
            Observation instance to serialize.

    Returns:
        `dict` compatible with `EnvClient._parse_result()`, with keys:
            - `observation` (`dict`): Observation fields.
            - `reward` (`float` or `None`): Reward value.
            - `done` (`bool`): Whether the episode is done.
            - `metadata` (`dict`, *optional*): Additional observation metadata.
    """
    # Keep metadata in the nested observation payload for backwards
    # compatibility with typed clients, and also surface it as a top-level
    # sibling for clients that read the generic wire format directly.
    obs_dict = observation.model_dump(
        exclude={
            "reward",
            "done",
        }  # Exclude these from observation dict
    )

    # Extract reward, done, and metadata directly from the observation.
    reward = observation.reward
    done = observation.done
    metadata = observation.metadata

    # Return in EnvClient expected format.
    result = {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }
    if metadata:
        result["metadata"] = metadata
    return result
