# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State for the τ²-bench environment.

The environment is MCP-only: use `ListToolsAction` and `CallToolAction` from
`openenv.core.env_server.mcp_types` to interact with it.
"""

from typing import Any

from openenv.core.env_server import State


class Tau2State(State):
    """
    Episode state.

    Attributes:
        domain: τ²-bench domain of this environment.
        split: Task split tasks are drawn from.
        task_id: The task of the current episode.
        done: Whether the conversation has ended.
        reward: τ²-bench's reward, set when the conversation ends.
        reward_info: τ²-bench's evaluation breakdown.
    """

    domain: str = ""
    split: str = ""
    task_id: str = ""
    done: bool = False
    reward: float = 0.0
    reward_info: dict[str, Any] = {}
