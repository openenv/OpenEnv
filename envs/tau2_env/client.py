# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the τ²-bench environment.

Example:
    >>> from tau2_env import Tau2Env
    >>>
    >>> async with Tau2Env(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     print(result.observation.metadata["user_message"])
    ...     reply = await env.call_tool("respond_to_user", message="Could you give me your user id?")
"""

from openenv.core.mcp_client import MCPToolClient


class Tau2Env(MCPToolClient):
    """Client for the τ²-bench environment; `MCPToolClient` provides everything."""

    pass
