# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the τ-emotion-bench environment.

Example:
    >>> from tau_emotion_env import TauEmotionEnv
    >>>
    >>> async with TauEmotionEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     print(result.observation.metadata["user_message"])
    ...     reply = await env.call_tool("respond_to_user", message="Could you give me your user id?")
"""

from openenv.core.mcp_client import MCPToolClient


class TauEmotionEnv(MCPToolClient):
    """Client for the τ-emotion-bench environment; `MCPToolClient` provides everything."""

    pass
