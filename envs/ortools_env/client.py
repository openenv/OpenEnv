# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OR-Tools Environment Client.

`call_tool` returns only the tool's text. The reward arrives on the observation of the submitting step, so use
`step(CallToolAction(...))` for `submit_solution` / `submit_algorithm` and read `result.reward`.

Examples:

```python
from openenv.core.env_server.mcp_types import CallToolAction

with ORToolsEnv(base_url="http://localhost:8000").sync() as env:
    prompt = env.reset(problem="jobshop", size="15x15", seed=0).observation.metadata["prompt"]
    print(env.call_tool("run_python", code="print(instance['num_jobs'])"))
    result = env.step(CallToolAction(tool_name="submit_solution", arguments={"solution": {"starts": [...]}}))
    print(result.reward, result.done)
```
"""

from typing import Any, Dict

from openenv.core.mcp_client import MCPToolClient

from .models import ORToolsState


class ORToolsEnv(MCPToolClient):
    """Client for the OR-Tools Environment; `MCPToolClient` provides `reset`, `step`, `call_tool` and `state`."""

    def _parse_state(self, payload: Dict[str, Any]) -> ORToolsState:
        return ORToolsState(**payload)
