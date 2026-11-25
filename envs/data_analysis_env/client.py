# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data-analysis Environment Client.

`call_tool` returns only the tool's text. The reward arrives on the observation of the step that
calls `submit_answer`, so use `step(CallToolAction(...))` for it and read `result.reward`.

Examples:

```python
from openenv.core.env_server.mcp_types import CallToolAction

from data_analysis_env import DataAnalysisEnv

with DataAnalysisEnv(base_url="http://localhost:8000").sync() as env:
    prompt = env.reset(split="eval", seed=0).observation.metadata["prompt"]
    print(env.call_tool("run_python", code="import pandas as pd\\nprint(1 + 1)"))
    result = env.step(CallToolAction(tool_name="submit_answer", arguments={"answer": "42"}))
    print(result.reward, result.done)
```
"""

from typing import Any, Dict

from openenv.core.mcp_client import MCPToolClient

from .models import DataAnalysisState


class DataAnalysisEnv(MCPToolClient):
    """Client for the data-analysis environment. `MCPToolClient` provides `reset`, `step`, `call_tool` and `state`."""

    def _parse_state(self, payload: Dict[str, Any]) -> DataAnalysisState:
        return DataAnalysisState(**payload)
