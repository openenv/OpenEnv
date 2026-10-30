# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OR-Tools Environment - design solutions and algorithms for hard operations research problems.

Tools: `run_python(code)`, `check_solution(solution)` and `submit_solution(solution)` (level 1), `test_algorithm(code)`
and `submit_algorithm(code)` (level 2).

Examples:

```python
from ortools_env import ORToolsEnv

with ORToolsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(problem="rcpsp", size="j120", level=1, seed=0)
    print(result.observation.metadata["prompt"])
```
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import ORToolsEnv
from .models import ORToolsState

__all__ = ["ORToolsEnv", "ORToolsState", "CallToolAction", "ListToolsAction"]
