# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Play one level-1 `ortools_env` episode through its MCP tools, with an earliest-due-date rule as the agent.

Examples:

```bash
# Terminal 1: the server
PYTHONPATH=src:envs uvicorn ortools_env.server.app:app --port 8000

# Terminal 2: the episode
PYTHONPATH=src:envs python examples/ortools_env_episode.py
```
"""

import json

from openenv.core.env_server.mcp_types import CallToolAction
from ortools_env import ORToolsEnv

SCRIPT = """
import json
jobs = range(len(instance["due_dates"]))
print(json.dumps({"sequence": sorted(jobs, key=lambda j: instance["due_dates"][j])}))
"""


def main():
    with ORToolsEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(problem="tardiness", size="n40", level=1, seed=0)
        print(result.observation.metadata["prompt"])

        output = env.call_tool("run_python", code=SCRIPT)
        solution = json.loads(output.splitlines()[0])
        print(env.call_tool("check_solution", solution=solution))

        # `call_tool` returns only the tool text; use `step` to read the reward of the submission.
        result = env.step(
            CallToolAction(
                tool_name="submit_solution", arguments={"solution": solution}
            )
        )
        print(f"reward={result.reward} done={result.done}")


if __name__ == "__main__":
    main()
