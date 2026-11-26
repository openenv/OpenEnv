# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Play one `data_analysis_env` episode through its MCP tools, with a scripted agent that loads the
task's first file and submits its row count.

Examples:

```bash
# Terminal 1: the server
docker build -t data-analysis-env:latest -f envs/data_analysis_env/server/Dockerfile envs/data_analysis_env
docker run --rm -p 8000:8000 data-analysis-env:latest

# Terminal 2: the episode
PYTHONPATH=src:envs python examples/data_analysis_env_episode.py
```
"""

from data_analysis_env import DataAnalysisEnv
from openenv.core.env_server.mcp_types import CallToolAction

LOAD = """
import pandas as pd
path = {path!r}
df = pd.read_json(path) if path.endswith(".json") else pd.read_csv(path, sep=None, engine="python")
len(df)
"""


def main():
    with DataAnalysisEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(split="eval", seed=0)
        meta = result.observation.metadata
        print(meta["prompt"])

        output = env.call_tool("run_python", code=LOAD.format(path=meta["files"][0]))
        print(output)

        # `call_tool` returns only the tool text, so submit through `step` to read the reward.
        result = env.step(
            CallToolAction(
                tool_name="submit_answer", arguments={"answer": output.strip()}
            )
        )
        print(f"reward={result.reward} done={result.done}")


if __name__ == "__main__":
    main()
