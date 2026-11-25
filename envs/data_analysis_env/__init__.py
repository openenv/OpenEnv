# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data-analysis Environment: answer questions about real data files from a live Python session.

Each episode is one [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) task:
a question about one or more CSV/TSV/JSON files. Tools: `run_python(code)` runs code in a
persistent, sandboxed CPython session with pandas and numpy, and `submit_answer(answer)` ends the
episode and grades the answer with the dataset's deterministic grader (reward 1.0 or 0.0).

Examples:

```python
from data_analysis_env import DataAnalysisEnv

with DataAnalysisEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(split="eval", seed=0)
    print(result.observation.metadata["prompt"])
```
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .answers import extract_answer
from .client import DataAnalysisEnv
from .models import AVAILABLE_TOOLS, DataAnalysisState

__all__ = [
    "AVAILABLE_TOOLS",
    "CallToolAction",
    "DataAnalysisEnv",
    "DataAnalysisState",
    "ListToolsAction",
    "extract_answer",
]
