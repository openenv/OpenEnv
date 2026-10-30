# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQL Query Optimization Environment for OpenEnv.

An execution-grounded RL environment: the agent rewrites slow SQL and is
rewarded by real DuckDB execution speedup plus result-correctness across five
tasks that scale from basic anti-patterns to expert window-function audits.

Examples:

```python
from envs.sql_optim_env import SQLOptimEnv, SQLOptimAction

with SQLOptimEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="task_1_basic_antipatterns").observation
    result = env.step(SQLOptimAction(optimized_query="SELECT id FROM orders ..."))
    print(result.reward, result.done)
```
"""

from .client import SQLOptimEnv
from .models import SQLOptimAction, SQLOptimObservation, SQLOptimState

__all__ = [
    "SQLOptimEnv",
    "SQLOptimAction",
    "SQLOptimObservation",
    "SQLOptimState",
]
