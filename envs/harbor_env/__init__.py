"""harbor_env: run Harbor tasks with token-level capture.

The implementation lives in `openenv.harbor` and `openenv.core.harness.capture`; this package is
deployment packaging only (manifest, Dockerfile, ASGI entry point). The client and result models are
re-exported here so `from harbor_env import HarborEnv` works, matching every other environment.

Examples:

```python
from harbor_env import HarborEnv

with HarborEnv(base_url="http://localhost:8000") as env:
    split = env.splits()[0]["name"]
    result = env.run_rollout(split=split, task_index=0, harness="opencode", sandbox="e2b")
    print(result.reward, result.n_turns)
```
"""

from openenv.harbor.client import HarborEnv
from openenv.harbor.models import (
    HarborConversation,
    HarborRolloutResult,
    HarborTaskRef,
    HarborTurn,
)

__all__ = [
    "HarborEnv",
    "HarborConversation",
    "HarborRolloutResult",
    "HarborTaskRef",
    "HarborTurn",
]
