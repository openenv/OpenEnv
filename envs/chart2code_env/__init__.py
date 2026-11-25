# SPDX-License-Identifier: BSD-3-Clause

"""Chart-to-code environment for OpenEnv.

The agent sees a chart and writes matplotlib code that draws it again. It can
render drafts to see what its code draws, then submits once. The reward
compares the submitted figure's artists with the target's: data, text, colors
and structure. No pixels are compared and no judge is called.

Examples:

```python
from envs.chart2code_env import Chart2CodeAction, Chart2CodeEnv

with Chart2CodeEnv(base_url="http://localhost:8000") as env:
    observation = env.reset(seed=3).observation
    code = my_model(observation.prompt, observation.target_png_base64)
    result = env.step(Chart2CodeAction(code=code, mode="submit"))
    print(result.reward, result.observation.scores)
```
"""

from .client import Chart2CodeEnv
from .models import Chart2CodeAction, Chart2CodeObservation, Chart2CodeState

__all__ = [
    "Chart2CodeEnv",
    "Chart2CodeAction",
    "Chart2CodeObservation",
    "Chart2CodeState",
]
