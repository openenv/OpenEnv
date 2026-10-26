# SPDX-License-Identifier: BSD-3-Clause

"""Pelican SVG environment for OpenEnv.

Turns Simon Willison's "generate an SVG of a pelican riding a bicycle" check
into an executable environment: the subject and vehicle are sampled from a
grid so the canonical prompt is not the only thing being measured, and scoring
runs in three layers of increasing cost.

Examples:

```python
from envs.pelican_svg_env import PelicanSvgAction, PelicanSvgEnv

with PelicanSvgEnv(base_url="http://localhost:8000") as env:
    observation = env.reset().observation
    result = env.step(PelicanSvgAction(response=my_model(observation.prompt)))
    print(result.reward, result.observation.feedback)
```
"""

from .client import PelicanSvgEnv
from .models import PelicanSvgAction, PelicanSvgObservation, PelicanSvgState

__all__ = [
    "PelicanSvgEnv",
    "PelicanSvgAction",
    "PelicanSvgObservation",
    "PelicanSvgState",
]
