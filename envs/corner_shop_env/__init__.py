# Copyright 2026 Kayam
# SPDX-License-Identifier: Apache-2.0

"""Corner Shop: run a small shop for 13 weeks, scored by a going-concern reward.

Example:
    >>> from corner_shop_env import CornerShopAction, CornerShopEnv
    >>>
    >>> with CornerShopEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(firm=7)
    ...     while not result.done:
    ...         reply = my_model(result.observation.system_prompt, result.observation.prompt)
    ...         result = env.step(CornerShopAction(message=reply))
    ...     print(result.reward, result.observation.metrics["cash_usd"])
"""

from .client import CornerShopEnv
from .models import CornerShopAction, CornerShopObservation, CornerShopState

__all__ = [
    "CornerShopAction",
    "CornerShopObservation",
    "CornerShopState",
    "CornerShopEnv",
]
