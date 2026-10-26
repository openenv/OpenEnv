# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CARLA environment for OpenEnv.

Embodied evaluation environment for testing LLM decision-making
in simulated scenarios with temporal flow and irreversible consequences.

Example usage:
    >>> from carla_env import CarlaEnv, CarlaAction
    >>> env = CarlaEnv(base_url="http://localhost:8000")
    >>> result = env.reset()
    >>> result = env.step(CarlaAction(action_type="emergency_stop"))
    >>> env.close()
"""

from .client import CarlaEnv
from .models import CarlaAction, CarlaObservation, CarlaState

__all__ = [
    "CarlaEnv",
    "CarlaAction",
    "CarlaObservation",
    "CarlaState",
]

__version__ = "0.1.0"
