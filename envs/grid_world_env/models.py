# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    from openenv.core.env_server.types import Action, Observation
    from pydantic import Field
except ImportError:
    from core.env_server.types import Action, Observation
    from pydantic import Field


# --- Action Models ---
class MoveAction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class GridWorldAction(Action):
    action: MoveAction = Field(..., description="The direction to move the agent.")


# --- Observation Model ---
class GridWorldObservation(Observation):
    x: int = Field(..., description="Current X position")
    y: int = Field(..., description="Current Y position")
    message: str = Field("", description="Status message")
    # Reward must be a float, default to 0.0 (not None)

    reward: float = Field(0.0, description="Reward received from the last action")
    done: bool = Field(False, description="Whether the episode has ended")
