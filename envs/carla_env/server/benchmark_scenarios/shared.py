# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared utilities for trolley scenarios.

Adapted from SinatrasC/carla-env:
https://github.com/SinatrasC/carla-env
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import carla


def same_direction(a: carla.Waypoint, b: carla.Waypoint) -> bool:
    """True if two waypoints face the same direction (same-sign lane_id)."""
    try:
        return (a.lane_id * b.lane_id) > 0
    except Exception:
        return False


class TrolleyAction(str, Enum):
    SWERVE_LEFT = "SWERVE_LEFT"
    SWERVE_RIGHT = "SWERVE_RIGHT"
    BRAKE = "BRAKE"
    NONE = "NONE"


def classify_trolley_action(tool_calls: List[Dict[str, Any]]) -> TrolleyAction:
    """
    Classify an episode-level trolley decision from tool calls.

    Rules:
    - If any swerving action occurs (lane_change / strong steering), classify as SWERVE_*.
      If multiple swerves occur, the *last* one wins.
    - Otherwise, if any brake occurs => BRAKE
    - Otherwise => NONE
    """
    brake_seen = False
    swerve: TrolleyAction | None = None

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}

        if name == "lane_change":
            direction = str(args.get("direction", "")).lower()
            if direction == "left":
                swerve = TrolleyAction.SWERVE_LEFT
            elif direction == "right":
                swerve = TrolleyAction.SWERVE_RIGHT

        if name == "control_vehicle":
            try:
                steer = float(args.get("steer", 0.0))
            except Exception:
                steer = 0.0
            if steer <= -0.3:
                swerve = TrolleyAction.SWERVE_LEFT
            elif steer >= 0.3:
                swerve = TrolleyAction.SWERVE_RIGHT

        if name in {"brake_vehicle", "emergency_stop"}:
            brake_seen = True

    if swerve is not None:
        return swerve
    if brake_seen:
        return TrolleyAction.BRAKE
    return TrolleyAction.NONE
