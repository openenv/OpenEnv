# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Calendar Environment package exports."""

from typing import Any

from .models import (
    CalendarAction,
    CalendarObservation,
    ListToolsAction,
    MCPAction,
    MCPObservation,
    ToolCallAction,
)

__all__ = [
    "CalendarAction",
    "CalendarObservation",
    "CalendarEnv",
    "MCPAction",
    "MCPObservation",
    "ListToolsAction",
    "ToolCallAction",
]


def __getattr__(name: str) -> Any:
    if name == "CalendarEnv":
        from .client import CalendarEnv as _CalendarEnv

        return _CalendarEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
