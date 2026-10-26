# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-RL (``TitanRL``) <-> OpenEnv integration example.

The OpenEnv-only bridge (:class:`OpenEnvBridge`, :class:`BridgeTurn`,
:func:`render_observation`) is importable directly and needs only ``openenv``.
The TorchTitan-RL components (``OpenEnvMessageEnv``, ``OpenEnvDataset``,
``OpenEnvReward``) require ``torchtitan`` and are exposed lazily so that
``import examples.titanrl_openenv`` (or ``titanrl_openenv``) does not pull in
``torch`` unless those names are actually used. The rollouter that wires them
together is a plain ``Rollouter.Config`` built in ``config_registry``.
"""

from __future__ import annotations

from typing import Any

from .openenv_bridge import (
    BridgeTurn,
    DEFAULT_ACT_TOOL,
    OpenEnvBridge,
    render_observation,
)
from .tasks import (
    CHESS_MOVE_TOOL,
    CHESS_TASK,
    get_task_profile,
    render_chess_observation,
    TASK_PROFILES,
    TaskProfile,
)

__all__ = [
    # OpenEnv-only bridge (no torchtitan required)
    "OpenEnvBridge",
    "BridgeTurn",
    "DEFAULT_ACT_TOOL",
    "render_observation",
    # Task profiles (no torchtitan required)
    "TaskProfile",
    "TASK_PROFILES",
    "get_task_profile",
    "CHESS_TASK",
    "CHESS_MOVE_TOOL",
    "render_chess_observation",
    # TorchTitan-RL components (lazy; require torchtitan)
    "OpenEnvMessageEnv",
    "OpenEnvSample",
    "OpenEnvDataset",
    "OpenEnvReward",
]

# Lazily import the torchtitan-dependent symbols on first access so bridge-only
# usage never imports torch.
_LAZY = {
    "OpenEnvMessageEnv": ("titanrl_env", "OpenEnvMessageEnv"),
    "OpenEnvSample": ("data", "OpenEnvSample"),
    "OpenEnvDataset": ("data", "OpenEnvDataset"),
    "OpenEnvReward": ("rubric", "OpenEnvReward"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule, attr = target
    module = importlib.import_module(f"{__name__}.{submodule}")
    return getattr(module, attr)
