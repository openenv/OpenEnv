# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Connect4 Environment for OpenEnv.

This module provides OpenEnv integration for the classic Connect4 board game.

Example:
    >>> from envs.Connect4_env import Connect4Env, Connect4Action
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = Connect4Env.from_docker_image("Connect4-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> result = env.step(Connect4Action(column=2))
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import Connect4Env
from .models import Connect4Action, Connect4Observation, Connect4State

__all__ = ["Connect4Env", "Connect4Action", "Connect4Observation", "Connect4State"]
