# SPDX-License-Identifier: BSD-3-Clause

"""
OpenEnv Auto Module
===================

Provides HuggingFace-style auto-discovery API for OpenEnv environments.

This module enables automatic environment and action class loading without
manual imports:

    >>> from openenv import AutoEnv, AutoAction
    >>>
    >>> # Load environment from installed package or HuggingFace Hub
    >>> env = AutoEnv.from_env("coding-env")
    >>>
    >>> # Get action class
    >>> CodeAction = AutoAction.from_env("coding")
    >>> action = CodeAction(code="print('Hello!')")

Classes:
    AutoEnv: Automatic environment client selection and instantiation
    AutoAction: Automatic action class selection

The auto-discovery system works by:
1. Discovering installed openenv-* packages via importlib.metadata
2. Loading environment manifests (openenv.yaml) from package resources
3. Supporting HuggingFace Hub repositories for remote environments
4. Caching discovery results for performance
"""

from .auto_action import AutoAction
from .auto_env import AutoEnv

__all__ = ["AutoEnv", "AutoAction"]
