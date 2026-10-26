# SPDX-License-Identifier: BSD-3-Clause

"""
AutoAction - Automatic Action Class Selection
==============================================

AutoAction provides a HuggingFace-style API for automatically retrieving the
correct Action class from installed packages or HuggingFace Hub.

This module simplifies working with environment actions by automatically
detecting and returning the appropriate Action class without requiring
manual imports.

Example:
    >>> from openenv import AutoEnv, AutoAction
    >>>
    >>> # Get Action class from environment name
    >>> CodeAction = AutoAction.from_env("coding")
    >>> action = CodeAction(code="print('Hello!')")
    >>>
    >>> # From HuggingFace Hub
    >>> CodeAction = AutoAction.from_env("openenv/coding_env")
    >>>
    >>> # Use with AutoEnv
    >>> env = AutoEnv.from_env("coding-env")
    >>> result = env.step(action)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

from ._discovery import _is_hub_url, get_discovery
from .auto_env import AutoEnv

logger = logging.getLogger(__name__)


class AutoAction:
    """
    AutoAction automatically retrieves the correct Action class based on
    environment names or HuggingFace Hub repositories.

    This class follows the HuggingFace AutoModel pattern, making it easy to
    get the right Action class without needing to know which module to import.

    The class provides factory methods that look up the Action class and
    return the class (not an instance) for you to instantiate.

    Example:
        >>> # From installed package
        >>> CodeAction = AutoAction.from_env("coding")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # From HuggingFace Hub
        >>> CodeAction = AutoAction.from_env("openenv/coding_env")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # Use with AutoEnv for a complete workflow
        >>> env = AutoEnv.from_env("coding-env")
        >>> ActionClass = AutoAction.from_env("coding-env")
        >>> action = ActionClass(code="print('Hello, AutoAction!')")
        >>> result = env.step(action)

    Note:
        AutoAction is not meant to be instantiated directly. Use the class
        method from_env() instead.
    """

    def __init__(self):
        """AutoAction should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoAction is a factory class and should not be instantiated directly. "
            "Use AutoAction.from_hub() or AutoAction.from_env() instead."
        )

    @classmethod
    def from_env(cls, name: str, skip_install: bool = False) -> Type:
        """
        Get the Action class from environment name or HuggingFace Hub repository.

        This method automatically:
        1. Checks if the name is a HuggingFace Hub URL/repo ID
        2. If Hub: downloads and installs the environment package
        3. If local: looks up the installed openenv-* package
        4. Imports and returns the Action class

        Args:
            name: Environment name or HuggingFace Hub repo ID
                  Examples:
                  - "coding" / "coding-env" / "coding_env"
                  - "openenv/coding_env" (Hub repo ID)
                  - "https://huggingface.co/openenv/coding_env" (Hub URL)
            skip_install: If True, skip package installation and return
                GenericAction class instead. Use this when working with
                GenericEnvClient to avoid installing remote packages.

        Returns:
            Action class (not an instance!). Returns GenericAction when
            skip_install=True.

        Raises:
            ValueError: If environment not found (only when skip_install=False)
            ImportError: If environment package is not installed (only when skip_install=False)

        Examples:
            >>> # From installed package
            >>> CodeAction = AutoAction.from_env("coding-env")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # From HuggingFace Hub
            >>> CodeAction = AutoAction.from_env("openenv/coding_env")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # Skip installation, use GenericAction (for GenericEnvClient)
            >>> ActionClass = AutoAction.from_env("user/repo", skip_install=True)
            >>> action = ActionClass(code="print('Hello!')")  # Returns GenericAction
            >>>
            >>> # Different name formats
            >>> EchoAction = AutoAction.from_env("echo")
            >>> EchoAction = AutoAction.from_env("echo-env")
            >>> EchoAction = AutoAction.from_env("echo_env")
        """
        # If skip_install is True, return GenericAction without any package lookup
        if skip_install:
            from openenv.core.generic_client import GenericAction

            logger.info(
                f"Returning GenericAction for '{name}' (skip_install=True). "
                f"Use keyword arguments to create actions: GenericAction(code='...')"
            )
            return GenericAction

        # Check if it's a HuggingFace Hub URL or repo ID
        if _is_hub_url(name):
            # Ensure package is installed (reuse AutoEnv logic, downloads only if needed)
            env_name = AutoEnv._ensure_package_from_hub(name)
        else:
            env_name = name

        # Get environment info from discovery
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(env_name)

        if not env_info:
            # Environment not found - provide helpful error message
            available_envs = discovery.discover()

            if not available_envs:
                raise ValueError(
                    "No OpenEnv environments found.\n"
                    "Install an environment with: pip install openenv-<env-name>\n"
                    "Or specify a HuggingFace Hub repository: AutoAction.from_env('openenv/echo_env')"
                )

            # Try to suggest similar environment names
            from difflib import get_close_matches

            env_keys = list(available_envs.keys())
            suggestions = get_close_matches(env_name, env_keys, n=3, cutoff=0.6)

            error_msg = f"Unknown environment '{env_name}'.\n"
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?\n"
            error_msg += f"Available environments: {', '.join(sorted(env_keys))}"

            raise ValueError(error_msg)

        # Get the action class
        try:
            action_class = env_info.get_action_class()
            return action_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import action class for '{env_name}'.\n"
                f"Package '{env_info.package_name}' appears to be installed but the module cannot be imported.\n"
                f"Try reinstalling: pip install --force-reinstall {env_info.package_name}\n"
                f"Original error: {e}"
            ) from e

    @classmethod
    def from_hub(cls, env_name: str, skip_install: bool = False) -> Type:
        """
        Get the Action class from environment name.

        This is an alias for from_env() for backward compatibility and clarity.

        Args:
            env_name: Environment name (e.g., "coding", "echo")
            skip_install: If True, skip package installation and return
                GenericAction class instead.

        Returns:
            Action class (not an instance!)

        Examples:
            >>> CodeAction = AutoAction.from_hub("coding")
            >>> action = CodeAction(code="print('Hello!')")
        """
        return cls.from_env(env_name, skip_install=skip_install)

    @classmethod
    def get_action_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an action class.

        Args:
            name: Environment name

        Returns:
            Dictionary with action class metadata

        Raises:
            ValueError: If environment not found

        Examples:
            >>> info = AutoAction.get_action_info("coding")
            >>> print(info['action_class'])
            'CodingAction'
            >>> print(info['module'])
            'coding_env.client'
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return {
            "env_key": env_info.env_key,
            "env_name": env_info.name,
            "package": env_info.package_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
        }

    @classmethod
    def list_actions(cls) -> None:
        """
        Print a formatted list of all available action classes.

        This discovers all installed openenv-* packages and displays
        their action class information in a user-friendly format.

        Examples:
            >>> AutoAction.list_actions()
            Available Action Classes:
            ----------------------------------------------------------------------
              echo           : EchoAction (from openenv-echo-env)
              coding         : CodingAction (from openenv-coding_env)
            ----------------------------------------------------------------------
            Total: 2 action classes
        """
        discovery = get_discovery()
        environments = discovery.discover()

        print("Available Action Classes:")
        print("-" * 70)

        if not environments:
            print("  No OpenEnv environments found.")
            print("  Install environments with: pip install openenv-<env-name>")
        else:
            for env_key in sorted(environments.keys()):
                env = environments[env_key]
                print(f"  {env_key:<15}: {env.action_class_name}")
                print(f"                   Package: {env.package_name}")

        print("-" * 70)
        print(f"Total: {len(environments)} action classes")
