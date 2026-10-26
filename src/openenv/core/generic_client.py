# SPDX-License-Identifier: BSD-3-Clause

"""
Generic environment client that works with raw dictionaries.

This module provides a GenericEnvClient that doesn't require installing
environment-specific packages. It's useful for connecting to remote servers
without running any untrusted code locally.
"""

from typing import Any, Dict

from .client_types import StepResult
from .env_client import EnvClient


class GenericEnvClient(EnvClient[Dict[str, Any], Dict[str, Any], Dict[str, Any]]):
    """
    Environment client that works with raw dictionaries instead of typed classes.

    This client doesn't require installing environment-specific packages, making it
    ideal for:
    - Connecting to remote servers without installing their packages
    - Quick prototyping and testing
    - Environments where type safety isn't needed
    - Security-conscious scenarios where you don't want to run remote code

    The trade-off is that you lose type safety and IDE autocomplete for actions
    and observations. Instead of typed objects, you work with plain dictionaries.

    Examples:

        ```python
        # Direct connection to a running server (no installation needed)
        with GenericEnvClient(base_url="http://localhost:8000") as env:
            result = env.reset()
            result = env.step({"code": "print('hello')"})
            print(result.observation)  # Dict[str, Any]
            print(result.observation.get("output"))

        # From local Docker image (chain .sync() for synchronous use)
        env = GenericEnvClient.from_docker_image("coding-env:latest").sync()
        result = env.reset()
        result = env.step({"code": "x = 1 + 2"})
        env.close()

        # From HuggingFace Hub (pulls Docker image, no pip install)
        env = GenericEnvClient.from_env("user/my-env", use_docker=True).sync()
        result = env.reset()
        env.close()
        ```

    `GenericEnvClient` inherits `from_docker_image()` and `from_env()` from
    `EnvClient`, so you can use it with Docker containers and HuggingFace
    Spaces without any package installation.
    """

    def _step_payload(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert action to payload for the server.

        For GenericEnvClient, this handles both raw dictionaries and
        typed Action objects (Pydantic models). If a Pydantic model is
        passed, it will be converted to a dictionary using model_dump().

        Args:
            action (`dict` or `BaseModel`):
                Action as a dictionary or Pydantic `BaseModel`.

        Returns:
            The action as a dictionary for the server
        """
        # If it's already a dict, return as-is
        if isinstance(action, dict):
            return action

        # If it's a Pydantic model (Action subclass), convert to dict
        if hasattr(action, "model_dump"):
            return action.model_dump()

        # Fallback for other objects with __dict__
        if hasattr(action, "__dict__"):
            return vars(action)

        # Last resort: try to convert to dict
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Dict[str, Any]]:
        """
        Parse server response into a StepResult.

        Extracts the observation, reward, and done fields from the
        server response.

        Args:
            payload (`dict`):
                Response payload from the server.

        Returns:
            StepResult with observation as a dictionary
        """
        return StepResult(
            observation=payload.get("observation", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse state response from the server.

        For GenericEnvClient, this returns the payload as-is since
        we're working with dictionaries.

        Args:
            payload (`dict`):
                State payload from the server.

        Returns:
            The state as a dictionary
        """
        return payload


class GenericAction(Dict[str, Any]):
    """
    A dictionary subclass for creating actions when using GenericEnvClient.

    This provides a semantic wrapper around dictionaries to make code more
    readable when working with GenericEnvClient. It behaves exactly like a
    dict but signals intent that this is an action for an environment.

    Examples:

        ```python
        # Without GenericAction (works fine)
        env.step({"code": "print('hello')"})

        # With GenericAction (more explicit)
        action = GenericAction(code="print('hello')")
        env.step(action)

        # With multiple fields
        action = GenericAction(code="x = 1", timeout=30, metadata={"tag": "test"})
        env.step(action)
        ```

    `GenericAction` is just a dict with a constructor that accepts keyword
    arguments. It's provided for symmetry with typed Action classes and
    to make code more readable.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Create a GenericAction from keyword arguments.

        Args:
            **kwargs:
                Action fields as keyword arguments.

        Examples:

            ```python
            action = GenericAction(code="print(1)", timeout=30)
            action["code"]  # 'print(1)'
            ```
        """
        super().__init__(kwargs)

    def __repr__(self) -> str:
        """Return a readable representation."""
        items = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"GenericAction({items})"
