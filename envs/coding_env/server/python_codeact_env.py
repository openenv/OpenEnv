# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python Code Action Environment.

This module provides a server-side environment implementation for executing
Python code actions using PyExecutor.
"""

import uuid

from openenv.core.env_server.interfaces import (
    Action,
    Environment,
    Observation,
    Transform,
)

from ..models import CodeAction, CodeObservation, CodeState
from .python_executor import DEFAULT_SAFE_IMPORTS, PyExecutor
from .transforms import create_safe_coding_transform


def _authorized_imports(additional_imports: list[str] | None) -> list[str] | None:
    """PyExecutor takes a complete allowlist, so keep the defaults in it."""
    if additional_imports is None:
        return None

    merged = list(DEFAULT_SAFE_IMPORTS)
    merged.extend(name for name in additional_imports if name not in merged)
    return merged


class PythonCodeActEnv(Environment):
    """
    Python Code Action Environment for executing code and tracking state.

    This environment executes Python code submitted as CodeAction during step,
    maintains the last exit code in its state, and returns results wrapped
    in CodeObservation.

    Args:
        transform: Optional transform to apply to observations
        additional_imports: List of additional module imports to authorize
                          on top of DEFAULT_SAFE_IMPORTS
                          (e.g., ["numpy", "pandas", "matplotlib"])

    Example:
        >>> env = PythonCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(code="print('Hello, World!')")
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, World!\n"
        >>> print(obs.exit_code)  # 0
        >>> print(env.state.last_exit_code)  # 0
    """

    def __init__(
        self,
        transform: Transform | None = None,
        additional_imports: list[str] | None = None,
    ):
        self._transform = transform
        self._additional_imports = _authorized_imports(additional_imports)
        self.transform = transform or create_safe_coding_transform()
        self._executor = PyExecutor(additional_imports=self._additional_imports)
        self._state = CodeState()

    def reset(self) -> Observation:
        """
        Reset environment and start fresh execution session.

        Returns:
            Initial observation with empty stdout/stderr and exit_code=0
        """
        # Initialize fresh state
        self._state = CodeState(episode_id=str(uuid.uuid4()), step_count=0)
        # Add last_exit_code to state
        self._state.last_exit_code = 0

        # Reset executor to clear any previously defined variables/functions
        self._executor = PyExecutor(additional_imports=self._additional_imports)

        # Reset transform to clear any accumulated state
        self.transform = self._transform or create_safe_coding_transform()

        # Return initial observation
        observation = CodeObservation(
            stdout="",
            stderr="",
            exit_code=0,
        )

        return self._apply_transform(observation)

    def step(self, action: Action) -> Observation:
        """
        Execute code action and return observation.

        Args:
            action: CodeAction containing the code to execute

        Returns:
            CodeObservation with execution results (stdout, stderr, exit_code)

        Raises:
            ValueError: If action is not a CodeAction instance
        """
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Execute the code using PyExecutor
        result = self._executor.run(action.code)

        # Update state
        self._state.step_count += 1
        self._state.last_exit_code = result.exit_code

        # Create observation from execution result
        # Include code in metadata for transform reward calculation
        observation = CodeObservation(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            metadata={"last_code": action.code},
        )

        return self._apply_transform(observation)

    @property
    def state(self) -> CodeState:
        """Get current environment state including last exit code."""
        return self._state
