# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sandboxed Python code executor for the REPL environment.

Uses smolagents.LocalPythonExecutor as the backend for battle-tested sandboxed
execution, with RLM-specific features on top:
- Context loading (set_context)
- Variable access (get_variable, list_variables)
- Function injection (inject_function for llm_query, llm_query_batched)
- Output capped at 8,192 characters per turn (configurable)
- Persistent namespace across code blocks
"""

import json
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any, Dict, List, Optional

from smolagents import LocalPythonExecutor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PythonExecutor:
    """Sandboxed Python code executor with persistent namespace.

    Wraps smolagents.LocalPythonExecutor with RLM-specific features:
    - Context loading for RLM tasks
    - Variable tracking for observation
    - Function injection for llm_query, llm_query_batched
    - Configurable output length limit (default 8192 chars per Prime Intellect)
    """

    def __init__(
        self,
        max_output_length: int = 8192,
        allowed_imports: Optional[List[str]] = None,
    ):
        """Initialize the executor.

        Args:
            max_output_length: Maximum characters for stdout/stderr (default 8192)
            allowed_imports: List of allowed module names for import

        Note:
            smolagents.LocalPythonExecutor does NOT support wall-clock timeouts.
            Instead, it limits operations (10M ops) and while iterations (1M).
        """
        self.max_output_length = max_output_length

        # Default allowed imports for RLM tasks
        default_imports = [
            "re",
            "json",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "operator",
            "string",
            "textwrap",
            "difflib",
            "statistics",
            "decimal",
            "fractions",
            "datetime",
            "copy",
            "pprint",
            "typing",
            "dataclasses",
            "enum",
            "bisect",
            "heapq",
            "array",
            "struct",
            "base64",
            "hashlib",
            "hmac",
            "uuid",
        ]

        self.allowed_imports = allowed_imports or default_imports

        # Initialize the smolagents executor
        self._executor = LocalPythonExecutor(
            additional_authorized_imports=self.allowed_imports
        )

        # Track variables we've set (for list_variables)
        self._user_variables: set[str] = set()

        # Track callable functions to register with send_tools
        self._callable_tools: Dict[str, Callable[..., Any]] = {}

        # Register helper utilities
        self._register_helpers()

    def _register_helpers(self) -> None:
        """Register helper functions with the executor."""
        helpers = {
            "format_exc": traceback.format_exc,
            "safe_json_dumps": lambda obj: json.dumps(obj, default=lambda o: repr(o)),
        }
        # Register helpers as callable tools
        for name, func in helpers.items():
            self.inject_function(name, func)

    def _sync_callable_tools(self) -> None:
        """Sync callable functions with the executor via send_tools."""
        if self._callable_tools:
            try:
                # Type ignore: smolagents accepts callables despite Tool type hint
                self._executor.send_tools(self._callable_tools)  # type: ignore[arg-type]
            except Exception:
                logger.debug(
                    "send_tools failed; continuing without extra tools",
                    exc_info=True,
                )

    def set_context(self, context: str, variable_name: str = "context") -> None:
        """Load context into namespace as a variable.

        Args:
            context: The context string to load
            variable_name: Name of the variable (default "context")
        """
        self.set_variable(variable_name, context)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace.

        Args:
            name: Variable name
            value: Variable value
        """
        self._executor.send_variables({name: value})
        self._user_variables.add(name)

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from namespace.

        Args:
            name: Variable name

        Returns:
            The variable value or None if not found
        """
        return self._executor.state.get(name)

    def list_variables(self) -> List[str]:
        """List non-private variables in namespace.

        Returns:
            List of variable names (excluding private and builtins)
        """
        variables = {key for key in self._executor.state if not key.startswith("_")}
        variables.update(self._user_variables)
        return list(variables)

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return results.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with stdout, stderr, locals_snapshot, execution_time,
            success, and exception fields
        """
        start_time = time.time()
        success = True
        exception_msg = None
        new_locals: Dict[str, str] = {}

        # Track state before execution
        pre_state_keys = set()
        if hasattr(self._executor, "state"):
            pre_state_keys = set(self._executor.state.keys())

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        try:
            exec_result = self._executor(code)

            # CodeOutput has: logs (str), output (Any), is_final_answer (bool)
            if exec_result.logs:
                stdout_parts.append(str(exec_result.logs))

            if exec_result.output is not None:
                try:
                    stdout_parts.append(json.dumps(exec_result.output))
                except Exception:
                    stdout_parts.append(repr(exec_result.output))

        except Exception as e:
            success = False
            exception_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            stderr_parts.append(exception_msg)

        execution_time = time.time() - start_time

        # Capture new/modified variables
        for key in self._executor.state:
            if key not in pre_state_keys and not key.startswith("_"):
                try:
                    val = self._executor.state[key]
                    val_repr = repr(val)
                    if len(val_repr) > 500:
                        val_repr = val_repr[:500] + "..."
                    new_locals[key] = val_repr
                    self._user_variables.add(key)
                except Exception:
                    new_locals[key] = "<unrepresentable>"

        # Compose stdout/stderr
        stdout = "\n".join(part for part in stdout_parts if part)
        stderr = "\n".join(part for part in stderr_parts if part)

        # Truncate output to max_output_length
        if len(stdout) > self.max_output_length:
            stdout = (
                stdout[: self.max_output_length]
                + f"\n... (truncated, total {len(stdout)} chars)"
            )

        if len(stderr) > self.max_output_length:
            stderr = (
                stderr[: self.max_output_length]
                + f"\n... (truncated, total {len(stderr)} chars)"
            )

        return {
            "stdout": stdout,
            "stderr": stderr,
            "locals_snapshot": new_locals,
            "execution_time": execution_time,
            "success": success,
            "exception": exception_msg,
        }

    def reset(self) -> None:
        """Reset namespace to initial state."""
        # Create a new executor instance
        self._executor = LocalPythonExecutor(
            additional_authorized_imports=self.allowed_imports
        )
        self._user_variables.clear()
        self._callable_tools.clear()
        self._register_helpers()

    def inject_function(self, name: str, func: Callable[..., Any]) -> None:
        """Inject a callable function into the namespace.

        Args:
            name: Function name in namespace
            func: The callable to inject
        """
        self._callable_tools[name] = func
        self._user_variables.add(name)
        self._sync_callable_tools()
