# SPDX-License-Identifier: BSD-3-Clause

"""Local Python Executor (enhanced).

This module provides a safer wrapper around smolagents.LocalPythonExecutor
with improved exception handling and a few helpful tools registered with
the executor to make debugging executed code easier.

Key improvements:
- Register a few helper utilities via send_tools so user code can use
  them for reporting (e.g. `format_exc`).
- More robust extraction of stdout/stderr/exit codes from the executor
  result object, tolerant to different versions of smolagents.
- Detailed stderr on unexpected exceptions including full traceback.
- Structured logging for operational visibility.
"""

from __future__ import annotations

import json
import logging
import traceback

from openenv.core.env_server.types import CodeExecResult
from smolagents import LocalPythonExecutor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PyExecutor:
    """Wrapper around smolagents LocalPythonExecutor.

    The wrapper registers a few non-privileged helper tools to the
    LocalPythonExecutor that can be used by the executed code to
    format exceptions and to safely stringify results for improved
    error reporting.
    """

    def __init__(self, additional_imports: list[str] | None = None):
        if additional_imports is None:
            additional_imports = []

        self._executor = LocalPythonExecutor(
            additional_authorized_imports=additional_imports
        )

        # Register helpful utilities exposed to the execution environment.
        # These are intentionally small, read-only helpers.
        tools = {
            # Provide a small helper to format the current exception in the
            # executed context. This is a *string formatting* helper only.
            "format_exc": traceback.format_exc,
            # Safe JSON dumps with a fallback for non-serializable objects.
            "safe_json_dumps": lambda obj: json.dumps(obj, default=lambda o: repr(o)),
        }

        # `send_tools` is the public API on LocalPythonExecutor to make
        # helper callables available to the sandboxed runtime. We don't
        # provide any builtins that could change the environment.
        try:
            self._executor.send_tools(tools)
        except Exception:
            # If the LocalPythonExecutor implementation doesn't support
            # send_tools or fails, log and continue — the executor is still usable.
            logger.debug(
                "LocalPythonExecutor.send_tools failed; continuing without extra tools",
                exc_info=True,
            )

    def run(self, code: str) -> CodeExecResult:
        """Execute Python code and return a CodeExecResult.

        This method is intentionally defensive: it attempts to extract
        meaningful stdout/stderr/exit_code information from a variety of
        possible return shapes that different versions of smolagents
        may provide.
        """
        try:
            exec_result = self._executor(code)

            # Default values
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []
            exit_code = 0

            # Extract logs/prints
            try:
                logs = getattr(exec_result, "logs", None)
                if logs:
                    stdout_parts.append(str(logs))
            except Exception:
                logger.debug("Failed to read exec_result.logs", exc_info=True)

            # Extract the result / output value
            try:
                if hasattr(exec_result, "output"):
                    out_val = exec_result.output
                    # If the output is not None, stringify it in a safe way
                    if out_val is not None:
                        # Prefer JSON if possible, otherwise repr
                        try:
                            stdout_parts.append(json.dumps(out_val))
                        except Exception:
                            stdout_parts.append(repr(out_val))
            except Exception:
                logger.debug("Failed to read exec_result.output", exc_info=True)

            # Some runtime implementations may put errors on `error` or `exception`
            try:
                err = getattr(exec_result, "error", None)
                if err:
                    stderr_parts.append(str(err))
            except Exception:
                logger.debug("Failed to read exec_result.error", exc_info=True)

            try:
                ex = getattr(exec_result, "exception", None)
                if ex:
                    stderr_parts.append(str(ex))
            except Exception:
                logger.debug("Failed to read exec_result.exception", exc_info=True)

            # Determine exit code if provided
            try:
                if hasattr(exec_result, "exit_code"):
                    exit_code = (
                        int(exec_result.exit_code)
                        if exec_result.exit_code is not None
                        else 0
                    )
                elif hasattr(exec_result, "success"):
                    # Some versions use `success` boolean
                    exit_code = 0 if exec_result.success else 1
                else:
                    # Fallback: if there were any stderr parts, treat as non-zero
                    exit_code = 1 if stderr_parts else 0
            except Exception:
                logger.debug("Failed to determine exec_result exit code", exc_info=True)
                exit_code = 1 if stderr_parts else 0

            # Compose the final stdout/stderr strings
            stdout = "\n".join(part for part in stdout_parts if part is not None)
            stderr = "\n".join(part for part in stderr_parts if part is not None)

            return CodeExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

        except Exception:
            # Any unexpected exception from the LocalPythonExecutor is
            # returned with a full traceback to make debugging easier.
            tb = traceback.format_exc()
            logger.exception("LocalPythonExecutor raised an exception during run")
            return CodeExecResult(stdout="", stderr=tb, exit_code=1)
