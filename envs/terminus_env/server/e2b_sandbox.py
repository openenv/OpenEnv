# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small E2B Code Interpreter wrapper for terminal-style environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]


@dataclass
class ShellResult:
    """Normalized result from a command executed in E2B."""

    stdout: str
    stderr: str
    error: str | None
    success: bool


class E2BSandbox:
    """Manages one E2B sandbox for one OpenEnv episode."""

    def __init__(self, api_key: str):
        if Sandbox is None:
            raise ImportError(
                "e2b-code-interpreter is not installed. Install the "
                "terminus_env package dependencies to use E2BSandbox. "
                f"Original import error: {_E2B_IMPORT_ERROR}"
            )
        self._sbx = Sandbox.create(api_key=api_key)
        self.sandbox_id: str = self._sbx.sandbox_id

    def run_shell(self, command: str, timeout_s: int = 120) -> ShellResult:
        shell_code = (
            "import subprocess, sys\n"
            f"_result = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={timeout_s})\n"
            "print(_result.stdout, end='')\n"
            "if _result.stderr: print(_result.stderr, end='', file=sys.stderr)\n"
            "if _result.returncode != 0:\n"
            "    raise SystemExit(_result.returncode)\n"
        )
        execution = self._sbx.run_code(shell_code)
        return _normalize(execution)

    def kill(self) -> None:
        try:
            self._sbx.kill()
        except Exception:
            logger.error("Failed to delete E2B sandbox %s", self.sandbox_id)
            raise


def _normalize(execution: Any) -> ShellResult:
    stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
    stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
    error = None
    if execution.error:
        error = (
            f"{execution.error.name}: {execution.error.value}\n"
            f"{execution.error.traceback}"
        )
    return ShellResult(
        stdout=stdout,
        stderr=stderr,
        error=error,
        success=execution.error is None,
    )
