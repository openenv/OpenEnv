# SPDX-License-Identifier: BSD-3-Clause

"""Core tools for code execution and other utilities."""

from .git_server_client import GitServerClient, RepoInfo

try:
    from .local_python_executor import PyExecutor
except ModuleNotFoundError:
    # smolagents is optional for environments that only need Git tooling.
    PyExecutor = None  # type: ignore[assignment]

__all__ = [
    "PyExecutor",
    "GitServerClient",
    "RepoInfo",
]
