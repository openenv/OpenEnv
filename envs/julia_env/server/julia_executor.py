# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Julia Executor with Process Pool Support.

This module provides a Julia code executor with:
- Proper process cleanup on timeout (no zombie processes)
- Robust error handling and logging
- Process group management for complete cleanup
- Automatic retry on transient failures
- Optional process pool for 50-100x speedup on repeated executions

Performance Modes:
- Standard mode: Spawn new process for each execution (default for single executions)
- Pool mode: Reuse persistent Julia processes (recommended for repeated executions)
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .julia_sandbox import (
    _socket_filter_unavailable,
    sandboxed,
    UNPRIVILEGED_UID,
    worker_env,
)

# Use julia_env hierarchy to inherit handlers from app.py's setup_logging()
logger = logging.getLogger("julia_env.executor")


@dataclass
class CodeExecResult:
    """Result of code execution."""

    stdout: str
    stderr: str
    exit_code: int


# Try to import process pool (optional dependency)
try:
    from .julia_process_pool import JuliaProcessPool

    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False
    JuliaProcessPool = None


class JuliaExecutor:
    """
    Executor for running Julia code with robust process management.

    This class provides a safe interface to execute Julia code in isolation
    and capture the results including stdout, stderr, and exit code.

    Features:
    - Proper timeout handling without zombie processes
    - Process group cleanup for nested processes
    - Automatic retry on transient failures
    - Comprehensive logging for debugging
    - Optional process pool for 50-100x speedup on repeated executions

    Example:
        >>> executor = JuliaExecutor()
        >>> result = executor.run('println("Hello, Julia!")')
        >>> print(result.stdout)  # "Hello, Julia!\\n"
        >>> print(result.exit_code)  # 0
        >>>
        >>> # With process pool (recommended for repeated executions)
        >>> JuliaExecutor.enable_process_pool(size=4)
        >>> executor = JuliaExecutor(use_process_pool=True)
        >>> for i in range(100):
        ...     result = executor.run(f'println({i})')  # 50-100x faster!
        >>> JuliaExecutor.shutdown_pool()  # Clean up when done
    """

    # Class-level process pool (shared across all instances if enabled)
    _shared_pool: Optional["JuliaProcessPool"] = None
    _pool_lock = threading.Lock()
    _pool_size: int = 0
    _pool_timeout: int = 120

    def __init__(
        self,
        timeout: Optional[int] = None,
        max_retries: int = 0,
        use_optimization_flags: bool = True,
        use_process_pool: bool = True,
    ):
        """
        Initialize the JuliaExecutor.

        Args:
            timeout: Maximum execution time in seconds. If None, reads from
                     JULIA_EXECUTION_TIMEOUT env var (default: 120 if not set)
            max_retries: Number of retry attempts on transient failures (default: 0)
            use_optimization_flags: Enable Julia performance flags (default: True)
            use_process_pool: Use process pool if available (default: True)

        Raises:
            RuntimeError: If Julia executable is not found in PATH
        """
        # Read timeout from env var if not explicitly provided
        if timeout is None:
            timeout = int(os.getenv("JULIA_EXECUTION_TIMEOUT", "120"))
            logger.debug(
                f"Executor timeout from JULIA_EXECUTION_TIMEOUT env var: {timeout}s"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_optimization_flags = use_optimization_flags
        self._use_process_pool = use_process_pool

        # Find Julia executable in PATH
        self.julia_path = shutil.which("julia")

        if not self.julia_path:
            # Try common installation paths
            common_paths = [
                os.path.expanduser("~/.juliaup/bin/julia"),
                os.path.expanduser("~/.julia/bin/julia"),
                "/usr/local/bin/julia",
                "/usr/bin/julia",
            ]

            for path in common_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    self.julia_path = path
                    break

        if not self.julia_path:
            logger.warning(
                "Julia executable not found in PATH or common locations. "
                "Please install Julia: https://julialang.org/downloads/"
            )

        # Build optimized Julia command with performance flags
        self.base_cmd = [self.julia_path] if self.julia_path else ["julia"]

        if self.use_optimization_flags:
            self.base_cmd.extend(
                [
                    "--compile=min",
                    "--optimize=2",
                    "--startup-file=no",
                    "--history-file=no",
                ]
            )

        logger.debug(f"JuliaExecutor initialized with Julia at: {self.julia_path}")
        if reason := _socket_filter_unavailable():
            logger.warning(f"Julia code has network access: {reason}")
        logger.debug(f"Timeout: {self.timeout}s, Max retries: {self.max_retries}")

    def _kill_process_tree(
        self, proc: subprocess.Popen, script_file: Optional[str] = None
    ) -> None:
        """
        Terminate a process and all its children.

        This prevents zombie processes by ensuring complete cleanup.

        Args:
            proc: The subprocess.Popen instance to terminate
            script_file: Optional script file path (for logging)
        """
        if proc.poll() is None:  # Process is still running
            try:
                # Try graceful termination first
                logger.warning(f"Terminating process {proc.pid} gracefully...")
                proc.terminate()

                # Wait up to 2 seconds for graceful termination
                try:
                    proc.wait(timeout=2.0)
                    logger.debug(f"Process {proc.pid} terminated gracefully")
                    return
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Process {proc.pid} did not terminate, forcing kill..."
                    )

                # Force kill if still running
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                    logger.debug(f"Process {proc.pid} killed forcefully")
                except subprocess.TimeoutExpired:
                    pass

            except Exception as e:
                logger.error(f"Error killing process {proc.pid}: {e}")

                # Last resort: try killing via process group
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        logger.debug(f"Killed process group for {proc.pid}")
                except Exception as pg_error:
                    logger.error(f"Failed to kill process group: {pg_error}")

    def run(self, code: str, timeout: Optional[int] = None) -> CodeExecResult:
        """
        Execute Julia code and return the result with robust error handling.

        This method provides:
        - Automatic retry on transient failures
        - Proper timeout handling without zombie processes
        - Process group cleanup for nested processes
        - Comprehensive error logging
        - Optional process pool for 50-100x speedup

        Args:
            code: Julia code string to execute
            timeout: Override default timeout (seconds). If None, uses pool's
                     configured timeout (when using pool) or instance timeout.

        Returns:
            CodeExecResult containing stdout, stderr, and exit_code
        """
        # Use process pool if enabled and available
        # Pass timeout as-is (None means use pool's configured default)
        if self._use_process_pool and JuliaExecutor._shared_pool is not None:
            try:
                return JuliaExecutor._shared_pool.execute(code, timeout=timeout)
            except Exception as e:
                logger.warning(
                    f"Process pool execution failed: {e}, falling back to subprocess"
                )
                # Fall through to standard execution

        # For subprocess fallback, apply instance default if timeout not specified
        if timeout is None:
            timeout = self.timeout

        # Check if Julia is available
        if not self.julia_path:
            return CodeExecResult(
                stdout="",
                stderr="Julia not found in PATH. Please install Julia.",
                exit_code=127,
            )

        code_file = None

        for attempt in range(self.max_retries + 1):
            proc = None

            try:
                # Create temporary file for Julia code
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jl", delete=False, encoding="utf-8"
                ) as f:
                    f.write(code)
                    code_file = f.name
                if os.geteuid() == 0:  # Julia runs as nobody (see julia_sandbox)
                    os.chown(code_file, UNPRIVILEGED_UID, UNPRIVILEGED_UID)

                script_name = Path(code_file).name
                logger.debug(
                    f"[Attempt {attempt + 1}/{self.max_retries + 1}] Executing: {script_name}"
                )

                # Start process with Popen for better control
                start_time = time.time()

                # On Unix systems, use process groups for better cleanup
                kwargs = {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }

                # Create new process group on Unix systems
                if hasattr(os, "setpgrp"):
                    kwargs["preexec_fn"] = os.setpgrp

                proc = subprocess.Popen(
                    sandboxed(self.base_cmd + [code_file]), env=worker_env(), **kwargs
                )

                logger.debug(f"Started Julia process {proc.pid}")

                # Wait for process with timeout
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    exit_code = proc.returncode
                    elapsed = time.time() - start_time

                    logger.debug(
                        f"Julia execution completed in {elapsed:.2f}s (exit: {exit_code})"
                    )

                    # Clean up temp file
                    self._cleanup_temp_file(code_file)

                    return CodeExecResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                    )

                except subprocess.TimeoutExpired:
                    logger.error(
                        f"Julia execution timed out after {timeout}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )

                    # CRITICAL: Kill the process AND all its children
                    self._kill_process_tree(proc, code_file)

                    # If this was our last retry, return timeout error
                    if attempt >= self.max_retries:
                        self._cleanup_temp_file(code_file)
                        return CodeExecResult(
                            stdout="",
                            stderr=f"Execution timed out after {timeout}s",
                            exit_code=124,  # Standard timeout exit code
                        )

                    # Wait before retry
                    time.sleep(1.0)
                    continue

            except FileNotFoundError:
                logger.error(f"Julia executable not found at {self.julia_path}")
                return CodeExecResult(
                    stdout="",
                    stderr=f"Julia executable not found: {self.julia_path}",
                    exit_code=127,
                )

            except Exception as e:
                logger.error(
                    f"Error executing Julia (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

                # Try to kill process if it exists
                if proc is not None and proc.poll() is None:
                    self._kill_process_tree(proc, code_file)

                # If this was our last retry, return error
                if attempt >= self.max_retries:
                    self._cleanup_temp_file(code_file)
                    return CodeExecResult(
                        stdout="",
                        stderr=f"Error executing Julia code: {str(e)}",
                        exit_code=1,
                    )

                # Wait before retry
                time.sleep(1.0)
                continue

            finally:
                # Always ensure temp file is cleaned up
                self._cleanup_temp_file(code_file)

        # Should never reach here
        return CodeExecResult(
            stdout="",
            stderr="Unexpected error: all retries exhausted",
            exit_code=1,
        )

    def _cleanup_temp_file(self, code_file: Optional[str]) -> None:
        """Clean up temporary file safely."""
        if code_file and Path(code_file).exists():
            try:
                Path(code_file).unlink()
            except Exception as e:
                logger.debug(f"Could not delete temp file {code_file}: {e}")

    @staticmethod
    def enable_process_pool(size: int = 4, timeout: Optional[int] = None) -> bool:
        """
        Enable the shared Julia process pool for all JuliaExecutor instances.

        This provides 50-100x speedup for repeated code executions by reusing
        persistent Julia processes instead of spawning new ones.

        Args:
            size: Number of worker processes to create (default: 4)
            timeout: Default timeout for code execution in seconds.
                     If None, reads from JULIA_EXECUTION_TIMEOUT env var (default: 120)

        Returns:
            True if pool was created successfully, False otherwise
        """
        if not POOL_AVAILABLE:
            logger.warning(
                "Process pool not available (julia_process_pool module not found). "
                "Falling back to subprocess execution."
            )
            return False

        # Read timeout from env var if not explicitly provided
        if timeout is None:
            timeout = int(os.getenv("JULIA_EXECUTION_TIMEOUT", "120"))

        with JuliaExecutor._pool_lock:
            if JuliaExecutor._shared_pool is not None:
                logger.debug("Process pool already enabled")
                return True

            try:
                logger.info(f"Enabling Julia process pool with {size} workers")
                JuliaExecutor._shared_pool = JuliaProcessPool(
                    size=size, timeout=timeout
                )
                JuliaExecutor._pool_size = size
                JuliaExecutor._pool_timeout = timeout
                logger.info("Julia process pool enabled successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to enable process pool: {e}")
                return False

    @staticmethod
    def shutdown_pool() -> None:
        """
        Shutdown the shared Julia process pool.

        This should be called when you're done with all Julia executions
        to properly clean up worker processes.
        """
        with JuliaExecutor._pool_lock:
            if JuliaExecutor._shared_pool is not None:
                logger.info("Shutting down Julia process pool")
                try:
                    JuliaExecutor._shared_pool.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down pool: {e}")
                finally:
                    JuliaExecutor._shared_pool = None

    @staticmethod
    def is_pool_enabled() -> bool:
        """Check if the process pool is currently enabled."""
        with JuliaExecutor._pool_lock:
            return JuliaExecutor._shared_pool is not None

    @staticmethod
    def get_pool_metrics() -> dict:
        """Get metrics about the process pool."""
        if JuliaExecutor._shared_pool is None:
            return {
                "enabled": False,
                "pool_size": 0,
                "available_workers": 0,
            }

        return {
            "enabled": True,
            "pool_size": JuliaExecutor._pool_size,
            "timeout": JuliaExecutor._pool_timeout,
            "available_workers": JuliaExecutor._pool_size,
        }
