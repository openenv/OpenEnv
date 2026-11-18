# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

`PyExecutor` interprets code inside the current process. smolagents is an
AST interpreter, not a sandbox: imported classes and module state are shared
by every executor in a process. `IsolatedPyExecutor` therefore runs one
`PyExecutor` per session in its own worker process, so sessions and
episodes never share interpreter state. The worker gets a minimal
environment (no server secrets), runs as an unprivileged user when the
server is root, and is killed on timeout. Server and worker exchange bounded
JSON only, never pickle, because the worker runs untrusted code.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
from multiprocessing.connection import Connection

from openenv.core.env_server.types import CodeExecResult
from smolagents import LocalPythonExecutor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_SAFE_IMPORTS = ["json"]

DEFAULT_EXEC_TIMEOUT_S = 30.0  # same as smolagents' MAX_EXECUTION_TIME_SECONDS
WORKER_STARTUP_TIMEOUT_S = 60.0
MAX_OUTPUT_CHARS = 1_000_000  # per stream and call, cut in the worker
MAX_REPLY_BYTES = 16 * 1024 * 1024
UNPRIVILEGED_UID = 65534  # nobody
# The only server environment variables a worker inherits. Anything else
# (tokens, API keys, credentials) stays in the server process.
WORKER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")

TIMEOUT_MESSAGE = (
    "TimeoutError: code execution exceeded the maximum execution time of "
    "{seconds:g} seconds. The Python session was restarted, so variables "
    "defined in earlier steps are gone."
)
CRASH_MESSAGE = (
    "RuntimeError: the Python session stopped unexpectedly and was restarted, "
    "so variables defined in earlier steps are gone."
)


class PyExecutor:
    """Wrapper around smolagents LocalPythonExecutor.

    The wrapper registers a few non-privileged helper tools to the
    LocalPythonExecutor that can be used by the executed code to
    format exceptions and to safely stringify results for improved
    error reporting.
    """

    def __init__(
        self,
        additional_imports: list[str] | None = None,
        timeout_seconds: int | None = 30,
    ):
        if additional_imports is None:
            additional_imports = DEFAULT_SAFE_IMPORTS.copy()

        self._executor = LocalPythonExecutor(
            additional_authorized_imports=additional_imports,
            timeout_seconds=timeout_seconds,
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

        except Exception as e:
            # Any unexpected exception from the LocalPythonExecutor is
            # returned with a full traceback to make debugging easier.
            tb = traceback.format_exc()
            logger.exception("LocalPythonExecutor raised an exception during run")
            return CodeExecResult(stdout="", stderr=tb, exit_code=1)


def set_non_dumpable() -> None:
    """Linux: no ptrace and no /proc/<pid>/{mem,environ} for same-uid processes."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        PR_SET_DUMPABLE = 4
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except Exception:
        logger.debug("prctl(PR_SET_DUMPABLE) failed", exc_info=True)


def _worker_env(home: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in WORKER_ENV_ALLOWLIST if k in os.environ}
    # The worker imports this module the way the server did.
    env["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(p) for p in sys.path if p and os.path.isdir(p)
    )
    # Some imports (fastmcp through openenv) need a home directory. Each worker
    # gets its own, so sessions can't pass files to each other through it.
    env["HOME"] = home
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


class IsolatedPyExecutor:
    """One Python session in its own worker process.

    Same `run(code) -> CodeExecResult` contract as `PyExecutor`, which runs
    inside the worker. The worker is started by `start()` (or the first
    `run()`) and killed by `close()`. A call that exceeds `timeout_s` kills
    the worker, and the next call starts a fresh one.
    """

    def __init__(
        self,
        additional_imports: list[str] | None = None,
        timeout_s: float | None = None,
    ):
        if timeout_s is None:
            timeout_s = float(
                os.environ.get("CODING_ENV_EXEC_TIMEOUT_S", DEFAULT_EXEC_TIMEOUT_S)
            )
        self._additional_imports = additional_imports
        self._timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._conn: Connection | None = None
        self._home: str | None = None
        self._ready = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        """Start the worker without waiting for it to finish importing."""
        if self._proc is not None:
            return
        parent_sock, child_sock = socket.socketpair()
        self._home = tempfile.mkdtemp(prefix="coding_env_")
        kwargs = {}
        if sys.platform != "win32" and os.geteuid() == 0:
            os.chown(self._home, UNPRIVILEGED_UID, UNPRIVILEGED_UID)
            kwargs = dict(
                user=UNPRIVILEGED_UID, group=UNPRIVILEGED_UID, extra_groups=[]
            )
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    f"import sys; from {__name__} import _worker_main; "
                    "_worker_main(int(sys.argv[1]))",
                    str(child_sock.fileno()),
                ],
                pass_fds=(child_sock.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                env=_worker_env(self._home),
                cwd=self._home,
                **kwargs,
            )
        except BaseException:
            parent_sock.close()
            raise
        finally:
            child_sock.close()
        self._conn = Connection(parent_sock.detach())
        self._conn.send_bytes(
            json.dumps({"imports": self._additional_imports}).encode()
        )

    def _wait_ready(self) -> None:
        if not self._conn.poll(WORKER_STARTUP_TIMEOUT_S):
            raise TimeoutError("the worker did not start")
        if json.loads(self._conn.recv_bytes(1024)) != {"ready": True}:
            raise ValueError("bad worker handshake")
        self._ready = True

    def run(self, code: str) -> CodeExecResult:
        try:
            self.start()
            if not self._ready:
                self._wait_ready()
        except Exception:
            logger.exception("could not start the coding_env worker process")
            self.close()
            return CodeExecResult(stdout="", stderr=CRASH_MESSAGE, exit_code=1)
        try:
            self._conn.send_bytes(json.dumps({"code": code}).encode())
            if not self._conn.poll(self._timeout_s):
                self.close()
                return CodeExecResult(
                    stdout="",
                    stderr=TIMEOUT_MESSAGE.format(seconds=self._timeout_s),
                    exit_code=1,
                )
            # Bounded JSON with typed fields only: the worker is not trusted.
            reply = json.loads(self._conn.recv_bytes(MAX_REPLY_BYTES))
            stdout, stderr = reply["stdout"], reply["stderr"]
            exit_code = reply["exit_code"]
            if not (
                isinstance(stdout, str)
                and isinstance(stderr, str)
                and type(exit_code) is int
            ):
                raise ValueError("malformed worker reply")
            return CodeExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)
        except (EOFError, OSError, ValueError, KeyError, TypeError):
            logger.warning("coding_env worker process failed, restarting it")
            self.close()
            return CodeExecResult(stdout="", stderr=CRASH_MESSAGE, exit_code=1)

    def close(self) -> None:
        """Kill the worker. Safe to call more than once."""
        proc, conn, home = self._proc, self._conn, self._home
        self._proc, self._conn, self._home, self._ready = None, None, None, False
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                logger.debug("could not reap the worker process", exc_info=True)
        if home is not None:
            shutil.rmtree(home, ignore_errors=True)


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"


def _worker_main(fd: int) -> None:
    """Worker process: run one session's code, one call at a time."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    set_non_dumpable()
    conn = Connection(fd)
    # Model code may patch json's classes in this process (it only hurts its
    # own session). Explicit separators make the replies immune to the
    # common case of patched class-level separators.
    encode = json.JSONEncoder(separators=(",", ":")).encode
    decode = json.JSONDecoder().decode
    config = decode(conn.recv_bytes().decode())
    # The server enforces the timeout by killing this process, so the
    # smolagents thread timeout (which leaves runaway threads) is off.
    executor = PyExecutor(additional_imports=config["imports"], timeout_seconds=None)
    conn.send_bytes(encode({"ready": True}).encode())
    while True:
        try:
            request = decode(conn.recv_bytes().decode())
        except EOFError:
            return
        result = executor.run(request["code"])
        reply = {
            "stdout": _cap(result.stdout),
            "stderr": _cap(result.stderr),
            "exit_code": int(result.exit_code),
        }
        conn.send_bytes(encode(reply).encode())
