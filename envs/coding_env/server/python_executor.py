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
episodes never share interpreter state. Workers are forked from a zygote, a
template process started once per server with a minimal environment (no
server secrets) and with smolagents already imported. They run as an
unprivileged user when the server is root and that user can run the
interpreter, and are killed on timeout. Server and workers exchange bounded
JSON only, never pickle, because workers run untrusted code.
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
import warnings
import weakref
from multiprocessing.connection import Connection
from multiprocessing.reduction import recv_handle, send_handle

from openenv.core.env_server.types import CodeExecResult
from smolagents import LocalPythonExecutor

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_SAFE_IMPORTS = ["json"]

DEFAULT_EXEC_TIMEOUT_S = 30.0  # same as smolagents' MAX_EXECUTION_TIME_SECONDS
ZYGOTE_STARTUP_TIMEOUT_S = 60.0  # one-time import of smolagents and openenv
ZYGOTE_REPLY_TIMEOUT_S = 10.0
MAX_OUTPUT_CHARS = 1_000_000  # per stream and call, cut in the worker
MAX_REPLY_BYTES = 16 * 1024 * 1024
UNPRIVILEGED_UID = 65534  # nobody
# The only server environment variables a worker inherits. Anything else
# (tokens, API keys, credentials) stays in the server process.
WORKER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4
# The zygote turns non-dumpable before importing anything, so a same-uid
# process can't read its memory while it loads. Workers inherit the flag.
_ZYGOTE_BOOT = (
    "import sys\n"
    "if sys.platform.startswith('linux'):\n"
    "    import ctypes\n"
    f"    ctypes.CDLL(None).prctl({PR_SET_DUMPABLE}, 0, 0, 0, 0)\n"
    f"from {__name__} import _zygote_main\n"
    "_zygote_main(int(sys.argv[1]))\n"
)

TIMEOUT_MESSAGE = (
    "TimeoutError: code execution exceeded the maximum execution time of "
    "{seconds:g} seconds. The Python session was restarted, so variables "
    "defined in earlier steps are gone."
)
CRASH_MESSAGE = (
    "RuntimeError: the Python session stopped unexpectedly and was restarted, "
    "so variables defined in earlier steps are gone."
)
START_FAILED_MESSAGE = (
    "RuntimeError: the Python session could not start. Try again later."
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


def _prctl(option: int, value: int) -> None:
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(option, value, 0, 0, 0)


def set_non_dumpable() -> None:
    """Linux: no ptrace and no /proc/<pid>/{mem,environ} for same-uid processes."""
    _prctl(PR_SET_DUMPABLE, 0)


def _worker_users() -> list[int | None]:
    """Users to try for the zygote and its workers, in order (None: the server's)."""
    mode = os.environ.get("CODING_ENV_WORKER_USER", "auto")
    if mode == "same":
        return [None]
    if mode == "nobody":
        return [UNPRIVILEGED_UID]
    if mode != "auto":
        raise ValueError(
            f"CODING_ENV_WORKER_USER must be auto, nobody or same, not {mode!r}"
        )
    return [UNPRIVILEGED_UID, None] if os.geteuid() == 0 else [None]


def _worker_env(home: str) -> dict[str, str]:
    env = {k: os.environ[k] for k in WORKER_ENV_ALLOWLIST if k in os.environ}
    # The zygote imports this module the way the server did.
    env["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(p) for p in sys.path if p and os.path.isdir(p)
    )
    # Some imports (fastmcp through openenv) need a home directory.
    env.update(HOME=home, TMPDIR=home, TMP=home, TEMP=home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _kill(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait()


class _Zygote:
    """Template process that forks one worker per session.

    It starts from `sys.executable` with a minimal environment, as `user` when
    set, becomes non-dumpable before its imports and then imports smolagents
    and this module once. Workers are forked from it, so they start in
    milliseconds, share its memory pages and are non-dumpable from their first
    instruction. The threaded server itself is never forked. The zygote also
    kills and reaps its workers, so a worker's pid can't be reused while the
    server still refers to it.
    """

    def __init__(self, user: int | None):
        self.user = user
        self._lock = threading.Lock()
        self._live: set[int] = set()
        self._dead: collections.deque[int] = collections.deque()
        cleanup = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, cleanup.close)
        try:
            home = tempfile.mkdtemp(prefix="coding_env_")
            cleanup.callback(shutil.rmtree, home, ignore_errors=True)
            parent_sock, child_sock = socket.socketpair()
            self._conn = Connection(parent_sock.detach())
            cleanup.callback(self._conn.close)
            kwargs = {}
            if user is not None:
                os.chown(home, user, user)
                kwargs = dict(user=user, group=user, extra_groups=[])
            with child_sock:
                self._proc = subprocess.Popen(
                    [sys.executable, "-c", _ZYGOTE_BOOT, str(child_sock.fileno())],
                    pass_fds=(child_sock.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    env=_worker_env(home),
                    cwd=home,
                    **kwargs,
                )
            cleanup.callback(_kill, self._proc)
            if self._receive(ZYGOTE_STARTUP_TIMEOUT_S) != {"ready": True}:
                raise ValueError("bad zygote handshake")
        except BaseException:
            self.close()
            raise

    @property
    def alive(self) -> bool:
        return self._finalizer.alive and self._proc.poll() is None

    def spawn(self, fd: int, config: dict) -> int:
        """Fork a worker that serves `fd` and return its pid."""
        with self._lock:
            pid = self._request(config, fd).get("pid")
        if type(pid) is not int or pid <= 0:
            raise ValueError("bad zygote reply")
        self._live.add(pid)
        return pid

    def kill(self, pid: int, wait: bool = True) -> None:
        """Kill and reap a worker. Safe to call more than once.

        Without `wait` (garbage collection, which may run while this thread
        holds the lock), a busy zygote kills it with its next request.
        """
        try:
            self._live.remove(pid)
        except KeyError:
            return
        self._dead.append(pid)
        if not self._lock.acquire(blocking=wait):
            return
        try:
            self._request(None)
        except Exception:
            # The zygote is gone, and its workers with it (PR_SET_PDEATHSIG).
            logger.debug("could not reach the coding_env zygote", exc_info=True)
        finally:
            self._lock.release()

    def close(self) -> None:
        self._finalizer()

    def _request(self, spawn: dict | None, fd: int | None = None) -> dict:
        """Send the pending kills and an optional spawn. Caller holds the lock."""
        kills = []
        while self._dead:
            kills.append(self._dead.popleft())
        try:
            self._conn.send_bytes(json.dumps({"kill": kills, "spawn": spawn}).encode())
            if fd is not None:
                send_handle(self._conn, fd, self._proc.pid)
            return self._receive(ZYGOTE_REPLY_TIMEOUT_S)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict:
        if not self._conn.poll(timeout):
            raise TimeoutError("the coding_env zygote did not reply")
        return json.loads(self._conn.recv_bytes(1024))


_zygote: _Zygote | None = None
_zygote_lock = threading.Lock()


def _start_zygote() -> _Zygote:
    *fallbacks, user = _worker_users()
    for candidate in fallbacks:
        try:
            return _Zygote(candidate)
        except Exception as e:
            # A warning, not a log record: nothing configures this logger in
            # a plain uvicorn server, and this must be seen.
            warnings.warn(
                f"coding_env workers run as the server's uid {os.geteuid()}, "
                f"because uid {candidate} can't start {sys.executable} with this "
                f"PYTHONPATH ({e!r}). Install the env where that uid can read "
                "it, or set CODING_ENV_WORKER_USER=same to skip this check.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _Zygote(user)


def _get_zygote() -> _Zygote:
    """This server's zygote, started on first use and restarted if it died."""
    global _zygote
    with _zygote_lock:
        if _zygote is None or not _zygote.alive:
            _zygote = _start_zygote()
        return _zygote


class IsolatedPyExecutor:
    """One Python session in its own worker process.

    Same `run(code) -> CodeExecResult` contract as `PyExecutor`, which runs
    inside the worker. The worker is forked by `start()` (or the first
    `run()`) and killed by `close()`, or when the executor is garbage
    collected. A call that exceeds `timeout_s` kills the worker, and the next
    call starts a fresh one.
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
        self._memory_mb = int(os.environ.get("CODING_ENV_WORKER_MEMORY_MB", "0"))
        self._zygote: _Zygote | None = None
        self._pid: int | None = None
        self._conn: Connection | None = None
        self._home: str | None = None
        self._finalizer: weakref.finalize | None = None

    @property
    def pid(self) -> int | None:
        return self._pid

    def start(self) -> bool:
        """Fork the worker if it is not running. False if it could not start."""
        if self._conn is not None:
            return True
        # Everything acquired below is released by close(), on failure, or
        # when this executor is garbage collected.
        cleanup = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, cleanup.close)
        try:
            self._zygote = zygote = _get_zygote()
            self._home = tempfile.mkdtemp(prefix="coding_env_")
            cleanup.callback(shutil.rmtree, self._home, ignore_errors=True)
            if zygote.user is not None:
                os.chown(self._home, zygote.user, zygote.user)
            parent_sock, child_sock = socket.socketpair()
            conn = Connection(parent_sock.detach())
            cleanup.callback(conn.close)
            config = {
                "home": self._home,
                "imports": self._additional_imports,
                "memory_mb": self._memory_mb,
            }
            with child_sock:
                self._pid = zygote.spawn(child_sock.fileno(), config)
            cleanup.callback(zygote.kill, self._pid, False)
            self._conn = conn
        except Exception:
            logger.exception("could not start the coding_env worker process")
            self.close()
            return False
        return True

    def run(self, code: str) -> CodeExecResult:
        if not self.start():
            return CodeExecResult(stdout="", stderr=START_FAILED_MESSAGE, exit_code=1)
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
        """Kill the worker and remove its home. Safe to call more than once."""
        if self._pid is not None:
            self._zygote.kill(self._pid)
        if self._finalizer is not None:
            self._finalizer()
        self._zygote = self._pid = self._conn = self._home = self._finalizer = None


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"


def _zygote_main(fd: int) -> None:
    """Zygote process: fork workers, and kill and reap them, on request."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    conn = Connection(fd)
    children: set[int] = set()
    conn.send_bytes(b'{"ready":true}')
    while True:
        try:
            request = json.loads(conn.recv_bytes())
        except (EOFError, OSError):
            return  # the server is gone, PR_SET_PDEATHSIG stops the workers
        for pid in request["kill"]:
            if pid in children:
                children.remove(pid)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
        reply = {}
        if request["spawn"] is not None:
            worker_fd = recv_handle(conn)
            pid = os.fork()
            if pid == 0:
                try:
                    conn.close()
                    _worker_main(worker_fd, request["spawn"])
                except BaseException:
                    traceback.print_exc()
                finally:
                    os._exit(0)
            os.close(worker_fd)
            children.add(pid)
            reply["pid"] = pid
        conn.send_bytes(json.dumps(reply).encode())


def _worker_main(fd: int, config: dict) -> None:
    """Forked worker: run one session's code, one call at a time."""
    _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)  # die with the zygote
    if config["memory_mb"]:
        import resource

        # On top of the address space the worker starts with: the zygote's
        # imports reserve far more of it than they use (about 2.3 GB).
        with open("/proc/self/statm") as statm:
            start = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        limit = start + config["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    home = config["home"]
    os.chdir(home)
    os.environ.update(HOME=home, TMPDIR=home, TMP=home, TEMP=home)
    tempfile.tempdir = None  # the zygote may have cached its own
    conn = Connection(fd)
    # Model code may patch json's classes in this process (it only hurts its
    # own session). Explicit separators make the replies immune to the
    # common case of patched class-level separators.
    encode = json.JSONEncoder(separators=(",", ":")).encode
    decode = json.JSONDecoder().decode
    # The server enforces the timeout by killing this process, so the
    # smolagents thread timeout (which leaves runaway threads) is off.
    executor = PyExecutor(additional_imports=config["imports"], timeout_seconds=None)
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
