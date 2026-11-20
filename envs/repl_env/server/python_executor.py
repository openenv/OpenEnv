# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python code executor for the REPL environment.

The model's code runs in smolagents' `LocalPythonExecutor`, an AST interpreter
that is not a sandbox. It runs in a worker process, one per executor, so the
code never shares memory with the server, which may hold tokens:
- the worker gets a minimal environment, without the server's variables,
- it runs as `nobody` when the server runs as root,
- the worker and the server are non-dumpable on Linux, so other processes of
  the same user can't read their `/proc/<pid>/{environ,mem}`.

Functions that need the server (`llm_query`, `rlm_query`, ...) are injected
with `inject_function` and run in the server: the worker only sends their
arguments and gets their results back. Server and worker exchange JSON only.

On top of that:
- Context loading (set_context)
- Variable access (get_variable, list_variables)
- Output capped at 8,192 characters per turn (configurable)
- Persistent namespace across code blocks
"""

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_IMPORTS = [
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

# The only server environment variables the worker gets.
WORKER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
UNPRIVILEGED_UID = 65534  # nobody
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
WORKER_STOPPED = (
    "RuntimeError: the Python session stopped. Call reset() to start a new one."
)

PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4


def _prctl(option: int, value: int) -> None:
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(option, value, 0, 0, 0)


def set_non_dumpable() -> None:
    """Linux: other processes of the same user can't read this one's /proc/<pid>/{environ,mem}."""
    _prctl(PR_SET_DUMPABLE, 0)


def _to_json(value: Any) -> Any:
    """`value` as JSON data; objects JSON can't hold become their `str()`."""
    return json.loads(json.dumps(value, default=str))


class PythonExecutor:
    """Python session with a persistent namespace, run in its own worker process.

    - Context loading for RLM tasks
    - Variable tracking for observation
    - Function injection for llm_query, llm_query_batched (run in the server)
    - Configurable output length limit (default 8192 chars per Prime Intellect)
    """

    def __init__(
        self,
        max_output_length: int = 8192,
        allowed_imports: Optional[List[str]] = None,
    ):
        """Initialize the executor and start its worker process.

        Args:
            max_output_length: Maximum characters for stdout/stderr (default 8192)
            allowed_imports: List of allowed module names for import

        Note:
            smolagents.LocalPythonExecutor does NOT support wall-clock timeouts.
            Instead, it limits operations (10M ops) and while iterations (1M).
        """
        self.max_output_length = max_output_length
        self.allowed_imports = allowed_imports or DEFAULT_IMPORTS
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[Connection] = None
        self._home: Optional[str] = None
        self._start()

    def _start(self) -> None:
        set_non_dumpable()
        self._home = tempfile.mkdtemp(prefix="repl_env_")
        uid = UNPRIVILEGED_UID if os.geteuid() == 0 else None
        if uid is not None:
            os.chown(self._home, uid, uid)
        env = {k: os.environ[k] for k in WORKER_ENV_ALLOWLIST if k in os.environ}
        # The worker imports smolagents the way the server did.
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p) for p in sys.path if p and os.path.isdir(p)
        )
        env.update(HOME=self._home, TMPDIR=self._home, PYTHONDONTWRITEBYTECODE="1")
        parent_sock, child_sock = socket.socketpair()
        with child_sock:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-s",
                    os.path.abspath(__file__),
                    str(child_sock.fileno()),
                ],
                pass_fds=(child_sock.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                env=env,
                cwd=self._home,
            )
        self._conn = Connection(parent_sock.detach())
        self._conn.send_bytes(
            json.dumps(
                {
                    "imports": self.allowed_imports,
                    "uid": uid,
                    "max_output_length": self.max_output_length,
                }
            ).encode()
        )

    def _request(self, message: dict) -> dict:
        """Send `message` and answer the worker's function calls until it replies."""
        if self._conn is None:
            raise RuntimeError(WORKER_STOPPED)
        try:
            self._conn.send_bytes(json.dumps(message).encode())
            while True:
                reply = json.loads(self._conn.recv_bytes(MAX_MESSAGE_BYTES))
                if "call" not in reply:
                    return reply
                self._conn.send_bytes(json.dumps(self._call(reply)).encode())
        except (EOFError, OSError, ValueError):
            logger.warning("repl_env worker process stopped", exc_info=True)
            self.close()
            raise RuntimeError(WORKER_STOPPED) from None

    def _call(self, request: dict) -> dict:
        """Run an injected function for the worker. Its arguments come from model code."""
        function = self._functions.get(request["call"])
        if function is None:
            return {"error": f"NameError: {request['call']!r} is not available"}
        try:
            return {"result": _to_json(function(*request["args"], **request["kwargs"]))}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

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
            value: Variable value (JSON data)
        """
        self._request({"op": "set", "name": name, "value": value})

    def get_variable(self, name: str) -> Optional[Any]:
        """Retrieve a variable from namespace.

        Args:
            name: Variable name

        Returns:
            The variable value as JSON data (other objects as their `str()`),
            or None if not found
        """
        return self._request({"op": "get", "name": name})["value"]

    def list_variables(self) -> List[str]:
        """List non-private variables in namespace.

        Returns:
            List of variable names (excluding private and builtins)
        """
        return self._request({"op": "list"})["names"]

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return results.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with stdout, stderr, locals_snapshot, execution_time,
            success, and exception fields
        """
        start_time = time.time()
        try:
            result = self._request({"op": "exec", "code": code})
        except RuntimeError as e:
            result = {
                "stdout": "",
                "stderr": str(e),
                "locals_snapshot": {},
                "success": False,
                "exception": str(e),
            }
        result["execution_time"] = time.time() - start_time
        return result

    def reset(self) -> None:
        """Reset namespace to initial state, in a new worker process."""
        self.close()
        self._functions.clear()
        self._start()

    def inject_function(self, name: str, func: Callable[..., Any]) -> None:
        """Make a server function callable from the namespace.

        The function runs in the server. Model code calls it with JSON
        arguments and gets its result as JSON data.

        Args:
            name: Function name in namespace
            func: The callable to inject
        """
        self._functions[name] = func
        self._request({"op": "function", "name": name})

    def close(self) -> None:
        """Stop the worker process. Safe to call more than once."""
        conn, proc, home = self._conn, self._proc, self._home
        self._conn = self._proc = self._home = None
        if conn is not None:
            conn.close()
        if proc is not None:
            proc.kill()
            proc.wait()
        if home is not None:
            shutil.rmtree(home, ignore_errors=True)

    def __del__(self):
        self.close()


def _worker(fd: int) -> None:
    """Worker process: the model's Python session, one request at a time."""
    import importlib
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    set_non_dumpable()
    conn = Connection(fd)
    # Model code may patch json's classes in this process; these were made before.
    encode = json.JSONEncoder(default=str).encode
    decode = json.JSONDecoder().decode
    config = decode(conn.recv_bytes().decode())

    from smolagents import LocalPythonExecutor

    # Import the allowed modules now, while the worker can still read the
    # server's virtual environment.
    for name in config["imports"]:
        try:
            importlib.import_module(name.removesuffix(".*"))
        except ImportError:
            pass
    if config["uid"] is not None:
        os.setgroups([])
        os.setgid(config["uid"])
        os.setuid(config["uid"])
        set_non_dumpable()  # changing the uid resets it, and the death signal
    _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    max_output_length = config["max_output_length"]
    executor = LocalPythonExecutor(additional_authorized_imports=config["imports"])
    user_variables: set[str] = set()

    def call_server(name: str) -> Callable[..., Any]:
        def call(*args, **kwargs):
            conn.send_bytes(
                encode({"call": name, "args": args, "kwargs": kwargs}).encode()
            )
            reply = decode(conn.recv_bytes().decode())
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply["result"]

        return call

    def names() -> List[str]:
        return sorted(
            {k for k in executor.state if not k.startswith("_")} | user_variables
        )

    def final_var(var_name: str) -> str:
        """Look up a variable by name and return FINAL(value) for detection."""
        var_name = str(var_name).strip().strip("\"'")
        value = executor.state.get(var_name)
        if value is not None:
            return f"FINAL({value})"
        return f"FINAL_VAR({var_name})"

    tools: Dict[str, Callable[..., Any]] = {
        "format_exc": traceback.format_exc,
        "safe_json_dumps": lambda obj: json.dumps(obj, default=repr),
        # FINAL(x) and print(f'FINAL({x})') both put the pattern in stdout.
        "FINAL": lambda value: f"FINAL({value})",
        "FINAL_VAR": final_var,
        "SHOW_VARS": names,
    }

    def send_tools() -> None:
        user_variables.update(tools)
        executor.send_tools(tools)

    send_tools()

    def cap(text: str) -> str:
        if len(text) <= max_output_length:
            return text
        return text[:max_output_length] + f"\n... (truncated, total {len(text)} chars)"

    while True:
        try:
            request = decode(conn.recv_bytes().decode())
        except EOFError:
            return
        op = request["op"]
        if op == "set":
            executor.send_variables({request["name"]: request["value"]})
            user_variables.add(request["name"])
            reply: Dict[str, Any] = {}
        elif op == "get":
            value = executor.state.get(request["name"])
            reply = {"value": _to_json(value)}
        elif op == "list":
            reply = {"names": names()}
        elif op == "function":
            tools[request["name"]] = call_server(request["name"])
            send_tools()
            reply = {}
        else:  # exec
            reply = _execute(executor, request["code"], user_variables)
            reply["stdout"] = cap(reply["stdout"])
            reply["stderr"] = cap(reply["stderr"])
        conn.send_bytes(encode(reply).encode())


def _execute(executor: Any, code: str, user_variables: set) -> Dict[str, Any]:
    success = True
    exception_msg = None
    new_locals: Dict[str, str] = {}
    pre_state_keys = set(executor.state.keys())
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    try:
        exec_result = executor(code)

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

    # Capture new/modified variables
    for key in executor.state:
        if key not in pre_state_keys and not key.startswith("_"):
            try:
                val_repr = repr(executor.state[key])
                if len(val_repr) > 500:
                    val_repr = val_repr[:500] + "..."
                new_locals[key] = val_repr
                user_variables.add(key)
            except Exception:
                new_locals[key] = "<unrepresentable>"

    return {
        "stdout": "\n".join(part for part in stdout_parts if part),
        "stderr": "\n".join(part for part in stderr_parts if part),
        "locals_snapshot": new_locals,
        "success": success,
        "exception": exception_msg,
    }


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
