"""
Subprocess lifecycle management for AWM sub-environments.
Each AWM scenario is a self-contained FastAPI application. This module handles:
- Patching generated code (DB path, FastApiMCP injection)
- Starting / stopping subprocess on a random port
- Persistent MCP connection for efficient tool calls

TRUST BOUNDARY:
    The ``full_code`` argument to ``ScenarioProcess.start`` originates from
    the curated HuggingFace dataset (Snowflake/AgentWorldModel-1K) and is
    treated as TRUSTED.

    This is *intentional*: scenario code is the data the env exists to serve,
    and the AWM design assumes it is benign. The container is the outer
    isolation boundary; per-subprocess sandboxing is not applied here.

    Verifier code (``_verifier_runner.py``) follows a different model with
    an explicit sandbox; that asymmetry is by design.

    If the dataset trust assumption changes, this module is the place that
    must be hardened (e.g. per-subprocess unshare/seccomp/non-root uid).
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import (
    MAX_PORT_RETRIES,
    READY_POLL_INTERVAL,
    READY_TIMEOUT,
    RETRY_READY_TIMEOUT,
)


logger = logging.getLogger(__name__)


def _get_random_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _patch_env_code(full_code: str, db_path: str, host: str, port: int) -> str:
    """
    Patch the generated FastAPI code:
    1. Replace create_engine() call to point to the session-specific DB.
    2. Inject FastApiMCP mount before uvicorn.run().
    3. Fix uvicorn.run() to use the assigned host/port.
    """
    new_lines = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
    ]

    sql_path = f"sqlite:///{db_path}"

    for line in full_code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = f"{left}create_engine('{sql_path}', connect_args={{'check_same_thread': False}})"

        if "uvicorn.run(app" in line:
            mcp_inject = textwrap.dedent("""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """)
            for inject_line in mcp_inject.strip().split("\n"):
                new_lines.append(f"    {inject_line}")

            line = f"    uvicorn.run(app, host='{host}', port={port})"

        new_lines.append(line)

    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Persistent MCP connection
# ---------------------------------------------------------------------------
class _MCPConnection:
    """Persistent MCP client session backed by AsyncExitStack.

    Keeps the streamable HTTP transport and ClientSession alive across
    multiple ``call_tool`` / ``list_tools`` invocations, avoiding the
    overhead of creating a new connection per call.
    """

    def __init__(self) -> None:
        self._stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self, mcp_url: str) -> None:
        self._stack = contextlib.AsyncExitStack()
        try:
            read_stream, write_stream, _ = await self._stack.enter_async_context(
                streamable_http_client(mcp_url)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self.close()
            raise

    async def list_tools(self) -> list[dict]:
        assert self._session is not None, "Not connected"
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        assert self._session is not None, "Not connected"
        result = await self._session.call_tool(tool_name, arguments)

        parts = []
        for c in result.content:
            if hasattr(c, "text"):
                parts.append(c.text)
            else:
                parts.append(str(c))

        text = "\n".join(parts)
        if result.isError:
            return {"success": False, "result": None, "error": text}
        return {"success": True, "result": text, "error": None}

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
            self._session = None


# ---------------------------------------------------------------------------
# ScenarioProcess
# ---------------------------------------------------------------------------
class ScenarioProcess:
    """Manages a single sub-environment subprocess with persistent MCP connection."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._temp_dir: str | None = None
        self._owns_temp_dir: bool = False
        self._server_py: str | None = None
        self._log_file = None
        self._log_path: str | None = None
        # Persistent MCP connection + dedicated event loop
        self._mcp: _MCPConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def mcp_url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}/mcp"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, full_code: str, db_path: str, session_dir: str) -> str:
        """
        Start the sub-environment subprocess and establish persistent MCP
        connection.

        Returns:
            The MCP URL of the started server.

        Raises:
            RuntimeError: If the server fails to start within the timeout.
        """
        self.stop()

        self._temp_dir = session_dir
        self._owns_temp_dir = False  # caller owns the directory
        host = "127.0.0.1"

        last_error = ""
        for attempt in range(1 + MAX_PORT_RETRIES):
            self._port = _get_random_port()
            timeout = READY_TIMEOUT if attempt == 0 else RETRY_READY_TIMEOUT

            patched_code = _patch_env_code(full_code, db_path, host, self._port)

            self._server_py = f"{self._temp_dir}/server.py"
            with open(self._server_py, "w", encoding="utf-8") as f:
                f.write(patched_code)

            if attempt > 0:
                logger.info(
                    f"Retry {attempt}/{MAX_PORT_RETRIES} on port {self._port} "
                    f"(timeout={timeout}s) ..."
                )
            else:
                logger.info(f"Starting sub-env on port {self._port} ...")

            self._log_path = f"{self._temp_dir}/server.log"
            self._log_file = open(self._log_path, "w", encoding="utf-8")

            self._process = subprocess.Popen(
                [sys.executable, self._server_py],
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )

            if self._wait_for_ready(timeout):
                # Establish persistent MCP connection
                try:
                    self._connect_mcp()
                    logger.info(
                        f"Sub-env ready on port {self._port}, "
                        f"mcp=persistent, log={self._log_path}"
                    )
                    return self.mcp_url
                except Exception as e:
                    last_error = (
                        f"Sub-env started on port {self._port} but "
                        f"MCP connection failed: {e}"
                    )
                    logger.warning(last_error)
                    self._disconnect_mcp()
                    self.stop()
                    continue

            # Failed — collect error output for diagnostics
            failed_port = self._port
            self._log_file.flush()
            try:
                with open(self._log_path, "r", encoding="utf-8") as lf:
                    logged_output = lf.read()
            except OSError:
                logged_output = ""
            last_error = (
                f"Sub-env failed to start on port {failed_port} "
                f"(timeout {timeout}s).\nOutput: {logged_output}"
            )
            logger.warning(last_error)
            self.stop()

        raise RuntimeError(
            f"Sub-env failed after {1 + MAX_PORT_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _wait_for_ready(self, timeout: float = READY_TIMEOUT) -> bool:
        """Poll the MCP endpoint until the server is ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self._process.poll() is not None:
                return False

            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1):
                    time.sleep(0.5)
                    return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                pass

            time.sleep(READY_POLL_INTERVAL)

        return False

    # -- Persistent MCP connection management ---------------------------------

    def _connect_mcp(self) -> None:
        """Create a dedicated event loop and establish persistent MCP session."""
        self._loop = asyncio.new_event_loop()
        self._mcp = _MCPConnection()
        self._loop.run_until_complete(self._mcp.connect(self.mcp_url))

    def _disconnect_mcp(self) -> None:
        """Tear down the persistent MCP session and event loop."""
        if self._mcp is not None:
            if self._loop is not None and not self._loop.is_closed():
                try:
                    self._loop.run_until_complete(self._mcp.close())
                except Exception:
                    pass
            self._mcp = None
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    def list_tools(self, timeout: float = 15.0) -> list[dict]:
        """List MCP tools via persistent connection (sync)."""
        if self._mcp is None or self._loop is None:
            raise RuntimeError("MCP connection not established")
        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(self._mcp.list_tools(), timeout=timeout)
            )
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []

    def call_tool(
        self, tool_name: str, arguments: dict, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Call an MCP tool via persistent connection (sync).

        Returns:
            dict with keys: "success" (bool), "result" (Any), "error" (str | None)
        """
        if self._mcp is None or self._loop is None:
            return {
                "success": False,
                "result": None,
                "error": "MCP connection not established",
            }
        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(
                    self._mcp.call_tool(tool_name, arguments),
                    timeout=timeout,
                )
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "result": None,
                "error": f"Tool call timed out after {timeout}s",
            }
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

    # -- Subprocess lifecycle --------------------------------------------------

    def stop(self) -> None:
        """Stop the subprocess and clean up resources.

        Thread-safe: captures ``self._process`` in a local variable and
        sets the attribute to ``None`` immediately so concurrent callers
        (e.g. cleanup thread + ``_handle_done``) don't double-kill.
        """
        # Close MCP connection first (before killing subprocess)
        self._disconnect_mcp()

        proc = self._process
        if proc is not None:
            self._process = None  # claim ownership immediately

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

        if self._owns_temp_dir and self._temp_dir and os.path.isdir(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._port = None
        self._server_py = None
