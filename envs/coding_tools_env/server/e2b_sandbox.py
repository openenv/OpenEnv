# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2B sandbox wrapper with shell and file-operation helpers."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]

# E2B's default code-interpreter template runs its Jupyter server, and so the
# kernel ``run_shell`` goes through, as root with the notebook in /home/user.
# ``run_command`` keeps the same user and directory, so verify commands see the
# same permissions they did when they ran from the kernel.
_COMMAND_USER = "root"
_COMMAND_CWD = "/home/user"


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] | None = None


def _command_result(
    stdout: str | None, stderr: str | None, exit_code: int
) -> ToolResult:
    """Describe a command that ran and exited, in the shape `run_shell` uses."""
    ok = exit_code == 0
    output = "\n".join(
        part for part in [(stdout or "").strip(), (stderr or "").strip()] if part
    )
    return ToolResult(
        ok=ok,
        output=output,
        error=None if ok else f"exit_code={exit_code}",
        metadata={"exit_code": exit_code, "stdout": stdout or ""},
    )


class E2BSandbox:
    """One E2B sandbox per OpenEnv episode."""

    def __init__(self, api_key: str):
        if Sandbox is None:
            raise ImportError(
                "e2b-code-interpreter is not installed. Install coding_tools_env "
                f"dependencies. Original import error: {_E2B_IMPORT_ERROR}"
            )
        self._sbx = Sandbox.create(api_key=api_key)
        self.sandbox_id: str = self._sbx.sandbox_id
        self.run_shell("mkdir -p /home/user/work")

    def run_shell(self, command: str, timeout_s: float = 30) -> ToolResult:
        code = (
            "import subprocess, json\n"
            f"_r = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={float(timeout_s)!r})\n"
            "print(json.dumps({'returncode': _r.returncode, 'stdout': _r.stdout, 'stderr': _r.stderr}))\n"
        )
        execution = self._sbx.run_code(code)
        result = _decode_json_line(execution.logs.stdout)
        if result is None:
            return ToolResult(ok=False, error=_format_error(execution))
        rc = int(result.get("returncode", 1))
        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        ok = rc == 0
        output = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
        return ToolResult(
            ok=ok,
            output=output,
            error=None if ok else f"exit_code={rc}",
            metadata={"exit_code": rc},
        )

    def run_command(self, command: str, timeout_s: float = 30) -> ToolResult:
        """
        Execute a shell command as its own sandbox process, outside the kernel.

        ``run_shell`` sends a ``subprocess.run`` cell to the code-interpreter
        kernel, so its result depends on the kernel's state: a rebound
        ``subprocess.run``, a changed working directory, edited ``os.environ``.
        That is fine for the agent's own ``bash`` tool and wrong for
        verification. This goes through E2B's process API instead and reports
        the command's real exit status.

        It removes the coupling to the kernel. It is not an isolation boundary:
        the agent's shell runs as root in the same sandbox, so it can still
        change what any process there sees, including this command's login
        shell.

        Args:
            command (`str`):
                Shell command to run.
            timeout_s (`float`, *optional*, defaults to `30`):
                Seconds to wait for the command before giving up on it.

        Returns:
            [`ToolResult`]: `ok` is `True` only for a zero exit status.

        Raises:
            `Exception`: whatever the SDK raises when the command cannot be
                started, or when the wait for it ends without an exit status.
                Only a command that ran and exited is reported as a result:
                a sandbox that is unreachable, expired or unauthenticated says
                nothing about the work being verified.
        """
        # Started in the background so there is a handle to kill if the wait
        # fails: E2B keeps a command running when its connection drops.
        handle = self._sbx.commands.run(
            command,
            background=True,
            user=_COMMAND_USER,
            cwd=_COMMAND_CWD,
            timeout=timeout_s,
        )
        try:
            result = handle.wait()
        except Exception as exc:
            # The SDK raises CommandExitException for a command that ran and
            # exited non-zero. It is told apart by its exit code rather than by
            # class so that this module keeps importing, and the tests keep
            # driving it, without the SDK installed.
            exit_code = getattr(exc, "exit_code", None)
            if exit_code is not None:
                return _command_result(
                    getattr(exc, "stdout", ""), getattr(exc, "stderr", ""), exit_code
                )
            # No exit status: the command timed out, or the connection to it
            # broke. Ask E2B to stop it, but a kill signals the command's own
            # process, so a child of it can stay alive and still write to the
            # sandbox. Nothing here can call the command finished, so the
            # failure is raised rather than counted as a verify result.
            with contextlib.suppress(Exception):
                handle.kill()
            raise
        return _command_result(result.stdout, result.stderr, result.exit_code)

    def read_file(
        self, file_path: str, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        code = (
            "from pathlib import Path\n"
            "import json\n"
            f"_p = Path({file_path!r})\n"
            "if not _p.exists():\n"
            "    print(json.dumps({'ok': False, 'error': 'file not found'}))\n"
            "else:\n"
            "    _t = _p.read_text(encoding='utf-8')\n"
            f"    _o = {offset if offset is not None else 'None'}\n"
            f"    _l = {limit if limit is not None else 'None'}\n"
            "    if _o is not None:\n"
            "        _t = _t[_o:]\n"
            "    if _l is not None:\n"
            "        _t = _t[:_l]\n"
            "    print(json.dumps({'ok': True, 'content': _t}))\n"
        )
        execution = self._sbx.run_code(code)
        result = _decode_json_line(execution.logs.stdout)
        if result is None:
            return ToolResult(ok=False, error=_format_error(execution))
        if not result.get("ok", False):
            return ToolResult(ok=False, error=str(result.get("error", "read failed")))
        return ToolResult(ok=True, output=str(result.get("content", "")))

    def write_file(self, file_path: str, content: str) -> ToolResult:
        try:
            self._sbx.files.write(file_path, content.encode("utf-8"))
            return ToolResult(ok=True, output="write ok", metadata={"bytes": len(content.encode("utf-8"))})
        except Exception as exc:
            return ToolResult(ok=False, error=f"write failed: {exc}")

    def glob_files(self, pattern: str, path: str | None = None) -> ToolResult:
        code = (
            "from pathlib import Path\n"
            "import json\n"
            f"_root = Path({(path or '.')!r})\n"
            f"_matches = sorted(str(p) for p in _root.glob({pattern!r}))\n"
            "print(json.dumps({'ok': True, 'matches': _matches}))\n"
        )
        execution = self._sbx.run_code(code)
        result = _decode_json_line(execution.logs.stdout)
        if result is None:
            return ToolResult(ok=False, error=_format_error(execution))
        matches = result.get("matches", [])
        return ToolResult(ok=True, output="\n".join(matches), metadata={"matches": matches})

    def list_dir(self, path: str = ".", ignore: list[str] | None = None) -> ToolResult:
        ignore = ignore or []
        code = (
            "from pathlib import Path\n"
            "import json\n"
            f"_ignore = set({ignore!r})\n"
            f"_p = Path({path!r})\n"
            "if not _p.exists():\n"
            "    print(json.dumps({'ok': False, 'error': 'path not found'}))\n"
            "else:\n"
            "    _items = []\n"
            "    for _x in sorted(_p.iterdir()):\n"
            "        if _x.name in _ignore:\n"
            "            continue\n"
            "        _items.append({'name': _x.name, 'is_dir': _x.is_dir()})\n"
            "    print(json.dumps({'ok': True, 'items': _items}))\n"
        )
        execution = self._sbx.run_code(code)
        result = _decode_json_line(execution.logs.stdout)
        if result is None:
            return ToolResult(ok=False, error=_format_error(execution))
        if not result.get("ok", False):
            return ToolResult(ok=False, error=str(result.get("error", "ls failed")))
        items = result.get("items", [])
        lines = [f"{'[dir]' if item['is_dir'] else '[file]'} {item['name']}" for item in items]
        return ToolResult(ok=True, output="\n".join(lines), metadata={"items": items})

    def grep(self, pattern: str, path: str | None = None, include: str | None = None) -> ToolResult:
        root = path or "."
        code = (
            "from pathlib import Path\n"
            "import fnmatch, json, re\n"
            f"_root = Path({root!r})\n"
            f"_pat = re.compile({pattern!r})\n"
            f"_include = {include!r}\n"
            "_out = []\n"
            "if not _root.exists():\n"
            "    print(json.dumps({'ok': False, 'error': 'path not found'}))\n"
            "else:\n"
            "    for _p in _root.rglob('*'):\n"
            "        if not _p.is_file():\n"
            "            continue\n"
            "        if _include and not fnmatch.fnmatch(_p.name, _include):\n"
            "            continue\n"
            "        try:\n"
            "            _lines = _p.read_text(encoding='utf-8').splitlines()\n"
            "        except Exception:\n"
            "            continue\n"
            "        for _i, _line in enumerate(_lines, start=1):\n"
            "            if _pat.search(_line):\n"
            "                _out.append(f'{_p}:{_i}:{_line}')\n"
            "    print(json.dumps({'ok': True, 'matches': _out}))\n"
        )
        execution = self._sbx.run_code(code)
        result = _decode_json_line(execution.logs.stdout)
        if result is None:
            return ToolResult(ok=False, error=_format_error(execution))
        if not result.get("ok", False):
            return ToolResult(ok=False, error=str(result.get("error", "grep failed")))
        matches = result.get("matches", [])
        return ToolResult(ok=True, output="\n".join(matches), metadata={"matches": matches})

    def kill(self) -> None:
        try:
            self._sbx.kill()
        except Exception:
            try:
                self._sbx.close()
            except Exception:
                pass


def _decode_json_line(lines: list[str] | None) -> dict[str, Any] | None:
    if not lines:
        return None
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def _format_error(execution: Any) -> str:
    if execution.error:
        return f"{execution.error.name}: {execution.error.value}"
    stderr = "\n".join(execution.logs.stderr or [])
    return stderr or "sandbox execution failed"
