# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2B sandbox wrapper with shell and file-operation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_E2B_IMPORT_ERROR: ImportError | None = None

try:
    from e2b_code_interpreter import Sandbox
except ImportError as _e2b_import_error:  # pragma: no cover
    _E2B_IMPORT_ERROR = _e2b_import_error
    Sandbox = None  # type: ignore[assignment]


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] | None = None


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
