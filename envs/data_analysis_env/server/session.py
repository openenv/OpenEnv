# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A notebook-like Python session: real CPython, one persistent namespace, captured output.

Runs inside a sandbox worker process (see sandbox.py), one session per worker. Model code runs
with `exec`, so any valid Python and pandas works (`df[['a', 'b']]`,
`s.str.split(',', expand=True)[0]`, ...), unlike an AST interpreter:

- one globals dict per session, so variables, imports and functions persist across calls;
- stdout and stderr captured, up to MAX_OUTPUT_CHARS per stream and call;
- if the last statement is an expression, its repr is printed, as in a notebook (not `None`);
- `exit_code` 1 only when the code raised (`SystemExit` included, the session goes on). The error
  is `ExceptionType: message`, after a short traceback of the model's own lines, never of
  library frames;
- warnings (pandas, numpy) are returned in stderr as `Category: message` lines and are not errors.
"""

from __future__ import annotations

import ast
import builtins
import io
import linecache
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Tuple

MAX_OUTPUT_CHARS = 20_000
MAX_WARNINGS = 5
MAX_WARNING_CHARS = 300
MAX_TRACEBACK_FRAMES = 4
MAX_ERROR_CHARS = 2_000


class _CappedWriter(io.TextIOBase):
    """A text stream that keeps the first `limit` characters and drops the rest."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, limit: int = MAX_OUTPUT_CHARS):
        self._parts: list[str] = []
        self._kept = 0
        self._limit = limit
        self.truncated = False

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s) -> int:
        s = str(s)
        room = self._limit - self._kept
        if room > 0:
            self._parts.append(s[:room])
            self._kept += min(len(s), room)
        if len(s) > max(room, 0):
            self.truncated = True
        return len(s)

    def getvalue(self) -> str:
        text = "".join(self._parts)
        return text + "\n... (output truncated)" if self.truncated else text


def _format_warnings(caught) -> str:
    """Warnings raised by the call, one line each, deduplicated and capped."""
    lines: list[str] = []
    for w in caught:
        line = f"{getattr(w.category, '__name__', 'Warning')}: {w.message}"[
            :MAX_WARNING_CHARS
        ]
        if line not in lines:
            lines.append(line)
        if len(lines) >= MAX_WARNINGS:
            break
    return "\n".join(lines)


class PythonSession:
    """Every call runs in the same globals dict, like the cells of a notebook."""

    def __init__(self):
        self.namespace: dict = {"__name__": "__main__", "__builtins__": builtins}
        self._cells = 0
        self._current = ""

    def _register(self, code: str) -> str:
        """Name the cell and register its source, so tracebacks show the model's lines."""
        self._cells += 1
        filename = f"<cell {self._cells}>"
        linecache.cache[filename] = (
            len(code),
            None,
            code.splitlines(keepends=True),
            filename,
        )
        self._current = filename
        return filename

    def _format_error(self, exc: BaseException) -> str:
        if isinstance(exc, SyntaxError) and (exc.filename or "").startswith("<cell "):
            where = f" (line {exc.lineno})" if exc.lineno else ""
            text = f"\n    {exc.text.strip()}" if exc.text and exc.text.strip() else ""
            return f"{type(exc).__name__}: {exc.msg}{where}{text}"[:MAX_ERROR_CHARS]
        frames = [
            f
            for f in traceback.extract_tb(exc.__traceback__)
            if f.filename.startswith("<cell ")
        ]
        lines = []
        if frames:
            lines.append("Traceback (your code only, most recent call last):")
            for f in frames[-MAX_TRACEBACK_FRAMES:]:
                if f.filename == self._current:
                    where = f"line {f.lineno}"
                else:
                    where = f"{f.filename[1:-1]}, line {f.lineno}"
                if f.name != "<module>":
                    where += f", in {f.name}"
                lines.append(f"  {where}: {(f.line or '').strip()}"[:300])
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        lines.append(message[:MAX_ERROR_CHARS])
        return "\n".join(lines)

    def run(self, code: str) -> Tuple[str, str, int]:
        """Execute one call. Returns (stdout, stderr, exit_code)."""
        out, err = _CappedWriter(), _CappedWriter()
        error = ""
        filename = self._register(code)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ResourceWarning)  # unclosed files: noise
            with redirect_stdout(out), redirect_stderr(err):
                try:
                    tree = ast.parse(code, filename, "exec")
                    last = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last = ast.Expression(tree.body.pop().value)
                    exec(compile(tree, filename, "exec"), self.namespace)
                    if last is not None:
                        value = eval(compile(last, filename, "eval"), self.namespace)
                        if value is not None:
                            self.namespace["_"] = value
                            print(repr(value))
                except BaseException as e:  # SystemExit too: the session goes on
                    error = self._format_error(e)
        parts = (err.getvalue().rstrip("\n"), _format_warnings(caught), error)
        stderr = "\n".join(part for part in parts if part)
        return out.getvalue(), stderr, 1 if error else 0
