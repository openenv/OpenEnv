# SPDX-License-Identifier: BSD-3-Clause

"""Runs chart code in a separate, short-lived process.

The process starts from an environment allowlist, never a copy of
`os.environ`, in an empty temporary directory, with a timeout and (on POSIX) a
memory cap. The task's spec and target image stay in the server process.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).with_name("runner.py")
WORKER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
MEMORY_LIMIT_BYTES = 2 * 1024**3


def _limit_memory() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def run_code(code: str, dpi: int, png: bool, timeout_s: float) -> dict[str, Any]:
    """Run `code` and summarize the figure it leaves open.

    Args:
        code (`str`):
            Python source that draws with matplotlib.
        dpi (`int`):
            Resolution of the returned PNG.
        png (`bool`):
            Whether to return the drawing as a base64 PNG.
        timeout_s (`float`):
            Seconds before the process is killed.

    Returns:
        `dict` with keys:
            - `error` (`str`): The traceback, or why the run was stopped; empty
              on success.
            - `figure` (`dict` or `None`): The figure's axes and artists, read
              by [`~envs.chart2code_env.server.scoring.score_figure`].
            - `png` (`str` or `None`): The drawing, when `png` was asked for.
    """
    nonce = secrets.token_hex(16)
    request = json.dumps({"code": code, "dpi": dpi, "png": png, "nonce": nonce})
    with tempfile.TemporaryDirectory(prefix="chart2code-") as workdir:
        env = {k: os.environ[k] for k in WORKER_ENV_ALLOWLIST if k in os.environ}
        env.update(HOME=workdir, MPLCONFIGDIR=workdir, MPLBACKEND="Agg")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(RUNNER)],
                input=request,
                capture_output=True,
                text=True,
                cwd=workdir,
                env=env,
                timeout=timeout_s,
                preexec_fn=_limit_memory if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {
                "error": f"the code ran past {timeout_s:g} s",
                "figure": None,
                "png": None,
            }
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(nonce):
            return json.loads(line[len(nonce) :])
    stderr = completed.stderr.strip().splitlines()
    return {
        "error": "the code stopped the process" + (f": {stderr[-1]}" if stderr else ""),
        "figure": None,
        "png": None,
    }
