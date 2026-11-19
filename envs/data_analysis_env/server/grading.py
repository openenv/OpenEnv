# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grade a submitted answer against a task's gold answer, inside the server.

The gold answer never leaves the server process: not in the prompt, the observations, the state
or the logs. The candidate is model output, so it is treated as untrusted text: length-capped and
only ever compared by the dataset's grader (grader.py), never executed.

math-verify, the grader's last tier, times out with signal.alarm, which only works in a process's
main thread, and anywhere else that tier would quietly fail. Tools run in worker threads, so grading
happens in a small helper process that runs the grader in its main thread, with an allowlisted
environment. A grade that takes longer than GRADE_TIMEOUT_S kills the helper and scores 0.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import sys
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAX_CANDIDATE_CHARS = 1000
GRADE_TIMEOUT_S = 30.0
GRADER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")

_GRADER_MODULE = __name__.rsplit(".", 1)[0] + ".grader"
_BOOT = f"from {__name__} import _grader_main\n_grader_main()\n"


def _tolerance(value) -> float:
    try:
        number = 1e-3 if value is None else float(value)
    except (TypeError, ValueError):
        return 1e-3
    return number if number == number else 1e-3  # NaN from the dataset: default


def _grader_main() -> None:
    """Helper process: one JSON request per line on stdin, one JSON reply per line on stdout."""
    from importlib import import_module

    grader = import_module(_GRADER_MODULE)
    for line in sys.stdin:
        try:
            r = json.loads(line)
            result = grader.grade(
                r["gold"],
                r["candidate"],
                reward_mode=r["reward_mode"],
                abs_tol=r["atol"],
                rel_tol=r["rtol"],
            )
            reply = {"reward": float(result.reward), "method": str(result.method)}
        except Exception as e:  # a grader bug must not take the helper down
            reply = {"reward": 0.0, "method": f"grader_error: {type(e).__name__}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


class _Grader:
    """The helper process, started on first use and restarted after a timeout or a crash."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    def _start(self) -> subprocess.Popen:
        env = {k: os.environ[k] for k in GRADER_ENV_ALLOWLIST if k in os.environ}
        env["PYTHONPATH"] = os.pathsep.join(
            os.path.abspath(p) for p in sys.path if p and os.path.isdir(p)
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.Popen(
            [sys.executable, "-c", _BOOT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd="/",
            text=True,
        )

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def grade(self, request: dict) -> Tuple[float, str]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._start()
            proc = self._proc
            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                with selectors.DefaultSelector() as sel:
                    sel.register(proc.stdout, selectors.EVENT_READ)
                    ready = sel.select(GRADE_TIMEOUT_S)
                if not ready:
                    logger.warning("data_analysis_env grader timed out")
                    self._stop()
                    return 0.0, "grader_timeout"
                reply = json.loads(proc.stdout.readline())
                reward, method = float(reply["reward"]), str(reply["method"])
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("data_analysis_env grader failed", exc_info=True)
                self._stop()
                return 0.0, "grader_error"
        return (1.0 if reward >= 1.0 else 0.0), method[:100]

    def close(self) -> None:
        with self._lock:
            self._stop()


_grader = _Grader()


def grade_answer(task: dict, candidate: Optional[str]) -> Tuple[float, str]:
    """Score a submission for a SmolDataEnvs row. Returns (reward, method).

    Uses the row's `reward_mode`, `atol` and `rtol`, as the dataset card does.
    """
    if candidate is None:
        return 0.0, "no_submission"
    candidate = str(candidate).strip()
    if not candidate:
        return 0.0, "empty"
    if len(candidate) > MAX_CANDIDATE_CHARS:
        return 0.0, "too_long"
    return _grader.grade(
        {
            "gold": str(task["answer"]),
            "candidate": candidate,
            "reward_mode": str(task.get("reward_mode") or ""),
            "atol": _tolerance(task.get("atol")),
            "rtol": _tolerance(task.get("rtol")),
        }
    )
