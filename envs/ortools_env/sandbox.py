# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run model-written Python in a subprocess with a wall-clock limit, a memory limit and blocked imports.

Keeping `ortools` and the environment's own packages out of reach is what keeps the reward honest: otherwise a policy
can call the reference solver. In the container (see `server/Dockerfile`), `SANDBOX_USER` is set and the code runs as
that unprivileged user with `SANDBOX_PYTHON`, an interpreter without `ortools`, while the environment source and the
server's packages are not readable by it. With `allow_ortools=True` it runs with `SANDBOX_ORTOOLS_PYTHON` instead,
which only the `ortools` group can read. Without `SANDBOX_USER` (local development), the code runs with the server's
interpreter and an import blocker that it can remove. Container UID/interpreter separation is necessary but does not
provide full containment: detached children escape process-group cleanup. See SECURITY.md for deployment holds.
"""

import json
import os
import pwd
import resource
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time

BLOCKED_ALWAYS = ("ortools_env", "problems", "server", "sandbox")
SANDBOX_USER = os.environ.get("SANDBOX_USER")
SANDBOX_PYTHON = os.environ.get("SANDBOX_PYTHON", sys.executable)
SANDBOX_ORTOOLS_PYTHON = os.environ.get("SANDBOX_ORTOOLS_PYTHON", sys.executable)

_RUNNER = """\
import importlib.abc, json, sys

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"import of {name!r} is not allowed in this sandbox")

BLOCKED = set(json.loads(sys.argv[1]))
sys.meta_path.insert(0, _Blocker())
payload = json.load(sys.stdin)
instance = payload["instance"]
# A submitted `solve` is imported like a module, so its `if __name__ == "__main__":` tests do not run.
namespace = {"__name__": "__main__" if sys.argv[3] == "run" else "solution", "instance": instance}
exec(compile(payload["code"], "<submission>", "exec"), namespace)
if sys.argv[3] == "solve":
    if "solve" not in namespace:
        raise NameError("the code must define solve(instance)")
    solution = namespace["solve"](instance)
    if not isinstance(solution, dict):
        raise TypeError(f"solve(instance) must return a dict, not {type(solution).__name__}")
    with open("solution.json", "w") as f:
        json.dump(solution, f, default=lambda o: o.tolist())
"""


def _limit_memory(memory_mb: int):
    def apply():
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply


def run(
    code: str,
    instance: dict,
    workdir: str,
    timeout_s: float,
    allow_ortools: bool = False,
    solve: bool = False,
    memory_mb: int = 4096,
) -> dict:
    """
    Run `code` with `instance` bound as a global, from `workdir` (where files persist between calls).

    Args:
        code (`str`):
            Python source to run.
        instance (`dict`):
            Instance bound to the global `instance` and written to `instance.json`.
        workdir (`str`):
            Working directory of the subprocess.
        timeout_s (`float`):
            Wall-clock limit of the subprocess.
        allow_ortools (`bool`, *optional*, defaults to `False`):
            Whether `code` may import `ortools`.
        solve (`bool`, *optional*, defaults to `False`):
            Whether to call `solve(instance)`, which `code` must define, and return its result as the solution.
        memory_mb (`int`, *optional*, defaults to `4096`):
            Address-space limit of the subprocess.

    Returns:
        `dict` with keys `output` (combined stdout and stderr, truncated), `elapsed_s`, `timed_out`, `returncode` and,
        with `solve=True`, `solution` (`None` if the run failed).
    """
    directory_fd = os.open(workdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        blocked = list(BLOCKED_ALWAYS) + ([] if allow_ortools else ["ortools"])
        # Never open an existing sandbox-owned entry for a privileged write. Replace
        # it atomically, and pass the authoritative input through a private descriptor.
        try:
            instance_name = f".instance-{secrets.token_hex(16)}"
            fd = os.open(
                instance_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=directory_fd,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(instance, f)
                os.replace(
                    instance_name,
                    "instance.json",
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                try:
                    os.unlink(instance_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            try:
                os.unlink("solution.json", dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        except OSError:
            # Leave adversarial directories untouched; never traverse them to recover.
            result = {
                "output": "failed to prepare sandbox files",
                "elapsed_s": 0.0,
                "timed_out": False,
                "returncode": 1,
            }
            if solve:
                result["solution"] = None
            return result
        user = {}
        if SANDBOX_USER is not None:
            account = pwd.getpwnam(SANDBOX_USER)
            os.fchown(directory_fd, account.pw_uid, account.pw_gid)
            user = {
                "user": SANDBOX_USER,
                "group": SANDBOX_USER,
                "extra_groups": ["ortools"] if allow_ortools else [],
            }
        start = time.monotonic()
        # Output goes to a file, so processes the code leaves behind cannot hold the call open through a pipe.
        with (
            tempfile.TemporaryFile("w+", errors="replace") as log,
            tempfile.TemporaryFile("w+") as input_file,
        ):
            json.dump({"code": code, "instance": instance}, input_file)
            input_file.seek(0)
            proc = subprocess.Popen(
                [
                    SANDBOX_ORTOOLS_PYTHON if allow_ortools else SANDBOX_PYTHON,
                    "-I",
                    "-c",
                    _RUNNER,
                    json.dumps(blocked),
                    "<submission>",
                    "solve" if solve else "run",
                ],
                cwd=workdir,
                stdin=input_file,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": workdir},
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                # RLIMIT_AS cannot be lowered on macOS.
                preexec_fn=_limit_memory(memory_mb)
                if sys.platform == "linux"
                else None,
                **user,
            )
            try:
                returncode, timed_out = proc.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
            # This only kills descendants that remain in the original process group.
            # A child that calls setsid() escapes; see SECURITY.md for the deployment hold.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            log.seek(0)
            output = log.read(8001)
            if len(output) > 8000:
                log.seek(0, os.SEEK_END)
                end = log.tell()
                log.seek(max(0, end - 4000))
                output = output[:4000] + "\n... [truncated] ...\n" + log.read(4000)
        result = {
            "output": output,
            "elapsed_s": time.monotonic() - start,
            "timed_out": timed_out,
            "returncode": returncode,
        }
        if solve:
            result["solution"] = None
            if returncode == 0:
                try:
                    # O_NONBLOCK prevents a FIFO from hanging the privileged reader.
                    fd = os.open(
                        "solution.json",
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory_fd,
                    )
                    with os.fdopen(fd, "rb") as f:
                        info = os.fstat(f.fileno())
                        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                            raise ValueError(
                                "solution must be a regular file with one link"
                            )
                        data = f.read(1024 * 1024 + 1)
                    if len(data) > 1024 * 1024:
                        raise ValueError("solution exceeds 1 MiB")
                    solution = json.loads(data)
                    if not isinstance(solution, dict):
                        raise ValueError("solution must be a JSON object")
                    result["solution"] = solution
                except (OSError, ValueError, UnicodeError, RecursionError):
                    result["output"] += "\nfailed to read a valid solution.json"
        return result
    finally:
        os.close(directory_fd)
