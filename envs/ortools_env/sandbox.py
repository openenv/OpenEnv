# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Run model-written Python in a subprocess with a wall-clock limit, a memory limit and blocked imports.

Keeping `ortools` and the environment's own packages out of reach is what keeps the reward honest: otherwise a policy
can call the reference solver. In the container (see `server/Dockerfile`), `SANDBOX_USER` is set and the code runs as
an unprivileged uid of its own with `SANDBOX_PYTHON`, an interpreter without `ortools`, while the environment source
and the server's packages are not readable by it. Each session gets its own uid (`new_uid`), so sessions can't signal
each other or read each other's files, and every process of that uid is killed when a call ends, including children
that left the process group. With `allow_ortools=True` it runs with `SANDBOX_ORTOOLS_PYTHON` instead,
which only the `ortools` group can read. Without `SANDBOX_USER` (local development), the code runs with the server's
interpreter and an import blocker that it can remove. See SECURITY.md for what is not contained.
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
import threading
import time

BLOCKED_ALWAYS = ("ortools_env", "problems", "server", "sandbox")
SANDBOX_USER = os.environ.get("SANDBOX_USER")
SANDBOX_PYTHON = os.environ.get("SANDBOX_PYTHON", sys.executable)
SANDBOX_ORTOOLS_PYTHON = os.environ.get("SANDBOX_ORTOOLS_PYTHON", sys.executable)
# One uid per session, with no account behind it. Below 65534 (nobody): containers on
# Hugging Face map only uids 0-65535.
_free_uids = list(range(60_000, 65_000))
_uids_lock = threading.Lock()
# Processes and threads per session uid (numpy's OpenBLAS starts one thread per core).
MAX_PROCESSES = 256

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


def _limit_resources(memory_mb: int, max_processes: int | None):
    def apply():
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        # RLIMIT_NPROC counts every process of the uid, so only for a session's own uid.
        if max_processes is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))

    return apply


def new_uid() -> int | None:
    """A uid for one session's code, used by no other session, or `None` without `SANDBOX_USER`.

    Give it back with `release_uid` when the session ends.
    """
    if SANDBOX_USER is None:
        return None
    with _uids_lock:
        if not _free_uids:
            raise RuntimeError("no free sandbox uid: too many sessions")
        return _free_uids.pop()


def release_uid(uid: int | None) -> None:
    """Stop every process of `uid` and make it available to another session."""
    if uid is None:
        return
    _kill_all(uid)
    with _uids_lock:
        _free_uids.append(uid)


def _processes_of(uid: int) -> list[tuple[int, int, str]]:
    """`(pid, parent pid, state)` of every process of `uid`."""
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                fields = dict(line.split(":", 1) for line in f if ":" in line)
        except OSError:
            continue
        if int(fields["Uid"].split()[0]) == uid:
            found.append((int(pid), int(fields["PPid"]), fields["State"].split()[0]))
    return found


def _kill_all(uid: int) -> None:
    """Stop every process of `uid`, from this (root) process, so the code can't interfere."""
    deadline = time.monotonic() + 10
    # Freeze first: a stopped process can't start more processes or send signals.
    while time.monotonic() < deadline:
        running = [pid for pid, _, state in _processes_of(uid) if state not in "TZX"]
        if not running:
            break
        for pid in running:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
    for pid, _, _ in _processes_of(uid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Orphans of the session were reparented to this process (a subreaper, see `run`).
    # Reap them, or their zombies would still count against the uid's process limit.
    while (processes := _processes_of(uid)) and time.monotonic() < deadline:
        for pid, parent, _ in processes:
            if parent == os.getpid():
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        time.sleep(0.01)


def run(
    code: str,
    instance: dict,
    workdir: str,
    timeout_s: float,
    allow_ortools: bool = False,
    solve: bool = False,
    memory_mb: int = 4096,
    uid: int | None = None,
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
        uid (`int`, *optional*):
            The session's uid from `new_uid`, which owns `workdir`. Defaults to a uid for this call only.
            Ignored without `SANDBOX_USER`.

    Returns:
        `dict` with keys `output` (combined stdout and stderr, truncated), `elapsed_s`, `timed_out`, `returncode` and,
        with `solve=True`, `solution` (`None` if the run failed).
    """
    own_uid = None
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
            # Orphans of the code come back to this process instead of PID 1, so they can be reaped.
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(
                36, 1, 0, 0, 0
            )  # PR_SET_CHILD_SUBREAPER
            gid = pwd.getpwnam(SANDBOX_USER).pw_gid
            if uid is None:
                uid = own_uid = new_uid()
            os.fchown(directory_fd, uid, gid)
            user = {
                "user": uid,
                "group": gid,
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
                preexec_fn=_limit_resources(memory_mb, MAX_PROCESSES if user else None)
                if sys.platform == "linux"
                else None,
                **user,
            )
            try:
                returncode, timed_out = proc.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if user:
                # Also the children that left the process group (setsid).
                _kill_all(user["user"])
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
        release_uid(own_uid)
