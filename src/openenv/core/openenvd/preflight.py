# SPDX-License-Identifier: BSD-3-Clause

"""Can openenvd isolate here? One yes or no, with the reasons when it's no.

openenvd ships in every image. It isolates when it can create containers: it
runs as root, the kernel lets it create user, mount, PID and network
namespaces, the cgroup v2 root is writable, and an OCI runtime is installed.
That holds inside a microVM, or in a container run with `--privileged`. When
any check fails, openenvd runs the environment exactly as it ran before.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000
OCI_RUNTIMES: tuple[str, ...] = ("crun", "runc", "runsc")
_CHILD_TIMEOUT_S = 5.0


@dataclass
class Preflight:
    """Whether openenvd can isolate this unit.

    Attributes:
        enabled (`bool`):
            Every check passed.
        reasons (`list[str]`):
            One line per failed check, for the log.
        runtime (`str`, *optional*):
            Path of the OCI runtime found.
    """

    enabled: bool
    reasons: list[str] = field(default_factory=list)
    runtime: str | None = None


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _in_child(body: Callable[[], None]) -> str | None:
    """Run `body` in a forked child. `None` on success, else why it failed."""
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.close(read_fd)
            try:
                body()
                msg = ""
                status = 0
            except BaseException as exc:  # noqa: BLE001 - report, never propagate
                msg = str(exc) or type(exc).__name__
            os.write(write_fd, msg.encode()[:512])
        finally:
            os._exit(status)
    os.close(write_fd)
    try:
        deadline = time.monotonic() + _CHILD_TIMEOUT_S
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return "check timed out"
            time.sleep(0.005)
        msg = os.read(read_fd, 512).decode(errors="replace")
    finally:
        os.close(read_fd)
    if os.waitstatus_to_exitcode(status) == 0:
        return None
    return msg or "check failed"


def check_namespaces() -> str | None:
    """`None` if a child can create user, mount, PID and network namespaces."""

    def body() -> None:
        libc = _libc()
        for flags, what in (
            (CLONE_NEWUSER, "user"),
            (CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET, "mount/pid/net"),
        ):
            if libc.unshare(flags) != 0:
                err = ctypes.get_errno()
                raise OSError(
                    err, f"cannot create {what} namespaces: {os.strerror(err)}"
                )

    try:
        return _in_child(body)
    except OSError as exc:
        return f"cannot fork a check: {exc.strerror}"


def check_cgroups(root: Path) -> str | None:
    """`None` if `root` is a writable cgroup v2 hierarchy."""
    control = Path(root) / "cgroup.subtree_control"
    if not control.exists():
        return f"no cgroup v2 hierarchy at {root}"
    probe = Path(root) / ".openenvd-preflight"
    try:
        probe.mkdir(exist_ok=True)
    except OSError as exc:
        return f"cgroup root is not writable ({exc.strerror})"
    try:
        probe.rmdir()
    except OSError:
        pass
    if not os.access(control, os.W_OK):
        return "cgroup.subtree_control is read-only"
    return None


def find_runtime(preferred: str = "crun") -> str | None:
    """Path of `preferred`, else of any supported OCI runtime, else `None`."""
    for name in (preferred, *OCI_RUNTIMES):
        path = shutil.which(name)
        if path:
            return path
    return None


def preflight(
    cgroup_root: Path = Path("/sys/fs/cgroup"), oci: str = "crun"
) -> Preflight:
    """Decide whether openenvd can isolate this unit.

    Args:
        cgroup_root (`Path`, *optional*, defaults to `/sys/fs/cgroup`):
            The unit's cgroup root.
        oci (`str`, *optional*, defaults to `"crun"`):
            The preferred OCI runtime.

    Returns:
        [`Preflight`]: `enabled` when every check passed, otherwise the reasons.
    """
    if not sys.platform.startswith("linux"):
        return Preflight(False, ["not Linux: no namespaces or cgroups"])
    reasons = []
    if os.geteuid() != 0:
        reasons.append("not running as root (run the container with --privileged)")
    problem = check_namespaces()
    if problem:
        reasons.append(problem + " (run the container with --privileged)")
    problem = check_cgroups(cgroup_root)
    if problem:
        reasons.append(problem + " (run with --privileged --cgroupns=private)")
    runtime = find_runtime(oci)
    if runtime is None:
        reasons.append(f"no OCI runtime installed (install {oci})")
    return Preflight(not reasons, reasons, runtime)
