# SPDX-License-Identifier: BSD-3-Clause

"""The container shim: PID 1 of every openenvd container.

Run as `python -m openenv.core.openenvd.shim --spec <path>`. It starts the
socket forwarders, then `exec`s the principal in a forked child. It reaps every child, forwards `SIGTERM`/`SIGINT` to the
principal's process group and exits with the principal's status.

Stdlib only (plus `forwarder`), since it runs inside minimal
containers.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import fcntl
import json
import os
import signal
import socket
import struct
import sys
import threading
from typing import Any

from . import forwarder

PR_SET_CHILD_SUBREAPER = 36

SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
IFF_UP = 0x1


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _warn(message: str) -> None:
    print(f"openenvd-shim: {message}", file=sys.stderr, flush=True)


def shim_spec(
    argv: list[str],
    env: dict[str, str],
    forwards: list[dict],
    workdir: str | None,
    uid: int | None = None,
    gid: int | None = None,
) -> dict:
    """Build the JSON spec the shim reads with `--spec`.

    Args:
        argv (`list[str]`):
            The principal's command, resolved against `PATH` in `env`.
        env (`dict[str, str]`):
            The principal's full environment.
        forwards (`list[dict]`):
            Forwarders the shim runs for the container's lifetime. Each is one of
            `{"kind": "tcp_to_unix", "host", "port", "path"}`,
            `{"kind": "unix_to_tcp", "path", "host", "port"}` or
            `{"kind": "unix_to_unix", "path", "target"}`.
        workdir (`str`, *optional*):
            Working directory of the principal.
        uid (`int`, *optional*):
            Switch the principal to this UID before `exec`.
        gid (`int`, *optional*):
            Switch the principal to this GID before `exec`.

    Returns:
        `dict`: The JSON-serializable shim spec.

    Raises:
        `ValueError`: If `argv` is empty or a forward is malformed.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    required = {
        "tcp_to_unix": ("host", "port", "path"),
        "unix_to_tcp": ("path", "host", "port"),
        "unix_to_unix": ("path", "target"),
    }
    for fwd in forwards:
        kind = fwd.get("kind")
        if kind not in required:
            raise ValueError(f"unknown forward kind {kind!r}")
        missing = [k for k in required[kind] if k not in fwd]
        if missing:
            raise ValueError(f"{kind} forward is missing {missing}")
    return {
        "argv": list(argv),
        "env": dict(env),
        "forwards": [dict(f) for f in forwards],
        "workdir": workdir,
        "uid": uid,
        "gid": gid,
    }


def _set_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        _warn("could not become a child subreaper")


def _bring_up_lo() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            ifreq = struct.pack("16sh22x", b"lo", 0)
            flags = struct.unpack("16sh22x", fcntl.ioctl(s, SIOCGIFFLAGS, ifreq))[1]
            if not flags & IFF_UP:
                fcntl.ioctl(
                    s, SIOCSIFFLAGS, struct.pack("16sh22x", b"lo", flags | IFF_UP)
                )
    except OSError:
        pass


async def _start_forward(fwd: dict) -> asyncio.base_events.Server:
    kind = fwd["kind"]
    if kind == "tcp_to_unix":
        return await forwarder.serve_tcp_to_unix(fwd["host"], fwd["port"], fwd["path"])
    if kind == "unix_to_tcp":
        return await forwarder.serve_unix_to_tcp(fwd["path"], fwd["host"], fwd["port"])
    if kind == "unix_to_unix":
        return await forwarder.serve_unix_to_unix(fwd["path"], fwd["target"])
    raise ValueError(f"unknown forward kind {kind!r}")


def _start_forwarders(forwards: list[dict]) -> asyncio.AbstractEventLoop | None:
    if not forwards:
        return None
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    errors: list[BaseException] = []

    async def start() -> None:
        try:
            for fwd in forwards:
                await _start_forward(fwd)
        except BaseException as exc:  # reported to the main thread
            errors.append(exc)
        finally:
            ready.set()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.create_task(start())
        loop.run_forever()

    threading.Thread(target=run, name="openenvd-forwarders", daemon=True).start()
    ready.wait()
    if errors:
        raise errors[0]
    return loop


def _exec_principal(spec: dict, go_fd: int) -> None:
    """Child side of the fork. Never returns."""
    try:
        os.read(go_fd, 1)
        os.close(go_fd)
        os.setsid()
        if spec.get("gid") is not None:
            os.setgroups([])
            os.setgid(spec["gid"])
        if spec.get("uid") is not None:
            os.setuid(spec["uid"])
        if spec.get("workdir"):
            os.chdir(spec["workdir"])
        argv = spec["argv"]
        os.execvpe(argv[0], argv, spec.get("env") or {})
    except BaseException as exc:
        _warn(f"failed to start the principal: {exc}")
        os._exit(127)


def _exit_code(status: int) -> int:
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def run(spec: dict[str, Any]) -> int:
    """Run the shim with a parsed spec until the principal exits.

    Args:
        spec (`dict`):
            A spec built by [`shim_spec`].

    Returns:
        `int`: The principal's exit code, or `128 + signal` if it was killed.
    """
    if _is_linux():
        _set_subreaper()
        _bring_up_lo()
    else:
        _warn("not Linux, skipping: child subreaper")

    # Fork before starting the forwarder thread; the child waits for the go byte.
    go_read, go_write = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(go_write)
        _exec_principal(spec, go_read)
    os.close(go_read)

    def forward_signal(signum: int, _frame: Any) -> None:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    loop = None
    try:
        loop = _start_forwarders(spec.get("forwards") or [])
    except BaseException as exc:
        _warn(f"failed to start forwarders: {exc}")
        os.kill(pid, signal.SIGKILL)
        os.close(go_write)
        os.waitpid(pid, 0)
        return 126
    os.write(go_write, b"g")
    os.close(go_write)

    code = 1
    while True:
        try:
            reaped, status = os.wait()
        except ChildProcessError:
            break
        if reaped == pid:
            code = _exit_code(status)
            break
    # Collect zombies that already exited; PID 1 leaving tears down the rest.
    while True:
        try:
            reaped, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if reaped == 0:
            break
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    return code


def main(argv: list[str] | None = None) -> int:
    """Entry point: `python -m openenv.core.openenvd.shim --spec <path>`.

    Args:
        argv (`list[str]`, *optional*):
            Command-line arguments. Defaults to `sys.argv[1:]`.

    Returns:
        `int`: The principal's exit code.
    """
    parser = argparse.ArgumentParser(prog="openenvd-shim")
    parser.add_argument("--spec", required=True, help="path to the shim JSON spec")
    args = parser.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    return run(spec)


if __name__ == "__main__":
    sys.exit(main())
