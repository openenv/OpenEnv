# SPDX-License-Identifier: BSD-3-Clause

"""Drivers for the OCI runtime and the mounts openenvd prepares for containers."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REMOUNT = 32
_MS_NOSYMFOLLOW = 256
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 262144
_MNT_DETACH = 2


class RuntimeFailure(RuntimeError):
    """The OCI runtime or a mount failed."""


class ContainerRuntime(Protocol):
    """What [`Unit`] needs from an OCI runtime."""

    name: str

    async def run(self, container_id: str, bundle: Path) -> None: ...
    async def kill(self, container_id: str, sig: int = signal.SIGKILL) -> None: ...
    async def delete(self, container_id: str) -> None: ...
    async def wait(self, container_id: str) -> int: ...


async def _exec(*argv: str, check: bool = True) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace").strip()
    if check and proc.returncode != 0:
        raise RuntimeFailure(f"{Path(argv[0]).name} {argv[1]} failed: {text}")
    return proc.returncode or 0, text


@dataclass
class OciRuntime:
    """Runs containers with `crun`, `runc` or `runsc`.

    Containers run detached, so their lifetime is not tied to openenvd's event
    loop, and openenvd waits on them through `state`.

    Args:
        binary (`str`):
            Path to the runtime binary.
        state_root (`Path`):
            The runtime's `--root` directory, private to openenvd.
        log_dir (`Path`):
            Where each container's stdout and stderr go.
    """

    binary: str
    state_root: Path
    log_dir: Path
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = Path(self.binary).name
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _base(self) -> list[str]:
        return [self.binary, "--root", str(self.state_root)]

    async def run(self, container_id: str, bundle: Path) -> None:
        log = self.log_dir / f"{container_id}.log"
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._base(),
                "run",
                "--detach",
                "--bundle",
                str(bundle),
                container_id,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=fd,
                stderr=fd,
            )
            code = await proc.wait()
        finally:
            os.close(fd)
        if code != 0:
            tail = log.read_text(errors="replace")[-2000:]
            raise RuntimeFailure(f"{self.name} run {container_id} failed: {tail}")

    async def state(self, container_id: str) -> dict | None:
        code, out = await _exec(*self._base(), "state", container_id, check=False)
        if code != 0:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return None

    async def kill(self, container_id: str, sig: int = signal.SIGKILL) -> None:
        await _exec(
            *self._base(), "kill", "--all", container_id, str(int(sig)), check=False
        )

    async def delete(self, container_id: str) -> None:
        await _exec(*self._base(), "delete", "--force", container_id, check=False)

    async def wait(self, container_id: str) -> int:
        while True:
            state = await self.state(container_id)
            if state is None or state.get("status") == "stopped":
                return 0
            await asyncio.sleep(0.05)

    async def list_ids(self) -> list[str]:
        code, out = await _exec(*self._base(), "list", "--format", "json", check=False)
        if code != 0 or not out:
            return []
        try:
            return [c["id"] for c in json.loads(out) or []]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []


def find_runtime(preferred: str = "crun") -> str | None:
    """The path of `preferred`, else of any supported runtime, else `None`."""
    for name in (preferred, "crun", "runc", "runsc"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _mount(
    source: str, target: str, fstype: str | None, flags: int, data: str | None
) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeFailure("mounts need Linux")
    libc = _libc()
    rc = libc.mount(
        source.encode(),
        target.encode(),
        fstype.encode() if fstype else None,
        ctypes.c_ulong(flags),
        data.encode() if data else None,
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise RuntimeFailure(f"mount {target}: {os.strerror(err)}")


def mount_overlay(lower: Path, upper: Path, work: Path, target: Path) -> None:
    """Mount `overlay(lower, upper)` at `target`.

    The lower layer is the read-only seed; the upper layer holds exactly what
    the episode changed, so it doubles as the file diff.
    """
    for d in (upper, work, target):
        d.mkdir(parents=True, exist_ok=True)
    options = f"lowerdir={lower},upperdir={upper},workdir={work}"
    _mount("overlay", str(target), "overlay", 0, options)


def bind_mount(
    source: Path,
    target: Path,
    *,
    read_only: bool = False,
    nosymfollow: bool = False,
    nodev: bool = True,
    nosuid: bool = True,
    noexec: bool = False,
) -> None:
    """Bind `source` at `target`, then remount with the requested flags."""
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
    _mount(str(source), str(target), None, _MS_BIND | _MS_REC, None)
    # A bind of a shared mount (systemd makes `/` shared) joins its peer group,
    # so unmounting the bind would also unmount the originals. Leave the group.
    _mount("none", str(target), None, _MS_REC | _MS_PRIVATE, None)
    flags = _MS_BIND | _MS_REMOUNT
    flags |= _MS_RDONLY if read_only else 0
    flags |= _MS_NOSYMFOLLOW if nosymfollow else 0
    flags |= _MS_NODEV if nodev else 0
    flags |= _MS_NOSUID if nosuid else 0
    flags |= _MS_NOEXEC if noexec else 0
    _mount("none", str(target), None, flags, None)


def unmount(target: Path) -> None:
    """Lazily detach `target`. Missing or unmounted targets are ignored."""
    if not sys.platform.startswith("linux") or not target.exists():
        return
    _libc().umount2(str(target).encode(), _MNT_DETACH)
