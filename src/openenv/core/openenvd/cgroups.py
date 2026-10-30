# SPDX-License-Identifier: BSD-3-Clause

"""The unit's cgroup v2 tree.

openenvd owns the root of the unit's cgroup namespace. It moves itself into a
`control/` leaf (cgroup v2 forbids processes in a group whose controllers are
delegated to children), then builds `zones/agent/{env,harness}`,
`zones/services/<name>` and `zones/observers/<name>` under it. Freezing, killing
and "is it empty yet" all act on a whole subtree, so nothing an agent forks can
slip out of a reset.
"""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .contract import Resources

CONTROLLERS: tuple[str, ...] = ("cpu", "memory", "pids")
CPU_PERIOD_US = 100_000

_POLL_S = 0.02
_KILL_PASSES = 50


class CgroupError(RuntimeError):
    """A cgroup operation the kernel refused."""


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _write(path: Path, value: str) -> None:
    # One open per value: cgroupfs takes one pid per write(2). Mode "w" also
    # works on a regular-file fake.
    try:
        with open(path, "w") as f:
            f.write(value)
    except OSError as exc:
        raise CgroupError(f"write {value!r} to {path}: {exc.strerror}") from exc


def _keyed(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (text or "").splitlines():
        key, _, value = line.partition(" ")
        if value.strip().lstrip("-").isdigit():
            out[key] = int(value)
    return out


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None


class CgroupTree:
    """Create, limit, freeze, kill and remove groups under a cgroup v2 root.

    Args:
        root (`Path`, *optional*, defaults to `Path("/sys/fs/cgroup")`):
            The root of the unit's cgroup namespace. Every `rel` path is
            relative to it.
    """

    def __init__(self, root: Path = Path("/sys/fs/cgroup")):
        self.root = Path(root)

    def path(self, rel: str) -> Path:
        """Absolute path of `rel`. Refuses paths that would leave the root.

        Args:
            rel (`str`):
                Relative cgroup path, e.g. `zones/agent/env`. Empty means the root.

        Returns:
            `Path`: The cgroup directory.
        """
        rel = rel.strip("/")
        parts = PurePosixPath(rel).parts if rel else ()
        if any(p in ("..", ".") for p in parts):
            raise CgroupError(f"cgroup path {rel!r} must stay under the root")
        return self.root.joinpath(*parts)

    def _enable(self, group: Path) -> None:
        available = set((_read(group / "cgroup.controllers") or "").split())
        wanted = [c for c in CONTROLLERS if c in available]
        enabled = {
            c.lstrip("+")
            for c in (_read(group / "cgroup.subtree_control") or "").split()
        }
        if any(c not in enabled for c in wanted):
            _write(
                group / "cgroup.subtree_control",
                " ".join(f"+{c}" for c in wanted),
            )

    def adopt_self(self, leaf: str = "control") -> None:
        """Move every process in the root into `leaf`, then enable controllers.

        Args:
            leaf (`str`, *optional*, defaults to `"control"`):
                The leaf group that holds openenvd itself.
        """
        self.path(leaf).mkdir(parents=True, exist_ok=True)
        for _ in range(10):  # a process may fork while we move its siblings
            procs = self._procs(self.root)
            if not procs:
                break
            for pid in procs:
                try:
                    self.move(pid, leaf)
                except CgroupError:
                    pass  # exited meanwhile, or a kernel thread
        self._enable(self.root)

    def create(self, rel: str) -> Path:
        """Create `rel` and every missing ancestor, delegating controllers down.

        Args:
            rel (`str`):
                Relative cgroup path.

        Returns:
            `Path`: The created (or existing) cgroup directory.
        """
        target = self.path(rel)
        group = self.root
        for part in target.relative_to(self.root).parts:
            self._enable(group)
            group = group / part
            try:
                group.mkdir(exist_ok=True)
            except OSError as exc:
                raise CgroupError(f"mkdir {group}: {exc.strerror}") from exc
        return target

    def set_limits(self, rel: str, resources: Resources) -> None:
        """Write `memory.max`, `memory.swap.max`, `pids.max` and `cpu.max`.

        Args:
            rel (`str`):
                Relative cgroup path.
            resources ([`Resources`]):
                Limits to apply. `None` fields are written as `max`.
        """
        group = self.path(rel)
        mem = resources.memory_mb
        _write(group / "memory.max", "max" if mem is None else str(mem * 1024 * 1024))
        if (group / "memory.swap.max").exists():
            _write(group / "memory.swap.max", "0")
        pids = resources.pids
        _write(group / "pids.max", "max" if pids is None else str(pids))
        cpu = resources.cpu
        # The kernel rejects quotas under 1ms.
        quota = "max" if cpu is None else str(max(1000, int(cpu * CPU_PERIOD_US)))
        _write(group / "cpu.max", f"{quota} {CPU_PERIOD_US}")

    def move(self, pid: int, rel: str) -> None:
        """Move `pid` into `rel`.

        Args:
            pid (`int`):
                Process to move.
            rel (`str`):
                Relative cgroup path.
        """
        _write(self.path(rel) / "cgroup.procs", str(pid))

    @staticmethod
    def _procs(group: Path) -> list[int]:
        return [int(t) for t in (_read(group / "cgroup.procs") or "").split()]

    def pids(self, rel: str) -> list[int]:
        """Every pid in `rel` and its descendants.

        Args:
            rel (`str`):
                Relative cgroup path.

        Returns:
            `list[int]`: Pids, empty if the group doesn't exist.
        """
        top = self.path(rel)
        if not top.is_dir():
            return []
        out: list[int] = []
        for dirpath, _, _ in os.walk(top):
            out.extend(self._procs(Path(dirpath)))
        return out

    def freeze(self, rel: str) -> None:
        """Ask the kernel to freeze every process under `rel`."""
        _write(self.path(rel) / "cgroup.freeze", "1")

    def thaw(self, rel: str) -> None:
        """Unfreeze every process under `rel`."""
        _write(self.path(rel) / "cgroup.freeze", "0")

    def _events(self, rel: str) -> dict[str, int] | None:
        text = _read(self.path(rel) / "cgroup.events")
        return None if text is None else _keyed(text)

    async def wait_frozen(self, rel: str, timeout: float) -> bool:
        """Wait until `cgroup.events` reports `frozen 1`.

        Args:
            rel (`str`):
                Relative cgroup path.
            timeout (`float`):
                Seconds to wait.

        Returns:
            `bool`: Whether the group froze in time.
        """
        deadline = time.monotonic() + timeout
        while True:
            events = self._events(rel)
            if events is not None and events.get("frozen") == 1:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_POLL_S)

    def kill(self, rel: str) -> bool:
        """SIGKILL everything under `rel`.

        Uses `cgroup.kill` when the kernel has it (5.14+): atomic, so a fork bomb
        can't outrun it. Otherwise SIGKILLs every pid repeatedly until a pass
        finds none.

        Args:
            rel (`str`):
                Relative cgroup path. A missing group is already empty.

        Returns:
            `bool`: Whether the kill was atomic (`cgroup.kill`, or nothing to kill).
        """
        group = self.path(rel)
        if not group.is_dir():
            return True
        if (group / "cgroup.kill").exists():
            _write(group / "cgroup.kill", "1")
            return True
        for _ in range(_KILL_PASSES):
            pids = self.pids(rel)
            if not pids:
                break
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.005)
        return False

    async def wait_empty(self, rel: str, timeout: float) -> bool:
        """Wait until `cgroup.events` reports `populated 0`.

        Args:
            rel (`str`):
                Relative cgroup path.
            timeout (`float`):
                Seconds to wait.

        Returns:
            `bool`: Whether the group emptied in time. A missing group is empty.
        """
        deadline = time.monotonic() + timeout
        while True:
            if not self.path(rel).is_dir():
                return True
            events = self._events(rel)
            if events is None:
                if not self.pids(rel):
                    return True
            elif events.get("populated") == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_POLL_S)

    def stats(self, rel: str) -> dict[str, Any]:
        """Current usage of `rel`. Missing files read as `None` or `{}`.

        Args:
            rel (`str`):
                Relative cgroup path.

        Returns:
            `dict` with keys:
                - `memory_current` (`int` or `None`)
                - `memory_events` (`dict[str, int]`)
                - `pids_current` (`int` or `None`)
                - `cpu_usage_usec` (`int` or `None`)
        """
        group = self.path(rel)
        return {
            "memory_current": _int(_read(group / "memory.current")),
            "memory_events": _keyed(_read(group / "memory.events")),
            "pids_current": _int(_read(group / "pids.current")),
            "cpu_usage_usec": _keyed(_read(group / "cpu.stat")).get("usage_usec"),
        }

    def remove(self, rel: str) -> None:
        """Remove `rel` and its descendants, children first.

        Args:
            rel (`str`):
                Relative cgroup path. A missing group is ignored.

        Raises:
            `CgroupError`: If a group is still populated (`EBUSY`).
        """
        top = self.path(rel)
        if not top.is_dir():
            return
        for dirpath, dirnames, _ in os.walk(top, topdown=False):
            for name in dirnames:
                _rmdir(Path(dirpath) / name)
        _rmdir(top)


def _rmdir(group: Path) -> None:
    try:
        group.rmdir()
    except FileNotFoundError:
        return
    except OSError as exc:
        if exc.errno == errno.EBUSY:
            raise CgroupError(f"{group} is still populated") from exc
        raise CgroupError(f"rmdir {group}: {exc.strerror}") from exc
