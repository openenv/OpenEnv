# SPDX-License-Identifier: BSD-3-Clause

"""Chain of custody for what the agent leaves behind.

Graders read the agent's outputs only through [`copy_validated`], which copies
regular files and directories and nothing else. Symlinks (which could point at
assets), FIFOs (which block a reader), sockets and device nodes are refused,
and the copy is bounded in size and entry count. Every open is relative to an
already-opened parent directory and refuses to follow symlinks, so swapping an
entry after it was listed doesn't redirect the copy.
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20
_MAX_DEPTH = 128
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class CustodyError(RuntimeError):
    """The agent's output contains something a grader must not read."""


@dataclass
class CustodyReport:
    """What [`copy_validated`] copied.

    Attributes:
        files (`int`):
            Regular files copied.
        bytes (`int`):
            Total bytes copied.
        dirs (`int`):
            Directories created below `dst`.
    """

    files: int = 0
    bytes: int = 0
    dirs: int = 0


def _list_dir(dfd: int) -> list[tuple[str, os.stat_result]]:
    names = sorted(os.listdir(dfd))
    return [(n, os.stat(n, dir_fd=dfd, follow_symlinks=False)) for n in names]


def _kind(mode: int) -> str:
    for check, name in (
        (stat.S_ISLNK, "symlink"),
        (stat.S_ISFIFO, "FIFO"),
        (stat.S_ISSOCK, "socket"),
        (stat.S_ISBLK, "block device"),
        (stat.S_ISCHR, "character device"),
    ):
        if check(mode):
            return name
    return "special file"


def _open_nofollow(name: str, dfd: int, rel: str, flags: int) -> int:
    try:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR, errno.ENXIO):
            raise CustodyError(f"{rel}: changed type while copying") from e
        raise


class _Copier:
    def __init__(self, max_bytes: int, max_files: int):
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.report = CustodyReport()

    def _count_entry(self, rel: str) -> None:
        if self.report.files + self.report.dirs >= self.max_files:
            raise CustodyError(f"{rel}: more than {self.max_files} entries")

    def copy_dir(self, sfd: int, dst: Path, rel: str, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise CustodyError(f"{rel}: nested deeper than {_MAX_DEPTH} directories")
        for name, st in _list_dir(sfd):
            child_rel = name if rel == "." else f"{rel}/{name}"
            if stat.S_ISREG(st.st_mode):
                self._count_entry(child_rel)
                self.copy_file(sfd, name, dst / name, child_rel)
            elif stat.S_ISDIR(st.st_mode):
                self._count_entry(child_rel)
                cfd = _open_nofollow(name, sfd, child_rel, os.O_RDONLY | _O_DIRECTORY)
                try:
                    if not stat.S_ISDIR(os.fstat(cfd).st_mode):
                        raise CustodyError(f"{child_rel}: changed type while copying")
                    os.mkdir(dst / name, 0o700)
                    os.chmod(dst / name, 0o700)
                    self.report.dirs += 1
                    self.copy_dir(cfd, dst / name, child_rel, depth + 1)
                finally:
                    os.close(cfd)
            else:
                raise CustodyError(f"{child_rel}: {_kind(st.st_mode)} not allowed")

    def copy_file(self, sfd: int, name: str, dst: Path, rel: str) -> None:
        rfd = _open_nofollow(name, sfd, rel, os.O_RDONLY)
        try:
            mode = os.fstat(rfd).st_mode
            if not stat.S_ISREG(mode):
                raise CustodyError(f"{rel}: {_kind(mode)} not allowed")
            wfd = os.open(
                dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            try:
                os.fchmod(wfd, 0o600)
                while chunk := os.read(rfd, _CHUNK):
                    if self.report.bytes + len(chunk) > self.max_bytes:
                        raise CustodyError(
                            f"{rel}: copy exceeds {self.max_bytes} bytes"
                        )
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(wfd, view) :]
                    self.report.bytes += len(chunk)
            finally:
                os.close(wfd)
            self.report.files += 1
        finally:
            os.close(rfd)


def copy_validated(
    src: Path, dst: Path, *, max_bytes: int, max_files: int
) -> CustodyReport:
    """
    Copy an agent-written tree for a grader, refusing anything but plain files.

    Ownership and permissions are not preserved: directories are created `0o700`
    and files `0o600`. On [`CustodyError`] the partial copy is left in `dst`.

    Args:
        src (`Path`):
            Directory the agent wrote. Must not itself be a symlink.
        dst (`Path`):
            Where to copy it. Must not exist, or be an empty directory.
        max_bytes (`int`):
            Total bytes allowed across all files.
        max_files (`int`):
            Total entries (files plus directories) allowed below `src`.

    Returns:
        [`CustodyReport`]: counts of what was copied.

    Raises:
        [`CustodyError`]: `src` contains a symlink, FIFO, socket or device node,
            exceeds a limit, or changed type during the copy. The message names
            the offending path relative to `src`.
        `ValueError`: `dst` exists and is not an empty directory.
    """
    dst = Path(dst)
    if dst.is_symlink() or (dst.exists() and (not dst.is_dir() or any(dst.iterdir()))):
        raise ValueError(f"custody destination {dst} must not exist or be empty")
    try:
        sfd = os.open(src, os.O_RDONLY | _O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise CustodyError(".: source is not a plain directory") from e
        raise
    try:
        if not stat.S_ISDIR(os.fstat(sfd).st_mode):
            raise CustodyError(".: source is not a plain directory")
        dst.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(dst, 0o700)
        copier = _Copier(max_bytes, max_files)
        copier.copy_dir(sfd, dst, ".", 0)
        return copier.report
    finally:
        os.close(sfd)
