# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Start Julia without network access and without the server's secrets.

`sandboxed(cmd)` turns a command into one that first runs this file as a
script: when the server is root it drops to `nobody`, on Linux it installs a
seccomp filter that refuses to create sockets or to reach into other
processes (no capability needed, so it works without CAP_SYS_ADMIN), then it execs `cmd`, which keeps both.
`worker_env()` is the environment to start it with: an allowlist, never a copy
of the server's, so no token or key reaches Julia code. `set_non_dumpable()`
keeps Julia code from reading the server's environment and memory through
`/proc` when both run as the same user. Standard library only, so it runs with
`python -I`.
"""

import os
import sys

PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
# Per architecture: (AUDIT_ARCH_*, syscalls the workers' seccomp filter denies). socket(2) for
# every address family; io_uring_setup/enter/register(2), which could open sockets without
# socket(2); and ptrace(2), process_vm_readv/writev(2) and pidfd_getfd(2), which could take
# a socket or memory from another process of the same uid (a non-root server's workers share
# its uid).
_SECCOMP_DENIED = {
    "x86_64": (0xC000003E, (41, 425, 426, 427, 101, 310, 311, 438)),
    "aarch64": (0xC00000B7, (198, 425, 426, 427, 117, 270, 271, 438)),
}

# The only server environment variables Julia code inherits, plus JULIA*
# settings (depot, threads, juliaup channel). Anything else stays in the server.
WORKER_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ")


UNPRIVILEGED_UID = 65534  # nobody


def set_non_dumpable() -> None:
    """Linux: other processes of the same user can't read this one's /proc/<pid>/{environ,mem}."""
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)


def worker_env() -> dict:
    return {
        k: v
        for k, v in os.environ.items()
        if k in WORKER_ENV_ALLOWLIST or k.startswith("JULIA")
    }


def sandboxed(cmd: list) -> list:
    """`cmd`, run through this file so it starts with the socket filter."""
    return [sys.executable, "-I", os.path.abspath(__file__), *cmd]


def _socket_filter(arch: int, denied: tuple) -> bytes:
    """A seccomp BPF program: EPERM for `denied` syscalls and for any other ABI, allow the rest."""
    import struct

    ld, jeq, jge, ret = (
        0x20,
        0x15,
        0x35,
        0x06,
    )  # BPF_LD|W|ABS, BPF_JMP|JEQ|K, JGE|K, RET|K
    allow, deny = (
        0x7FFF0000,
        0x00050000 | 1,
    )  # SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO | EPERM
    prog = [(ld, 0, 0, 4), (jeq, 1, 0, arch), (ret, 0, 0, deny), (ld, 0, 0, 0)]
    checks = [(jge, 0x40000000)] + [(jeq, nr) for nr in denied]  # x32 syscalls too
    for i, (op, k) in enumerate(checks):
        prog.append((op, len(checks) - i, 0, k))  # to the final deny
    prog += [(ret, 0, 0, allow), (ret, 0, 0, deny)]
    return b"".join(struct.pack("=HBBI", *ins) for ins in prog)


def _socket_filter_unavailable() -> str:
    """Why this platform has no socket filter, or "" if it has one."""
    import platform

    if not sys.platform.startswith("linux"):
        return f"seccomp needs Linux (this is {sys.platform})"
    if platform.machine() not in _SECCOMP_DENIED:
        return f"no seccomp socket filter for {platform.machine()}"
    return ""


def block_sockets() -> None:
    """Linux: install a seccomp filter so this process and its children can't create sockets
    or reach into other processes (see `_SECCOMP_DENIED`).

    It needs no capability (no_new_privs is set first) and can't be undone from
    inside the process. Raises OSError if not possible.
    """
    import platform

    reason = _socket_filter_unavailable()
    if reason:
        raise OSError(reason)
    import ctypes

    arch, denied = _SECCOMP_DENIED[platform.machine()]
    code = _socket_filter(arch, denied)
    buf = ctypes.create_string_buffer(code, len(code))

    class _Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    prog = _Prog(len(code) // 8, ctypes.addressof(buf))
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
    prctl.argtypes += [ctypes.c_ulong, ctypes.c_ulong]  # unused arguments must be 0
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_NO_NEW_PRIVS, 1, None, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")


if __name__ == "__main__":
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(UNPRIVILEGED_UID)
        os.setuid(UNPRIVILEGED_UID)
        # Julia reads ~/.julia/config: root's home is not readable now. Packages
        # come from JULIA_DEPOT_PATH, which nobody must be able to read.
        os.environ["HOME"] = "/nonexistent"
    # Where the filter exists it must apply: an error here stops before Julia starts.
    if not _socket_filter_unavailable():
        block_sockets()
    os.execvp(sys.argv[1], sys.argv[1:])
