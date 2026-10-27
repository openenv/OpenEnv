# SPDX-License-Identifier: BSD-3-Clause

"""Run the agent's script in its HF Job without network access.

The broker sends this file as the job's script, with the agent's script and
requirements embedded below. As root, the launcher:

1. installs the requirements into a fresh venv, from wheels only. Installing a
   wheel runs none of its code; its `.pth` files and modules first run when the
   agent's script starts, under the filter;
2. gives an unprivileged uid `/data/outputs`;
3. installs a seccomp filter that refuses every socket except `AF_UNIX`, and
   io_uring, which can open sockets without calling `socket()`, and the calls
   that read or write another process's memory;
4. closes inherited file descriptors, drops to that uid and runs the agent's
   script. As root the script could still write the unfiltered uv parent's
   `/proc/<pid>/mem`.

HF Jobs containers have no `CAP_NET_ADMIN` and cannot create namespaces, but
any process may filter its own system calls; systemd's
`RestrictAddressFamilies=` works the same way. The filter is inherited by every
child process and cannot be removed. If a step fails, the agent's script does
not run.
"""

import base64
import ctypes
import errno
import os
import socket
import subprocess
import sys
from pathlib import Path

AGENT_SCRIPT = ""  # base64, filled in by the broker
REQUIREMENTS: list[str] = []  # filled in by the broker
# Used by no other process in the job, so the script can't reach an unfiltered one.
AGENT_UID = 10000
WORKDIR = Path("/tmp/posttrain")
VENV = WORKDIR / "venv"
OUTPUTS = Path("/data/outputs")
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_EACCES = 0x00050000 | errno.EACCES
SCMP_CMP_NE = 1
DENIED_SYSCALLS = (
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "pidfd_getfd",
)


class ArgCmp(ctypes.Structure):
    """libseccomp's `struct scmp_arg_cmp`."""

    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def install_requirements() -> None:
    """Install `REQUIREMENTS` from PyPI wheels into `VENV`, without running their code."""
    uv = ["uv", "--no-config", "--quiet"]
    subprocess.run([*uv, "venv", "--python", sys.executable, str(VENV)], check=True)
    if REQUIREMENTS:
        python = str(VENV / "bin" / "python")
        install = [*uv, "pip", "install", "--python", python, "--no-build"]
        subprocess.run([*install, *REQUIREMENTS], check=True)


def install_filter() -> None:
    """Refuse every socket family but `AF_UNIX`, io_uring and cross-process memory access."""
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgCmp),
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        raise OSError("seccomp_init failed")
    not_unix = (ArgCmp * 1)(ArgCmp(0, SCMP_CMP_NE, socket.AF_UNIX, 0))
    rules = [("socket", not_unix), ("socketpair", not_unix)]
    rules += [(name, None) for name in DENIED_SYSCALLS]
    for name, args in rules:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if lib.seccomp_rule_add_array(
            ctx, SCMP_ACT_EACCES, number, len(args or ()), args
        ):
            raise OSError(f"cannot filter {name}")
    # Sets no_new_privs first, so loading needs no privileges.
    if lib.seccomp_load(ctx):
        raise OSError("seccomp_load failed")
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
    except PermissionError:
        return
    raise OSError("the filter did not block an AF_INET socket")


def main() -> None:
    try:
        WORKDIR.mkdir(mode=0o755, exist_ok=True)
        script = WORKDIR / "job.py"
        script.write_bytes(base64.b64decode(AGENT_SCRIPT))
        install_requirements()
        home = WORKDIR / "home"
        home.mkdir(exist_ok=True)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        for path in (home, OUTPUTS):
            os.chown(path, AGENT_UID, AGENT_UID)
        install_filter()
        os.closerange(3, os.sysconf("SC_OPEN_MAX"))
        os.setgroups([])
        os.setgid(AGENT_UID)
        os.setuid(AGENT_UID)
    except Exception as e:
        sys.exit(f"posttrain launcher failed, the script did not run: {e!r}")
    os.chdir(home)
    env = {**os.environ, "HOME": str(home), "HF_HUB_OFFLINE": "1"}
    python = str(VENV / "bin" / "python")
    os.execve(python, [python, str(script)], env)


if __name__ == "__main__":
    main()
