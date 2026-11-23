# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandboxed worker processes for model-written Python, one per session.

Model code runs as real CPython (see session.py), so it can do whatever its process can. The
layers below decide what that is, from the strongest to defense in depth:

1. **A user per session** (Linux, server running as root, as in the Docker image). Each worker
   switches to its own unprivileged uid (`DATA_ANALYSIS_SANDBOX_UID_BASE` + n) right after the
   fork, before any model code runs. It can't read the server's `/proc/<pid>/environ` or memory,
   files only root can read, or another session's home, and it can't signal other sessions. When
   a session ends, every process left with its uid is killed and its files in /tmp, /var/tmp and
   /dev/shm are removed before the uid is used again.
2. **No network.** On Linux every worker installs a seccomp filter (after no_new_privs, so no
   capability is needed) that refuses to create any socket, before model code runs. With root and
   CAP_SYS_ADMIN the spawner also moves itself into a new network namespace before it forks any
   worker, so workers have no usable interface, not even loopback. The server logs which of the
   two apply ("network: blocked by seccomp" without the namespace, e.g. Docker's default or a
   runtime that refuses unshare), or "network: NOT isolated" when neither does (not Linux, a CPU
   without a filter here, or a kernel that refuses it), and then only the audit hook (4) blocks
   sockets.
3. **Process limits and hygiene.** A hard per-call timeout (the worker is killed and the next
   call starts a fresh session), an address-space limit, a cap on file size, open files and
   (with a uid per session) processes, no core files. The worker's environment is built from an
   allowlist, never copied from the server, so no token or key reaches it. Server and spawner are
   non-dumpable (no ptrace, no `/proc/<pid>/mem`) and workers die with the spawner. Server and
   workers exchange bounded JSON only, never pickle.
4. **An audit hook** (PEP 578), armed after the worker's imports. File reads only under the
   task's data directory, the session home and the Python installation. Writes only under the
   session home. No sockets, subprocesses, signals, ctypes/cffi, pyarrow file readers, or
   introspection escapes. Audit hooks can be bypassed from Python, so this is not a boundary on
   its own: it is what stands when the server can't run workers as other users (not root, not
   Linux), as in local development.

Workers are forked from a *spawner*, a single-threaded process started once per server with an
allowlisted environment and pandas, numpy and openpyxl already imported, so a session starts in
milliseconds. The threaded server itself is never forked.
"""

from __future__ import annotations

import collections
import contextlib
import gc
import importlib
import json
import logging
import os
import shutil
import signal
import socket
import sys
import sysconfig
import tempfile
import threading
import time
import weakref
from multiprocessing.connection import Connection
from multiprocessing.reduction import recv_handle, send_handle
from typing import Dict, Iterable, List, Optional, Tuple

# No NullHandler: the startup line saying how sessions are isolated must reach the server log.
logger = logging.getLogger(__name__)

DEFAULT_EXEC_TIMEOUT_S = 30.0
DEFAULT_MEM_LIMIT_MB = 4096
DEFAULT_FILE_LIMIT_MB = 64  # largest file model code may write in its home
DEFAULT_UID_BASE = 61000
DEFAULT_UID_COUNT = 1024
NPROC_LIMIT = 64  # processes and threads per session uid
NOFILE_LIMIT = 256
SPAWNER_STARTUP_TIMEOUT_S = 120.0
SPAWNER_REPLY_TIMEOUT_S = 30.0
WORKER_START_TIMEOUT_S = 30.0
# Longest a session start can take: a spawner (re)start, a uid sweep, then the fork.
MAX_START_S = (
    SPAWNER_STARTUP_TIMEOUT_S + 2 * SPAWNER_REPLY_TIMEOUT_S + WORKER_START_TIMEOUT_S
)
MAX_CODE_BYTES = 1_000_000
MAX_REPLY_BYTES = 1_000_000
CLONE_NEWNET = 0x40000000
PR_SET_PDEATHSIG = 1
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
# Per architecture: (AUDIT_ARCH_*, syscalls the workers' seccomp filter denies). socket(2) for
# every address family, and io_uring_setup(2), which could open sockets without socket(2).
_SECCOMP_DENIED = {
    "x86_64": (0xC000003E, (41, 425)),
    "aarch64": (0xC00000B7, (198, 425)),
}

# The only server environment variables a worker inherits. Anything else (tokens, API keys,
# credentials) stays in the server process.
WORKER_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
# One thread per numerical library: the spawner imports them, so this must be set before.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_SESSION_MODULE = __name__.rsplit(".", 1)[0] + ".session"
# Imported once by the spawner, so every worker starts with them loaded.
PRELOAD = [
    "numpy",
    "pandas",
    "openpyxl",
    "pandas.io.excel",
    "pandas.io.json",
    "pandas.io.parsers",
    # Some tasks fit a model. scikit-learn and its helpers (threadpoolctl uses ctypes) are
    # imported here, before the audit hook, because a worker can't import ctypes.
    "sklearn",
    "sklearn.base",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.feature_extraction",
    "sklearn.impute",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.naive_bayes",
    "sklearn.neighbors",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "sklearn.svm",
    "sklearn.tree",
    "collections",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "math",
    "re",
    "statistics",
    _SESSION_MODULE,
]
# Modules model code may not import. pyarrow reads files with its own C++ code, which raises no
# audit event, so it is kept out of the workers entirely (pandas then uses its own readers).
BLOCKED_IMPORTS = (
    "ctypes",
    "_ctypes",
    "cffi",
    "_cffi_backend",
    "_posixsubprocess",
    "pyarrow",
)

TIMEOUT_MESSAGE = (
    "TimeoutError: this call ran for more than {seconds:g} seconds and was stopped. "
    "The Python session was restarted, so variables and imports from earlier calls are gone."
)
CRASH_MESSAGE = (
    "RuntimeError: the Python session crashed (for example it ran out of memory) and was "
    "restarted, so variables and imports from earlier calls are gone."
)
START_FAILED_MESSAGE = (
    "RuntimeError: the Python session could not start. Try again later."
)


def settings() -> Dict[str, object]:
    """Sandbox settings, from the server's environment."""
    return {
        "exec_timeout_s": float(
            os.environ.get("DATA_ANALYSIS_EXEC_TIMEOUT_S", DEFAULT_EXEC_TIMEOUT_S)
        ),
        "mem_limit_mb": int(
            os.environ.get("DATA_ANALYSIS_MEM_LIMIT_MB", DEFAULT_MEM_LIMIT_MB)
        ),
        "uid_base": int(
            os.environ.get("DATA_ANALYSIS_SANDBOX_UID_BASE", DEFAULT_UID_BASE)
        ),
        "uid_count": int(
            os.environ.get("DATA_ANALYSIS_SANDBOX_UID_COUNT", DEFAULT_UID_COUNT)
        ),
        "isolate_network": os.environ.get("DATA_ANALYSIS_ISOLATE_NETWORK", "1") != "0",
    }


def drops_privileges() -> bool:
    """Whether workers run as their own users: only a root server on Linux can switch uid."""
    return sys.platform.startswith("linux") and os.geteuid() == 0


# --- The audit hook ----------------------------------------------------------------------------

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

# Events allowed only on paths under the session home: (path arguments, dir_fd arguments).
_HOME_EVENTS = {
    "os.mkdir": ((0,), (2,)),
    "os.remove": ((0,), (1,)),
    "os.rmdir": ((0,), (1,)),
    "os.rename": ((0, 1), (2, 3)),
    "os.utime": ((0,), (3,)),
    "os.chmod": ((0,), (2,)),
    "os.truncate": ((0,), ()),
    "tempfile.mkstemp": ((0,), ()),
    "tempfile.mkdtemp": ((0,), ()),
}

# Events refused outright once the hook is armed (prefix match).
_BLOCKED_EVENT_PREFIXES = (
    "socket.",  # any socket: no network, no DNS, no Unix sockets
    "urllib.",
    "http.client.",
    "ftplib.",
    "smtplib.",
    "poplib.",
    "imaplib.",
    "nntplib.",
    "telnetlib.",
    "webbrowser.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.putenv",
    "os.unsetenv",
    "os.lchmod",
    "os.chflags",
    "os.lchflags",
    "os.chown",
    "os.lchown",
    "os.link",
    "os.symlink",
    "os.mkfifo",
    "os.mknod",
    "os.setxattr",
    "os.removexattr",
    "shutil.",
    "fcntl.",
    "pty.",
    "ctypes.",
    "resource.setrlimit",
    "resource.prlimit",
    "sys.addaudithook",
    "sys.setprofile",
    "sys.settrace",
    "sys.monitoring.",
    "gc.get_objects",
    "gc.get_referrers",
    "gc.get_referents",
    "code.__new__",
    "builtins.input",
    "sqlite3.enable_load_extension",
    "sqlite3.load_extension",
)

_NEVER_READABLE = (
    "",
    "/proc",
    "/etc",
    "/root",
    "/home",
    "/sys",
    "/dev",
    "/tmp",
    "/var",
    "/usr",
)


class SandboxViolation(PermissionError):
    """Raised inside the worker when model code tries something the sandbox doesn't allow."""


SandboxViolation.__module__ = (
    "builtins"  # shown to the model as `SandboxViolation: ...`
)


def _python_roots() -> List[str]:
    """Directories of the Python installation and of the packages the workers use."""
    paths = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        with contextlib.suppress(KeyError):
            paths.append(sysconfig.get_paths()[name])
    for name in (
        "numpy",
        "pandas",
        "openpyxl",
        "sklearn",
        "scipy",
        __name__.split(".")[0],
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            # The directory holding the package: site-packages, or the source tree in dev.
            paths.append(os.path.dirname(os.path.dirname(module.__file__)))
    paths.append("/usr/share/zoneinfo")
    return paths


def read_roots(extra: Iterable[str] = ()) -> Tuple[str, ...]:
    """Directories a worker may read: `extra` (task data, home) and the Python installation.

    The Python directories never include the filesystem root, the server user's home or one of
    its parents, or /proc, /etc, /root, /home, /sys, /dev, /var, /tmp and /usr themselves.
    """
    roots = set()
    home = os.path.realpath(os.path.expanduser("~")).rstrip(os.sep) + os.sep
    for path in _python_roots():
        real = os.path.realpath(path).rstrip(os.sep) + os.sep
        # Too broad: the root, a system directory itself, the server user's home or a parent of it.
        if real.rstrip(os.sep) in _NEVER_READABLE or home.startswith(real):
            continue
        roots.add(real)
    for path in extra:
        roots.add(os.path.realpath(path).rstrip(os.sep) + os.sep)
    return tuple(sorted(roots))


def make_audit_hook(roots: Tuple[str, ...], home: str):
    """The hook: reads under `roots`, writes under `home`, and nothing from the blocked list.

    Model code shares this process, so it can patch any Python-level function or module global
    the hook would look up at call time (`os.path.realpath`, `builtins.isinstance`, this module's
    tables). Everything the hook uses is therefore bound here, once: builtins, C functions from
    `posix`, and private copies of the tables. Paths are resolved with those C functions only,
    and only plain `str` and `bytes` paths are accepted, because a path-like object could return
    one path to the hook and another to the code that opens the file.
    """
    import posix

    lstat, stat, readlink, getcwd = (
        posix.lstat,
        posix.stat,
        posix.readlink,
        posix.getcwd,
    )
    str_, bytes_, int_, type_, tuple_, list_, len_ = (
        str,
        bytes,
        int,
        type,
        tuple,
        list,
        len,
    )
    oserror, violation = OSError, SandboxViolation
    encoding, errors = sys.getfilesystemencoding(), sys.getfilesystemencodeerrors()
    prefixes = tuple(_BLOCKED_EVENT_PREFIXES)
    blocked_imports = frozenset(BLOCKED_IMPORTS)
    home_events = dict(_HOME_EVENTS)
    write_flags = _WRITE_FLAGS
    read_roots = tuple(roots)
    home_root = (os.path.realpath(home).rstrip("/") + "/",)
    allowed_files = frozenset(("/dev/null", "/dev/urandom", "/etc/localtime"))

    def as_text(path):
        """The path as a str, or None if it is not a plain str or bytes path."""
        if type_(path) is str_:
            return path
        if type_(path) is bytes_:
            return path.decode(encoding, errors)
        return None

    def resolve(path: str) -> str:
        """Like os.path.realpath, with C functions only. Relative paths may not use '..'."""
        if not path.startswith("/"):
            if ".." in path.split("/"):
                raise violation("sandbox: relative paths with '..' are not allowed")
            path = getcwd() + "/" + path
        pending = list_(path.split("/"))
        pending.reverse()
        resolved = []
        links = 0
        while pending:
            name = pending.pop()
            if name in ("", "."):
                continue
            if name == "..":
                if resolved:
                    resolved.pop()
                continue
            candidate = "/" + "/".join(resolved + [name])
            try:
                mode = lstat(candidate).st_mode
            except oserror:
                resolved.append(name)  # doesn't exist (yet): keep it as written
                continue
            if mode & 0o170000 == 0o120000:  # a symlink: follow it
                links += 1
                if links > 40:
                    raise violation("sandbox: too many levels of symbolic links")
                target = readlink(candidate)
                if target.startswith("/"):
                    resolved = []
                parts = list_(target.split("/"))
                parts.reverse()
                pending.extend(parts)
                continue
            resolved.append(name)
        return "/" + "/".join(resolved)

    def checked(path, allowed, what: str) -> str:
        """The resolved path, if it is under `allowed`."""
        text = as_text(path)
        if text is None:
            raise violation(f"sandbox: {what} needs a plain str path")
        real = resolve(text)
        if real in allowed_files and allowed is read_roots:
            return real
        if not (real.rstrip("/") + "/").startswith(allowed):
            if allowed is home_root:
                raise violation(
                    f"sandbox: {what} {text!r} is not allowed, "
                    "only under the session's working directory"
                )
            raise violation(
                f"sandbox: {what} {text!r} is not allowed, only the task's data files"
            )
        return real

    def is_dir(real: str) -> bool:
        try:
            return stat(real).st_mode & 0o170000 == 0o040000
        except oserror:
            return False

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = (tuple_(args) + (None, None, None))[:3]
            if type_(path) is int_:  # an already-open descriptor
                return
            writing = (
                type_(mode) is str_
                and ("w" in mode or "a" in mode or "x" in mode or "+" in mode)
            ) or (type_(flags) is int_ and flags & write_flags)
            if writing:
                checked(path, home_root, "writing")
                return
            real = checked(path, read_roots, "reading")
            # The open event doesn't carry dir_fd, and the hook resolves relative paths against
            # the working directory. So the only directories that can be opened (for use as a
            # dir_fd) are under the session home, where any relative path stays in the home.
            if is_dir(real) and not (real.rstrip("/") + "/").startswith(home_root):
                raise violation(
                    "sandbox: opening a directory is only allowed "
                    "under the session's working directory"
                )
            return
        if event in ("os.listdir", "os.scandir", "os.chdir"):
            path = args[0] if args else None
            if path is None:
                path = "."
            if type_(path) is not int_:
                checked(path, read_roots, event)
            return
        if event in home_events:
            path_args, fd_args = home_events[event]
            for i in fd_args:
                if i < len_(args) and args[i] is not None and args[i] != -1:
                    raise violation(f"sandbox: {event} with dir_fd is not allowed")
            for i in path_args:
                if i < len_(args) and type_(args[i]) is not int_:
                    checked(args[i], home_root, event)
            return
        if event == "sqlite3.connect":
            database = args[0] if args else None
            if as_text(database) in (":memory:", ""):
                return
            checked(database, home_root, "a sqlite3 database at")
            return
        if event == "import":
            name = args[0] if args else ""
            if type_(name) is str_ and name.split(".")[0] in blocked_imports:
                raise violation(f"sandbox: import of {name} is not allowed")
            return
        if event.startswith(prefixes):
            raise violation(f"sandbox: {event} is not allowed")

    return hook


# --- Process helpers ---------------------------------------------------------------------------


def _prctl(option: int, value: int) -> None:
    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(option, value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), f"prctl({option}, {value}) failed")


def set_non_dumpable() -> None:
    """Linux: no ptrace and no /proc/<pid>/{mem,environ,fd} for other processes of the same uid."""
    _prctl(PR_SET_DUMPABLE, 0)


def _isolate_network() -> Tuple[bool, str]:
    """Move this process into a new, empty network namespace. Returns (isolated, why not)."""
    if not sys.platform.startswith("linux"):
        return False, f"network namespaces need Linux (this is {sys.platform})"
    if os.geteuid() != 0:
        return False, "the server is not root"
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(CLONE_NEWNET) != 0:
        err = ctypes.get_errno()
        return False, f"unshare(CLONE_NEWNET) failed: {os.strerror(err)}"
    return True, ""


def _socket_filter(arch: int, denied: Tuple[int, ...]) -> bytes:
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
    """Linux: install a seccomp filter so this process and its children can't create sockets.

    Unlike a network namespace it needs no capability (no_new_privs is set first), and unlike
    the audit hook it can't be undone from inside the process. Raises OSError if not possible.
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


def _probe_socket_filter() -> Tuple[bool, str]:
    """Whether the filter works here: a child installs it and must be refused an AF_INET socket."""
    reason = _socket_filter_unavailable()
    if reason:
        return False, reason
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            block_sockets()
            try:
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
            except PermissionError:
                code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return True, ""
    return False, "the kernel refused the seccomp filter"


def _base_env(home: str) -> Dict[str, str]:
    env = {k: os.environ[k] for k in WORKER_ENV_ALLOWLIST if k in os.environ}
    env.update(THREAD_ENV)
    env.update(HOME=home, TMPDIR=home, TMP=home, TEMP=home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _spawner_env(home: str) -> Dict[str, str]:
    env = _base_env(home)
    # The spawner imports this module the way the server did.
    env["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(p) for p in sys.path if p and os.path.isdir(p)
    )
    return env


def _kill(proc) -> None:
    proc.kill()
    proc.wait()


# --- Spawner (server side) ---------------------------------------------------------------------

# Non-dumpable before any import, so a same-uid process can't read the spawner's memory while it
# loads. pyarrow is kept out before anything imports pandas: pandas then uses its own readers.
_SPAWNER_BOOT = (
    "import sys\n"
    "if sys.platform.startswith('linux'):\n"
    "    import ctypes\n"
    f"    ctypes.CDLL(None).prctl({PR_SET_DUMPABLE}, 0, 0, 0, 0)\n"
    "class _NoPyarrow:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] == 'pyarrow':\n"
    "            raise ImportError('pyarrow is not available in the sandbox')\n"
    "sys.meta_path.insert(0, _NoPyarrow())\n"
    f"from {__name__} import _spawner_main\n"
    "_spawner_main(int(sys.argv[1]), sys.argv[2] == '1')\n"
)


class _Spawner:
    """Handle on the spawner process: forks, kills and reaps workers, and sweeps session uids.

    The spawner kills and reaps its own workers, so a worker's pid can't be reused while the
    server still refers to it. It exits when the server closes its end of the connection.
    """

    def __init__(self, isolate_network: bool):
        import subprocess

        self._lock = threading.Lock()
        cleanup = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, cleanup.close)
        try:
            home = tempfile.mkdtemp(prefix="data_analysis_env_spawner_")
            cleanup.callback(shutil.rmtree, home, ignore_errors=True)
            parent_sock, child_sock = socket.socketpair()
            self._conn = Connection(parent_sock.detach())
            cleanup.callback(self._conn.close)
            with child_sock:
                self._proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _SPAWNER_BOOT,
                        str(child_sock.fileno()),
                        "1" if isolate_network else "0",
                    ],
                    pass_fds=(child_sock.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    env=_spawner_env(home),
                    cwd=home,
                )
            cleanup.callback(_kill, self._proc)
            hello = self._receive(SPAWNER_STARTUP_TIMEOUT_S)
            if hello.get("ready") is not True:
                raise ValueError("bad spawner handshake")
            self.network_isolated = hello.get("network_isolated") is True
            self.network_reason = str(hello.get("network_reason", ""))[:300]
            self.socket_filter = hello.get("socket_filter") is True
            self.socket_filter_reason = str(hello.get("socket_filter_reason", ""))[:300]
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def alive(self) -> bool:
        return self._finalizer.alive and self._proc.poll() is None

    def spawn(self, fd: int, config: dict) -> int:
        """Fork a worker that serves `fd` and return its pid."""
        with self._lock:
            pid = self._request({"op": "spawn", "config": config}, fd).get("pid")
        if type(pid) is not int or pid <= 1:
            raise ValueError("bad spawner reply")
        return pid

    def reap(self, pid: int) -> None:
        """Kill a worker (if still alive) and have the spawner reap it."""
        with self._lock:
            self._request({"op": "reap", "pids": [pid]})

    def sweep(self, uid: int) -> None:
        """Kill every process of `uid` and remove its files in the shared temp directories."""
        with self._lock:
            if self._request({"op": "sweep", "uid": uid}).get("swept") is not True:
                raise RuntimeError(f"could not clean up sandbox uid {uid}")

    def close(self) -> None:
        self._finalizer()

    def _request(self, request: dict, fd: Optional[int] = None) -> dict:
        """Caller holds the lock."""
        try:
            self._conn.send_bytes(json.dumps(request).encode())
            if fd is not None:
                send_handle(self._conn, fd, self._proc.pid)
            return self._receive(SPAWNER_REPLY_TIMEOUT_S)
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout: float) -> dict:
        if not self._conn.poll(timeout):
            raise TimeoutError("the data_analysis_env sandbox spawner did not reply")
        reply = json.loads(self._conn.recv_bytes(4096))
        if not isinstance(reply, dict):
            raise ValueError("bad spawner reply")
        return reply


class _UidPool:
    """Unprivileged uids for sessions, one per live session, swept before reuse."""

    def __init__(self, base: int, count: int):
        if base < 1000 or count < 1 or base + count > 65534:
            raise ValueError(f"bad sandbox uid range {base}..{base + count - 1}")
        # Never a uid that belongs to a real account: a sweep kills all of its processes.
        self._free = collections.deque(
            uid for uid in range(base, base + count) if not _has_account(uid)
        )
        self._clean: set[int] = set()
        self._lock = threading.Lock()

    def acquire(self) -> Tuple[int, bool]:
        """A free uid, and whether it must be swept first (not yet swept by this server)."""
        with self._lock:
            if not self._free:
                raise RuntimeError("no free sandbox uid: too many concurrent sessions")
            uid = self._free.popleft()
            return uid, uid not in self._clean

    def release(self, uid: int, swept: bool) -> None:
        with self._lock:
            if swept:
                self._clean.add(uid)
            else:
                self._clean.discard(uid)
            self._free.append(uid)


def _has_account(uid: int) -> bool:
    import pwd

    try:
        pwd.getpwuid(uid)
    except KeyError:
        return False
    return True


_spawner: Optional[_Spawner] = None
_uids: Optional[_UidPool] = None
_global_lock = threading.Lock()


def _get_spawner() -> _Spawner:
    """This server's spawner, started on first use and restarted if it died."""
    global _spawner
    with _global_lock:
        if _spawner is None or not _spawner.alive:
            conf = settings()
            _spawner = _Spawner(bool(conf["isolate_network"]))
            if drops_privileges():
                users = f"a uid per session (from {conf['uid_base']})"
            else:
                users = "the server's uid (not root: audit hook only)"
            if _spawner.network_isolated:
                network = "isolated" + (
                    " (namespace and seccomp)"
                    if _spawner.socket_filter
                    else f" (namespace, {_spawner.socket_filter_reason})"
                )
            elif _spawner.socket_filter:
                network = (
                    f"blocked by seccomp (no namespace: {_spawner.network_reason})"
                )
            else:
                network = (
                    f"NOT isolated ({_spawner.network_reason}, "
                    f"{_spawner.socket_filter_reason})"
                )
            logger.warning(
                "data_analysis_env sandbox: workers run as %s, network: %s",
                users,
                network,
            )
        return _spawner


def _get_uid_pool() -> _UidPool:
    global _uids
    with _global_lock:
        if _uids is None:
            conf = settings()
            _uids = _UidPool(int(conf["uid_base"]), int(conf["uid_count"]))
        return _uids


def sandbox_status() -> Dict[str, object]:
    """How sessions are isolated on this server (starts the spawner if needed)."""
    spawner = _get_spawner()
    return {
        "user_per_session": drops_privileges(),
        "network_isolated": spawner.network_isolated,
        "network_reason": spawner.network_reason,
        "socket_filter": spawner.socket_filter,
        "socket_filter_reason": spawner.socket_filter_reason,
        "spawner_pid": spawner.pid,
    }


class SandboxSession:
    """One Python session in its own worker process.

    The worker is forked by `start()` (or the first `run()`) and killed by `close()`, or when the
    session is garbage collected. A call that exceeds `timeout_s` kills the worker, and the next
    call starts a fresh one.

    Args:
        read_dirs: Directories model code may read (the task's data).
        timeout_s: Hard limit per call (default `DATA_ANALYSIS_EXEC_TIMEOUT_S` or 30 s).
        mem_limit_mb: Address-space limit of the worker (default `DATA_ANALYSIS_MEM_LIMIT_MB`).
    """

    def __init__(
        self,
        read_dirs: Iterable[str] = (),
        timeout_s: Optional[float] = None,
        mem_limit_mb: Optional[int] = None,
    ):
        conf = settings()
        self._read_dirs = [os.path.realpath(d) for d in read_dirs]
        self._timeout_s = float(timeout_s or conf["exec_timeout_s"])
        self._mem_limit_mb = int(mem_limit_mb or conf["mem_limit_mb"])
        self._spawner: Optional[_Spawner] = None
        self._pid: Optional[int] = None
        self._uid: Optional[int] = None
        self._home: Optional[str] = None
        self._conn: Optional[Connection] = None
        self._finalizer: Optional[weakref.finalize] = None

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    @property
    def uid(self) -> Optional[int]:
        return self._uid

    @property
    def home(self) -> Optional[str]:
        return self._home

    def start(self) -> bool:
        """Fork the worker if it is not running. False if it could not start."""
        if self._conn is not None:
            return True
        cleanup = contextlib.ExitStack()
        self._finalizer = weakref.finalize(self, cleanup.close)
        try:
            self._spawner = spawner = _get_spawner()
            swept = [False]
            if drops_privileges():
                pool = _get_uid_pool()
                uid, dirty = pool.acquire()
                self._uid = uid
                cleanup.callback(lambda: pool.release(uid, swept[0]))
                if dirty:
                    spawner.sweep(uid)
            self._home = tempfile.mkdtemp(prefix="data_analysis_env_")
            cleanup.callback(shutil.rmtree, self._home, ignore_errors=True)
            if self._uid is not None:
                os.chown(self._home, self._uid, self._uid)
                # On close: after the worker is killed, before its home is removed.
                cleanup.callback(_sweep_quietly, spawner, self._uid, swept)
            os.chmod(self._home, 0o700)
            config = {
                "uid": self._uid,
                "home": self._home,
                "env": dict(_base_env(self._home), no_proxy="*"),
                "roots": list(read_roots(self._read_dirs + [self._home])),
                "mem_limit_mb": self._mem_limit_mb,
                "file_limit_mb": DEFAULT_FILE_LIMIT_MB,
                "block_sockets": spawner.socket_filter,
            }
            parent_sock, child_sock = socket.socketpair()
            conn = Connection(parent_sock.detach())
            cleanup.callback(conn.close)
            with child_sock:
                pid = spawner.spawn(child_sock.fileno(), config)
            self._pid = pid
            cleanup.callback(_reap_quietly, spawner, pid)
            if not conn.poll(WORKER_START_TIMEOUT_S):
                raise TimeoutError("the sandbox worker did not start")
            if json.loads(conn.recv_bytes(1024)) != {"ready": True}:
                raise ValueError("bad worker handshake")
            self._conn = conn
        except Exception:
            logger.exception("could not start a data_analysis_env sandbox worker")
            self.close()
            return False
        return True

    def run(self, code: str) -> Tuple[str, str, int]:
        """Run one call. Returns (stdout, stderr, exit_code)."""
        if not self.start():
            return "", START_FAILED_MESSAGE, 1
        deadline = time.monotonic() + self._timeout_s
        # Kill the worker at the deadline, so a reply that stalls (even mid-frame) ends in EOF.
        timer = threading.Timer(self._timeout_s, os.kill, (self._pid, signal.SIGKILL))
        timer.start()
        try:
            self._conn.send_bytes(json.dumps({"code": str(code)}).encode())
            # Bounded JSON with typed fields only: the worker runs model code and is not trusted.
            reply = json.loads(self._conn.recv_bytes(MAX_REPLY_BYTES))
            stdout, stderr = reply["stdout"], reply["stderr"]
            exit_code = reply["exit_code"]
            if not (
                isinstance(stdout, str)
                and isinstance(stderr, str)
                and exit_code in (0, 1)
                and type(exit_code) is int
            ):
                raise ValueError("malformed worker reply")
            return stdout, stderr, exit_code
        except (EOFError, OSError, ValueError, KeyError, TypeError):
            self.close()
            if time.monotonic() >= deadline:
                return "", TIMEOUT_MESSAGE.format(seconds=self._timeout_s), 1
            logger.warning("data_analysis_env worker process failed, restarting it")
            return "", CRASH_MESSAGE, 1
        finally:
            timer.cancel()

    def close(self) -> None:
        """Kill the worker, sweep its uid and remove its home. Safe to call more than once."""
        if self._finalizer is not None:
            self._finalizer()
        self._spawner = self._pid = self._uid = self._home = None
        self._conn = self._finalizer = None


def _reap_quietly(spawner: _Spawner, pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)  # now, even if the spawner is busy
    try:
        spawner.reap(pid)
    except Exception:
        # The spawner is gone, and its workers with it (PR_SET_PDEATHSIG).
        logger.debug("could not reach the data_analysis_env spawner", exc_info=True)


def _sweep_quietly(spawner: _Spawner, uid: int, swept: List[bool]) -> None:
    try:
        spawner.sweep(uid)
        swept[0] = True
    except Exception:
        # Not marked clean: the next session that gets this uid sweeps it first.
        logger.warning("could not sweep sandbox uid %d", uid, exc_info=True)


# --- Spawner and worker (child side) -----------------------------------------------------------


def _spawner_main(fd: int, isolate_network: bool) -> None:
    """Spawner process: fork workers, kill and reap them, and sweep uids, on request."""
    import resource

    sys.dont_write_bytecode = True
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if isolate_network:
        isolated, reason = _isolate_network()
    else:
        isolated, reason = False, "disabled with DATA_ANALYSIS_ISOLATE_NETWORK=0"
    filtered, filter_reason = _probe_socket_filter()
    for name in PRELOAD:
        importlib.import_module(name)
    conn = Connection(fd)
    children: set[int] = set()
    # Keep the imported objects out of the workers' garbage collections, so those don't write to
    # (and copy) the memory pages they share with the spawner.
    gc.freeze()
    hello = {
        "ready": True,
        "network_isolated": isolated,
        "network_reason": reason,
        "socket_filter": filtered,
        "socket_filter_reason": filter_reason,
    }
    conn.send_bytes(json.dumps(hello).encode())
    while True:
        try:
            request = json.loads(conn.recv_bytes(64 * 1024))
        except (EOFError, OSError):
            return  # the server is gone, PR_SET_PDEATHSIG stops the workers
        op = request.get("op")
        reply: dict = {}
        if op == "spawn":
            worker_fd = recv_handle(conn)
            pid = os.fork()
            if pid == 0:
                try:
                    conn.close()
                    _worker_main(worker_fd, request["config"])
                finally:
                    os._exit(0)  # never return into the spawner's loop
            os.close(worker_fd)
            children.add(pid)
            reply["pid"] = pid
        elif op == "reap":
            for pid in request.get("pids", []):
                if pid in children:
                    children.discard(pid)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(pid, 0)
        elif op == "sweep":
            reply["swept"] = _sweep_uid(int(request["uid"]))
        conn.send_bytes(json.dumps(reply).encode())


_SWEEP_DIRS = ("/tmp", "/var/tmp", "/dev/shm")


def _sweep_uid(uid: int) -> bool:
    """As `uid`, in a child: kill every process of that uid and remove its top-level entries in
    the shared temp directories. Running as the uid itself, not root, means nothing here can be
    tricked into touching files that uid could not touch anyway."""
    if not drops_privileges() or uid < 1000:
        return False
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.setgroups([])
            os.setresgid(uid, uid, uid)
            os.setresuid(uid, uid, uid)
            with contextlib.suppress(ProcessLookupError):
                os.kill(
                    -1, signal.SIGKILL
                )  # every process this uid may signal, not itself
            for directory in _SWEEP_DIRS:
                with contextlib.suppress(OSError):
                    for entry in os.scandir(directory):
                        with contextlib.suppress(OSError):
                            if entry.stat(follow_symlinks=False).st_uid != uid:
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                shutil.rmtree(entry.path, ignore_errors=True)
                            else:
                                os.unlink(entry.path)
            code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def _apply_limits(config: dict) -> None:
    import resource

    # A write past RLIMIT_FSIZE raises SIGXFSZ, so ignore it and get EFBIG instead.
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limits = [
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_FSIZE, int(config["file_limit_mb"]) * 1024 * 1024),
        (resource.RLIMIT_NOFILE, NOFILE_LIMIT),
    ]
    if sys.platform.startswith("linux"):  # macOS doesn't enforce RLIMIT_AS
        limits.append((resource.RLIMIT_AS, int(config["mem_limit_mb"]) * 1024 * 1024))
    if config["uid"] is not None:  # counted per uid, so only with a uid per session
        limits.append((resource.RLIMIT_NPROC, NPROC_LIMIT))
    for which, value in limits:
        _, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(which, (value, value))


def _drop_blocked_modules() -> None:
    """Remove ctypes & co. from sys.modules (so a fresh import reaches the hook) and cut the raw
    fork+exec that skips the subprocess audit event."""
    import subprocess as _subprocess

    def _refused(*args, **kwargs):
        raise SandboxViolation("sandbox: subprocesses are not allowed")

    if hasattr(_subprocess, "_fork_exec"):
        _subprocess._fork_exec = _refused
    posixsub = sys.modules.get("_posixsubprocess")
    if posixsub is not None and hasattr(posixsub, "fork_exec"):
        with contextlib.suppress(AttributeError):
            del posixsub.fork_exec
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED_IMPORTS:
            del sys.modules[name]


def _worker_main(fd: int, config: dict) -> None:
    """Forked worker: drop privileges, lock down, then run one session's calls in order."""
    uid = config["uid"]
    if uid is not None:
        os.setgroups([])
        os.setresgid(uid, uid, uid)
        os.setresuid(uid, uid, uid)
        if os.getresuid() != (uid, uid, uid) or os.getresgid() != (uid, uid, uid):
            raise PermissionError("could not switch to the session uid")
    set_non_dumpable()  # changing uid resets it, so set it in every case
    _prctl(PR_SET_PDEATHSIG, signal.SIGKILL)  # die with the spawner
    if config["block_sockets"]:
        block_sockets()  # no socket from here on, whatever happens to the audit hook
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    # Model code writing to fd 0-2 directly must not reach the server's log.
    devnull = os.open(os.devnull, os.O_RDWR)
    for target in (0, 1, 2):
        os.dup2(devnull, target)
    os.close(devnull)
    home = config["home"]
    os.chdir(home)
    os.environ.clear()
    os.environ.update(config["env"])
    tempfile.tempdir = None  # the spawner may have cached its own
    _apply_limits(config)
    roots = tuple(config["roots"])
    # Only import from directories the hook lets the worker read.
    sys.path[:] = [
        p for p in sys.path if p and (os.path.realpath(p) + os.sep).startswith(roots)
    ]
    conn = Connection(fd)
    # Model code may patch json's classes in this process (it only hurts its own session).
    encode = json.JSONEncoder(separators=(",", ":")).encode
    decode = json.JSONDecoder().decode
    session = importlib.import_module(_SESSION_MODULE).PythonSession()
    session.run(
        "pass"
    )  # warm the cell machinery (ast, compile, linecache) before arming
    _drop_blocked_modules()
    sys.addaudithook(make_audit_hook(roots, home))
    conn.send_bytes(b'{"ready":true}')
    while True:
        try:
            request = decode(conn.recv_bytes(MAX_CODE_BYTES).decode())
            code = request["code"]
            if not isinstance(code, str):
                return
        except (EOFError, OSError, ValueError, KeyError, TypeError):
            return
        stdout, stderr, exit_code = session.run(code)
        reply = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
        try:
            conn.send_bytes(encode(reply).encode())
        except (OSError, ValueError):
            return
