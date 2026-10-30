# SPDX-License-Identifier: BSD-3-Clause

"""Seccomp profiles for openenvd containers.

The container-wide profile (`linux.seccomp` in the OCI config) allows by
default and denies the syscalls that reach kernel attack surface or leave the
container's namespaces. It never denies `AF_INET`: the shim's loopback
forwarders need it, and the network namespace already holds nothing else.
Every zone gets the same profile.
"""

from __future__ import annotations

import platform
from typing import Any

EPERM = 1
ENOSYS = 38

AF_PACKET = 17

CLONE_NAMESPACE_FLAGS: dict[str, int] = {
    "CLONE_NEWUSER": 0x10000000,
    "CLONE_NEWNS": 0x00020000,
    "CLONE_NEWPID": 0x20000000,
    "CLONE_NEWNET": 0x40000000,
    "CLONE_NEWIPC": 0x08000000,
    "CLONE_NEWUTS": 0x04000000,
    "CLONE_NEWCGROUP": 0x02000000,
}

DENIED_SYSCALLS: tuple[str, ...] = (
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "kexec_load",
    "kexec_file_load",
    "bpf",
    "perf_event_open",
    "keyctl",
    "add_key",
    "request_key",
    "mount",
    "umount2",
    "pivot_root",
    "unshare",
    "setns",
    "open_by_handle_at",
    "name_to_handle_at",
    "init_module",
    "finit_module",
    "delete_module",
    "reboot",
    "swapon",
    "swapoff",
    "acct",
    "userfaultfd",
    "fanotify_init",
    "mount_setattr",
    "move_mount",
    "open_tree",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    # The i386 socket multiplexer would bypass the `socket` argument filters.
    "socketcall",
)

_ARCHITECTURES = {
    "x86_64": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86"],
    "amd64": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86"],
    "aarch64": ["SCMP_ARCH_AARCH64"],
    "arm64": ["SCMP_ARCH_AARCH64"],
}


def seccomp_architectures(arch: str | None = None) -> list[str]:
    """Return the libseccomp architecture names for a machine.

    Args:
        arch (`str`, *optional*):
            A `platform.machine()` value. Defaults to the running machine.

    Returns:
        `list[str]`: `SCMP_ARCH_*` names.

    Raises:
        `ValueError`: If the architecture is not x86_64 or aarch64.
    """
    machine = (arch or platform.machine()).lower()
    try:
        return list(_ARCHITECTURES[machine])
    except KeyError:
        raise ValueError(f"unsupported architecture {machine!r}") from None


def _deny(names: list[str], errno: int = EPERM, args: list | None = None) -> dict:
    rule: dict[str, Any] = {
        "names": names,
        "action": "SCMP_ACT_ERRNO",
        "errnoRet": errno,
    }
    if args:
        rule["args"] = args
    return rule


def seccomp_profile(arch: str | None = None) -> dict:
    """Build the container-wide OCI `linux.seccomp` object.

    Args:
        arch (`str`, *optional*):
            A `platform.machine()` value. Defaults to the running machine.

    Returns:
        `dict`: An OCI runtime-spec `LinuxSeccomp` object.
    """
    syscalls = [
        _deny(list(DENIED_SYSCALLS)),
        # glibc falls back to clone(2) on ENOSYS, where the flag filter applies.
        _deny(["clone3"], errno=ENOSYS),
    ]
    for flag in CLONE_NAMESPACE_FLAGS.values():
        syscalls.append(
            _deny(
                ["clone"],
                args=[
                    {
                        "index": 0,
                        "value": flag,
                        "valueTwo": flag,
                        "op": "SCMP_CMP_MASKED_EQ",
                    }
                ],
            )
        )
    syscalls.append(
        _deny(
            ["socket"],
            args=[{"index": 0, "value": AF_PACKET, "op": "SCMP_CMP_EQ"}],
        )
    )
    return {
        "defaultAction": "SCMP_ACT_ALLOW",
        "architectures": seccomp_architectures(arch),
        "syscalls": syscalls,
    }
