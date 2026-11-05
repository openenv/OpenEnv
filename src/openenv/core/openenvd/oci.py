# SPDX-License-Identifier: BSD-3-Clause

"""OCI runtime bundles for openenvd containers.

Every container runs the openenvd shim as its PID 1 in fresh pid, mount, net,
ipc, uts and cgroup namespaces (plus a user namespace when allowed), on a
read-only root, with no capabilities and the container-wide seccomp profile.
The shim then `exec`s the principal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .contract import Resources, ZoneKind
from .seccomp import seccomp_profile

OCI_VERSION = "1.0.2"
CPU_PERIOD_US = 100_000
USERNS_SIZE = 65536
PRINCIPAL_ID = 1000
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

SHIM_DIR = str(Path(__file__).resolve().parent)
# Loads only the shim and its two stdlib-only helpers, under a private package
# name, so the `openenv` package (and the threads its imports start) never
# loads in a process that forks the principal.
SHIM_BOOTSTRAP = (
    "import sys,types;"
    "p=types.ModuleType('openenvd_shim');p.__path__=[sys.argv.pop(1)];"
    "sys.modules['openenvd_shim']=p;"
    "from openenvd_shim.shim import main;sys.exit(main())"
)

MASKED_PATHS = [
    "/proc/acpi",
    "/proc/asound",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/sys/firmware",
    "/sys/devices/virtual/powercap",
]
READONLY_PATHS = [
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
]


@dataclass
class MountPlan:
    """One mount added after the default mounts.

    Attributes:
        source (`str`):
            Host path (or filesystem source for non-bind mounts).
        destination (`str`):
            Absolute path inside the container.
        type (`str`, *optional*, defaults to `"bind"`):
            Filesystem type.
        options (`list[str]`, *optional*, defaults to `["rbind", "nosuid", "nodev"]`):
            Mount options, passed through unchanged (e.g. `ro`, `nosymfollow`).
    """

    source: str
    destination: str
    type: str = "bind"
    options: list[str] = field(default_factory=lambda: ["rbind", "nosuid", "nodev"])

    def to_oci(self) -> dict:
        return {
            "destination": self.destination,
            "type": self.type,
            "source": self.source,
            "options": list(self.options),
        }


@dataclass
class ContainerPlan:
    """Everything openenvd decided about one container.

    Attributes:
        name (`str`):
            Container name, unique within the unit.
        zone ([`ZoneKind`]):
            The container's zone.
        argv (`list[str]`):
            The principal's command. Carried in the shim spec, not the OCI config.
        env (`dict[str, str]`):
            Environment of the shim process.
        rootfs (`str`):
            Directory used as the read-only root.
        uid (`int`):
            First host UID of the container's range.
        gid (`int`):
            First host GID of the container's range.
        hostname (`str`):
            The container's hostname.
        cgroups_path (`str`):
            Absolute cgroup path within the unit's cgroup namespace, e.g. `/zones/agent/env`.
        mounts (`list[MountPlan]`):
            Mounts after the defaults, in order. Must include the socket directory.
        resources (`Resources`):
            Effective cgroup limits.
        shim_spec_path (`str`):
            Path inside the container to the shim's JSON spec.
        python (`str`, *optional*, defaults to `"/usr/bin/python3"`):
            Interpreter inside the rootfs that runs the shim.
        userns (`bool`, *optional*, defaults to `True`):
            Create a user namespace. Without one the shim runs as host `uid`/`gid`.
    """

    name: str
    zone: ZoneKind
    argv: list[str]
    env: dict[str, str]
    rootfs: str
    uid: int
    gid: int
    hostname: str
    cgroups_path: str
    mounts: list[MountPlan]
    resources: Resources
    shim_spec_path: str
    python: str = "/usr/bin/python3"
    userns: bool = True


def _default_mounts() -> list[dict]:
    return [
        {
            "destination": "/proc",
            "type": "proc",
            "source": "proc",
            "options": ["nosuid", "noexec", "nodev"],
        },
        {
            "destination": "/dev",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "strictatime", "mode=755", "size=65536k"],
        },
        {
            "destination": "/dev/pts",
            "type": "devpts",
            "source": "devpts",
            "options": [
                "nosuid",
                "noexec",
                "newinstance",
                "ptmxmode=0666",
                "mode=0620",
            ],
        },
        {
            "destination": "/dev/shm",
            "type": "tmpfs",
            "source": "shm",
            "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=65536k"],
        },
        {
            "destination": "/dev/mqueue",
            "type": "mqueue",
            "source": "mqueue",
            "options": ["nosuid", "noexec", "nodev"],
        },
        {
            "destination": "/sys",
            "type": "sysfs",
            "source": "sysfs",
            "options": ["nosuid", "noexec", "nodev", "ro"],
        },
        {
            "destination": "/tmp",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "nodev", "mode=1777"],
        },
    ]


def _resources(resources: Resources) -> dict:
    out: dict = {"devices": [{"allow": False, "access": "rwm"}]}
    if resources.memory_mb is not None:
        out["memory"] = {"limit": resources.memory_mb * 1024 * 1024}
    if resources.pids is not None:
        out["pids"] = {"limit": resources.pids}
    if resources.cpu is not None:
        out["cpu"] = {
            "quota": int(resources.cpu * CPU_PERIOD_US),
            "period": CPU_PERIOD_US,
        }
    return out


def build_spec(plan: ContainerPlan) -> dict:
    """Build an OCI runtime-spec `config.json` for a container.

    Args:
        plan ([`ContainerPlan`]):
            The container's plan.

    Returns:
        `dict`: The OCI 1.0.2 runtime config.
    """
    env = {"PATH": DEFAULT_PATH, **plan.env}
    if plan.userns:
        user = {"uid": PRINCIPAL_ID, "gid": PRINCIPAL_ID}
    else:
        user = {"uid": plan.uid, "gid": plan.gid}
    no_caps: dict[str, list[str]] = {
        k: [] for k in ("bounding", "effective", "inheritable", "permitted", "ambient")
    }

    namespaces = [
        {"type": t} for t in ("pid", "network", "ipc", "uts", "mount", "cgroup")
    ]
    linux: dict = {
        "namespaces": namespaces,
        "resources": _resources(plan.resources),
        "cgroupsPath": plan.cgroups_path,
        "seccomp": seccomp_profile(),
        "sysctl": {"net.ipv4.ip_unprivileged_port_start": "0"},
        "maskedPaths": list(MASKED_PATHS),
        "readonlyPaths": list(READONLY_PATHS),
    }
    if plan.userns:
        namespaces.append({"type": "user"})
        linux["uidMappings"] = [
            {"containerID": 0, "hostID": plan.uid, "size": USERNS_SIZE}
        ]
        linux["gidMappings"] = [
            {"containerID": 0, "hostID": plan.gid, "size": USERNS_SIZE}
        ]

    return {
        "ociVersion": OCI_VERSION,
        "process": {
            "terminal": False,
            "user": user,
            "args": [
                plan.python,
                "-I",
                "-c",
                SHIM_BOOTSTRAP,
                SHIM_DIR,
                "--spec",
                plan.shim_spec_path,
            ],
            "env": [f"{k}={v}" for k, v in env.items()],
            "cwd": "/",
            "capabilities": no_caps,
            "rlimits": [
                {"type": "RLIMIT_NOFILE", "hard": 4096, "soft": 4096},
                {"type": "RLIMIT_CORE", "hard": 0, "soft": 0},
            ],
            "noNewPrivileges": True,
        },
        "root": {"path": plan.rootfs, "readonly": True},
        "hostname": plan.hostname,
        "mounts": _default_mounts() + [m.to_oci() for m in plan.mounts],
        "annotations": {
            "org.openenv.openenvd.zone": plan.zone.value,
            "org.openenv.openenvd.container": plan.name,
        },
        "linux": linux,
    }


def write_bundle(bundle_dir: Path, plan: ContainerPlan) -> Path:
    """Write `config.json` for a container into an OCI bundle directory.

    The shim spec is not part of the bundle: openenvd writes it into the
    container's socket directory, which is mounted at `/run/openenvd`.

    Args:
        bundle_dir (`Path`):
            Bundle directory, created if missing.
        plan ([`ContainerPlan`]):
            The container's plan.

    Returns:
        `Path`: The path of the written `config.json`.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / "config.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(build_spec(plan), f, indent=2)
    os.chmod(path, 0o600)
    return path
