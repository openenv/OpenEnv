# SPDX-License-Identifier: BSD-3-Clause

"""`openenv check enforcement`: test a live unit instead of trusting its configuration.

For each zone, openenvd starts a throwaway container the way it starts that
zone's real containers and runs a probe that tries what the zone must not be able to
do. Any success is a failure of the check.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .contract import ContainerSpec, Resources, ZoneKind

PROBE = textwrap.dedent(
    r"""
    import json, os, socket, sys
    out = {}
    def attempt(name, fn):
        try:
            fn()
            out[name] = "allowed"
        except Exception as exc:
            out[name] = "denied:" + type(exc).__name__

    def read_hidden(path):
        if not os.listdir(path):
            raise PermissionError("masked")

    for i, path in enumerate(json.loads(sys.argv[1])):
        attempt(f"read_hidden_{i}", lambda p=path: read_hidden(p))
    def connect(host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.5)
        try:
            s.connect((host, port))
        finally:
            s.close()
    for host, port in json.loads(sys.argv[2]):
        attempt(f"connect_{host}_{port}", lambda h=host, p=port: connect(h, p))
    attempt("write_root", lambda: open("/probe-write", "w").write("x"))
    attempt("write_usr", lambda: open("/usr/probe-write", "w").write("x"))
    out["visible_pids"] = len([p for p in os.listdir("/proc") if p.isdigit()])
    out["uid"] = os.getuid()
    def fork_many():
        pids = []
        try:
            for _ in range(int(sys.argv[3]) + 16):
                pid = os.fork()
                if pid == 0:
                    import time; time.sleep(2); os._exit(0)
                pids.append(pid)
        finally:
            for pid in pids:
                try:
                    os.kill(pid, 9); os.waitpid(pid, 0)
                except OSError:
                    pass
    attempt("fork_past_pids_max", fork_many)
    with open("/run/openenvd/probe.json", "w") as f:
        json.dump(out, f)
    """
)

_PIDS_LIMIT = 32
_TARGETS = [("1.1.1.1", 53), ("8.8.8.8", 443), ("169.254.169.254", 80)]


@dataclass
class ZoneCheck:
    zone: ZoneKind
    results: dict[str, Any]
    failures: list[str] = field(default_factory=list)


@dataclass
class CheckReport:
    """What a live unit actually enforces."""

    zones: list[ZoneCheck]

    @property
    def failures(self) -> list[str]:
        return [f"{z.zone.value}: {f}" for z in self.zones for f in z.failures]

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": not self.failures,
            "zones": {
                z.zone.value: {"results": z.results, "failures": z.failures}
                for z in self.zones
            },
        }


def evaluate(zone: ZoneKind, results: dict[str, Any]) -> list[str]:
    """Turn one zone's probe results into a list of violated expectations."""
    failures = []
    for key, value in results.items():
        allowed = value == "allowed"
        if key.startswith("read_hidden") and allowed:
            failures.append(f"{key}: control-plane state is readable")
        elif key.startswith("connect_") and allowed:
            failures.append(f"{key}: reached a host outside the unit")
        elif key in ("write_root", "write_usr") and allowed:
            failures.append(f"{key}: root filesystem is writable")
        elif key == "fork_past_pids_max" and allowed:
            failures.append("fork_past_pids_max: pids.max is not enforced")
    if results.get("visible_pids", 0) > 4:
        failures.append("visible_pids: processes outside the container are visible")
    if results.get("uid") == 0:
        failures.append("uid: principal runs as root inside its namespace")
    return failures


async def run_check(unit: Any) -> CheckReport:
    """Run a probe container in every zone of `unit` and evaluate the results.

    Args:
        unit ([`Unit`]):
            A unit that is not running an episode.

    Returns:
        `CheckReport`: Per-zone results and every violated expectation.
    """
    from .unit import _Container

    checks: list[ZoneCheck] = []
    hidden = getattr(unit.launcher, "hidden", None) or unit.paths.hidden
    unit.episode_id = "check"
    for index, kind in enumerate(ZoneKind):
        spec = ContainerSpec(
            argv=[
                unit.python,
                "-c",
                PROBE,
                json.dumps(hidden),
                json.dumps(_TARGETS),
                str(_PIDS_LIMIT),
            ],
            resources=Resources(pids=_PIDS_LIMIT),
        )
        c = _Container(kind, f"probe-{kind.value}", spec, 1000 + index)
        sock = unit._sock_dir(c)
        if sock.exists():
            shutil.rmtree(sock)
        sock.mkdir(parents=True)
        sock.parent.chmod(0o755)
        sock.chmod(0o755)
        unit._chown(sock, c)
        (unit._episode_dir() / "workspace" / "merged").mkdir(
            parents=True, exist_ok=True
        )
        (sock / "hosts").write_text("127.0.0.1 localhost\n")
        out = unit._episode_dir() / "observers" / c.name / "out"
        out.mkdir(parents=True, exist_ok=True)
        unit._chown(out, c)
        unit._containers[c.key] = c
        try:
            await unit._start(c)
            await unit.launcher.wait(c.key, timeout=60)
            result_file = sock / "probe.json"
            results = (
                json.loads(result_file.read_text()) if result_file.exists() else {}
            )
        finally:
            await unit._stop(c)
            unit._containers.pop(c.key, None)
            unit.cgroups.kill(c.cgroup)
            await unit.cgroups.wait_empty(c.cgroup, 5)
            unit.cgroups.remove(c.cgroup)
        failures = evaluate(kind, results)
        if not results:
            failures.append("probe produced no result")
        checks.append(ZoneCheck(kind, results, failures))
    return CheckReport(checks)
