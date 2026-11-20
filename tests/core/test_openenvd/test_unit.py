# SPDX-License-Identifier: BSD-3-Clause

"""Unit assembly without a kernel: plans, ordering and verdicts.

The live behaviour (real containers, and the unprivileged fallback) is covered by
`integration/test_live_unit.py`, which runs in Docker.
"""

import json

import pytest
from openenv.core.openenvd.check import CheckReport, evaluate, ZoneCheck
from openenv.core.openenvd.contract import parse_manifest, Phase, ZoneKind
from openenv.core.openenvd.oci import MountPlan
from openenv.core.openenvd.unit import (
    _ensure_mountpoints,
    HARNESS_PORTS,
    PLACEHOLDER_KEY,
    SOCKET_DIR,
    Unit,
    UnitPaths,
)

MANIFEST = {
    "app": "echo_env.server.app:app",
    "privileged_assets": {"oracle": "assets/oracle.json"},
    "zones": {
        "agent": {"containers": {"env": {}, "harness": {"argv": ["harness"]}}},
        "services": {
            "containers": {
                "sim": {
                    "argv": ["sim", "--state", "/var/lib/sim"],
                    "expose": {"host": "sim.example", "port": 80},
                    "port": 8080,
                    "state": "/var/lib/sim",
                }
            }
        },
        "observers": {
            "containers": {
                "grader": {
                    "argv": ["grade"],
                    "reads": ["assets.oracle", "runner.output"],
                    "phases": ["grading"],
                    "after": "runner",
                },
                "runner": {
                    "argv": ["run"],
                    "reads": ["workspace"],
                    "phases": ["grading"],
                    "output": "/out/report.json",
                },
            }
        },
    },
}


class FakeLauncher:
    async def start(self, plan, bundle_dir):
        pass

    async def stop(self, name):
        pass

    async def wait(self, name, timeout):
        return True


def _unit(tmp_path, monkeypatch):
    paths = UnitPaths(
        state=tmp_path / "s" / "state", run=tmp_path / "s" / "run", assets=tmp_path
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "oracle.json").write_text("{}")
    unit = Unit(
        parse_manifest(MANIFEST),
        paths,
        FakeLauncher(),
        cgroups=None,
        python="/usr/bin/python3",
    )
    monkeypatch.setattr(unit, "_rootfs", lambda rootfs: rootfs)
    unit.episode_id = "ep1"
    unit._prepare_dirs()
    return unit


def _shim(unit, key):
    _, spec = unit._plan(unit._containers[key])
    return spec


def test_harness_sees_only_loopback_relays_and_a_placeholder_key(tmp_path, monkeypatch):
    unit = _unit(tmp_path, monkeypatch)
    spec = _shim(unit, "agent-harness")
    assert spec["env"]["ANTHROPIC_API_KEY"] == PLACEHOLDER_KEY
    assert spec["env"]["MCP_URL"].startswith("http://127.0.0.1:")
    ports = sorted(HARNESS_PORTS.values())
    assert sorted(f["port"] for f in spec["forwards"]) == ports


def test_env_reaches_services_through_its_own_socket_dir(tmp_path, monkeypatch):
    unit = _unit(tmp_path, monkeypatch)
    spec = _shim(unit, "agent-env")
    assert spec["forwards"] == [
        {
            "kind": "tcp_to_unix",
            "host": "127.0.0.10",
            "port": 80,
            "path": f"{SOCKET_DIR}/svc-sim.sock",
        }
    ]
    assert spec["env"]["OPENENVD_SERVICE_SIM"] == "http://sim.example:80"
    hosts = (unit._sock_dir(unit._containers["agent-env"]) / "hosts").read_text()
    assert "127.0.0.10 sim.example" in hosts
    assert "--uds" in spec["argv"] and f"{SOCKET_DIR}/env.sock" in spec["argv"]


def test_services_and_observers_cannot_connect_anywhere(tmp_path, monkeypatch):
    unit = _unit(tmp_path, monkeypatch)
    for key in ("services-sim", "observers-grader"):
        outbound = [
            f for f in _shim(unit, key)["forwards"] if f["kind"] == "tcp_to_unix"
        ]
        assert outbound == [], key


def test_observer_mounts_are_read_only_and_nosymfollow(tmp_path, monkeypatch):
    unit = _unit(tmp_path, monkeypatch)
    plan, _ = unit._plan(unit._containers["observers-grader"])
    by_dest = {m.destination: m.options for m in plan.mounts}
    for dest in ("/assets", "/inputs"):
        assert {"ro", "nosymfollow", "nodev", "nosuid", "noexec"} <= set(by_dest[dest])
    assert "ro" not in by_dest["/out"]
    assert "/workspace" not in by_dest


def test_missing_mountpoints_are_created_on_the_root(tmp_path):
    """A fresh tmpfs on /run hides the image's /run/openenvd (as in a microVM)."""
    src_dir, src_file = tmp_path / "sock", tmp_path / "hosts"
    src_dir.mkdir()
    src_file.write_text("")
    root = tmp_path / "root"
    _ensure_mountpoints(
        [MountPlan(str(src_dir), SOCKET_DIR), MountPlan(str(src_file), "/etc/hosts")],
        root,
    )
    assert (root / SOCKET_DIR.lstrip("/")).is_dir()
    assert (root / "etc/hosts").is_file()


def test_runner_runs_before_the_verdict(tmp_path, monkeypatch):
    unit = _unit(tmp_path, monkeypatch)
    observers = unit._observers_for(Phase.GRADING)
    assert [c.name for c in unit._order(observers)] == ["runner", "grader"]


def test_verdict_reports_observer_results_and_no_internals(tmp_path, monkeypatch):
    from openenv.core.openenvd.unit import Verdict

    unit = _unit(tmp_path, monkeypatch)
    assert unit.verdict_info() == {"episode_id": "ep1", "status": None}
    unit.verdict = Verdict("graded", observers={"grader": {"score": 1.0}})
    info = unit.verdict_info()
    assert info["status"] == "graded"
    assert info["observers"] == {"grader": {"score": 1.0}}
    assert str(tmp_path) not in json.dumps(info)


@pytest.mark.parametrize(
    "results, failure",
    [
        ({"read_hidden_0": "allowed"}, "read_hidden_0"),
        ({"connect_1.1.1.1_53": "allowed"}, "connect_1.1.1.1_53"),
        ({"fork_past_pids_max": "allowed"}, "fork_past_pids_max"),
        ({"visible_pids": 40}, "visible_pids"),
        ({"uid": 0}, "uid"),
    ],
)
def test_check_evaluation(results, failure):
    failures = evaluate(ZoneKind.AGENT, results)
    assert [f.split(":")[0] for f in failures] == [failure]


def test_check_report_passes_only_without_failures():
    clean = CheckReport([ZoneCheck(ZoneKind.AGENT, {"uid": 1000})])
    assert clean.to_json()["passed"] is True and clean.failures == []
    dirty = CheckReport([ZoneCheck(ZoneKind.AGENT, {}, ["visible_pids: x"])])
    assert dirty.to_json()["passed"] is False
    assert dirty.failures == ["agent: visible_pids: x"]
