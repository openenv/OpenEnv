# SPDX-License-Identifier: BSD-3-Clause

import json
import stat

from openenv.core.openenvd.contract import Resources, ZoneKind
from openenv.core.openenvd.oci import (
    build_spec,
    ContainerPlan,
    MountPlan,
    SHIM_BOOTSTRAP,
    SHIM_DIR,
    write_bundle,
)

# A hand-written subset of the OCI runtime-spec 1.0.2 schema: required keys and types.
_SCHEMA = {
    "ociVersion": str,
    "root": {"path": str},
    "process": {"args": list, "cwd": str, "user": {"uid": int, "gid": int}},
    "mounts": list,
    "linux": {"namespaces": list},
}


def _check_schema(obj, schema, where="config"):
    for key, expected in schema.items():
        assert key in obj, f"{where}.{key} missing"
        if isinstance(expected, dict):
            _check_schema(obj[key], expected, f"{where}.{key}")
        else:
            assert isinstance(obj[key], expected), f"{where}.{key} wrong type"
    for mount in obj.get("mounts", []):
        assert isinstance(mount["destination"], str)
        assert mount["destination"].startswith("/")
    for ns in obj.get("linux", {}).get("namespaces", []):
        assert ns["type"] in {"pid", "network", "mount", "ipc", "uts", "user", "cgroup"}


def _plan(**overrides):
    fields = dict(
        name="env",
        zone=ZoneKind.AGENT,
        argv=["python", "-m", "server"],
        env={"FOO": "bar"},
        rootfs="/srv/rootfs",
        uid=200000,
        gid=200000,
        hostname="env",
        cgroups_path="/zones/agent/env",
        mounts=[
            MountPlan("/run/u/sock/env", "/run/openenvd"),
            MountPlan(
                "/var/obs",
                "/obs",
                options=["rbind", "ro", "nosymfollow", "nodev", "nosuid", "noexec"],
            ),
        ],
        resources=Resources(memory_mb=512, pids=128, cpu=1.5),
        shim_spec_path="/run/openenvd/shim.json",
    )
    fields.update(overrides)
    return ContainerPlan(**fields)


def test_spec_matches_schema_subset():
    spec = build_spec(_plan())
    _check_schema(spec, _SCHEMA)
    assert spec["ociVersion"] == "1.0.2"
    json.loads(json.dumps(spec))


def test_process_runs_shim_unprivileged():
    proc = build_spec(_plan(python="/opt/py/bin/python3"))["process"]
    assert proc["args"] == [
        "/opt/py/bin/python3",
        "-I",
        "-c",
        SHIM_BOOTSTRAP,
        SHIM_DIR,
        "--spec",
        "/run/openenvd/shim.json",
    ]
    assert proc["noNewPrivileges"] is True
    assert proc["user"] == {"uid": 1000, "gid": 1000}
    assert set(proc["capabilities"]) == {
        "bounding",
        "effective",
        "inheritable",
        "permitted",
        "ambient",
    }
    assert all(v == [] for v in proc["capabilities"].values())
    rlimits = {r["type"]: r for r in proc["rlimits"]}
    assert rlimits["RLIMIT_NOFILE"]["hard"] == 4096
    assert rlimits["RLIMIT_CORE"]["hard"] == 0
    assert "FOO=bar" in proc["env"]
    assert any(e.startswith("PATH=") for e in proc["env"])


def test_namespaces_and_id_maps_with_userns():
    spec = build_spec(_plan())
    types = {ns["type"] for ns in spec["linux"]["namespaces"]}
    assert types == {"pid", "network", "ipc", "uts", "mount", "cgroup", "user"}
    assert spec["linux"]["uidMappings"] == [
        {"containerID": 0, "hostID": 200000, "size": 65536}
    ]
    assert spec["linux"]["gidMappings"][0]["hostID"] == 200000


def test_without_userns_runs_as_host_uid():
    spec = build_spec(_plan(userns=False))
    types = {ns["type"] for ns in spec["linux"]["namespaces"]}
    assert "user" not in types and "network" in types
    assert "uidMappings" not in spec["linux"]
    assert spec["process"]["user"] == {"uid": 200000, "gid": 200000}


def test_root_hostname_and_mounts():
    spec = build_spec(_plan())
    assert spec["root"] == {"path": "/srv/rootfs", "readonly": True}
    assert spec["hostname"] == "env"
    dests = [m["destination"] for m in spec["mounts"]]
    assert dests[:7] == [
        "/proc",
        "/dev",
        "/dev/pts",
        "/dev/shm",
        "/dev/mqueue",
        "/sys",
        "/tmp",
    ]
    assert dests[-2:] == ["/run/openenvd", "/obs"]
    obs = spec["mounts"][-1]
    assert obs["type"] == "bind" and obs["source"] == "/var/obs"
    assert obs["options"] == ["rbind", "ro", "nosymfollow", "nodev", "nosuid", "noexec"]
    tmp = spec["mounts"][6]
    assert "nosuid" in tmp["options"] and "nodev" in tmp["options"]


def test_resources_cgroups_and_hardening():
    linux = build_spec(_plan())["linux"]
    res = linux["resources"]
    assert res["memory"] == {"limit": 512 * 1024 * 1024}
    assert res["pids"] == {"limit": 128}
    assert res["cpu"] == {"quota": 150000, "period": 100000}
    assert linux["cgroupsPath"] == "/zones/agent/env"
    assert linux["seccomp"]["defaultAction"] == "SCMP_ACT_ALLOW"
    assert linux["sysctl"] == {"net.ipv4.ip_unprivileged_port_start": "0"}
    for path in ("/proc/kcore", "/proc/keys", "/proc/timer_list", "/sys/firmware"):
        assert path in linux["maskedPaths"]
    assert "/proc/sys" in linux["readonlyPaths"]


def test_unlimited_resources_are_omitted():
    res = build_spec(_plan(resources=Resources()))["linux"]["resources"]
    assert not {"memory", "pids", "cpu"} & set(res)


def test_write_bundle(tmp_path):
    path = write_bundle(tmp_path / "bundle", _plan())
    assert path == tmp_path / "bundle" / "config.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == build_spec(_plan())
