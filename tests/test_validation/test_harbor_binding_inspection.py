"""Trusted host checks against fake Docker replies and real local output files."""

import json
import os
import stat
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from openenv.validation.harbor import (
    _MAX_REWARD_BYTES,
    _read_output,
    DockerInspector,
    NativeEvidenceError,
)

PROJECT = "openenv-validation-" + "d" * 32 + "-nop__env"
IMAGE = "sha256:" + "a" * 64
TAG = "openenv-validation:" + "d" * 32


@pytest.fixture
def inspection(monkeypatch):
    inspector = DockerInspector()
    container = {
        "Id": "c" * 64,
        "Image": IMAGE,
        "State": {"Running": True},
        "Config": {
            "Labels": {
                "com.docker.compose.project": PROJECT,
                "com.docker.compose.service": "main",
            },
            "User": "1000",
            "WorkingDir": "/app",
            "Env": ["B=2", "A=1"],
            "Entrypoint": ["/bin/sh"],
            "Cmd": ["-c", "sleep infinity"],
        },
        "HostConfig": {
            "NetworkMode": PROJECT + "_default",
            "Memory": 512 * 1024 * 1024,
        },
        "Mounts": [
            {
                "Type": "volume",
                "Name": "owned-volume",
                "Destination": "/data",
                "RW": True,
            },
        ],
    }
    retained = {"Id": IMAGE, "Os": "linux"}
    containers = [container["Id"]]
    commands = []

    def docker(*args):
        commands.append(args)
        if args[0] == "ps":
            return "\n".join(containers)
        if args[:2] in (("network", "ls"), ("volume", "ls")):
            return ""
        if args == ("inspect", container["Id"]):
            return json.dumps([container])
        if args[:2] == ("image", "tag"):
            assert args[3] == TAG
            return ""
        if args == ("image", "inspect", TAG):
            return json.dumps([retained])
        pytest.fail(f"unexpected Docker command: {args}")

    monkeypatch.setattr(inspector, "_docker", docker)
    return inspector, container, retained, containers, commands


@pytest.mark.parametrize("retain", [False, True])
def test_capture_binds_the_image_configuration_and_anonymous_volumes(
    inspection, retain
):
    inspector, container, _, _, commands = inspection
    identity = inspector.capture(PROJECT, TAG, retain=retain)
    assert identity["container_id"] == container["Id"]
    assert identity["image_ref"] == IMAGE
    assert identity["retained_image_tag"] == TAG
    assert identity["configuration"]["network_mode"] == "compose-default"
    assert identity["owned_volumes"] == ["owned-volume"]
    assert inspector._owned_volumes[PROJECT] == ["owned-volume"]
    assert (("image", "tag", IMAGE, TAG) in commands) is retain
    container["Config"]["Env"].reverse()
    assert (
        inspector.capture(PROJECT, TAG, retain=False)["configuration_digest"]
        == identity["configuration_digest"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "multiple",
        "project",
        "service",
        "stopped",
        "mutable",
        "tag",
        "os",
        "volume",
    ],
)
def test_capture_rejects_untrusted_identity_and_binding(inspection, defect):
    inspector, container, retained, containers, _ = inspection
    if defect == "empty":
        containers.clear()
    elif defect == "multiple":
        containers.append("e" * 64)
    elif defect == "project":
        container["Config"]["Labels"]["com.docker.compose.project"] = "other"
    elif defect == "service":
        container["Config"]["Labels"]["com.docker.compose.service"] = "other"
    elif defect == "stopped":
        container["State"]["Running"] = False
    elif defect == "mutable":
        container["Image"] = "registry.example/mutable:latest"
    elif defect == "tag":
        retained["Id"] = "sha256:" + "b" * 64
    elif defect == "os":
        retained["Os"] = "windows"
    elif defect == "volume":
        container["Mounts"][0]["Name"] = "--unowned"
    reason = {
        "empty": "exactly one main container",
        "multiple": "exactly one main container",
        "project": "native main service",
        "service": "native main service",
        "stopped": "not running with an immutable image",
        "mutable": "not running with an immutable image",
        "tag": "retained image binding changed",
        "os": "retained image binding changed",
        "volume": "invalid mounted volume identity",
    }[defect]
    with pytest.raises(NativeEvidenceError, match=reason):
        inspector.capture(PROJECT, TAG)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "Image", "sha256:" + "b" * 64),
        ("Config", "User", "root"),
        ("Config", "WorkingDir", "/other"),
        ("Config", "Env", ["A=changed"]),
        ("Config", "Entrypoint", ["/other"]),
        ("Config", "Cmd", ["other"]),
        ("HostConfig", "NanoCpus", 2_000_000_000),
        ("HostConfig", "Memory", 1024),
        ("HostConfig", "MemorySwap", -1),
        ("HostConfig", "PidsLimit", 512),
        ("HostConfig", "ReadonlyRootfs", True),
        ("HostConfig", "Privileged", True),
        ("HostConfig", "CapAdd", ["ALL"]),
        ("HostConfig", "CapDrop", ["ALL"]),
        ("HostConfig", "SecurityOpt", ["no-new-privileges"]),
        ("HostConfig", "Devices", ["/dev/example"]),
        ("HostConfig", "NetworkMode", "host"),
        ("Mounts", "Type", "bind"),
        ("Mounts", "Destination", "/other"),
        ("Mounts", "RW", False),
    ],
)
def test_capture_digest_covers_effective_configuration(inspection, section, key, value):
    inspector, container, _, _, _ = inspection
    before = inspector.capture(PROJECT, TAG)
    target = container if section is None else container[section]
    if section == "Mounts":
        target = target[0]
    target[key] = value
    after = inspector.capture(PROJECT, TAG, retain=False)
    assert after["configuration_digest"] != before["configuration_digest"]


@pytest.mark.parametrize("linux", [False, True])
def test_preflight_requires_linux_and_compose(monkeypatch, linux):
    inspector = DockerInspector()
    commands = []

    def docker(*args):
        commands.append(args)
        if args == ("info", "--format", "{{json .}}"):
            return json.dumps(
                {
                    "OSType": "linux" if linux else "windows",
                    "ID": "daemon",
                    "ServerVersion": "29",
                }
            )
        assert args == ("compose", "version")
        return "Docker Compose"

    monkeypatch.setattr(inspector, "_docker", docker)
    if linux:
        assert inspector.preflight() == {
            "os": "linux",
            "daemon_id": "daemon",
            "server_version": "29",
        }
        assert commands[-1] == ("compose", "version")
    else:
        with pytest.raises(NativeEvidenceError, match="requires a Linux Docker daemon"):
            inspector.preflight()
        assert len(commands) == 1


@pytest.mark.parametrize("matches", [False, True])
def test_verify_image_checks_the_retained_tag(inspection, matches):
    inspector, _, retained, _, commands = inspection
    retained["Id"] = IMAGE if matches else "sha256:" + "b" * 64
    assert inspector.verify_image(TAG, IMAGE) is matches
    assert commands == [("image", "inspect", TAG)]


@pytest.mark.parametrize("kind", ["file", "parent", "broken"])
def test_read_output_refuses_symlinks(tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reward.json").write_bytes(b"secret")
    root = tmp_path / "trial"
    root.mkdir()
    if kind == "parent":
        (root / "verifier").symlink_to(outside, target_is_directory=True)
        path = root / "verifier/reward.json"
    else:
        path = root / "reward.json"
        path.symlink_to(outside / ("missing" if kind == "broken" else "reward.json"))
    with pytest.raises(NativeEvidenceError, match="symbolic links"):
        _read_output(path, root)


def test_read_output_refuses_a_real_fifo_without_blocking(tmp_path, monkeypatch):
    path = tmp_path / "reward.json"
    os.mkfifo(path)
    real_open = os.open

    def opened(path, flags):
        assert flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW
        return real_open(path, flags)

    monkeypatch.setattr(os, "open", opened)
    with pytest.raises(NativeEvidenceError, match="bounded regular file"):
        _read_output(path, tmp_path)


@pytest.mark.parametrize("grows_after_stat", [False, True])
def test_read_output_bounds_both_file_size_and_bytes_read(
    tmp_path, monkeypatch, grows_after_stat
):
    path = tmp_path / "reward.json"
    path.write_bytes(b"x" * (_MAX_REWARD_BYTES + 1))
    if grows_after_stat:
        actual_stat = os.fstat

        def fstat(fd):
            info = list(actual_stat(fd))
            info[stat.ST_SIZE] = _MAX_REWARD_BYTES
            return os.stat_result(info)

        monkeypatch.setattr(os, "fstat", fstat)
    reason = "exceeded its size bound" if grows_after_stat else "bounded regular file"
    with pytest.raises(NativeEvidenceError, match=reason):
        _read_output(path, tmp_path)


def test_read_output_preserves_raw_bytes_and_missing_files(tmp_path, monkeypatch):
    real_fdopen = os.fdopen

    @contextmanager
    def fdopen(fd, mode):
        with real_fdopen(fd, mode) as stream:

            def read(size=-1):
                assert size == _MAX_REWARD_BYTES + 1
                return stream.read(size)

            yield SimpleNamespace(fileno=stream.fileno, read=read)

    monkeypatch.setattr(os, "fdopen", fdopen)
    path = tmp_path / "reward.json"
    assert _read_output(path, tmp_path) is None
    raw = b'{"reward":0}\n'
    path.write_bytes(raw)
    assert _read_output(path, tmp_path) == raw
