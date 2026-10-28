"""Exercise the trusted device inventory against local files and injected read errors."""

import builtins
import errno
import io
import json
import os
import sys

import pytest
from openenv.validation.providers._pidns import PROBE_SCRIPT


@pytest.fixture
def device_probe(tmp_path, monkeypatch, capsys):
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "shm").mkdir()
    identity = dev.stat().st_dev
    device = f"{os.major(identity)}:{os.minor(identity)}"
    mountinfo = (
        "1 0 0:999 / / ro - overlay overlay ro\n"
        f"2 1 {device} / /dev rw - tmpfs tmpfs rw\n"
        "3 2 0:998 / /dev/shm rw - tmpfs tmpfs rw\n"
    )
    original_open = builtins.open
    original_os_open = os.open
    original_listdir = os.listdir
    monkeypatch.setattr(os, "geteuid", lambda: dev.stat().st_uid + 1)

    def open_file(path, *args, **kwargs):
        if path == "/proc/1/mountinfo":
            return io.StringIO(mountinfo)
        return original_open(path, *args, **kwargs)

    def open_fd(path, flags, *args, **kwargs):
        # Denied creation isolates the regular-file and incomplete-inventory cases,
        # including when the test process itself runs as root.
        if os.path.basename(path).startswith(".openenv-containment-"):
            writable = dev / "writable"
            parent_fd = kwargs.get("dir_fd")
            if not (
                writable.exists()
                and parent_fd is not None
                and os.fstat(parent_fd).st_ino == writable.stat().st_ino
            ):
                raise PermissionError(errno.EACCES, "denied")
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_file)
    monkeypatch.setattr(os, "open", open_fd)
    monkeypatch.setattr(
        os, "listdir", lambda path: [] if path == "/proc" else original_listdir(path)
    )
    monkeypatch.setattr(sys, "argv", ["probe"])

    def run(separate_shm=True, alias=None):
        nonlocal mountinfo
        if not separate_shm:
            mountinfo = "\n".join(mountinfo.splitlines()[:2]) + "\n"
        if alias:
            mountinfo += f"4 1 {device} / {alias} rw - tmpfs tmpfs rw\n"
        script = PROBE_SCRIPT.replace(
            'root = "/proc/1/root"', f"root = {str(tmp_path)!r}"
        )
        namespace = {}
        exec(compile(script, "<pidns-probe>", "exec"), namespace)
        measured = json.loads(capsys.readouterr().out)
        return measured, measured["device_storage"][device]

    return dev, run


def test_device_probe_finds_regular_file_without_changing_it(device_probe):
    dev, run = device_probe
    (dev / "nested").mkdir()
    target = dev / "nested" / "data"
    target.write_bytes(b"preserve data")
    measured, reading = run()
    assert reading == {"complete": True, "writable": True}
    assert measured["writable"] == ["/dev/nested/data"]
    assert target.read_bytes() == b"preserve data"


def test_device_probe_skips_symlinks_and_separate_allotted_mount(device_probe):
    dev, run = device_probe
    (dev / "shm" / "data").write_bytes(b"allotted")
    (dev / "link").symlink_to(dev / "shm", target_is_directory=True)
    (dev / "file-link").symlink_to(dev / "shm" / "data")
    (dev / "dangling").symlink_to("missing")
    measured, reading = run()
    assert reading == {"complete": True, "writable": False}
    assert measured["writable"] == []


def test_device_probe_bounds_witnesses_without_losing_positive_evidence(device_probe):
    dev, run = device_probe
    for index in range(12):
        (dev / str(index)).write_bytes(b"data")
    measured, reading = run()
    assert reading == {"complete": True, "writable": True}
    assert len(measured["writable"]) == 8


def test_allotted_directory_without_separate_mount_is_inventoried(device_probe):
    dev, run = device_probe
    (dev / "shm" / "data").write_bytes(b"storage")
    measured, reading = run(separate_shm=False)
    assert reading == {"complete": True, "writable": True}
    assert measured["writable"] == ["/dev/shm/data"]


def test_device_probe_never_recurses_proc_directory_alias(device_probe, monkeypatch):
    dev, run = device_probe
    alias = dev.parent / "proc" / "alias"
    alias.mkdir(parents=True)
    original_scandir = os.scandir

    def scandir(fd):
        assert os.fstat(fd).st_ino != alias.stat().st_ino
        return original_scandir(fd)

    monkeypatch.setattr(os, "scandir", scandir)
    _, reading = run(alias="/proc/alias")
    assert reading == {"complete": False, "writable": False}


def test_successful_write_survives_probe_cleanup_failure(device_probe, monkeypatch):
    dev, run = device_probe
    (dev / "writable").mkdir()

    def unlink(*args, **kwargs):
        raise OSError(errno.EIO, "cleanup failed")

    monkeypatch.setattr(os, "unlink", unlink)
    measured, reading = run()
    assert reading == {"complete": False, "writable": True}
    assert measured["writable"] == ["/dev/writable"]


@pytest.mark.parametrize("failure", ["directory", "file", "race", "depth", "exhausted"])
def test_device_probe_incomplete_inventory_is_explicit(
    device_probe, monkeypatch, failure
):
    dev, run = device_probe
    if failure == "exhausted":
        for index in range(1025):
            (dev / str(index)).symlink_to("missing")
    elif failure == "depth":
        path = dev
        for _ in range(34):
            path = path / "nested"
            path.mkdir()
    elif failure == "directory":
        original_scandir = os.scandir

        def unreadable(path):
            if isinstance(path, int):
                raise PermissionError(errno.EACCES, "unreadable directory")
            return original_scandir(path)

        monkeypatch.setattr(os, "scandir", unreadable)
    else:
        (dev / "data").write_bytes(b"data")
        original_open = os.open

        def io_error(path, *args, **kwargs):
            if path == "data":
                raise OSError(
                    errno.ELOOP if failure == "race" else errno.EIO,
                    "unreadable file access",
                )
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", io_error)
    _, reading = run()
    assert reading == {"complete": False, "writable": False}


@pytest.mark.parametrize("storage", ["directory", "file"])
@pytest.mark.parametrize("denial", [errno.EACCES, errno.EPERM])
def test_owner_changeable_denial_is_incomplete(
    device_probe, monkeypatch, storage, denial
):
    dev, run = device_probe
    path = dev / "subject-owned"
    if storage == "directory":
        path.mkdir(mode=0o500)
    else:
        path.write_bytes(b"preserve data")
        path.chmod(0o400)
    original = os.open

    def denied(target, flags, *args, **kwargs):
        if target == path.name and storage == "file":
            raise PermissionError(denial, "denied")
        return original(target, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", denied)
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid)
    mode = path.stat().st_mode
    _, reading = run()
    assert reading == {"complete": False, "writable": False}
    assert path.stat().st_mode == mode
    if storage == "file":
        assert path.read_bytes() == b"preserve data"
