# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import pytest
from openenv.core.openenvd import preflight as pf


def _cgroup_root(tmp_path, writable=True):
    root = tmp_path / "cg"
    root.mkdir()
    control = root / "cgroup.subtree_control"
    control.write_text("")
    if not writable:
        control.chmod(0o444)
    return root


def test_off_linux_isolation_is_off_with_one_reason(monkeypatch):
    monkeypatch.setattr(pf.sys, "platform", "darwin")
    result = pf.preflight()
    assert result.enabled is False
    assert result.reasons == ["not Linux: no namespaces or cgroups"]


def test_every_failure_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(pf.sys, "platform", "linux")
    monkeypatch.setattr(pf.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(pf, "check_namespaces", lambda: "cannot create user namespaces")
    monkeypatch.setattr(pf, "find_runtime", lambda oci: None)
    result = pf.preflight(tmp_path / "nowhere", "crun")
    assert result.enabled is False
    text = "\n".join(result.reasons)
    for expected in ("not running as root", "user namespaces", "no cgroup v2", "crun"):
        assert expected in text
    assert all("--privileged" in r for r in result.reasons[:3])


def test_all_checks_passing_enables_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(pf.sys, "platform", "linux")
    monkeypatch.setattr(pf.os, "geteuid", lambda: 0)
    monkeypatch.setattr(pf, "check_namespaces", lambda: None)
    monkeypatch.setattr(pf, "find_runtime", lambda oci: "/usr/bin/crun")
    result = pf.preflight(_cgroup_root(tmp_path), "crun")
    assert (result.enabled, result.reasons, result.runtime) == (
        True,
        [],
        "/usr/bin/crun",
    )


def test_cgroup_root_checks(tmp_path):
    assert pf.check_cgroups(tmp_path / "missing").startswith("no cgroup v2")
    assert pf.check_cgroups(_cgroup_root(tmp_path)) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write read-only files")
def test_read_only_subtree_control_is_refused(tmp_path):
    assert pf.check_cgroups(_cgroup_root(tmp_path, writable=False)) == (
        "cgroup.subtree_control is read-only"
    )


def test_find_runtime_prefers_the_named_one(monkeypatch):
    found = {"runc": "/usr/bin/runc", "crun": "/usr/bin/crun"}
    monkeypatch.setattr(pf.shutil, "which", found.get)
    assert pf.find_runtime("runc") == "/usr/bin/runc"
    assert pf.find_runtime("runsc") == "/usr/bin/crun"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux")
def test_namespace_check_runs_in_a_child():
    result = pf.check_namespaces()
    assert result is None or "namespaces" in result
