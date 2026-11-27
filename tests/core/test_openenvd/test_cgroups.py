# SPDX-License-Identifier: BSD-3-Clause

"""CgroupTree against a fake cgroupfs, plus one real-kernel test on Linux."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from openenv.core.openenvd.cgroups import CgroupError, CgroupTree
from openenv.core.openenvd.contract import Resources


def fake_group(path: Path, controllers: str = "cpu memory pids") -> Path:
    """Make `path` look like a cgroup v2 directory."""
    path.mkdir(parents=True, exist_ok=True)
    files = {
        "cgroup.controllers": controllers,
        "cgroup.subtree_control": "",
        "cgroup.procs": "",
        "cgroup.events": "populated 0\nfrozen 0\n",
        "cgroup.freeze": "0",
    }
    for name, text in files.items():
        (path / name).write_text(text)
    return path


@pytest.fixture
def tree(tmp_path: Path, monkeypatch) -> CgroupTree:
    real_rmdir = Path.rmdir

    def rmdir(self: Path) -> None:
        # cgroupfs removes a group's interface files with it; a plain dir doesn't.
        for f in self.iterdir():
            if f.is_file():
                f.unlink()
        real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", rmdir)
    fake_group(tmp_path / "cg")
    return CgroupTree(tmp_path / "cg")


def test_path_refuses_escape(tree: CgroupTree):
    assert tree.path("zones/agent") == tree.root / "zones" / "agent"
    assert tree.path("/zones/agent/") == tree.root / "zones" / "agent"
    assert tree.path("") == tree.root
    with pytest.raises(CgroupError):
        tree.path("zones/../../etc")


def test_create_enables_controllers_on_every_ancestor(tree: CgroupTree):
    fake_group(tree.root / "zones")
    fake_group(tree.root / "zones" / "agent", controllers="memory pids")
    created = tree.create("zones/agent/env")
    assert created.is_dir()

    def sc(rel: str) -> str:
        return tree.path(rel).joinpath("cgroup.subtree_control").read_text()

    assert sc("") == "+cpu +memory +pids"
    assert sc("zones") == "+cpu +memory +pids"
    assert sc("zones/agent") == "+memory +pids"  # only what the level offers
    assert not (created / "cgroup.subtree_control").exists()  # leaf left alone


def test_create_is_idempotent(tree: CgroupTree):
    tree.create("zones")
    (tree.root / "cgroup.subtree_control").write_text("cpu memory pids")
    tree.create("zones")
    assert (tree.root / "cgroup.subtree_control").read_text() == "cpu memory pids"


def test_adopt_self_moves_root_procs_then_enables(tree: CgroupTree, monkeypatch):
    (tree.root / "cgroup.procs").write_text("1\n42\n")
    moved: list[tuple[int, str]] = []

    def kernel_move(pid: int, rel: str) -> None:
        # Simulate the kernel: a moved pid leaves the root.
        moved.append((pid, rel))
        procs = [p for p in tree._procs(tree.root) if p != pid]
        (tree.root / "cgroup.procs").write_text("\n".join(map(str, procs)))

    monkeypatch.setattr(tree, "move", kernel_move)
    tree.adopt_self()
    assert moved == [(1, "control"), (42, "control")]
    assert (tree.root / "control").is_dir()
    assert (tree.root / "cgroup.subtree_control").read_text() == "+cpu +memory +pids"


def test_set_limits_writes_kernel_units(tree: CgroupTree):
    group = fake_group(tree.root / "env")
    (group / "memory.swap.max").write_text("max")
    tree.set_limits("env", Resources(memory_mb=512, pids=64, cpu=1.5))
    assert (group / "memory.max").read_text() == str(512 * 1024 * 1024)
    assert (group / "memory.swap.max").read_text() == "0"
    assert (group / "pids.max").read_text() == "64"
    assert (group / "cpu.max").read_text() == "150000 100000"


def test_set_limits_none_means_max_and_skips_missing_swap(tree: CgroupTree):
    group = fake_group(tree.root / "env")
    tree.set_limits("env", Resources())
    assert (group / "memory.max").read_text() == "max"
    assert (group / "pids.max").read_text() == "max"
    assert (group / "cpu.max").read_text() == "max 100000"
    assert not (group / "memory.swap.max").exists()


def test_move_and_recursive_pids(tree: CgroupTree):
    fake_group(tree.root / "zones" / "agent" / "env")
    fake_group(tree.root / "zones" / "agent" / "harness")
    tree.move(101, "zones/agent/env")
    tree.move(202, "zones/agent/harness")
    assert sorted(tree.pids("zones/agent")) == [101, 202]
    assert tree.pids("zones/agent/env") == [101]
    assert tree.pids("missing") == []


def test_move_into_missing_group_raises(tree: CgroupTree):
    with pytest.raises(CgroupError):
        tree.move(1, "nope/deeper")


async def test_freeze_and_wait_frozen(tree: CgroupTree):
    group = fake_group(tree.root / "agent")
    tree.freeze("agent")
    assert (group / "cgroup.freeze").read_text() == "1"
    assert await tree.wait_frozen("agent", timeout=0.05) is False

    async def kernel_freezes():
        await asyncio.sleep(0.03)
        (group / "cgroup.events").write_text("populated 1\nfrozen 1\n")

    task = asyncio.create_task(kernel_freezes())
    assert await tree.wait_frozen("agent", timeout=1.0) is True
    await task
    tree.thaw("agent")
    assert (group / "cgroup.freeze").read_text() == "0"


def test_kill_uses_cgroup_kill_when_available(tree: CgroupTree, monkeypatch):
    group = fake_group(tree.root / "agent")
    (group / "cgroup.kill").write_text("0")
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("must not signal"))
    assert tree.kill("agent") is True
    assert (group / "cgroup.kill").read_text() == "1"


def test_kill_fallback_sweeps_until_empty(tree: CgroupTree, monkeypatch):
    group = fake_group(tree.root / "agent")
    child = fake_group(group / "env")
    (group / "cgroup.procs").write_text("10\n")
    (child / "cgroup.procs").write_text("11\n")
    killed: list[int] = []
    forks = iter([12])  # 11 forks 12 before dying

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)
        if pid == 11:
            (child / "cgroup.procs").write_text(f"{next(forks, '')}\n")
        else:
            for g in (group, child):
                procs = [p for p in tree._procs(g) if p != pid]
                (g / "cgroup.procs").write_text("\n".join(map(str, procs)))
        if pid == 10:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)
    assert tree.kill("agent") is False
    assert set(killed) == {10, 11, 12}
    assert tree.pids("agent") == []


def test_kill_missing_group(tree: CgroupTree):
    assert tree.kill("missing") is True


async def test_wait_empty(tree: CgroupTree):
    group = fake_group(tree.root / "agent")
    (group / "cgroup.events").write_text("populated 1\nfrozen 0\n")
    assert await tree.wait_empty("agent", timeout=0.05) is False

    async def last_exit():
        await asyncio.sleep(0.03)
        (group / "cgroup.events").write_text("populated 0\nfrozen 0\n")

    task = asyncio.create_task(last_exit())
    assert await tree.wait_empty("agent", timeout=1.0) is True
    await task
    assert await tree.wait_empty("missing", timeout=0.0) is True


def test_stats_tolerates_missing_files(tree: CgroupTree):
    group = fake_group(tree.root / "agent")
    assert tree.stats("agent") == {
        "memory_current": None,
        "memory_events": {},
        "pids_current": None,
        "cpu_usage_usec": None,
    }
    (group / "memory.current").write_text("4096\n")
    (group / "memory.events").write_text("low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\n")
    (group / "pids.current").write_text("3\n")
    (group / "cpu.stat").write_text("usage_usec 1234\nuser_usec 1000\n")
    stats = tree.stats("agent")
    assert stats["memory_current"] == 4096
    assert stats["memory_events"]["oom_kill"] == 1
    assert stats["pids_current"] == 3
    assert stats["cpu_usage_usec"] == 1234


def test_remove_children_first_and_ignores_missing(tree: CgroupTree):
    fake_group(tree.root / "zones" / "agent" / "env")
    fake_group(tree.root / "zones" / "agent" / "harness")
    tree.remove("zones/agent")
    assert not (tree.root / "zones" / "agent").exists()
    assert (tree.root / "zones").is_dir()
    tree.remove("zones/agent")  # already gone


def test_remove_busy_raises(tree: CgroupTree, monkeypatch):
    fake_group(tree.root / "agent")

    def busy(self):
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(Path, "rmdir", busy)
    with pytest.raises(CgroupError, match="still populated"):
        tree.remove("agent")


def _real_cgroup_root() -> Path | None:
    root = Path("/sys/fs/cgroup")
    if sys.platform != "linux" or os.geteuid() != 0:
        return None
    if not os.access(root / "cgroup.controllers", os.R_OK):
        return None
    if not os.access(root / "cgroup.subtree_control", os.W_OK):
        return None
    return root


@pytest.mark.skipif(_real_cgroup_root() is None, reason="needs root and cgroup v2")
async def test_real_subtree_lifecycle():
    real = _real_cgroup_root()
    # Work under a fresh child: it holds no processes, so controllers can be
    # delegated below it even when the namespace root itself is populated.
    top = real / f"openenvd-test-{os.getpid()}"
    top.mkdir()
    tree = CgroupTree(top)
    proc = None
    try:
        if "pids" not in (top / "cgroup.controllers").read_text().split():
            # In a throwaway container the namespace root holds this process, so
            # it can't delegate controllers until processes move out of it.
            # Never do this on a real host: it moves every process in the root.
            if os.environ.get("OPENENVD_TEST_ADOPT_CGROUP_ROOT") != "1":
                pytest.skip("pids controller not delegated to this cgroup")
            CgroupTree(real).adopt_self("pytest")
        tree.create("zones/agent")
        tree.set_limits("zones/agent", Resources(pids=16))
        assert tree.path("zones/agent").joinpath("pids.max").read_text() == "16\n"
        proc = subprocess.Popen(["sleep", "60"])
        tree.move(proc.pid, "zones/agent")
        assert tree.pids("zones") == [proc.pid]
        membership = Path(f"/proc/{proc.pid}/cgroup").read_text()
        assert membership.strip().endswith(f"{top.name}/zones/agent")
        tree.kill("zones/agent")
        proc.wait(timeout=5)
        assert await tree.wait_empty("zones/agent", timeout=5.0)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        CgroupTree(real).remove(top.name)
    assert not top.exists()
