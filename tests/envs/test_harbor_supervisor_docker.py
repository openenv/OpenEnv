"""The sandbox supervisor against Harbor's real docker backend.

Opt-in, like the other real-Docker tests:

    OPENENV_DOCKER_INTEGRATION=1 PYTHONPATH=src:envs uv run pytest \
        tests/envs/test_harbor_supervisor_docker.py -v

Needs Harbor (Python >= 3.12) and a locally cached `alpine:latest`; nothing is pulled. Every
container is found by its compose project, so nothing else on the host is touched.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.docker,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENENV_DOCKER_INTEGRATION") != "1"
        or shutil.which("docker") is None,
        reason="Set OPENENV_DOCKER_INTEGRATION=1 to run real Docker lifecycle tests",
    ),
]

pytest.importorskip("harbor")
from harbor.environments.docker.docker import (  # noqa: E402
    _sanitize_docker_compose_project_name,
)
from openenv.harbor.supervisor import SandboxSupervisor  # noqa: E402

IMAGE = os.environ.get("OPENENV_DOCKER_TEST_IMAGE", "alpine:latest")


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Starting a sandbox installs handlers in this (main) thread; keep them out of other tests."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


# Builds a real Harbor trial, starts its sandbox through the supervisor, then waits to be signalled.
CHILD = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from harbor.models.trial.config import (
        AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig,
    )
    from harbor.trial.trial import Trial
    from openenv.harbor.supervisor import get_supervisor

    async def main():
        task, trials, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
        trial = await Trial.create(TrialConfig(
            task=TaskConfig(path=task), agent=AgentConfig(name="nop"),
            environment=EnvironmentConfig(type="docker", delete=True),
            verifier=VerifierConfig(), trial_name=name, trials_dir=trials,
        ))
        get_supervisor().adopt(trial)
        await trial.agent_environment.start(force_build=False)
        print("READY", trial.agent_environment.session_id, flush=True)
        await asyncio.sleep(3600)

    asyncio.run(main())
    """
)


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("noop\n")
    (task / "task.toml").write_text("")
    (task / "environment" / "Dockerfile").write_text(
        f'FROM {IMAGE}\nCMD ["sleep", "infinity"]\n'
    )
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    return task


def containers(session_id: str) -> list[str]:
    project = _sanitize_docker_compose_project_name(session_id)
    out = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.fixture
def bystander():
    """A container this process did not start; cleanup must leave it running."""
    name = f"oe-supervisor-bystander-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, IMAGE, "sleep", "300"],
        check=True,
        capture_output=True,
    )
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_tears_down_owned_sandbox_only(sig, task_dir, tmp_path, bystander):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    name = f"oe-sup-{uuid.uuid4().hex[:8]}"
    child = subprocess.Popen(
        [sys.executable, str(script), str(task_dir), str(tmp_path / "trials"), name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    session_id = None
    try:
        for line in child.stdout:
            if line.startswith("READY "):
                session_id = line.split()[1]
                break
        assert session_id, "child exited before its sandbox was up"
        assert containers(session_id), "sandbox should be running before the signal"

        child.send_signal(sig)
        child.wait(timeout=120)

        # The signal still reached its original handler: SIG_DFL for SIGTERM, and for SIGINT
        # asyncio's KeyboardInterrupt, which CPython turns into death-by-SIGINT on exit.
        assert child.returncode == -sig
        assert containers(session_id) == []
        running = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", bystander],
            capture_output=True,
            text=True,
        )
        assert running.stdout.strip() == "true"
    finally:
        if child.poll() is None:
            child.kill()
        if session_id:
            for container in containers(session_id):
                subprocess.run(["docker", "rm", "-f", container], capture_output=True)


async def _trial(task_dir: Path, trials: Path):
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.trial.trial import Trial

    return await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_dir),
            agent=AgentConfig(name="nop"),
            environment=EnvironmentConfig(type="docker", delete=True),
            verifier=VerifierConfig(),
            trial_name=f"oe-sup-{uuid.uuid4().hex[:8]}",
            trials_dir=trials,
        )
    )


async def test_creation_limit_serialises_real_starts(task_dir, tmp_path):
    sup = SandboxSupervisor(max_starts=1)
    windows: list[tuple[float, float]] = []
    trials = [await _trial(task_dir, tmp_path / "trials") for _ in range(3)]
    for trial in trials:
        real = trial.agent_environment.start

        async def timed(force_build: bool, real=real) -> None:
            began = time.monotonic()
            await real(force_build=force_build)
            windows.append((began, time.monotonic()))

        trial.agent_environment.start = timed
        sup.adopt(trial)
    try:
        await asyncio.gather(
            *(t.agent_environment.start(force_build=False) for t in trials)
        )
        assert sup.owned == 3
    finally:
        await sup.aclose()

    windows.sort()
    assert all(prev[1] <= nxt[0] for prev, nxt in zip(windows, windows[1:]))
    assert sup.owned == 0
    for trial in trials:
        assert containers(trial.agent_environment.session_id) == []


async def test_cancel_during_creation_leaves_no_container(task_dir, tmp_path):
    sup = SandboxSupervisor()
    trial = await _trial(task_dir, tmp_path / "trials")
    sup.adopt(trial)
    env = trial.agent_environment

    starting = asyncio.ensure_future(env.start(force_build=False))
    # Cancel once `docker compose up` is under way, so the start is abandoned mid-flight.
    deadline = time.monotonic() + 120
    while not containers(env.session_id):
        assert time.monotonic() < deadline and not starting.done()
        await asyncio.sleep(0.05)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    # Harbor's `_finalize` stops the environment after a cancelled start.
    await env.stop(delete=True)

    assert sup.owned == 0
    assert containers(env.session_id) == []
