"""Compose cancellation tests use fake processes and never contact Docker."""

import asyncio
import signal
from types import SimpleNamespace

import pytest
from openenv.validation import harbor

pytest.importorskip("harbor.trial.trial")


@pytest.mark.asyncio
@pytest.mark.parametrize("during_spawn", [False, True])
@pytest.mark.parametrize("command", ["build", "up", "pull"])
async def test_compose_cancellation_reaps_the_process_group(
    monkeypatch, tmp_path, during_spawn, command
):
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.trial.trial import Trial

    started = asyncio.Event()
    release_spawn = asyncio.Event()
    events = []
    options = {}

    class Process:
        pid = 123456
        returncode = None

        async def communicate(self, input=None):
            started.set()
            await asyncio.Event().wait()

        async def wait(self):
            assert self.returncode == -signal.SIGKILL
            events.append("reaped")

    process = Process()

    async def spawn(*args, **kwargs):
        options.update(kwargs)
        if during_spawn:
            started.set()
            await release_spawn.wait()
        return process

    class Environment:
        session_id = "openenv-validation-" + "d" * 32 + "-nop__env"
        environment_name = "test"
        environment_dir = tmp_path
        _docker_compose_paths = [tmp_path / "compose.yaml"]
        _run_docker_compose_command = DockerEnvironment._run_docker_compose_command
        _collect_buffered_output = staticmethod(
            DockerEnvironment._collect_buffered_output
        )

        @classmethod
        def runtime(cls):
            return SimpleNamespace(
                compose=["docker", "compose"], supports_compose_project_directory=True
            )

        def _compose_env_vars(self, *, include_os_env):
            return {}

    environment = Environment()
    trial = SimpleNamespace(agent_environment=environment)

    async def create(config):
        return trial

    def killpg(pid, sig):
        assert pid == process.pid and sig == signal.SIGKILL
        process.returncode = -sig
        events.append("killed")

    monkeypatch.setattr(Trial, "create", create)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(harbor.os, "killpg", killpg)
    created = await harbor._create_trial(None)
    running = asyncio.create_task(
        created.agent_environment._run_docker_compose_command([command])
    )
    await started.wait()
    running.cancel()
    release_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert options.get("start_new_session") is True
    assert events == ["killed", "reaped"]
    assert created._openenv_cancelled_compose == [
        {
            "command": command,
            "project": environment.session_id,
            "client_group_killed": True,
            "client_reaped": True,
            "daemon_completion_verified": False,
        }
    ]
