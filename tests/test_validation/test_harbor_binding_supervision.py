"""Compose wrapper tests use fake processes and never contact Docker."""

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


@pytest.mark.asyncio
async def test_compose_runs_without_an_environment_directory(monkeypatch, tmp_path):
    options = {}

    async def spawn(*argv, **kwargs):
        options.update(kwargs, argv=argv)
        return object()

    class Environment:
        session_id = "openenv-validation-" + "d" * 32 + "-nop__env"
        environment_dir = tmp_path / "environment"
        _docker_compose_paths = []

        @classmethod
        def runtime(cls):
            return SimpleNamespace(
                compose=["docker", "compose"], supports_compose_project_directory=True
            )

        def _compose_env_vars(self, *, include_os_env):
            return {}

        async def _collect_buffered_output(self, process, *, timeout_sec, stdin_data):
            return SimpleNamespace(return_code=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await harbor._run_compose(Environment(), ["up"], cancelled=[])
    # A missing working directory would stop the Compose client from starting.
    assert options["cwd"] is None
    # As in Harbor, the project directory never falls back to the task root,
    # where Compose would read a task-authored .env file.
    argv = options["argv"]
    assert argv[argv.index("--project-directory") + 1] == str(
        Environment.environment_dir.resolve()
    )


@pytest.mark.asyncio
async def test_hf_sandbox_start_labels_its_job_and_keeps_harbor_setup(
    monkeypatch, tmp_path
):
    import huggingface_hub
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )

    calls = []

    class Sandbox:
        id = "job-1"
        files = SimpleNamespace(mkdir=lambda path: calls.append(("mkdir", path)))

        def run(self, argv, **kwargs):
            calls.append(("run", argv))
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

        def kill(self):
            calls.append(("kill",))

    def create(**kwargs):
        calls.append(("create", kwargs))
        return Sandbox()

    monkeypatch.setattr(huggingface_hub.Sandbox, "create", create)
    image = "registry.example/image@sha256:" + "d" * 64
    task = tmp_path / "task"
    for directory in ("solution", "tests"):
        (task / directory).mkdir(parents=True)
    (task / "task.toml").write_text(
        f'schema_version = "1.4"\n[environment]\ndocker_image = "{image}"\n'
        + 'workdir = "/app"\n'
    )
    (task / "instruction.md").write_text("Write the correct answer.\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "solution/solve.sh").write_text("#!/bin/sh\nexit 0\n")
    trial = await harbor._create_trial(
        TrialConfig(
            task=TaskConfig(path=task),
            agent=AgentConfig(name="oracle"),
            environment=EnvironmentConfig(
                type=EnvironmentType.HF_SANDBOX,
                kwargs={
                    "flavor": "cpu-upgrade",
                    "namespace": "org",
                    "job_timeout": 900,
                },
            ),
            trial_name="openenv-validation-" + "d" * 32 + "-oracle-replay",
            trials_dir=tmp_path / "trials",
        )
    )
    await trial.agent_environment.start(force_build=False)
    assert calls[0] == (
        "create",
        {
            "image": image,
            "flavor": "cpu-upgrade",
            "idle_timeout": 900,
            "namespace": "org",
            "labels": {
                "openenv-validation-run": "d" * 32,
                "openenv-validation-role": "oracle-replay",
            },
            "start_timeout": harbor._SANDBOX_START_TIMEOUT_S,
        },
    )
    assert calls[1] == ("mkdir", "/app")
    # Harbor's own setup still creates the log directories the verifier writes.
    assert "/logs/verifier" in calls[2][1][-1]
    await trial.agent_environment.stop(delete=True)
    assert calls[-1] == ("kill",)
