import io
import json
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

# Add the project root to the path for envs imports, and envs/ itself so the
# server module's Spaces-style absolute imports (``tbench2_env.models``)
# resolve when this file runs standalone (in the full suite another test
# happens to insert envs/ first — don't rely on collection order).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import camel  # noqa: F401
except Exception:
    camel = None

from envs.tbench2_env.models import Tbench2Action
from envs.tbench2_env.server import tbench2_env_environment
from envs.tbench2_env.server.tbench2_env_environment import (
    Tbench2DockerEnvironment,
    Tbench2Environment,
)


class _FakeTerminalToolkit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def shell_exec(self, **kwargs):
        return ""


def test_tbench2_reset_uses_default_task_id(monkeypatch, tmp_path: Path):
    """HTTP resets without kwargs should land on the default demo task."""
    task_dir = tmp_path / "headless-terminal"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Solve the terminal task.\n")

    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )

    env = Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id="headless-terminal",
    )

    observation = env.reset()

    assert observation.success is True
    assert observation.task_id == "headless-terminal"
    assert "terminal task" in observation.instruction


def test_tbench2_reset_reports_execution_budgets(monkeypatch, tmp_path: Path):
    """reset() surfaces the budgets that bound a single env op, so clients can
    derive per-message deadlines from the server instead of guessing."""
    task_dir = tmp_path / "budget-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "task.toml").write_text("[verifier]\ntimeout_sec = 3600\n")

    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )
    env = Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        command_timeout_s=42.0,
        default_task_id="budget-task",
    )

    observation = env.reset()

    assert observation.info["verifier_timeout_sec"] == 3600.0
    assert observation.info["command_timeout_s"] == 42.0


def test_tbench2_reset_budget_falls_back_without_task_toml(monkeypatch, tmp_path: Path):
    """A task with no [verifier] budget reports the same fallback that
    evaluate enforces, not a missing key."""
    task_dir = tmp_path / "plain-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("do it\n")

    env = _make_env(monkeypatch, tmp_path, "plain-task")

    observation = env.reset()

    assert (
        observation.info["verifier_timeout_sec"]
        == tbench2_env_environment._DEFAULT_VERIFIER_TIMEOUT_S
    )


@pytest.mark.parametrize(
    "verifier_config",
    [
        f"[verifier]\ntimeout_sec = {value}\n"
        for value in ['"slow"', "[1, 2, 3]", "nan", "inf", "-inf", "0", "-1"]
    ]
    + ['verifier = "slow"\n', "verifier = [1, 2, 3]\n"],
)
@pytest.mark.parametrize("docker_mode", [False, True])
def test_reset_invalid_verifier_budget_uses_finite_default(
    monkeypatch, tmp_path: Path, verifier_config: str, docker_mode: bool
):
    task = _make_task_dir(tmp_path)
    (task / "task.toml").write_text(
        verifier_config + '\n[environment]\ndocker_image = "debian:12"\n'
    )
    if docker_mode:
        monkeypatch.setattr(
            Tbench2DockerEnvironment, "_start_container", lambda self, *args: None
        )
        env = Tbench2DockerEnvironment(tasks_dir=str(tmp_path))
    else:
        env = _make_env(monkeypatch, tmp_path, task.name)

    observation = env.reset(task_id=task.name)

    assert observation.info["verifier_timeout_sec"] == 900.0
    assert (
        json.loads(observation.model_dump_json())["info"]["verifier_timeout_sec"]
        == 900.0
    )


def _make_env(monkeypatch, tmp_path: Path, task_id: str) -> Tbench2Environment:
    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )
    return Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id=task_id,
    )


def test_tbench2_workdir_from_task_dockerfile(monkeypatch, tmp_path: Path):
    """A parsed WORKDIR that exists and isn't the server tree becomes the cwd."""
    task_dir = tmp_path / "demo-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it\n")
    image_workdir = tmp_path / "image-workdir"
    image_workdir.mkdir()
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM python:3.13\nWORKDIR {image_workdir}\n"
    )

    env = _make_env(monkeypatch, tmp_path, "demo-task")
    env.reset()

    assert env._workdir == str(image_workdir)


def test_tbench2_workdir_rejects_server_tree(monkeypatch, tmp_path: Path):
    """A parsed WORKDIR that is the server's own install tree (e.g. /app on
    the standard server image) falls back to the task source dir."""
    server_tree = Path(tbench2_env_environment.__file__).resolve().parent
    task_dir = tmp_path / "demo-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it\n")
    (task_dir / "environment" / "Dockerfile").write_text(
        f"FROM python:3.13\nWORKDIR {server_tree}\n"
    )

    env = _make_env(monkeypatch, tmp_path, "demo-task")
    env.reset()

    assert env._workdir == str(task_dir)


# ---------------------------------------------------------------------------
# Verifier-asset withholding (TB2_WITHHOLD_TESTS) and stage-at-verify
# ---------------------------------------------------------------------------


def _make_task_dir(root: Path, name: str = "my-task") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "tests" / "test_outputs.py").write_text("EXPECTED = 'secret42'\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("echo the-answer\n")
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    (task / "instruction.md").write_text("do the thing\n")
    return task


@pytest.fixture()
def staged_paths(monkeypatch, tmp_path: Path):
    """Redirect the official fixed paths (/tests, /logs/verifier) into tmp."""
    tests = tmp_path / "stage" / "tests"
    logs = tmp_path / "stage" / "logs" / "verifier"
    monkeypatch.setattr(tbench2_env_environment, "_VERIFY_TESTS_DIR", str(tests))
    monkeypatch.setattr(tbench2_env_environment, "_VERIFIER_LOG_DIR", str(logs))
    return tests, logs


@pytest.fixture(autouse=True)
def _fresh_withhold_cache(monkeypatch):
    monkeypatch.setattr(Tbench2Environment, "_WITHHELD_TESTS", {})


def test_withhold_removes_assets_and_caches_tests(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)

    env._withhold_verifier_assets(task)

    assert not (task / "solution").exists()
    assert not (task / "tests").exists()
    assert (task / "task.toml").is_file()
    assert (task / "instruction.md").is_file()
    assert str(task) in Tbench2Environment._WITHHELD_TESTS

    # Second reset of the same task (dir already gone) must keep the cache.
    env._withhold_verifier_assets(task)
    assert str(task) in Tbench2Environment._WITHHELD_TESTS


def test_stage_from_memory_and_wipe_planted_files(tmp_path: Path, staged_paths):
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    # An agent pre-plants a conftest.py (pytest would auto-load it) and a
    # fake reward before verification.
    stage_tests.mkdir(parents=True)
    (stage_tests / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_logs.mkdir(parents=True)
    (stage_logs / "reward.txt").write_text("1\n")

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test.sh").is_file()
    assert (stage_tests / "test_outputs.py").read_text() == "EXPECTED = 'secret42'\n"
    assert not (stage_tests / "conftest.py").exists()
    assert not (stage_logs / "reward.txt").exists()
    assert stage_logs.is_dir() and not list(stage_logs.iterdir())


def test_stage_from_disk_without_gate(tmp_path: Path, staged_paths):
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test_outputs.py").is_file()
    # Without the gate the source checkout is never touched.
    assert (task / "tests").is_dir()
    assert (task / "solution").is_dir()


def test_stage_reports_missing_tests(tmp_path: Path, staged_paths):
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert not env._stage_tests_for_verify()


class _RecordingToolkit:
    def __init__(self, output: str):
        self.calls: list[dict] = []
        self.output = output

    def shell_exec(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


@pytest.mark.parametrize("docker_mode", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "timeout_value, expected_timeout", [("37.5", 37.5), ("nan", 900.0), (None, 900.0)]
)
def test_verifier_budget_frozen_until_next_reset(
    monkeypatch,
    tmp_path: Path,
    staged_paths,
    docker_mode,
    canonical,
    timeout_value,
    expected_timeout,
):
    task = _make_task_dir(tmp_path)
    if not canonical:
        (task / "tests" / "test.sh").unlink()
    original_config = (task / "task.toml").read_text()
    if timeout_value is not None:
        (task / "task.toml").write_text(
            original_config + f"\n[verifier]\ntimeout_sec = {timeout_value}\n"
        )
    output = "__TB2_REWARD__:1" if canonical else "__TB2_EXIT_CODE__:0"
    if docker_mode:
        monkeypatch.setattr(
            Tbench2DockerEnvironment,
            "_start_container",
            lambda self, *args: setattr(
                self, "_container", _FakeContainer(output.encode())
            ),
        )
        env = Tbench2DockerEnvironment(tasks_dir=str(tmp_path))
    else:
        toolkit = _RecordingToolkit(output)
        monkeypatch.setattr(
            tbench2_env_environment,
            "_require_terminal_toolkit",
            lambda: lambda **kwargs: toolkit,
        )
        env = Tbench2Environment(tasks_dir=str(tmp_path), withhold_tests=False)

    observation = env.reset(task_id=task.name)
    assert observation.info["verifier_timeout_sec"] == expected_timeout
    (task / "task.toml").write_text(
        original_config + "\n[verifier]\ntimeout_sec = 7200\n"
    )

    for reset_again, budget in enumerate((expected_timeout, 7200.0)):
        if reset_again:
            observation = env.reset(task_id=task.name)
            assert observation.info["verifier_timeout_sec"] == budget
        if docker_mode:
            _, reward, _ = env._evaluate_docker()
            command = env._container.events[-2][1]
            assert f"timeout {budget:g} " in command
        else:
            _, reward, _ = env._evaluate_task()
            assert toolkit.calls[-1]["timeout"] == budget
        assert reward == 1.0


def test_evaluate_canonical_from_withheld_copy(tmp_path: Path, staged_paths):
    """Full evaluate flow: staging from memory + canonical test.sh scoring."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    env._terminal_toolkit = _RecordingToolkit(output="__TB2_REWARD__:1")
    env._withhold_verifier_assets(task)

    output, reward, info = env._evaluate_task()

    assert reward == 1.0
    assert info["harness"] == "tests/test.sh"
    # The verify command ran test.sh from the staged fixed path, and wrote its
    # log under the wiped verifier dir — not a fixed /tmp path that outlives it.
    command = env._terminal_toolkit.calls[0]["command"]
    assert f"{stage_tests}/test.sh" in command
    assert "/tmp/tb2_testsh.log" not in command
    assert f"{stage_logs}/testsh.log" in command
    # Both the staged tests and the verifier log/reward dir are gone once
    # scoring returns.
    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_evaluate_missing_verdict_raises_not_zero(tmp_path: Path, staged_paths):
    """test.sh ran but wrote no reward.txt (crash / kill before the verifier
    wrote one): scoring must error out — reward None on the observation — not
    return a 0.0 indistinguishable from tests genuinely failing. The staged
    assets are still wiped."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    # Marker line present but empty: reward.txt was never written.
    env._terminal_toolkit = _RecordingToolkit(output="__TB2_REWARD__:")
    env._withhold_verifier_assets(task)

    with pytest.raises(RuntimeError, match="produced no verdict"):
        env._evaluate_task()

    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_evaluate_without_tests_scores_zero(tmp_path: Path, staged_paths):
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task
    env._terminal_toolkit = _RecordingToolkit(output="")

    output, reward, info = env._evaluate_task()

    assert reward == 0.0
    assert info["tests_passed"] is False
    assert env._terminal_toolkit.calls == []  # nothing was run


def test_evaluate_wipes_partial_stage_on_error(
    tmp_path: Path, staged_paths, monkeypatch
):
    """A stage that errors partway must be wiped like a completed one — the
    session survives a scoring error, so no partial tests/ copy may remain."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    env._terminal_toolkit = _RecordingToolkit(output="")
    env._withhold_verifier_assets(task)

    def _explode(tar, dest):
        (Path(dest) / "test_outputs.py").write_text("EXPECTED = 'secret42'\n")
        raise OSError("disk full")

    monkeypatch.setattr(tbench2_env_environment, "_extractall", _explode)

    with pytest.raises(OSError, match="disk full"):
        env._evaluate_task()

    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_stage_wipes_symlinked_fixed_path(tmp_path: Path, staged_paths):
    """A symlinked /tests must not let the official suite merge into a dir the
    agent controls (rmtree silently no-ops on a symlink; _hard_remove doesn't)."""
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    evil = tmp_path / "agent_controlled"
    evil.mkdir()
    (evil / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_tests.parent.mkdir(parents=True, exist_ok=True)
    stage_tests.symlink_to(evil, target_is_directory=True)

    assert env._stage_tests_for_verify()

    assert not stage_tests.is_symlink()
    assert (stage_tests / "test.sh").is_file()
    # The official suite never leaked into the agent's directory.
    assert not (evil / "test.sh").exists()
    assert (evil / "conftest.py").exists()  # their dir left alone


def test_withhold_removes_symlinked_tests(tmp_path: Path):
    """tests/ as a symlink is removed too — the link, not just its target."""
    task = tmp_path / "task"
    task.mkdir()
    real = tmp_path / "real_tests"
    real.mkdir()
    (real / "test_outputs.py").write_text("x = 1\n")
    (task / "tests").symlink_to(real, target_is_directory=True)

    env = Tbench2Environment(withhold_tests=True)
    env._withhold_verifier_assets(task)

    assert not (task / "tests").exists()  # symlink gone
    assert real.exists()  # its target is outside the task dir; untouched
    # The cache captured the target's real files, not a bare link member that
    # would restore an empty suite (tar.add doesn't follow a top-level link).
    import io
    import tarfile

    blob = Tbench2Environment._WITHHELD_TESTS[str(task)]
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        assert any(n.endswith("test_outputs.py") for n in tar.getnames())


# ---------------------------------------------------------------------------
# Docker mode: copy exclusions + stage-at-verify (mock container, no docker)
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Records exec/put_archive calls in one ordered event log.

    exec_run receives argv form (["bash", "-c", payload]); the log records
    the shell payload, and raw_cmds keeps the argv for shape assertions.
    """

    def __init__(self, exec_output: bytes = b"__TB2_REWARD__:1\n"):
        self.events: list[tuple] = []
        self.raw_cmds: list = []
        self.exec_output = exec_output

    def put_archive(self, dest, data):
        self.events.append(("put", dest, data))
        return True

    def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
        self.raw_cmds.append(cmd)
        self.events.append(("exec", cmd[-1] if isinstance(cmd, list) else cmd))
        return 0, self.exec_output


def _tar_names(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return tar.getnames()


def test_docker_copy_excludes_verifier_assets(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM debian:12\n")

    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._copy_dir_to_container(task, "/task", exclude_top=("solution", "tests"))

    ((kind, dest, data),) = container.events
    assert (kind, dest) == ("put", "/task")
    names = _tar_names(data)
    assert "task.toml" in names
    assert "environment/Dockerfile" in names
    assert not any(n.startswith(("solution", "tests")) for n in names)
    assert len(names) == len(set(names))  # recursive=False: no duplicates


def test_evaluate_docker_missing_verdict_raises_not_zero(tmp_path: Path):
    """Docker-mode twin of the local missing-verdict case: an empty reward
    marker errors out (with cleanup) instead of scoring 0.0."""
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    container = _FakeContainer(exec_output=b"__TB2_REWARD__:\n")
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="produced no verdict"):
        env._evaluate_docker()

    assert container.events[-1][0] == "exec"
    assert "rm -rf /tests /logs/verifier" in container.events[-1][1]


@pytest.mark.parametrize("timeout_s", [900.0, 0.5])
def test_evaluate_docker_stages_tests_at_verify(tmp_path: Path, timeout_s: float):
    task = _make_task_dir(tmp_path)
    with (task / "task.toml").open("a") as handle:
        handle.write(f"\n[verifier]\ntimeout_sec = {timeout_s}\n")
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task
    env._verifier_timeout_s = timeout_s

    output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert info == {"tests_passed": True, "harness": "tests/test.sh"}
    kinds = [e[0] for e in container.events]
    # rm -rf the agent-writable fixed paths BEFORE the fresh copy goes in,
    # the copy lands before test.sh runs, and both are removed again after.
    assert kinds == ["exec", "put", "exec", "exec"]
    assert "rm -rf /tests /logs/verifier" in container.events[0][1]
    assert container.events[1][1] == "/tests"
    assert "test.sh" in _tar_names(container.events[1][2])
    eval_cmd = container.events[2][1]
    # Canonical harness from the agent's workdir, bounded by the verifier
    # budget, verdict read from reward.txt — not bare pytest in /task.
    assert "bash /tests/test.sh" in eval_cmd
    assert eval_cmd.startswith("cd /task && ")  # no resolved workdir → /task
    assert f"timeout {timeout_s:g}" in eval_cmd
    assert "/logs/verifier/reward.txt" in eval_cmd
    assert "rm -rf /tests /logs/verifier" in container.events[3][1]


def test_evaluate_docker_runs_in_resolved_workdir(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    env._container = _FakeContainer()
    env._task_dir = task
    env._workdir = "/app"

    _, reward, _ = env._evaluate_docker()

    assert reward == 1.0
    eval_cmd = env._container.events[2][1]
    assert eval_cmd.startswith("cd /app && ")


def test_evaluate_docker_fallback_without_testsh(tmp_path: Path):
    """A task dir whose tests/ ships no canonical harness scores via pytest,
    still against the staged /tests copy."""
    task = _make_task_dir(tmp_path)
    (task / "tests" / "test.sh").unlink()
    with (task / "task.toml").open("a") as handle:
        handle.write("\n[verifier]\ntimeout_sec = 0.5\n")
    env = Tbench2DockerEnvironment()
    container = _FakeContainer(exec_output=b"__TB2_EXIT_CODE__:0\n")
    env._container = container
    env._task_dir = task
    env._verifier_timeout_s = 0.5

    output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert info == {"tests_passed": True, "exit_code": 0}
    eval_cmd = container.events[2][1]
    assert "pytest -q /tests -rA" in eval_cmd
    assert "test.sh" not in eval_cmd
    assert "timeout 0.5" in eval_cmd


@pytest.mark.parametrize("has_timeout", [False, True])
def test_fallback_timeout_preserves_exit_marker(tmp_path: Path, has_timeout: bool):
    """Timeout failure or absence must never run an unbounded verifier."""
    timeout_args = tmp_path / "timeout-args"
    if has_timeout:
        timeout = tmp_path / "timeout"
        timeout.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" > "$TIMEOUT_ARGS"\nexit 124\n'
        )
        timeout.chmod(0o755)
    verifier_ran = tmp_path / "verifier-ran"
    uvx = tmp_path / "uvx"
    uvx.write_text('#!/bin/sh\nprintf ran > "$VERIFIER_RAN"\n')
    uvx.chmod(0o755)
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            tbench2_env_environment._fallback_eval_cmd(str(tmp_path), timeout_s=0.5),
        ],
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "TIMEOUT_ARGS": str(timeout_args),
            "VERIFIER_RAN": str(verifier_ran),
        },
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert not verifier_ran.exists()
    if has_timeout:
        assert timeout_args.read_text().splitlines()[:3] == ["0.5", "bash", "-c"]
    expected_exit = 124 if has_timeout else 127
    assert (
        tbench2_env_environment._parse_exit_code_marker(result.stdout) == expected_exit
    )


def test_evaluate_docker_cleans_up_when_scoring_raises(tmp_path: Path):
    """The staged /task/tests must not outlive a verify that errors out —
    step() reports the failure without ending the episode, so the agent
    keeps its session."""
    task = _make_task_dir(tmp_path)

    class _ExplodingContainer(_FakeContainer):
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            if "test.sh" in payload:
                self.events.append(("exec", payload))
                raise RuntimeError("docker daemon hiccup")
            return super().exec_run(cmd, workdir=workdir, stdout=stdout, stderr=stderr)

    env = Tbench2DockerEnvironment()
    container = _ExplodingContainer()
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="hiccup"):
        env._evaluate_docker()

    assert container.events[-1][0] == "exec"
    assert "rm -rf /tests /logs/verifier" in container.events[-1][1]


def test_evaluate_docker_fails_closed_when_prestage_wipe_fails(tmp_path: Path):
    """If the pre-stage rm -rf fails, scoring must error out rather than
    put_archive into a dir the agent still controls."""
    task = _make_task_dir(tmp_path)

    class _StubbornContainer(_FakeContainer):
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            self.events.append(("exec", payload))
            if "rm -rf /tests" in payload and "mkdir" in payload:
                return 1, b"rm: cannot remove '/tests': busy\n"
            return 0, self.exec_output

    env = Tbench2DockerEnvironment()
    container = _StubbornContainer()
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="could not reset /tests"):
        env._evaluate_docker()

    assert not any(e[0] == "put" for e in container.events)


def test_evaluate_docker_warns_when_cleanup_fails(tmp_path: Path, caplog):
    """A post-verify rm -rf that exits nonzero must not stay silent."""
    task = _make_task_dir(tmp_path)

    class _LeakyContainer(_FakeContainer):
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            payload = cmd[-1] if isinstance(cmd, list) else cmd
            self.events.append(("exec", payload))
            if "rm -rf /tests" in payload and "mkdir" not in payload:
                return 1, b"rm: cannot remove '/tests': busy\n"
            return 0, self.exec_output

    env = Tbench2DockerEnvironment()
    container = _LeakyContainer()
    env._container = container
    env._task_dir = task

    with caplog.at_level(logging.WARNING):
        output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    assert any(
        "failed to remove staged /tests" in record.getMessage()
        for record in caplog.records
    )


def test_evaluate_docker_missing_tests_scores_zero(tmp_path: Path):
    task = tmp_path / "no-tests-task"
    task.mkdir()
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 0.0
    assert container.events == []


def test_docker_reset_rejects_task_without_image(tmp_path: Path):
    """No host-execution fallback: a task dir that declares no docker_image
    must fail reset loudly, not silently run agent commands on the server."""
    task = tmp_path / "imageless-task"
    task.mkdir()
    (task / "task.toml").write_text("[metadata]\n")
    (task / "instruction.md").write_text("do it\n")

    env = Tbench2DockerEnvironment(
        tasks_dir=str(tmp_path), output_dir=str(tmp_path / "runs")
    )

    with pytest.raises(RuntimeError, match="docker_image"):
        env.reset(task_id="imageless-task")


def test_docker_failed_reset_closes_previous_container(tmp_path: Path):
    """A rejected reset must not leave the previous container usable with
    metadata from the new task."""
    task = tmp_path / "imageless-task"
    task.mkdir()
    (task / "task.toml").write_text("[metadata]\n")
    (task / "instruction.md").write_text("do it\n")

    class _PreviousContainer:
        def __init__(self):
            self.events = []

        def stop(self, timeout):
            self.events.append(("stop", timeout))

        def remove(self, force):
            self.events.append(("remove", force))

    env = Tbench2DockerEnvironment(
        tasks_dir=str(tmp_path), output_dir=str(tmp_path / "runs")
    )
    previous_container = _PreviousContainer()
    env._container = previous_container
    env._task_dir = tmp_path / "previous-task"
    env._instruction = "previous task"
    env._workdir = "/previous-workdir"

    with pytest.raises(RuntimeError, match="docker_image"):
        env.reset(task_id="imageless-task")

    assert previous_container.events == [("stop", 10), ("remove", True)]
    assert env._container is None
    assert env._task_dir is None
    assert env._instruction == ""
    assert env._workdir == ""


def _docker_reset_info(monkeypatch, tmp_path: Path, task_toml: str) -> dict:
    task = tmp_path / "docker-task"
    task.mkdir()
    (task / "task.toml").write_text(task_toml)
    (task / "instruction.md").write_text("do it\n")
    monkeypatch.setattr(
        Tbench2DockerEnvironment, "_start_container", lambda self, *args: None
    )
    env = Tbench2DockerEnvironment(
        tasks_dir=str(tmp_path), output_dir=str(tmp_path / "runs")
    )
    return env.reset(task_id="docker-task").info


def test_docker_reset_reports_verifier_budget(monkeypatch, tmp_path: Path):
    """Docker mode enforces the task's verifier budget in-shell, so reset()
    reports it alongside the image. It must not advertise command_timeout_s:
    exec_run has no server-side timeout, so that value is never enforced."""
    info = _docker_reset_info(
        monkeypatch,
        tmp_path,
        '[environment]\ndocker_image = "tb2/demo:1"\n\n[verifier]\ntimeout_sec = 1800\n',
    )

    assert info == {"docker_image": "tb2/demo:1", "verifier_timeout_sec": 1800.0}


def test_docker_reset_budget_falls_back_without_verifier_section(
    monkeypatch, tmp_path: Path
):
    info = _docker_reset_info(
        monkeypatch, tmp_path, '[environment]\ndocker_image = "tb2/demo:1"\n'
    )

    assert (
        info["verifier_timeout_sec"]
        == tbench2_env_environment._DEFAULT_VERIFIER_TIMEOUT_S
    )


class _FakeImage:
    def __init__(self, working_dir: str):
        self.attrs = {"Config": {"WorkingDir": working_dir}}


def test_docker_workdir_prefers_image_metadata(tmp_path: Path):
    """Image Config.WorkingDir wins (it sees base-image WORKDIRs); the task
    Dockerfile is the fallback, /task the last resort."""
    task = _make_task_dir(tmp_path)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM debian:12\nWORKDIR /from-dockerfile\n"
    )
    env = Tbench2DockerEnvironment()

    assert env._resolve_workdir(_FakeImage("/from-image"), task) == "/from-image"
    assert env._resolve_workdir(_FakeImage(""), task) == "/from-dockerfile"

    bare = tmp_path / "bare-task"
    bare.mkdir()
    assert env._resolve_workdir(_FakeImage(""), bare) == "/task"


def test_docker_exec_passes_command_as_argv(tmp_path: Path):
    """Agent commands ride as a bash -c argv element, so shell quoting in the
    command (a single quote, say) survives byte-identical; the exec cd's into
    the resolved image workdir."""
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._workdir = "/app"

    env._exec_in_container("echo 'hi there'")

    (raw,) = container.raw_cmds
    assert isinstance(raw, list) and raw[:2] == ["bash", "-c"]
    assert raw[2] == "cd /app && echo 'hi there'"


@pytest.mark.skipif(camel is None, reason="camel-ai not installed")
@pytest.mark.skipif(
    os.environ.get("TB2_ENABLE_TESTS", "0") != "1",
    reason="TB2_ENABLE_TESTS not enabled",
)
def test_tbench2_env_smoke():
    env = Tbench2Environment(tasks_dir=os.environ.get("TB2_TASKS_DIR"))
    obs = env.reset(task_id=os.environ.get("TB2_TASK_ID", "headless-terminal"))
    assert obs.instruction

    result = env.step(Tbench2Action(action_type="exec", command="pwd"))
    assert result.success
    assert result.output

    env.step(Tbench2Action(action_type="close"))
    env.close()
