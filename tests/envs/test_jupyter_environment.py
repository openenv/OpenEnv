# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jupyter_env.models import JupyterState
from jupyter_env.server import submission
from jupyter_env.server.e2b_sandbox import CellResult, E2BSandbox
from jupyter_env.server.jupyter_environment import JupyterEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.killed = False
        self.shell_commands: list[str] = []
        self.process_commands: list[str] = []

    def run_code(self, code: str) -> CellResult:
        return CellResult(
            stdout=f"ran: {code}",
            stderr="",
            error=None,
            error_name=None,
            text_results=[],
            images=[],
            execution_count=1,
            success=True,
        )

    def run_shell(self, command: str) -> CellResult:
        self.shell_commands.append(command)
        return self._respond(command)

    def run_command(self, command: str, timeout_s: float = 120) -> CellResult:
        self.process_commands.append(command)
        return self._respond(command)

    def _respond(self, command: str) -> CellResult:
        if "exit 1" in command:
            return CellResult(
                stdout="",
                stderr="failed",
                error="failed",
                error_name="CommandError",
                text_results=[],
                images=[],
                execution_count=1,
                success=False,
            )
        if command.startswith("cat /home/user/logs/verifier/reward.txt"):
            return CellResult(
                stdout="",
                stderr="",
                error=None,
                error_name=None,
                text_results=[],
                images=[],
                execution_count=1,
                success=True,
            )
        return CellResult(
            stdout=f"shell: {command}",
            stderr="",
            error=None,
            error_name=None,
            text_results=[],
            images=[],
            execution_count=1,
            success=True,
        )

    def kill(self) -> None:
        self.killed = True


def _extract_text(result) -> str:
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def test_lists_notebook_tools_without_reset():
    env = JupyterEnvironment()

    obs = env.step(ListToolsAction())

    tool_names = {tool.name for tool in obs.tools}
    assert "add_and_execute_code_cell" in tool_names
    assert "edit_and_execute_current_cell" in tool_names
    assert "execute_shell_command" in tool_names
    assert "get_notebook_state" in tool_names
    assert "final_answer" in tool_names


def test_reset_without_e2b_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    env = JupyterEnvironment()

    obs = env.reset()

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert "E2B_API_KEY" in obs.metadata["error"]


def test_code_tool_updates_notebook_state():
    env = JupyterEnvironment()
    env._sandbox = FakeSandbox()
    env._state = JupyterState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="add_and_execute_code_cell",
            arguments={"code": "print('hello')"},
        )
    )

    assert obs.error is None
    assert "ran: print('hello')" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert len(env.state.cells) == 1
    assert env.state.cells[0].code == "print('hello')"


def test_shell_tool_updates_notebook_state():
    env = JupyterEnvironment()
    env._sandbox = FakeSandbox()
    env._state = JupyterState(episode_id="episode-1", sandbox_id="fake-sandbox")

    obs = env.step(
        CallToolAction(
            tool_name="execute_shell_command",
            arguments={"command": "pwd"},
        )
    )

    assert obs.error is None
    assert "shell: pwd" in _extract_text(obs.result)
    assert env.state.step_count == 1
    assert env.state.cells[0].cell_type == "shell"


def test_reset_runs_setup_and_stores_verify_commands(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    env = JupyterEnvironment()
    fake_sandbox = FakeSandbox()
    monkeypatch.setattr(
        "jupyter_env.server.jupyter_environment.E2BSandbox",
        lambda api_key: fake_sandbox,
    )

    obs = env.reset(setup=["echo setup"], verify=["test -f answer.py"])

    assert obs.done is False
    assert fake_sandbox.shell_commands == [
        "mkdir -p /home/user/logs/verifier",
        "echo setup",
    ]
    assert env.state.setup_results[0].command == "echo setup"
    assert env.state.setup_results[0].success is True
    assert env.state.verify_commands == ["test -f answer.py"]
    assert obs.metadata["verify_commands"] == ["test -f answer.py"]


def test_reset_fails_when_setup_command_fails(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake-key")
    env = JupyterEnvironment()
    monkeypatch.setattr(
        "jupyter_env.server.jupyter_environment.E2BSandbox",
        lambda api_key: FakeSandbox(),
    )

    obs = env.reset(setup=["exit 1"], verify=["test -f answer.py"])

    assert obs.done is True
    assert obs.metadata["status"] == "error"
    assert obs.metadata["setup_results"][0]["success"] is False


def test_final_answer_runs_verify_commands():
    env = JupyterEnvironment()
    fake_sandbox = FakeSandbox()
    env._sandbox = fake_sandbox
    env._state = JupyterState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=["test -f answer.py", "exit 1"],
    )

    obs = env.step(
        CallToolAction(
            tool_name="final_answer",
            arguments={"answer": "done"},
        )
    )

    assert obs.error is None
    assert "Verification: 1/2 passed; reward=0.5" in _extract_text(obs.result)
    assert env.state.submitted_answer == "done"
    assert env.state.last_reward == 0.5
    assert [result.command for result in env.state.verify_results] == [
        "test -f answer.py",
        "exit 1",
    ]
    assert "exit 1" in fake_sandbox.process_commands
    assert "exit 1" not in fake_sandbox.shell_commands


REWARD_FILE = "/home/user/logs/verifier/reward.txt"


def _cell(stdout: str = "", success: bool = True) -> CellResult:
    return CellResult(
        stdout=stdout,
        stderr="",
        error=None if success else "SystemExit: 1",
        error_name=None if success else "SystemExit",
        text_results=[],
        images=[],
        execution_count=1,
        success=success,
    )


class RewardFileSandbox(FakeSandbox):
    """FakeSandbox whose reward file any command can write, as in E2B.

    The agent writes to it through `run_shell`; verification clears, writes and
    reads it as sandbox processes through `run_command`.
    """

    def __init__(self, *, removable: bool = True) -> None:
        super().__init__()
        self.reward_file: str | None = None
        self.removable = removable

    def _reward_file_command(self, command: str) -> CellResult | None:
        if command.startswith("echo ") and command.endswith(f" > {REWARD_FILE}"):
            self.reward_file = command[len("echo ") : -len(f" > {REWARD_FILE}")]
            return _cell()
        if command.startswith(f"rm -f {REWARD_FILE}"):
            if self.removable:
                self.reward_file = None
            return _cell(success=self.reward_file is None)
        if command.startswith(f"cat {REWARD_FILE}"):
            return _cell(stdout=self.reward_file or "")
        return None

    def run_shell(self, command: str) -> CellResult:
        result = self._reward_file_command(command)
        if result is None:
            return super().run_shell(command)
        self.shell_commands.append(command)
        return result

    def run_command(self, command: str, timeout_s: float = 120) -> CellResult:
        result = self._reward_file_command(command)
        if result is None:
            return super().run_command(command)
        self.process_commands.append(command)
        return result


def _submit(sandbox, verify_commands, agent_commands=()) -> JupyterState:
    env = JupyterEnvironment()
    env._sandbox = sandbox
    env._state = JupyterState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=verify_commands,
    )
    for command in agent_commands:
        env.step(
            CallToolAction(
                tool_name="execute_shell_command", arguments={"command": command}
            )
        )
    env.step(CallToolAction(tool_name="final_answer", arguments={"answer": "done"}))
    return env.state


def test_reward_file_written_by_the_agent_is_ignored():
    state = _submit(
        RewardFileSandbox(),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert state.last_reward == 0.5
    assert state.reward_override_ignored is None


def test_reward_file_written_by_a_verify_command_overrides_the_pass_rate():
    state = _submit(RewardFileSandbox(), [f"echo 0.8 > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == 0.8
    assert state.reward_override_ignored is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "passed"])
def test_invalid_reward_file_falls_back_to_the_pass_rate(value):
    state = _submit(RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == 0.5
    assert repr(value) in state.reward_override_ignored


@pytest.mark.parametrize("value", ["-0.5", "1.5"])
def test_reward_file_outside_the_unit_interval_is_kept(value):
    state = _submit(RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"])

    assert state.last_reward == float(value)
    assert state.reward_override_ignored is None


def test_reward_file_that_cannot_be_removed_is_ignored():
    state = _submit(
        RewardFileSandbox(removable=False),
        ["true", "exit 1"],
        agent_commands=[f"echo 1.0 > {REWARD_FILE}"],
    )

    assert state.last_reward == 0.5
    assert "could not be removed" in state.reward_override_ignored


class _LocalKernel:
    """Runs E2B cell code in-process; an exception becomes execution.error."""

    def run_code(self, code: str):
        stdout, stderr = io.StringIO(), io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, {})
        except BaseException as exc:  # SystemExit is reported, not raised, by E2B
            error = SimpleNamespace(
                name=type(exc).__name__, value=str(exc), traceback=""
            )
        return SimpleNamespace(
            error=error,
            logs=SimpleNamespace(
                stdout=[stdout.getvalue()] if stdout.getvalue() else [],
                stderr=[stderr.getvalue()] if stderr.getvalue() else [],
            ),
            results=[],
            execution_count=1,
        )


@pytest.mark.parametrize(
    ("command", "success"),
    [("true", True), ("exit 3", False), ("test -e /nonexistent/reward.txt", False)],
)
def test_run_shell_reports_the_exit_code(command, success):
    sandbox = E2BSandbox.__new__(E2BSandbox)
    sandbox._sbx = _LocalKernel()

    assert sandbox.run_shell(command).success is success


class _Kernel:
    """A kernel whose namespace persists between cells, as E2B's run_code does."""

    def __init__(self) -> None:
        self.namespace: dict = {}
        self.cells: list[str] = []

    def run_code(self, code: str, **kwargs):
        self.cells.append(code)
        stdout, stderr, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, self.namespace)
        except BaseException as exc:  # a kernel reports SystemExit, it doesn't exit
            error = SimpleNamespace(
                name=type(exc).__name__, value=str(exc), traceback=""
            )
        return SimpleNamespace(
            error=error,
            logs=SimpleNamespace(
                stdout=[stdout.getvalue()] if stdout.getvalue() else [],
                stderr=[stderr.getvalue()] if stderr.getvalue() else [],
            ),
            results=[],
            execution_count=len(self.cells),
        )


class _CommandExit(Exception):
    """Stands in for `e2b.CommandExitException`, raised on a non-zero exit."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(f"Command exited with code {exit_code}")
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


class _Processes:
    """E2B's process API: each command is a fresh process with a real exit status.

    envd starts commands as `/bin/bash -l -c`, so the login files in `HOME` are
    sourced here too. `/home/user` is mapped to a temporary directory, which is
    also that `HOME`, so the commands can run here.
    """

    def __init__(self, home) -> None:
        self.home = str(home)
        self.calls: list[dict] = []
        # Captured now, before any notebook cell gets a chance to rebind it.
        self._run = subprocess.run

    def run(self, cmd, background=None, user=None, cwd=None, timeout=None, **kwargs):
        self.calls.append(
            {
                "cmd": cmd,
                "background": background,
                "user": user,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        mapped = cmd.replace("/home/user", self.home).replace(
            "/submission", f"{self.home}/submission"
        )
        done = self._run(
            ["/bin/bash", "-l", "-c", mapped],
            cwd=(cwd or "").replace("/home/user", self.home) or None,
            env={
                "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                "HOME": self.home,
            },
            capture_output=True,
            text=True,
        )
        return _Handle(done=done)


class _Handle:
    """What `commands.run(background=True)` returns: wait for it, or kill it."""

    def __init__(
        self, done=None, wait_error=None, kill_result=True, kill_error=None
    ) -> None:
        self._done, self._wait_error = done, wait_error
        self._kill_result, self._kill_error = kill_result, kill_error
        self.killed = False

    def wait(self):
        if self._wait_error is not None:
            raise self._wait_error
        if self._done.returncode:
            raise _CommandExit(
                self._done.returncode, self._done.stdout, self._done.stderr
            )
        return SimpleNamespace(
            stdout=self._done.stdout, stderr=self._done.stderr, exit_code=0
        )

    def kill(self) -> bool:
        self.killed = True
        if self._kill_error is not None:
            raise self._kill_error
        return self._kill_result


class _Files:
    """E2B's file API over the temporary directory standing in for the sandbox."""

    def __init__(self, home) -> None:
        self.home = str(home)

    def _local(self, path: str) -> Path:
        if path.startswith("/submission"):
            return Path(self.home) / "submission" / path[len("/submission/") :]
        return Path(path.replace("/home/user", self.home, 1))

    def read(self, path: str, format: str = "text"):
        data = self._local(path).read_bytes()
        return data if format == "bytes" else data.decode()

    def write(self, path: str, content) -> None:
        target = self._local(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)


def _sandbox_on(home, sandbox_id="fake-sandbox"):
    kernel, processes = _Kernel(), _Processes(home)
    sandbox = E2BSandbox.__new__(E2BSandbox)
    sandbox._sbx = SimpleNamespace(
        run_code=kernel.run_code, commands=processes, files=_Files(home)
    )
    sandbox.sandbox_id = sandbox_id
    return sandbox, kernel, processes


def _environment_on_real_sandbox(home, verify_commands):
    sandbox, kernel, processes = _sandbox_on(home)
    env = JupyterEnvironment()
    env._sandbox = sandbox
    env._state = JupyterState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=verify_commands,
    )
    return env, kernel, processes


def _environment_with_a_verifier_sandbox(
    agent_home, verifier_home, verify_commands, submit_paths, setup_commands=()
):
    """An episode whose verification runs in a second, untouched sandbox.

    `/submission` and `/tests` live outside the mapped workspace, so the
    verifier's temporary directory is its parent here.
    """
    env, kernel, processes = _environment_on_real_sandbox(agent_home, verify_commands)
    env._state.submit_paths = list(submit_paths)
    env._state.setup_commands = list(setup_commands)
    verifier, verifier_kernel, verifier_processes = _sandbox_on(
        verifier_home, "fake-verifier-sandbox"
    )
    env._new_sandbox = lambda: verifier
    return env, processes, verifier, verifier_processes


def _run_cell(env, code):
    env.step(
        CallToolAction(tool_name="add_and_execute_code_cell", arguments={"code": code})
    )


def _submit_answer(env):
    env.step(CallToolAction(tool_name="final_answer", arguments={"answer": "done"}))


# Reproduction from #1210: a cell that rebinds subprocess.run used to decide the
# outcome of every verify command, and of the reward-file read.
_REBIND_SUBPROCESS = """\
import subprocess

class _Fake:
    returncode = 0
    stdout = "1.0"
    stderr = ""

subprocess.run = lambda *a, **k: _Fake()
"""


def test_verification_ignores_a_rebound_subprocess(monkeypatch, tmp_path):
    # Restores the real function after the cell below replaces it.
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    env, kernel, processes = _environment_on_real_sandbox(
        tmp_path, ["exit 1", "exit 1"]
    )

    _run_cell(env, _REBIND_SUBPROCESS)
    _submit_answer(env)

    assert [result.success for result in env.state.verify_results] == [False, False]
    assert env.state.last_reward == 0.0
    assert not any("exit 1" in cell for cell in kernel.cells)


def test_reward_file_is_read_outside_the_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", subprocess.run)
    env, _, _ = _environment_on_real_sandbox(
        tmp_path,
        ["echo 0.8 > /home/user/logs/verifier/reward.txt", "exit 1"],
    )

    _run_cell(env, _REBIND_SUBPROCESS)
    _submit_answer(env)

    # The rebound function would have reported "1.0" for the read.
    assert env.state.last_reward == 0.8


def test_verification_does_not_inherit_the_notebook_directory(monkeypatch, tmp_path):
    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    # Restores the working directory after the cell changes it.
    monkeypatch.chdir(tmp_path)
    env, _, processes = _environment_on_real_sandbox(home, ["pwd"])

    _run_cell(env, f"import os\nos.chdir({str(elsewhere)!r})\n")
    _submit_answer(env)

    assert [result.output for result in env.state.verify_results] == [str(home)]
    assert {
        (call["background"], call["user"], call["cwd"], call["timeout"])
        for call in processes.calls
    } == {(True, "root", "/home/user", 120)}


def _sandbox_with_handle(handle=None, start_error=None):
    sandbox = E2BSandbox.__new__(E2BSandbox)

    def run(cmd, **kwargs):
        if start_error is not None:
            raise start_error
        return handle

    sandbox._sbx = SimpleNamespace(commands=SimpleNamespace(run=run))
    return sandbox


def test_run_command_reports_a_non_zero_exit():
    handle = _Handle(wait_error=_CommandExit(3, "partial output", "boom"))

    result = _sandbox_with_handle(handle).run_command("false")

    assert not result.success
    assert result.error == "exit code 3"
    assert (result.stdout, result.stderr) == ("partial output", "boom")
    # The command exited, so there is nothing to stop.
    assert not handle.killed


def test_run_command_kills_a_command_it_stops_waiting_for_and_raises():
    # The kill reports that it stopped nothing, as E2B's does when the command
    # has already left a child behind: either way the command is not a result.
    handle = _Handle(
        wait_error=TimeoutError("context deadline exceeded"), kill_result=False
    )
    sandbox = _sandbox_with_handle(handle)

    with pytest.raises(TimeoutError):
        sandbox.run_command("sleep 999")

    assert handle.killed


def test_run_command_raises_the_wait_failure_even_when_the_kill_fails():
    handle = _Handle(
        wait_error=TimeoutError("context deadline exceeded"),
        kill_error=RuntimeError("process not found"),
    )

    with pytest.raises(TimeoutError):
        _sandbox_with_handle(handle).run_command("sleep 999")


def test_run_command_propagates_a_command_that_could_not_start():
    sandbox = _sandbox_with_handle(start_error=ConnectionError("sandbox unreachable"))

    with pytest.raises(ConnectionError):
        sandbox.run_command("true")


def test_run_command_matches_the_sdk_exception():
    e2b = pytest.importorskip("e2b")
    exc = e2b.CommandExitException(stderr="boom", stdout="", exit_code=2, error=None)
    handle = _Handle(wait_error=exc)

    result = _sandbox_with_handle(handle).run_command("false")

    assert not result.success
    assert result.error == "exit code 2"
    assert result.stderr == "boom"
    assert not handle.killed


def test_a_sandbox_failure_during_verification_is_not_a_reward(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    env, _, processes = _environment_on_real_sandbox(home, ["true", "true"])

    def unreachable(*args, **kwargs):
        raise ConnectionError("sandbox unreachable")

    # The sandbox goes away after the agent's work, while verification runs.
    processes.run = unreachable
    obs = env.step(
        CallToolAction(tool_name="final_answer", arguments={"answer": "done"})
    )

    # Reported as an error rather than scored: no verify results, no reward,
    # and the episode is not finished.
    assert "sandbox unreachable" in obs.error.message
    assert env.state.verify_results == []
    assert env.state.last_reward is None
    assert not obs.done


# What a root agent can write into the profile a login shell sources: an EXIT
# trap replaces the status of whatever the shell was asked to run.
_POISONED_PROFILE = "trap 'exit 0' EXIT\n"


def test_a_poisoned_login_profile_still_defeats_a_failing_verify_command(
    monkeypatch, tmp_path
):
    """The limit this change does not close, pinned so it cannot move unnoticed.

    E2B runs each command as `bash -l -c` and the notebook kernel runs as root,
    so a cell can write the profile that login shell sources. Verification runs
    outside the kernel, not outside the agent's reach: the boundary that would
    stop this is tracked in #1232.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    env, _, _ = _environment_on_real_sandbox(home, ["exit 1"])

    _run_cell(
        env,
        f"open({str(home / '.bash_profile')!r}, 'w').write({_POISONED_PROFILE!r})",
    )
    _submit_answer(env)

    # The verify command fails, the profile reports success for it anyway.
    assert [result.success for result in env.state.verify_results] == [True]
    assert env.state.last_reward == 1.0


# The cases @zuraiz-anjum ran against `coding_tools_env` on #1232. The task
# ships its own tests from setup; the agent submits an answer and nothing else.
_WRONG_ANSWER = """\
def add(a, b):
    return a - b
"""
_RIGHT_ANSWER = """\
def add(a, b):
    return a + b
"""
_TASK_TESTS = """\
from solution import add


def test_add():
    assert add(2, 3) == 5
"""
_CHEATING_CONFTEST = """\
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    outcome.get_result().outcome = "passed"
"""


def _setup_commands():
    """What the task does in whichever sandbox verification runs in."""
    return [
        f"mkdir -p /home/user/tests && cat > /home/user/tests/test_add.py <<'PYEOF'\n{_TASK_TESTS}PYEOF"
    ]


def _verify_commands(submission_subdir=""):
    source = f"/submission/{submission_subdir}".rstrip("/")
    return [
        f"cd /home/user/tests && cp {source}/* . 2>/dev/null; "
        "python -m pytest -q -p no:cacheprovider test_add.py"
    ]


def _write_cell(path, text):
    return f"open({str(path)!r}, 'w').write({text!r})"


def _submission_episode(
    tmp_path, cells, submit_paths, submission_subdir="", setup=None
):
    agent_home, verifier_home = tmp_path / "agent", tmp_path / "verifier"
    agent_home.mkdir()
    verifier_home.mkdir()
    env, processes, verifier, verifier_processes = _environment_with_a_verifier_sandbox(
        agent_home,
        verifier_home,
        _verify_commands(submission_subdir),
        submit_paths,
        setup_commands=_setup_commands() if setup is None else setup,
    )
    for code in cells:
        _run_cell(env, code)
    _submit_answer(env)
    return env, agent_home, verifier_home, verifier_processes


def test_a_wrong_answer_scores_zero_in_a_fresh_sandbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, _, _ = _submission_episode(
        tmp_path,
        [_write_cell(agent / "solution.py", _WRONG_ANSWER)],
        submit_paths=("solution.py",),
    )

    assert env.state.last_reward == 0.0
    assert env.state.submission_refused == []


def test_a_right_answer_scores_one_in_a_fresh_sandbox(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, verifier_home, _ = _submission_episode(
        tmp_path,
        [_write_cell(agent / "solution.py", _RIGHT_ANSWER)],
        submit_paths=("solution.py",),
    )

    assert env.state.last_reward == 1.0
    assert (verifier_home / "submission/solution.py").read_text() == _RIGHT_ANSWER


def test_a_cheating_conftest_under_a_declared_directory_is_left_behind(
    monkeypatch, tmp_path
):
    """The case that takes a wrong answer to 1.0 when the verifier reads a
    `conftest.py` the agent wrote."""
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, verifier_home, _ = _submission_episode(
        tmp_path,
        [
            f"import os\nos.makedirs({str(agent / 'work')!r}, exist_ok=True)",
            _write_cell(agent / "work/solution.py", _WRONG_ANSWER),
            _write_cell(agent / "work/conftest.py", _CHEATING_CONFTEST),
        ],
        submit_paths=("work",),
        submission_subdir="work",
    )

    assert env.state.last_reward == 0.0
    assert any("conftest.py" in reason for reason in env.state.submission_refused)
    assert not (verifier_home / "submission/work/conftest.py").exists()


def test_a_rewritten_task_test_does_not_replace_the_one_setup_wrote(
    monkeypatch, tmp_path
):
    """Setup runs before the submission arrives, and the submission lands
    outside the workspace, so the task's own test file is what runs."""
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, verifier_home, _ = _submission_episode(
        tmp_path,
        [
            _write_cell(agent / "solution.py", _WRONG_ANSWER),
            f"import os\nos.makedirs({str(agent / 'tests')!r}, exist_ok=True)",
            _write_cell(
                agent / "tests/test_add.py", "def test_add():\n    assert True\n"
            ),
        ],
        submit_paths=("solution.py", "tests"),
    )

    assert env.state.last_reward == 0.0
    assert (verifier_home / "tests/test_add.py").read_text() == _TASK_TESTS
    assert (verifier_home / "submission/tests/test_add.py").exists()


def test_a_poisoned_login_profile_does_not_reach_a_fresh_sandbox(monkeypatch, tmp_path):
    """What the fresh sandbox buys over #1233: the poisoned profile is in the
    sandbox the verifier never uses."""
    monkeypatch.chdir(tmp_path)
    agent_home, verifier_home = tmp_path / "agent", tmp_path / "verifier"
    agent_home.mkdir()
    verifier_home.mkdir()
    env, _, _, _ = _environment_with_a_verifier_sandbox(
        agent_home, verifier_home, ["exit 1"], ["answer.txt"]
    )

    _run_cell(env, _write_cell(agent_home / ".bash_profile", _POISONED_PROFILE))
    _run_cell(env, _write_cell(agent_home / "answer.txt", "42\n"))
    _submit_answer(env)

    assert [result.success for result in env.state.verify_results] == [False]
    assert env.state.last_reward == 0.0


@pytest.mark.parametrize(
    "declared", ["/etc/passwd", "../outside.txt", "work/../../outside.txt"]
)
def test_a_path_outside_the_workspace_is_refused(monkeypatch, tmp_path, declared):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, _, _ = _submission_episode(
        tmp_path,
        [_write_cell(agent / "solution.py", _RIGHT_ANSWER)],
        submit_paths=("solution.py", declared),
    )

    assert any(declared in reason for reason in env.state.submission_refused)
    assert env.state.last_reward == 1.0


def test_a_symlink_is_reported_rather_than_followed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent"
    env, _, verifier_home, _ = _submission_episode(
        tmp_path,
        [
            _write_cell(agent / "solution.py", _RIGHT_ANSWER),
            f"import os\nos.symlink('/etc/passwd', {str(agent / 'secrets.txt')!r})",
        ],
        submit_paths=("solution.py", "secrets.txt"),
    )

    assert any("not a regular file" in r for r in env.state.submission_refused)
    assert not (verifier_home / "submission/secrets.txt").exists()


def test_no_declared_paths_keeps_verification_in_the_agent_sandbox(
    monkeypatch, tmp_path
):
    """The opt-in guarantee: a task that declares nothing scores as it did."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    env, _, _ = _environment_on_real_sandbox(home, ["true", "exit 1"])

    def no_second_sandbox():
        raise AssertionError("verification started a sandbox it should not have")

    env._new_sandbox = no_second_sandbox
    _submit_answer(env)

    assert env.state.last_reward == 0.5
    assert env.state.submission_refused == []


def test_setup_failing_in_the_verifier_sandbox_is_not_a_reward(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    agent_home, verifier_home = tmp_path / "agent", tmp_path / "verifier"
    agent_home.mkdir()
    verifier_home.mkdir()
    env, _, _, _ = _environment_with_a_verifier_sandbox(
        agent_home, verifier_home, ["true"], ["answer.txt"], setup_commands=["exit 3"]
    )

    _run_cell(env, _write_cell(agent_home / "answer.txt", "42\n"))
    obs = env.step(
        CallToolAction(tool_name="final_answer", arguments={"answer": "done"})
    )

    assert "setup failed in the verifier sandbox" in obs.error.message
    assert env.state.verify_results == []
    assert env.state.last_reward is None


@pytest.mark.parametrize(
    "name",
    [
        ".bash_profile",
        ".bash_login",
        ".profile",
        ".bashrc",
        "sitecustomize.py",
        "answer.pth",
        "conftest.py",
        "pytest.ini",
        ".pytest.ini",
        "pytest.toml",
        ".pytest.toml",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
    ],
)
def test_a_file_read_before_the_first_assertion_stays_behind(name):
    """@zuraiz-anjum measured the pytest half of this list on #1579: each of
    those files can carry `addopts = --co`, which collects without running and
    reports success."""
    assert submission.excluded_file(f"work/{name}")
    # Naming one exactly is the task author's decision, so it is still allowed.
    assert submission.check_declared_path(f"work/{name}") is None
