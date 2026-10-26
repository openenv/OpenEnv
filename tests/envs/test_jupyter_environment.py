# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest
from jupyter_env.models import JupyterState
from jupyter_env.server.e2b_sandbox import CellResult, E2BSandbox
from jupyter_env.server.jupyter_environment import JupyterEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.killed = False
        self.shell_commands: list[str] = []

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
    """FakeSandbox whose reward file any shell command can write, as in E2B."""

    def __init__(self, *, removable: bool = True) -> None:
        super().__init__()
        self.reward_file: str | None = None
        self.removable = removable

    def run_shell(self, command: str) -> CellResult:
        if command.startswith("echo ") and command.endswith(f" > {REWARD_FILE}"):
            self.shell_commands.append(command)
            self.reward_file = command[len("echo ") : -len(f" > {REWARD_FILE}")]
            return _cell()
        if command.startswith(f"rm -f {REWARD_FILE}"):
            self.shell_commands.append(command)
            if self.removable:
                self.reward_file = None
            return _cell(success=self.reward_file is None)
        if command.startswith(f"cat {REWARD_FILE}"):
            self.shell_commands.append(command)
            return _cell(stdout=self.reward_file or "")
        return super().run_shell(command)


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
