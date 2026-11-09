# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import subprocess
from types import SimpleNamespace

import pytest
from coding_tools_env.models import CodingToolsState
from coding_tools_env.server.coding_tools_env_environment import CodingToolsEnvironment
from coding_tools_env.server.e2b_sandbox import E2BSandbox, ToolResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.files = {}
        self.killed = False
        self.shell_commands: list[str] = []
        self.process_commands: list[str] = []

    def run_shell(self, command: str, timeout_s: float = 30) -> ToolResult:
        self.shell_commands.append(command)
        return self._respond(command)

    def run_command(self, command: str, timeout_s: float = 30) -> ToolResult:
        self.process_commands.append(command)
        return self._respond(command)

    def _respond(self, command: str) -> ToolResult:
        if "exit 1" in command:
            return ToolResult(
                ok=False, output="", error="exit_code=1", metadata={"exit_code": 1}
            )
        if command.startswith("cat /home/user/logs/verifier/reward.txt"):
            return ToolResult(ok=True, output="", error=None, metadata={"exit_code": 0})
        return ToolResult(
            ok=True, output=f"shell: {command}", error=None, metadata={"exit_code": 0}
        )

    def read_file(self, file_path: str, offset=None, limit=None) -> ToolResult:
        if file_path not in self.files:
            return ToolResult(ok=False, output="", error="file not found", metadata={})
        value = self.files[file_path]
        if offset is not None:
            value = value[offset:]
        if limit is not None:
            value = value[:limit]
        return ToolResult(ok=True, output=value, error=None, metadata={})

    def write_file(self, file_path: str, content: str) -> ToolResult:
        self.files[file_path] = content
        return ToolResult(
            ok=True, output="write ok", error=None, metadata={"bytes": len(content)}
        )

    def glob_files(self, pattern: str, path: str | None = None) -> ToolResult:
        return ToolResult(
            ok=True,
            output="\n".join(sorted(self.files.keys())),
            error=None,
            metadata={},
        )

    def grep(
        self, pattern: str, path: str | None = None, include: str | None = None
    ) -> ToolResult:
        matches = [
            f"{name}:1:{text}" for name, text in self.files.items() if pattern in text
        ]
        return ToolResult(ok=True, output="\n".join(matches), error=None, metadata={})

    def list_dir(self, path: str = ".", ignore=None) -> ToolResult:
        names = sorted(name.rsplit("/", 1)[-1] for name in self.files.keys())
        return ToolResult(
            ok=True,
            output="\n".join(f"[file] {name}" for name in names),
            error=None,
            metadata={},
        )

    def kill(self) -> None:
        self.killed = True


def _extract_text(result) -> str:
    if hasattr(result, "content") and result.content:
        return result.content[0].text
    if hasattr(result, "data"):
        return str(result.data)
    return str(result)


def test_lists_expected_tools():
    env = CodingToolsEnvironment()
    obs = env.step(ListToolsAction())
    names = {tool.name for tool in obs.tools}
    assert names == {
        "bash",
        "edit",
        "glob",
        "grep",
        "ls",
        "multi_edit",
        "read",
        "submit_solution",
        "todo_write",
        "write",
    }


def test_reset_without_e2b_key_fails(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    env = CodingToolsEnvironment()
    obs = env.reset()
    assert obs.done is True
    assert obs.metadata["status"] == "error"


def test_write_read_edit_and_submit(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake")
    fake = FakeSandbox()
    monkeypatch.setattr(
        "coding_tools_env.server.coding_tools_env_environment.E2BSandbox",
        lambda api_key: fake,
    )
    env = CodingToolsEnvironment()
    reset_obs = env.reset(setup=["echo setup"], verify=["test -f answer.txt"])
    assert reset_obs.done is False

    write_obs = env.step(
        CallToolAction(
            tool_name="write",
            arguments={"file_path": "/home/user/work/a.txt", "content": "hello world"},
        )
    )
    assert "write ok" in _extract_text(write_obs.result)

    read_obs = env.step(
        CallToolAction(
            tool_name="read",
            arguments={"file_path": "/home/user/work/a.txt"},
        )
    )
    assert _extract_text(read_obs.result) == "hello world"

    edit_obs = env.step(
        CallToolAction(
            tool_name="edit",
            arguments={
                "file_path": "/home/user/work/a.txt",
                "old_string": "hello",
                "new_string": "hi",
                "replace_all": False,
            },
        )
    )
    assert "edit ok" in _extract_text(edit_obs.result)
    assert fake.files["/home/user/work/a.txt"] == "hi world"

    todo_obs = env.step(
        CallToolAction(
            tool_name="todo_write",
            arguments={
                "todos": [
                    {
                        "id": "1",
                        "content": "a",
                        "status": "completed",
                        "priority": "high",
                    },
                    {
                        "id": "2",
                        "content": "b",
                        "status": "in_progress",
                        "priority": "medium",
                    },
                ]
            },
        )
    )
    assert "stored 2 todos" in _extract_text(todo_obs.result)
    assert len(env.state.todos) == 2

    submit_obs = env.step(CallToolAction(tool_name="submit_solution", arguments={}))
    assert "Verification: 1/1 passed; reward=1.0" in _extract_text(submit_obs.result)
    assert env.state.last_reward == 1.0
    assert "test -f answer.txt" in fake.process_commands
    assert "test -f answer.txt" not in fake.shell_commands


def test_todo_write_rejects_multiple_in_progress(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "fake")
    monkeypatch.setattr(
        "coding_tools_env.server.coding_tools_env_environment.E2BSandbox",
        lambda api_key: FakeSandbox(),
    )
    env = CodingToolsEnvironment()
    env.reset()
    obs = env.step(
        CallToolAction(
            tool_name="todo_write",
            arguments={
                "todos": [
                    {
                        "id": "1",
                        "content": "a",
                        "status": "in_progress",
                        "priority": "high",
                    },
                    {
                        "id": "2",
                        "content": "b",
                        "status": "in_progress",
                        "priority": "medium",
                    },
                ]
            },
        )
    )
    assert "only one todo item can be in_progress" in _extract_text(obs.result)
    env._state = CodingToolsState(episode_id="x")


REWARD_FILE = "/home/user/logs/verifier/reward.txt"


class RewardFileSandbox(FakeSandbox):
    """FakeSandbox whose reward file shell commands and file tools can write.

    The agent writes to it through `run_shell` and the file tools; verification
    clears, writes and reads it as sandbox processes through `run_command`.
    """

    def __init__(self, *, removable: bool = True) -> None:
        super().__init__()
        self.removable = removable

    def _reward_file_command(self, command: str) -> ToolResult | None:
        if command.startswith("echo ") and command.endswith(f" > {REWARD_FILE}"):
            self.files[REWARD_FILE] = command[len("echo ") : -len(f" > {REWARD_FILE}")]
            return ToolResult(ok=True, output="", error=None, metadata={"exit_code": 0})
        if command.startswith(f"rm -f {REWARD_FILE}"):
            if self.removable:
                self.files.pop(REWARD_FILE, None)
            gone = REWARD_FILE not in self.files
            return ToolResult(
                ok=gone,
                output="",
                error=None if gone else "exit_code=1",
                metadata={"exit_code": 0 if gone else 1},
            )
        if command.startswith(f"cat {REWARD_FILE}"):
            return ToolResult(
                ok=True,
                output=self.files.get(REWARD_FILE, ""),
                error=None,
                metadata={"exit_code": 0},
            )
        return None

    def run_shell(self, command: str, timeout_s: float = 30) -> ToolResult:
        result = self._reward_file_command(command)
        if result is None:
            return super().run_shell(command, timeout_s)
        self.shell_commands.append(command)
        return result

    def run_command(self, command: str, timeout_s: float = 30) -> ToolResult:
        result = self._reward_file_command(command)
        if result is None:
            return super().run_command(command, timeout_s)
        self.process_commands.append(command)
        return result


def _submit(monkeypatch, sandbox, verify_commands, agent_actions=()):
    monkeypatch.setenv("E2B_API_KEY", "fake")
    monkeypatch.setattr(
        "coding_tools_env.server.coding_tools_env_environment.E2BSandbox",
        lambda api_key: sandbox,
    )
    env = CodingToolsEnvironment()
    env.reset(verify=verify_commands)
    for tool_name, arguments in agent_actions:
        env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
    env.step(CallToolAction(tool_name="submit_solution", arguments={}))
    return env.state


@pytest.mark.parametrize(
    "agent_action",
    [
        ("bash", {"command": f"echo 1.0 > {REWARD_FILE}"}),
        ("write", {"file_path": REWARD_FILE, "content": "1.0"}),
    ],
)
def test_reward_file_written_by_the_agent_is_ignored(monkeypatch, agent_action):
    state = _submit(
        monkeypatch, RewardFileSandbox(), ["true", "exit 1"], [agent_action]
    )

    assert state.last_reward == 0.5
    assert state.reward_override_ignored is None


def test_reward_file_written_by_a_verify_command_overrides_the_pass_rate(monkeypatch):
    state = _submit(
        monkeypatch, RewardFileSandbox(), [f"echo 0.8 > {REWARD_FILE}", "exit 1"]
    )

    assert state.last_reward == 0.8
    assert state.reward_override_ignored is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "passed"])
def test_invalid_reward_file_falls_back_to_the_pass_rate(monkeypatch, value):
    state = _submit(
        monkeypatch, RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"]
    )

    assert state.last_reward == 0.5
    assert repr(value) in state.reward_override_ignored


@pytest.mark.parametrize("value", ["-0.5", "1.5"])
def test_reward_file_outside_the_unit_interval_is_kept(monkeypatch, value):
    state = _submit(
        monkeypatch, RewardFileSandbox(), [f"echo {value} > {REWARD_FILE}", "exit 1"]
    )

    assert state.last_reward == float(value)
    assert state.reward_override_ignored is None


def test_reward_file_that_cannot_be_removed_is_ignored(monkeypatch):
    state = _submit(
        monkeypatch,
        RewardFileSandbox(removable=False),
        ["true", "exit 1"],
        [("write", {"file_path": REWARD_FILE, "content": "1.0"})],
    )

    assert state.last_reward == 0.5
    assert "could not be removed" in state.reward_override_ignored


class _SandboxPaths:
    """Maps the sandbox's /home/user and /root onto temporary directories."""

    def __init__(self, tmp_path) -> None:
        self.home = tmp_path / "home"
        self.root = tmp_path / "root"
        self.home.mkdir()
        self.root.mkdir()

    def __call__(self, text: str) -> str:
        mapped = {"/home/user": str(self.home), "/root": str(self.root)}
        return re.sub("/home/user|/root", lambda m: mapped[m.group()], text)


class _Kernel:
    """A kernel whose namespace persists between cells, as E2B's run_code does.

    It runs in the test process, whose working directory stands in for the
    kernel's (the notebook directory, /home/user).
    """

    def __init__(self, paths: _SandboxPaths) -> None:
        self.paths = paths
        self.namespace: dict = {}
        self.cells: list[str] = []

    def run_code(self, code: str, **kwargs):
        self.cells.append(code)
        stdout, stderr, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(self.paths(code), self.namespace)
        except BaseException as exc:  # a kernel reports SystemExit, it doesn't exit
            error = SimpleNamespace(name=type(exc).__name__, value=str(exc))
        return SimpleNamespace(
            error=error,
            logs=SimpleNamespace(
                stdout=[stdout.getvalue()] if stdout.getvalue() else [],
                stderr=[stderr.getvalue()] if stderr.getvalue() else [],
            ),
        )


class _CommandExit(Exception):
    """Stands in for `e2b.CommandExitException`, raised on a non-zero exit."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(f"Command exited with code {exit_code}")
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


class _Processes:
    """E2B's process API: each command is a fresh process with a real exit status.

    envd starts commands as `/bin/bash -l -c`, so the login files in root's
    `HOME` are sourced here too.
    """

    def __init__(self, paths: _SandboxPaths) -> None:
        self.paths = paths
        self.calls: list[dict] = []
        # Captured now, before anything in the kernel gets a chance to rebind it.
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
        done = self._run(
            ["/bin/bash", "-l", "-c", self.paths(cmd)],
            cwd=self.paths(cwd) if cwd else None,
            env={"PATH": "/usr/bin:/bin", "HOME": str(self.paths.root)},
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


def _environment_on_real_sandbox(monkeypatch, tmp_path, verify_commands):
    paths = _SandboxPaths(tmp_path)
    # The kernel starts in the notebook directory; restored after the test.
    monkeypatch.chdir(paths.home)
    kernel, processes = _Kernel(paths), _Processes(paths)
    sandbox = E2BSandbox.__new__(E2BSandbox)
    sandbox._sbx = SimpleNamespace(run_code=kernel.run_code, commands=processes)
    sandbox.sandbox_id = "fake-sandbox"
    env = CodingToolsEnvironment()
    env._sandbox = sandbox
    env._state = CodingToolsState(
        episode_id="episode-1",
        sandbox_id="fake-sandbox",
        verify_commands=verify_commands,
    )
    return env, kernel, processes, paths


def _submit_solution(env):
    return env.step(CallToolAction(tool_name="submit_solution", arguments={}))


# The agent has no cell tool here, but its shell runs as root next to the
# kernel, so code can still reach the kernel by another route. This is the
# reproduction from #1210: a cell that rebinds subprocess.run used to decide the
# outcome of every verify command.
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
    env, kernel, _, _ = _environment_on_real_sandbox(
        monkeypatch, tmp_path, ["exit 1", "exit 1"]
    )

    kernel.run_code(_REBIND_SUBPROCESS)
    _submit_solution(env)

    assert [result.ok for result in env.state.verify_results] == [False, False]
    assert env.state.last_reward == 0.0
    assert not any("exit 1" in cell for cell in kernel.cells)


def test_reward_file_is_read_outside_the_kernel(monkeypatch, tmp_path):
    # Restores the real method after the cell below replaces it.
    monkeypatch.setattr(pathlib.Path, "read_text", pathlib.Path.read_text)
    env, kernel, _, _ = _environment_on_real_sandbox(
        monkeypatch,
        tmp_path,
        ["echo 0.8 > /home/user/logs/verifier/reward.txt", "exit 1"],
    )

    kernel.run_code("import pathlib\npathlib.Path.read_text = lambda *a, **k: '1.0'\n")
    _submit_solution(env)

    # A read through the kernel would have reported the forged "1.0".
    assert env.state.last_reward == 0.8


def test_verification_does_not_inherit_the_kernel_directory(monkeypatch, tmp_path):
    env, kernel, processes, paths = _environment_on_real_sandbox(
        monkeypatch, tmp_path, ["pwd"]
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    kernel.run_code(f"import os\nos.chdir({str(elsewhere)!r})\n")
    _submit_solution(env)

    assert [result.output for result in env.state.verify_results] == [str(paths.home)]
    assert {
        (call["background"], call["user"], call["cwd"]) for call in processes.calls
    } == {(True, "root", "/home/user")}
    # Verify commands keep their 120 s timeout.
    assert [call["timeout"] for call in processes.calls if call["cmd"] == "pwd"] == [
        120
    ]


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

    assert not result.ok
    assert result.error == "exit_code=3"
    assert result.output == "partial output\nboom"
    assert result.metadata == {"exit_code": 3}
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

    assert not result.ok
    assert result.error == "exit_code=2"
    assert result.output == "boom"
    assert not handle.killed


def test_a_sandbox_failure_during_verification_is_not_a_reward(monkeypatch, tmp_path):
    env, _, processes, _ = _environment_on_real_sandbox(
        monkeypatch, tmp_path, ["true", "true"]
    )

    def unreachable(*args, **kwargs):
        raise ConnectionError("sandbox unreachable")

    # The sandbox goes away after the agent's work, while verification runs.
    processes.run = unreachable
    obs = _submit_solution(env)

    # Reported as an error rather than scored: no verify results, no reward,
    # and the episode is not finished.
    assert "sandbox unreachable" in obs.error.message
    assert env.state.verify_results == []
    assert env.state.last_reward is None
    assert not obs.done


def test_a_poisoned_login_profile_still_defeats_a_failing_verify_command(
    monkeypatch, tmp_path
):
    """The limit this change does not close, pinned so it cannot move unnoticed.

    E2B runs each command as `bash -l -c`, and the agent's own shell runs as
    root, so the agent can write the profile that login shell sources.
    Verification runs outside the kernel, not outside the agent's reach: the
    boundary that would stop this is tracked in #1232.
    """
    env, _, _, _ = _environment_on_real_sandbox(monkeypatch, tmp_path, ["exit 1"])

    env.step(
        CallToolAction(
            tool_name="bash",
            arguments={"command": "echo \"trap 'exit 0' EXIT\" > /root/.bash_profile"},
        )
    )
    _submit_solution(env)

    # The verify command fails, the profile reports success for it anyway.
    assert [result.ok for result in env.state.verify_results] == [True]
    assert env.state.last_reward == 1.0
