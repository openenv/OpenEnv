# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
from coding_tools_env.models import CodingToolsState
from coding_tools_env.server.coding_tools_env_environment import CodingToolsEnvironment
from coding_tools_env.server.e2b_sandbox import ToolResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class FakeSandbox:
    sandbox_id = "fake-sandbox"

    def __init__(self) -> None:
        self.files = {}
        self.killed = False

    def run_shell(self, command: str, timeout_s: float = 30) -> ToolResult:
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
    """FakeSandbox whose reward file shell commands and file tools can write."""

    def __init__(self, *, removable: bool = True) -> None:
        super().__init__()
        self.removable = removable

    def run_shell(self, command: str, timeout_s: float = 30) -> ToolResult:
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
        return super().run_shell(command, timeout_s)


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
