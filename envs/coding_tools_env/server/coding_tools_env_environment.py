# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SETA-style multi-tool coding environment backed by E2B."""

from __future__ import annotations

import math
import os
from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

try:
    from .e2b_sandbox import E2BSandbox
    from ..models import CodingToolsState, CommandResult, EditSpec, TodoItem
except ImportError:  # pragma: no cover
    from models import CodingToolsState, CommandResult, EditSpec, TodoItem
    from server.e2b_sandbox import E2BSandbox


REWARD_FILE = "/home/user/logs/verifier/reward.txt"
_CLEAR_REWARD_FILE = f"rm -f {REWARD_FILE} && test ! -e {REWARD_FILE}"


class CodingToolsEnvironment(MCPEnvironment):
    """Tool-centric coding environment with one sandbox per episode."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._sandbox: Optional[E2BSandbox] = None
        self._state = CodingToolsState(episode_id=str(uuid4()), step_count=0)

        mcp = FastMCP("coding_tools_env")

        @mcp.tool
        def bash(command: str, timeout: float | None = 30) -> str:
            """Execute bash commands using the computer instance."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            timeout_value = 30 if timeout is None else float(timeout)
            result = self._sandbox.run_shell(command, timeout_s=timeout_value)
            self._record("bash", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}\n{result.output}".strip()

        @mcp.tool
        def read(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
            """Read file contents using computer instance."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.read_file(file_path=file_path, offset=offset, limit=limit)
            self._record("read", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}"

        @mcp.tool
        def write(file_path: str, content: str) -> str:
            """Write content to a file using computer instance."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.write_file(file_path=file_path, content=content)
            self._record("write", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}"

        @mcp.tool
        def edit(
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> str:
            """Perform exact string replacement in a file."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            read_result = self._sandbox.read_file(file_path=file_path)
            if not read_result.ok:
                self._record("edit", False, "", read_result.error, None)
                return f"ERROR: {read_result.error}"
            original = read_result.output
            if old_string not in original:
                self._record("edit", False, "", "old_string not found", None)
                return "ERROR: old_string not found"
            if replace_all:
                updated = original.replace(old_string, new_string)
            else:
                updated = original.replace(old_string, new_string, 1)
            write_result = self._sandbox.write_file(file_path=file_path, content=updated)
            ok = write_result.ok
            msg = "edit ok" if ok else ""
            self._record("edit", ok, msg, write_result.error, {"replace_all": replace_all})
            return msg if ok else f"ERROR: {write_result.error}"

        @mcp.tool
        def multi_edit(file_path: str, edits: list[dict[str, Any]]) -> str:
            """Perform multiple edits on a single file."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            read_result = self._sandbox.read_file(file_path=file_path)
            if not read_result.ok:
                self._record("multi_edit", False, "", read_result.error, None)
                return f"ERROR: {read_result.error}"
            text = read_result.output
            applied = 0
            for raw in edits:
                spec = EditSpec.model_validate(raw)
                if spec.old_string not in text:
                    self._record(
                        "multi_edit",
                        False,
                        "",
                        f"old_string not found: {spec.old_string[:80]}",
                        {"applied": applied},
                    )
                    return f"ERROR: old_string not found: {spec.old_string[:80]}"
                if spec.replace_all:
                    text = text.replace(spec.old_string, spec.new_string)
                else:
                    text = text.replace(spec.old_string, spec.new_string, 1)
                applied += 1
            write_result = self._sandbox.write_file(file_path=file_path, content=text)
            self._record(
                "multi_edit",
                write_result.ok,
                f"applied {applied} edits" if write_result.ok else "",
                write_result.error,
                {"applied": applied},
            )
            return f"applied {applied} edits" if write_result.ok else f"ERROR: {write_result.error}"

        @mcp.tool
        def glob(pattern: str, path: str | None = None) -> str:
            """Find files matching a glob pattern."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.glob_files(pattern=pattern, path=path)
            self._record("glob", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}"

        @mcp.tool
        def grep(pattern: str, path: str | None = None, include: str | None = None) -> str:
            """Search for patterns in files."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.grep(pattern=pattern, path=path, include=include)
            self._record("grep", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}\n{result.output}".strip()

        @mcp.tool
        def ls(path: str = ".", ignore: list[str] | None = None) -> str:
            """List files and directories."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            result = self._sandbox.list_dir(path=path, ignore=ignore)
            self._record("ls", result.ok, result.output, result.error, result.metadata)
            return result.output if result.ok else f"ERROR: {result.error}"

        @mcp.tool
        def todo_write(todos: list[dict[str, Any]]) -> str:
            """Manage todo list for planning and progress tracking."""
            validated = [TodoItem.model_validate(todo) for todo in todos]
            in_progress = [item for item in validated if item.status == "in_progress"]
            if len(in_progress) > 1:
                msg = "ERROR: only one todo item can be in_progress"
                self._record("todo_write", False, "", msg, None)
                return msg
            for item in validated:
                if item.status not in {"pending", "in_progress", "completed"}:
                    msg = f"ERROR: invalid status {item.status}"
                    self._record("todo_write", False, "", msg, None)
                    return msg
                if item.priority not in {"high", "medium", "low"}:
                    msg = f"ERROR: invalid priority {item.priority}"
                    self._record("todo_write", False, "", msg, None)
                    return msg
            self._state.todos = validated
            self._record("todo_write", True, f"stored {len(validated)} todos", None, None)
            return f"stored {len(validated)} todos"

        @mcp.tool
        def submit_solution() -> str:
            """Submit solution and run test suite via verify commands."""
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            self._state.submitted = True
            if not self._state.verify_commands:
                self._state.last_reward = 0.0
                self._record(
                    "submit_solution",
                    True,
                    "No verify commands configured. reward=0.0",
                    None,
                    {"reward": 0.0, "finished": True},
                )
                return "No verify commands configured. reward=0.0"
            summary = self._run_verify_commands()
            self._record(
                "submit_solution",
                True,
                (
                    f"Verification: {summary['passed']}/{summary['total']} passed; "
                    f"reward={summary['reward']}"
                ),
                None,
                {"reward": summary["reward"], "finished": True},
            )
            return (
                f"Verification: {summary['passed']}/{summary['total']} passed; "
                f"reward={summary['reward']}"
            )

        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

        api_key = os.environ.get("E2B_API_KEY")
        self._state = CodingToolsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        if not api_key:
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": "E2B_API_KEY is not set. Configure it before reset.",
                },
            )

        try:
            self._sandbox = E2BSandbox(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": f"failed to create E2B sandbox: {type(exc).__name__}: {exc}",
                },
            )

        self._state.sandbox_id = self._sandbox.sandbox_id
        setup_commands = _coerce_commands(
            kwargs.get("setup", kwargs.get("setup_scripts", []))
        )
        verify_commands = _coerce_commands(
            kwargs.get("verify", kwargs.get("verify_scripts", []))
        )
        self._state.verify_commands = verify_commands

        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        if setup_commands:
            for command in setup_commands:
                result = self._sandbox.run_shell(command, timeout_s=60)
                command_result = CommandResult(
                    tool="setup",
                    ok=result.ok,
                    output=result.output,
                    error=result.error,
                    metadata={"command": command},
                )
                self._state.setup_results.append(command_result)
                if not result.ok:
                    return Observation(
                        done=True,
                        reward=None,
                        metadata={
                            "status": "error",
                            "sandbox_id": self._state.sandbox_id,
                            "message": "Setup command failed.",
                            "setup_results": [
                                entry.model_dump() for entry in self._state.setup_results
                            ],
                        },
                    )

        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "sandbox_id": self._state.sandbox_id,
                "message": "coding_tools_env ready.",
                "verify_commands": verify_commands,
                "setup_results": [
                    entry.model_dump() for entry in self._state.setup_results
                ],
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction for MCP interactions."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        if self._state.submitted and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        return obs

    @property
    def state(self) -> CodingToolsState:
        return self._state

    def close(self) -> None:
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _record(
        self,
        tool: str,
        ok: bool,
        output: str,
        error: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        result = CommandResult(
            tool=tool,
            ok=ok,
            output=output,
            error=error,
            metadata=metadata or {},
        )
        self._state.tool_history.append(result)
        self._state.last_error = error

    def _run_verify_commands(self) -> dict[str, Any]:
        if not self._sandbox:
            return {"passed": 0, "total": 0, "reward": None}
        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        # The override is documented as verify-written, so drop anything the
        # policy wrote there during the episode before verification starts.
        cleared = self._sandbox.run_shell(_CLEAR_REWARD_FILE).ok
        self._state.verify_results = []
        passed = 0
        for command in self._state.verify_commands:
            result = self._sandbox.run_shell(command, timeout_s=120)
            record = CommandResult(
                tool="verify",
                ok=result.ok,
                output=result.output,
                error=result.error,
                metadata={"command": command},
            )
            self._state.verify_results.append(record)
            if result.ok:
                passed += 1
        total = len(self._state.verify_commands)
        reward, self._state.reward_override_ignored = _read_reward_override(
            self._sandbox, cleared
        )
        if reward is None:
            reward = (passed / total) if total else 0.0
        self._state.last_reward = reward
        return {"passed": passed, "total": total, "reward": reward}


def _coerce_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _read_reward_override(
    sandbox: E2BSandbox, cleared: bool
) -> tuple[float | None, str | None]:
    if not cleared:
        return None, "reward file could not be removed before verification"
    result = sandbox.read_file(REWARD_FILE)
    if not result.ok:
        return None, None
    return _parse_reward_override(result.output or "")


def _parse_reward_override(raw: str) -> tuple[float | None, str | None]:
    """Parse the reward file written by a verify command.

    The value comes from the task author rather than the agent, so any finite
    reward is accepted, including negative ones and values above 1. Only a
    value that would poison training is rejected.

    Returns `(reward, None)` for a finite value, `(None, None)` when the file
    is empty, and `(None, reason)` when the value is rejected, in which case
    the caller falls back to the verify pass rate.
    """
    raw = raw.strip()
    if not raw:
        return None, None
    try:
        value = float(raw)
    except ValueError:
        return None, f"not a number: {raw[:40]!r}"
    if not math.isfinite(value):
        return None, f"not finite: {raw[:40]!r}"
    return value, None
