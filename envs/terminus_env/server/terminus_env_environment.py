# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2B-backed single-tool coding environment inspired by Terminus."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

try:
    from .e2b_sandbox import E2BSandbox
    from ..models import CommandResult, TerminusState
except ImportError:  # pragma: no cover
    from models import CommandResult, TerminusState
    from server.e2b_sandbox import E2BSandbox


REWARD_FILE = "/home/user/logs/verifier/reward.txt"
_CLEAR_REWARD_FILE = f"rm -f {REWARD_FILE} && test ! -e {REWARD_FILE}"


class TerminusEnvironment(MCPEnvironment):
    """Single-tool terminal environment with one E2B sandbox per episode."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._sandbox: Optional[E2BSandbox] = None
        self._state = TerminusState(episode_id=str(uuid4()), step_count=0)

        mcp = FastMCP("terminus_env")

        @mcp.tool
        def terminal(command: str = "", final_answer: str = "") -> str:
            """Run a shell command or submit a final answer inside the sandbox.

            Args:
                command: Shell command to execute in the episode's E2B sandbox.
                final_answer: Optional answer string. When provided, stored
                    as the final answer and any reset-time verify commands run.

            Returns:
                Command output, or final-answer verification summary.
            """
            if not self._sandbox:
                return "Error: environment not reset. Call reset() first."
            if final_answer:
                self._state.submitted_answer = final_answer
                if not self._state.verify_commands:
                    return f"Answer submitted: {final_answer}"
                summary = self._run_verify_commands()
                return (
                    f"Answer submitted: {final_answer}\n"
                    f"Verification: {summary['passed']}/{summary['total']} passed; "
                    f"reward={summary['reward']}"
                )
            if not command.strip():
                return "Error: command or final_answer is required."
            result = self._run_shell_command(command)
            self._state.commands.append(result)
            return result.output

        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Create a fresh E2B sandbox and run optional setup commands."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

        api_key = os.environ.get("E2B_API_KEY")
        self._state = TerminusState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        if not api_key:
            return Observation(
                done=True,
                reward=None,
                metadata={
                    "status": "error",
                    "error": (
                        "E2B_API_KEY is not set. Configure it before resetting "
                        "terminus_env."
                    ),
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
            setup_results = self._run_shell_commands(setup_commands)
            self._state.setup_results = setup_results
            failed = [result for result in setup_results if not result.success]
            if failed:
                return Observation(
                    done=True,
                    reward=None,
                    metadata={
                        "status": "error",
                        "sandbox_id": self._state.sandbox_id,
                        "message": "Setup command failed.",
                        "setup_results": [
                            result.model_dump() for result in setup_results
                        ],
                    },
                )

        msg = "Terminus environment ready. Use terminal(command=...) to work."
        if setup_commands:
            msg += f" Setup commands run: {len(setup_commands)}."
        if verify_commands:
            msg += f" Verify commands registered: {len(verify_commands)}."
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "sandbox_id": self._state.sandbox_id,
                "message": msg,
                "setup_results": [
                    result.model_dump() for result in self._state.setup_results
                ],
                "verify_commands": verify_commands,
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
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
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
        if self._state.submitted_answer is not None and self._state.last_reward is not None:
            obs.done = True
            obs.reward = self._state.last_reward
        return obs

    @property
    def state(self) -> TerminusState:
        return self._state

    def close(self) -> None:
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None

    def _run_shell_commands(self, commands: Iterable[str]) -> list[CommandResult]:
        return [self._run_shell_command(command) for command in commands]

    def _run_shell_command(self, command: str) -> CommandResult:
        result = self._sandbox.run_shell(command)
        output = _format_for_llm(result)
        return CommandResult(
            command=command,
            output=output,
            error=result.error,
            success=result.success,
        )

    def _run_verify_commands(self) -> dict[str, Any]:
        if not self._sandbox:
            return {"passed": 0, "total": 0, "reward": None}

        self._sandbox.run_shell("mkdir -p /home/user/logs/verifier")
        # The override is documented as verify-written, so drop anything the
        # policy wrote there during the episode before verification starts.
        cleared = self._sandbox.run_shell(_CLEAR_REWARD_FILE).success
        verify_results = self._run_shell_commands(self._state.verify_commands)
        self._state.verify_results = verify_results
        passed = sum(1 for result in verify_results if result.success)
        total = len(verify_results)
        reward, self._state.reward_override_ignored = _read_reward_override(
            self._sandbox, cleared
        )
        if reward is None and total:
            reward = passed / total
        self._state.last_reward = reward
        return {"passed": passed, "total": total, "reward": reward}


def _coerce_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _format_for_llm(result) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.strip())
    if result.stderr:
        parts.append(result.stderr.strip())
    if result.error:
        parts.append(f"ERROR:\n{result.error}")
    return "\n".join(parts) if parts else "(no output)"


def _read_reward_override(
    sandbox: E2BSandbox, cleared: bool
) -> tuple[Optional[float], Optional[str]]:
    if not cleared:
        return None, "reward file could not be removed before verification"
    result = sandbox.run_shell(f"cat {REWARD_FILE} 2>/dev/null || true")
    return _parse_reward_override(result.stdout or "")


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
