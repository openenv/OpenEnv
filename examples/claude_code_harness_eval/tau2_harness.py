# SPDX-License-Identifier: BSD-3-Clause

"""Claude Code as the agent of a τ²-bench conversation, through `HarnessEnvironment`.

`Tau2Harness` runs Claude Code with the domain's tools injected over MCP, and plays
the τ²-bench task around it: each `step()` sends the customer's message to Claude
Code, passes its reply back to τ²-bench's simulated customer, and returns the
customer's next message. A rubric reads τ²-bench's score when the customer ends
the conversation, so the reward is in `observation.reward`.

In production mode (`WS /harness`), messages go straight to Claude Code, so a person
plays the customer and nothing is scored.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from claude_code_adapter import ClaudeCodeAdapter
from fastmcp import FastMCP
from openenv.core.env_server.types import Observation
from openenv.core.harness import HarnessAction, HarnessConfig, HarnessEnvironment
from openenv.core.rubrics import Rubric
from openenv.core.utils import run_async_safely
from tau2_env.server.tau2_environment import Tau2Environment, without_end_tokens

# Claude Code talks to the customer through its replies, not through these tools.
CONVERSATION_TOOLS = ("respond_to_user", "done")
AGENT_PROMPT = (
    "You are a customer service agent. Each message you receive is from the "
    "customer, and your reply is sent back to them. Use the tools to look things "
    "up and to make changes, and follow this policy. When the policy states the "
    "current time, use it instead of today's date:\n\n"
)


def domain_tools(tau2: Tau2Environment) -> FastMCP:
    """The domain's tools of a τ²-bench environment, to inject into the harness."""
    mcp = FastMCP(f"tau2_{tau2.domain}")
    for tool in run_async_safely(tau2.mcp_server.list_tools()):
        if tool.name not in CONVERSATION_TOOLS:
            mcp.add_tool(tool)
    return mcp


class Tau2Score(Rubric):
    """τ²-bench's own score, once the customer has ended the conversation."""

    def __init__(self, tau2: Tau2Environment):
        super().__init__()
        self.tau2 = tau2

    def forward(self, action: Any, observation: Observation) -> float:
        return self.tau2.state.reward if self.tau2.state.done else 0.0


class Tau2Harness(HarnessEnvironment):
    """
    Claude Code as the agent of one τ²-bench task.

    Args:
        tau2 (`Tau2Environment`):
            The τ²-bench environment. Each `reset()` starts the task on it, and it is
            closed with the harness.
        task_id (`str`):
            The τ²-bench task to play.
        config (`HarnessConfig`):
            How to launch Claude Code.
    """

    def __init__(self, tau2: Tau2Environment, task_id: str, config: HarnessConfig):
        super().__init__(
            adapter=ClaudeCodeAdapter(config),
            mcp=domain_tools(tau2),
            rubric=Tau2Score(tau2),
        )
        self.tau2 = tau2
        self.task_id = task_id

    async def reset_async(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs
    ) -> Observation:
        # A fresh copy of the task's database, and the customer's opening message.
        task = (await asyncio.to_thread(self.tau2.reset, task_id=self.task_id)).metadata
        self.adapter.system_prompt = AGENT_PROMPT + task["policy"]
        observation = await super().reset_async(seed, episode_id, **kwargs)
        observation.metadata["customer"] = task["user_message"]
        return observation

    # Every turn, sync or async, goes through `_run_turn`, which scores it before
    # returning. The customer answers after Claude Code, so score again after that.
    async def _run_turn(
        self, action: HarnessAction, timeout_s: Optional[float] = None
    ) -> Observation:
        observation = await super()._run_turn(action, timeout_s=timeout_s)
        if observation.done:  # e.g. Claude Code crashed
            return observation
        reply = observation.metadata["response"] or "(no reply)"
        try:
            customer = await asyncio.to_thread(self.tau2.act, reply)
        except RuntimeError as error:  # e.g. the simulated customer's model failed
            return await self._terminal_error_observation(
                str(error), error_type="customer_failed"
            )
        observation.metadata["customer"] = without_end_tokens(customer)
        observation.done = self.tau2.state.done
        observation.reward = await self._apply_rubric_async(action, observation)
        return observation

    def close(self) -> None:
        super().close()
        self.tau2.close()
