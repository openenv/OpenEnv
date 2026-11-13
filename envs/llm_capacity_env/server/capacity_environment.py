"""OpenEnv MCP boundary. Agents receive tools, never simulation controls."""

import os
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State

from .engine import Replay
from .trajectory import TrajectoryRecorder


class CapacityEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, split=None, trace_dir=None):
        self.replay = Replay(split or os.getenv("CAPACITY_SPLIT", "train"))
        self._recorder = TrajectoryRecorder(trace_dir or os.getenv("CAPACITY_TRACE_DIR"))
        self._state = State(episode_id=str(uuid4()), step_count=0)
        mcp = FastMCP("llm_capacity_env")

        @mcp.tool
        def observe_demand_history() -> dict:
            """Read the last 14 realized days, cost rules and previous outcome; time does not advance."""
            return self.replay.observe()

        @mcp.tool
        def reserve_capacity(reservations: dict[str, list[float]]) -> dict:
            """Commit next day's hourly token reservations: ChatGPT and GPT-4, each 24 values in [0,1000000].

            Advances exactly one day and reveals that day's outcome. Decisions cannot be undone.
            Each day's reward is negative modeled cost / 1000. Minimize total modeled cost.
            """
            result = self.replay.reserve(reservations)
            self._state.step_count = self.replay.turn
            return result

        super().__init__(mcp)

    def reset(self, seed=None, episode_id=None, **kwargs):
        view = self.replay.reset(seed)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        observation = Observation(reward=0.0, done=False, metadata=view)
        self._recorder.start(seed, self._state.episode_id, view)
        return observation

    def _finish(self, observation, before, action):
        observation.done = self.replay.done
        observation.reward = self.replay.previous["reward"] if self.replay.turn > before else 0.0
        self._recorder.append(action, observation, self._state.step_count)
        return observation

    def step(self, action, timeout_s=None, **kwargs):
        before = getattr(self.replay, "turn", 0)
        return self._finish(super().step(action, timeout_s=timeout_s, **kwargs), before, action)

    async def step_async(self, action, timeout_s=None, **kwargs):
        before = getattr(self.replay, "turn", 0)
        result = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._finish(result, before, action)

    def _step_impl(self, action, timeout_s=None, **kwargs):
        raise ValueError("Use MCP tools: observe_demand_history or reserve_capacity")

    @property
    def state(self):
        return self._state.model_copy(deep=True)
