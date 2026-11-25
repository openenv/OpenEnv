"""Gymnasium-style wrapper (no hard dependency on gymnasium).

    env = GoingConcernGym(tier=1, obs_format="text")
    obs, info = env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(action)   # action: dict or JSON string

Reward is 0 until the episode ends, then the chosen terminal reward (going_concern | cash | equity),
in thousands of dollars. Observations: "text" (compact text), "dict" (structured)."""
from __future__ import annotations

from .env import GoingConcernEnv
from .observe import render_compact
from .rewards import cash_reward, equity_reward, going_concern


class GoingConcernGym:
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, tier: int = 1, reward: str = "going_concern", obs_format: str = "text",
                 gc_rollouts: int = 16, gc_weeks: int = 13):
        self.tier, self.reward, self.obs_format = tier, reward, obs_format
        self.gc_rollouts, self.gc_weeks = gc_rollouts, gc_weeks
        self._env = GoingConcernEnv(record=True)

    def _obs(self):
        o = self._env.observe()
        return render_compact(o) if self.obs_format == "text" else o

    def reset(self, seed: int | None = None, options: dict | None = None):
        tier = (options or {}).get("tier", self.tier)
        self._env.reset(seed=seed or 0, tier=tier)
        return self._obs(), {"seed": seed or 0, "tier": tier}

    def step(self, action):
        _, done, info = self._env.step(action)
        reward = 0.0
        if done:
            sim = self._env.sim
            reward = {"cash": cash_reward, "equity": equity_reward}.get(self.reward, None)
            reward = (reward(sim) if reward else going_concern(sim, n=self.gc_rollouts, k_weeks=self.gc_weeks)) / 1000
        terminated = done and (self._env.sim.bankrupt or self._env.violation is not None)
        truncated = done and not terminated
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return render_compact(self._env.observe())

    def close(self):
        pass
