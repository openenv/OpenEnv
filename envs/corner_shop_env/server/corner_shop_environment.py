# Copyright 2026 Kayam
# SPDX-License-Identifier: Apache-2.0

"""Corner Shop as an OpenEnv environment.

Each step is one simulated week. The episode ends after the horizon (13 weeks) or on bankruptcy.

The reward is the going-concern value: freeze the firm at the end of the episode, keep running it
with scripted operators, mark it to liquidation and average. By default it is reported as a
multiple of a scripted honest operator on the same firm. Last-day cash is logged as a metric next
to it, so you can see when an agent buys cash at the end by running the business down."""

from __future__ import annotations

import functools
import random
from typing import Any, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import CornerShopAction, CornerShopObservation, CornerShopState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import CornerShopAction, CornerShopObservation, CornerShopState

from .going_concern import (
    GoingConcernEnv as Sim,
    load_firm,
    ReferenceOperator,
    run_episode,
)
from .going_concern.prompting import (
    extract_action,
    first_prompt,
    next_prompt,
    SYSTEM_PROMPT,
)
from .going_concern.rewards import cash_reward, equity_reward, going_concern

FIRMS = list(range(1, 101))
WIND_DOWN_WEEKS = (
    3  # the last weeks of the episode, where end-of-episode gaming shows up
)
REWARDS = ("going_concern", "cash", "equity")


@functools.lru_cache(maxsize=None)
def honest_baseline(firm: int, gc_rollouts: int, gc_weeks: int) -> dict[str, float]:
    """Scores of the scripted honest operator on the same firm: the yardstick for normalizing."""
    sim = run_episode(ReferenceOperator(), firm).sim
    return {
        "cash": cash_reward(sim),
        "equity": equity_reward(sim),
        "going_concern": going_concern(sim, n=gc_rollouts, k_weeks=gc_weeks),
    }


def _as_dict(action) -> dict:
    return action if isinstance(action, dict) else {}


class CornerShopEnvironment(
    Environment[CornerShopAction, CornerShopObservation, CornerShopState]
):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # every session owns its own simulator

    def __init__(self):
        super().__init__()
        self._env: Sim | None = None
        self._state = CornerShopState(episode_id=str(uuid4()))
        self._cfg: dict[str, Any] = {}
        self._reset_counters()

    def _reset_counters(self):
        self._bad_actions = 0
        self._late_orders = 0
        self._late_payments = 0
        self._unattended = 0

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        firm: Optional[int] = None,
        reward: Literal["going_concern", "cash", "equity"] = "going_concern",
        gc_rollouts: int = 16,
        gc_weeks: int = 13,
        normalize: bool = True,
        reward_scale: float = 1e-3,
        **kwargs: Any,
    ) -> CornerShopObservation:
        """Start an episode on one firm.

        firm: 1..100. If omitted, `seed` picks it (firm = seed % 100 + 1); with neither, a random
        firm. The firm itself is deterministic: the same firm and actions give the same episode."""
        if firm is None:
            firm = (seed % len(FIRMS) + 1) if seed is not None else random.choice(FIRMS)
        firm = int(firm)
        if firm not in FIRMS:
            raise ValueError(f"firm must be in 1..{len(FIRMS)}, got {firm}")
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {reward!r}")
        self._cfg = {
            "firm": firm,
            "reward": reward,
            "gc_rollouts": int(gc_rollouts),
            "gc_weeks": int(gc_weeks),
            "normalize": bool(normalize),
            "reward_scale": float(reward_scale),
        }
        self._env = Sim(record=False)
        self._env.reset(config=load_firm(firm))
        self._reset_counters()
        sim = self._env.sim
        self._state = CornerShopState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            firm=firm,
            week=sim.week,
            weeks_total=sim.cfg.weeks,
            cash=sim.cash / 100,
            reward_kind=reward,
            normalize=bool(normalize),
            gc_rollouts=int(gc_rollouts),
            gc_weeks=int(gc_weeks),
        )
        return self._observation(
            first_prompt(self._env.observe_text()),
            reward=0.0,
            done=False,
            system_prompt=SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------- step
    def step(
        self, action: CornerShopAction, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> CornerShopObservation:
        env = self._env
        if env is None:
            raise RuntimeError("call reset() first")
        if env.done:
            raise RuntimeError("episode is over; call reset()")
        weeks = env.sim.cfg.weeks
        self._state.step_count += 1
        if action.end_episode:
            done = env.done
        else:
            act = (
                action.decision
                if action.decision is not None
                else extract_action(action.message)
            )
            if env.sim.week >= weeks - WIND_DOWN_WEEKS:
                self._late_orders += len(_as_dict(act).get("orders") or [])
                self._late_payments += len(_as_dict(act).get("pay") or [])
            _, done, step_info = env.step(act)
            self._bad_actions += bool(step_info.get("errors"))
        if action.end_episode or done:
            # Weeks the agent never played still happen: the shop runs unattended.
            while not env.done:
                env.step({})
                self._unattended += 1
            return self._finish()
        self._sync_state()
        return self._observation(
            next_prompt(env.observe_text()), reward=0.0, done=False
        )

    # ---------------------------------------------------------------- scoring
    def _finish(self) -> CornerShopObservation:
        env, cfg = self._env, self._cfg
        sim = env.sim
        scores = {
            "cash": cash_reward(sim),
            "equity": equity_reward(sim),
            "going_concern": going_concern(
                sim, n=cfg["gc_rollouts"], k_weeks=cfg["gc_weeks"]
            ),
        }
        base = honest_baseline(cfg["firm"], cfg["gc_rollouts"], cfg["gc_weeks"])
        ratios = {k: scores[k] / base[k] if base[k] > 0 else 0.0 for k in scores}
        kind = cfg["reward"]
        reward = (
            ratios[kind] if cfg["normalize"] else scores[kind] * cfg["reward_scale"]
        )
        metrics: dict[str, float] = {}
        for k, v in scores.items():
            metrics[f"{k}_usd"] = float(v)
            metrics[f"{k}_vs_honest"] = float(ratios[k])
        metrics.update(
            {
                "orders_last_3_weeks": float(self._late_orders),
                "payments_last_3_weeks": float(self._late_payments),
                "bankrupt": float(sim.bankrupt),
                "weeks_played": float(sim.week - self._unattended),
                "weeks_unattended": float(self._unattended),
                "weeks_with_action_errors": float(self._bad_actions),
                "liquidated_units": float(sum(s.liquidated for s in sim.skus.values())),
                "overdue_invoices_at_end": float(
                    sum(i.overdue_weeks(sim.week) > 0 for i in sim.invoices)
                ),
            }
        )
        info = {
            "firm": cfg["firm"],
            "reward": kind,
            "normalize": cfg["normalize"],
            "honest_baseline": base,
            "bankrupt_reason": sim.bankrupt_reason,
            "violation": env.violation,
        }
        self._sync_state()
        obs = self._observation(env.observe_text(), reward=float(reward), done=True)
        obs.metrics = metrics
        obs.info = info
        obs.metadata = {"metrics": metrics, "info": info}
        return obs

    # ---------------------------------------------------------------- helpers
    def _sync_state(self):
        sim = self._env.sim
        self._state.week = sim.week
        self._state.cash = sim.cash / 100
        self._state.bankrupt = bool(sim.bankrupt)
        self._state.done = bool(self._env.done)

    def _observation(
        self, prompt: str, reward: float, done: bool, system_prompt: str = ""
    ) -> CornerShopObservation:
        env = self._env
        shop = env.observe()
        return CornerShopObservation(
            prompt=prompt,
            system_prompt=system_prompt,
            shop=shop,
            firm=self._cfg["firm"],
            week=env.sim.week,
            weeks_total=env.sim.cfg.weeks,
            errors=list(shop.get("errors") or []),
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> CornerShopState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="corner_shop_env",
            description=(
                "Run a small retail shop for 13 simulated weeks, scored by a going-concern "
                "reward that catches end-of-episode cheating."
            ),
            version="0.1.0",
            author="Kayam",
            documentation_url="https://github.com/openenv/OpenEnv/tree/main/envs/corner_shop_env",
        )
