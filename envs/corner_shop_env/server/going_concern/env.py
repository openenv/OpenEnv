"""Environment API.

    env = GoingConcernEnv()
    obs = env.reset(seed=7, tier=1)          # or env.reset(config=FirmConfig(...))
    while not done:
        obs, done, info = env.step(action)   # action: dict or JSON string
    env.log                                  # full episode log (replayable, viewer input)

Every episode is reproducible from (seed, tier, version) plus the action sequence."""
from __future__ import annotations

import json

from .actions import parse_action
from .config import FirmConfig
from .engine import Sim
from .ledger import InvariantViolation
from .observe import d, observe, render_text
from .version import ENGINE_VERSION


def snapshot(sim: Sim) -> dict:
    L = sim.ledger.bal
    return {
        "week": sim.week, "cash": sim.cash, "debt": sim.credit.drawn, "ap": sim.ap_total(),
        "inventory_value": sim.inventory_value(), "prepaid": sim.prepaid_total(),
        "equity": sim.ledger.equity(), "headcount": sim.headcount, "goodwill": round(sim.goodwill, 4),
        "on_hand": {k: s.on_hand for k, s in sim.skus.items()},
        "prices": {k: s.price for k, s in sim.skus.items()},
        "revenue_total": L["sales_retail"] + L["sales_bulk"] + L["liquidation_proceeds"],
        "frozen": sim.credit.frozen, "in_breach": sim.credit.in_breach,
        "bankrupt": sim.bankrupt,
    }


class GoingConcernEnv:
    def __init__(self, record: bool = True):
        self.record = record
        self.sim: Sim | None = None
        self.log: dict = {}
        self.violation: str | None = None

    def reset(self, seed: int = 0, tier: int = 1, config: FirmConfig | None = None, weeks: int | None = None) -> dict:
        if config is None:
            raise ValueError("Corner Shop: pass config=load_firm(i), i in 1..10")
        cfg = config
        self.sim = Sim(cfg)
        self.violation = None
        self._last_errors: list[str] = []
        self.log = {"engine_version": ENGINE_VERSION, "seed": cfg.seed, "tier": cfg.tier,
                    "config": cfg.to_dict(), "initial": snapshot(self.sim), "steps": []}
        return self.observe()

    def observe(self) -> dict:
        return observe(self.sim, self._last_errors)

    def observe_text(self) -> str:
        return render_text(self.observe())

    @property
    def done(self) -> bool:
        return self.sim is None or self.sim.done or self.violation is not None

    def step(self, action) -> tuple[dict, bool, dict]:
        if self.done:
            raise RuntimeError("call reset() first")
        act = parse_action(action)
        try:
            info = self.sim.step(act)
        except Exception as e:  # InvariantViolation, or any engine bug: stop, log, never crash the caller
            self.violation = f"{type(e).__name__}: {e}"
            # a crashed episode is scored like a bankruptcy (no continuation): crashing never pays
            self.sim.bankrupt, self.sim.done = True, True
            self.sim.bankrupt_reason = f"run stopped: {self.violation}"
            info = {"invariant_violation": self.violation, "errors": act.get("_errors", [])}
            if self.record:
                self.log["violation"] = self.violation
            return self.observe(), True, info
        self._last_errors = info["errors"]
        if self.record:
            self.log["steps"].append({
                "action": {k: v for k, v in act.items() if k != "_errors" and v not in ([], {}, 0)},
                "errors": info["errors"], "events": info["events"],
                "flows": info["flows"], "state": snapshot(self.sim),
            })
        if self.sim.done and self.record:
            self.log["final"] = snapshot(self.sim)
            self.log["bankrupt_reason"] = self.sim.bankrupt_reason
        return self.observe(), self.sim.done, info

    def save_log(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.log, f, default=str)




__all__ = ["GoingConcernEnv", "run_episode", "snapshot", "d"]


def load_firm(i: int) -> FirmConfig:
    """One of the 100 Corner Shop firms (1..100)."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "firms", f"corner_shop_{i:02d}.json")
    return FirmConfig.from_json(open(path).read())


def run_episode(agent, firm: int, record: bool = False):
    env = GoingConcernEnv(record=record)
    obs = env.reset(config=load_firm(firm))
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    while not done:
        obs, done, info = env.step(agent.act(obs))
    return env
