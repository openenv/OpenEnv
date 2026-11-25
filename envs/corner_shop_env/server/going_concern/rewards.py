"""Rewards.

cash_reward(sim)       cash at episode end (the gameable baseline)
equity_reward(sim)     book equity at episode end (second baseline; gameable by stuffing)
going_concern(sim)     freeze the firm, run N Monte Carlo rollouts of scripted reference operators
                       for K more weeks inside the simulator, mark the firm to liquidation at the end
                       (cash + inventory at the liquidator's price + prepaid - all liabilities),
                       discount, and average. Pure simulator ground truth: no forecasts, nothing the
                       agent can talk its way into.

All values are in dollars."""
from __future__ import annotations

import statistics

from .engine import Sim
from .ledger import InvariantViolation
from .observe import observe
from .operators import ReferenceOperator
from .rng import make_rng

ENSEMBLE_SPECS = {
    "balanced": dict(),
    "lean": dict(safety_weeks=0.5, cash_buffer_weeks=0.7, ema=0.5),
    "cautious": dict(safety_weeks=1.5, cash_buffer_weeks=2.0, ema=0.25),
    "growth": dict(safety_weeks=1.2, accept_margin=1.1, staff_slack=1.2),
}


def make_operator(name: str) -> ReferenceOperator:
    return ReferenceOperator(name=name, **ENSEMBLE_SPECS[name])


def liquidation_mark(sim: Sim) -> int:
    """Cents a creditor would see if the firm stopped now: cash + inventory at the liquidator's
    price + prepaid deposits (refundable at most) - every liability."""
    inv = sum(int(round(st.value * sim.cfg.sku(k).liquidation_frac)) for k, st in sim.skus.items())
    return sim.cash + inv + sim.prepaid_total() - sim.liabilities()


def bankruptcy_value(sim: Sim) -> int:
    """A bankrupt firm is worth its liquidation mark, less the bill it failed to pay and
    administration costs (two weeks of payroll and rent). Default is never a free exit."""
    admin = 2 * (sim.headcount * sim.cfg.staff.wage + sim.cfg.rent)
    return liquidation_mark(sim) - sim.bankrupt_unpaid - admin


def market_anchor(sim: Sim) -> dict[str, float]:
    """Shelf price (dollars) the continuation operators use: the firm's original markup from its
    config applied to its main supplier's current cost. Independent of anything the agent priced,
    so final-week price moves can't leak into the score (H6), and margins survive cost inflation."""
    out = {}
    for s in sim.cfg.skus:
        main = next((c for c in sim.cfg.suppliers if c.sku == s.id), None)  # the markup was set on this one
        cost = sim.sup[main.id].unit_cost if main else s.start_unit_cost
        out[s.id] = cost * (s.start_price / s.start_unit_cost) / 100
    return out


def cash_reward(sim: Sim) -> float:
    return sim.cash / 100


def equity_reward(sim: Sim) -> float:
    return sim.ledger.equity() / 100


def _rollout(sim: Sim, op_name: str, k_weeks: int, tag: tuple) -> tuple[float, bool, int]:
    c = sim.clone()
    c.reseed(*tag)
    c.done = False
    c.horizon = c.week + k_weeks
    op = make_operator(op_name)
    op.anchor = market_anchor(c)  # the agent's final shelf prices must not carry into the score
    start = c.week
    while not c.done:
        from .actions import parse_action
        c.step(parse_action(op.act(observe(c))))
    return (bankruptcy_value(c) if c.bankrupt else liquidation_mark(c)), c.bankrupt, c.week - start


def going_concern(sim: Sim, n: int = 16, k_weeks: int = 13, operators=tuple(ENSEMBLE_SPECS),
                  discount_apr: float = 0.10, detail: bool = False):
    """Going-concern value in dollars. If the firm is already bankrupt, its value is its
    liquidation mark now (usually negative)."""
    if sim.bankrupt:
        v = bankruptcy_value(sim) / 100
        return {"value": v, "bankrupt_at_freeze": True, "rollouts": []} if detail else v
    weekly = (1 + discount_apr) ** (1 / 52)
    pick = make_rng(sim.cfg.seed, sim.cfg.tier, sim.cfg.version, "gc-operator-pick", sim.week)
    rolls = []
    for i in range(n):
        op_name = operators[pick.randrange(len(operators))]
        try:
            mark, bankrupt, weeks = _rollout(sim, op_name, k_weeks, ("gc", sim.week, i))
        except InvariantViolation as e:  # a broken continuation is a bug, never a score
            raise InvariantViolation(f"in going-concern rollout {i}: {e}") from e
        rolls.append({"operator": op_name, "value": mark / 100 / weekly ** weeks, "bankrupt": bankrupt})
    value = statistics.fmean(r["value"] for r in rolls)
    if not detail:
        return value
    by_op: dict[str, list[float]] = {}
    for r in rolls:
        by_op.setdefault(r["operator"], []).append(r["value"])
    return {
        "value": value, "bankrupt_at_freeze": False,
        "std": statistics.pstdev(r["value"] for r in rolls),
        "rollout_bankrupt_rate": sum(r["bankrupt"] for r in rolls) / n,
        "by_operator": {k: statistics.fmean(v) for k, v in sorted(by_op.items())},
        "rollouts": rolls,
    }


def operator_spread(sim: Sim, n_per_op: int = 8, k_weeks: int = 13, discount_apr: float = 0.10) -> dict:
    """Score the same frozen firm with each reference operator separately, to show the reward
    does not hinge on one operator's quirks."""
    out = {}
    for name in ENSEMBLE_SPECS:
        out[name] = going_concern(sim, n=n_per_op, k_weeks=k_weeks, operators=(name,), discount_apr=discount_apr)
    vals = list(out.values())
    return {"by_operator": out, "range": max(vals) - min(vals), "mean": statistics.fmean(vals)}
