"""Deterministic chronological replay. No OpenEnv imports or network access."""

import csv
import math
import random
from pathlib import Path

MODELS = ("ChatGPT", "GPT-4")
UNIT_COST = {"ChatGPT": 0.002, "GPT-4": 0.0375}
SHORTFALL_MULTIPLIER = 4.0
RESERVATION_CAP = 1_000_000.0  # Numerical action bound, not hardware capacity.
REWARD_SCALE = 1000.0
HISTORY_DAYS = 14
EPISODE_DAYS = 14
DATA = Path(__file__).resolve().parent / "data/hourly_demand.csv"


def load_trace(path=DATA):
    trace = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            day, hour = int(row["log_day"]), int(row["hour"])
            model, demand = row["model"], int(row["total_tokens"])
            if model not in MODELS or not 0 <= hour < 24 or demand < 0:
                raise ValueError("Invalid trace row")
            bucket = trace.setdefault(day, {m: [None] * 24 for m in MODELS})
            if bucket[model][hour] is not None:
                raise ValueError("Duplicate trace row")
            bucket[model][hour] = demand
    if not trace or any(v is None for d in trace.values() for xs in d.values() for v in xs):
        raise ValueError("Incomplete hourly trace")
    # Under this recorded bound, 48 hourly costs / 1000 are strictly below 2.
    if max(v for d in trace.values() for xs in d.values() for v in xs) > 500_000:
        raise ValueError("Trace exceeds declared reward-bound assumptions")
    return trace


def eligible_starts(trace, split):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    target_ranges = ((14, 89), (239, 304)) if split == "train" else ((90, 120), (305, 334))
    return [
        start
        for a, b in target_ranges
        for start in range(a, b - EPISODE_DAYS + 2)
        if all(day in trace for day in range(start - HISTORY_DAYS, start + EPISODE_DAYS))
    ]


def validate_reservations(reservations):
    if not isinstance(reservations, dict) or set(reservations) != set(MODELS):
        raise ValueError("Provide exactly ChatGPT and GPT-4")
    result = {}
    for model in MODELS:
        xs = reservations[model]
        if not isinstance(xs, list) or len(xs) != 24:
            raise ValueError("Each deployment requires exactly 24 hourly reservations")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or not 0 <= x <= RESERVATION_CAP
            for x in xs
        ):
            raise ValueError("Reservations must be finite numbers between 0 and 1000000")
        result[model] = [float(x) for x in xs]
    return result


def modeled_cost(reservations, demand):
    return sum(
        UNIT_COST[m] * (r + SHORTFALL_MULTIPLIER * max(d - r, 0)) / 1000
        for m in MODELS
        for r, d in zip(reservations[m], demand[m], strict=True)
    )


class Replay:
    def __init__(self, split="train"):
        self.trace = load_trace()
        self.split = split
        self.starts = eligible_starts(self.trace, split)
        self.active = False
        self.turn = 0

    def reset(self, seed=None):
        self.start = random.Random(seed).choice(self.starts)
        self.turn = 0
        self.total_cost = 0.0
        self.previous = None
        self.active = True
        return self.observe()

    @property
    def done(self):
        return self.active and self.turn == EPISODE_DAYS

    def observe(self):
        if not self.active:
            raise RuntimeError("Environment has not been initialized by orchestration")
        day = self.start + self.turn
        # Copy nested structures; consumers must not mutate replay data.
        return {
            "episode_day": self.turn,
            "episode_days": EPISODE_DAYS,
            "next_log_day": None if self.done else day,
            "done": self.done,
            "split": self.split,
            "history": [
                {"log_day": d, "tokens": {m: list(self.trace[d][m]) for m in MODELS}}
                for d in range(day - HISTORY_DAYS, day)
            ],
            "unit_cost_per_1000_tokens": dict(UNIT_COST),
            "shortfall_multiplier": SHORTFALL_MULTIPLIER,
            "reservation_cap": RESERVATION_CAP,
            "reward_scale": REWARD_SCALE,
            "cumulative_modeled_cost": self.total_cost,
            "previous": None
            if self.previous is None
            else {
                **self.previous,
                "realized_tokens": {
                    m: list(xs) for m, xs in self.previous["realized_tokens"].items()
                },
            },
        }

    def reserve(self, reservations):
        if not self.active or self.done:
            raise RuntimeError("No active decision day")
        plan = validate_reservations(reservations)  # Validate before any mutation.
        day = self.start + self.turn
        demand = self.trace[day]
        cost = modeled_cost(plan, demand)
        self.total_cost += cost
        self.turn += 1
        self.previous = {
            "log_day": day,
            "realized_tokens": demand,
            "modeled_cost": cost,
            "reward": -cost / REWARD_SCALE,
        }
        return self.observe()
